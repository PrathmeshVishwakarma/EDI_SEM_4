"""
Motor Frequency Analyzer — Flask Backend
ESP32 + ADXL345 → Real-time FFT + ML anomaly detection
"""

# ── eventlet MUST be monkey-patched before ANY other import ──
# Importing serial/threading before this causes EINVAL (errno 22)
# on macOS when pyserial's tcsetattr runs inside the patched I/O layer.
import eventlet
eventlet.monkey_patch()
import eventlet.tpool          # must import after monkey_patch; not auto-loaded as attribute

import os
import sys
import time
import json
import pickle
import threading
import yaml
import serial
import numpy as np
from numpy import polyfit
from scipy import signal
from scipy.fft import rfft, rfftfreq

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from feature_pipeline import extract_features, dominant_frequency, build_fft_payload, order_domain
from ml_models import (
    CompanyClassifier, EnsembleAnomalyModel,
    GMMAnomalyDetector, AutoencoderAnomalyDetector,
    PerCompanyModelRegistry,
    TransferLearningAdapter, TORCH_AVAILABLE
)
from rul_model import RULPredictor
from motor_classifier import MotorClassifier

# ─────────────────────────────────────────────────────────────
#  App Setup
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'motor_analyzer_secret_2024'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
COMPANY_MODELS_DIR = os.path.join(MODELS_DIR, 'companies')
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(COMPANY_MODELS_DIR, exist_ok=True)

# Load config
try:
    with open(CONFIG_PATH, 'r') as f:
        APP_CONFIG = yaml.safe_load(f) or {}
except Exception:
    APP_CONFIG = {}

COMPANY_CLASSES = APP_CONFIG.get('companies', {}).get('classes', ['Unknown'])
FEATURE_DIM = APP_CONFIG.get('feature', {}).get('dimension', 28)
NUM_COMPANIES = len(COMPANY_CLASSES)
SAMPLE_RATE = APP_CONFIG.get('streaming', {}).get('sample_rate', 100)
WINDOW_SIZE = APP_CONFIG.get('streaming', {}).get('window_size', 128)

# Company-aware model registry
company_registry = PerCompanyModelRegistry(models_dir=COMPANY_MODELS_DIR)

# RUL degradation predictor (optional)
rul_predictor = RULPredictor.load()

# Motor characteristics classifier (optional)
motor_classifier = MotorClassifier.load()

# ─────────────────────────────────────────────────────────────
#  Shared Application State
# ─────────────────────────────────────────────────────────────
state = {
    'serial_connected': False,
    'serial_port': '/dev/cu.usbserial-0001',
    'baud_rate': 115200,
    'training': False,
    'evaluating': False,
    'train_duration': 20,
    'train_start': 0,
    'baseline': None,
    'baseline_samples': [],
    'baseline_ready': False,
    'training_features': [],
    'active_model': None,
    'active_model_name': None,
    'is_anomaly': False,
    'anomaly_score': 0.0,
    'dominant_freq': 0.0,
    'current_model_obj': None,
    'status': 'IDLE — Connect a serial port to begin',
    'status_level': 'info',   # info | warning | error | success | training
    'company_idx': 0,
    'company_name': 'Unknown',
    'company_confidence': 0.0,
    'company_identified': False,
    'company_fingerprint_buffer': [],

    # ── Order tracking ──
    'rpm': 0.0,
    'order_spectrum': {},

    # ── Degradation trending ──
    'rms_history': [],              # (timestamp, rms) pairs for trending
    'rms_slope': 0.0,               # slope of last 30min RMS (change/hour)
    'trend_alert': False,           # True if RMS trending upward significantly

    # ── Auto-stop (sustained anomaly) ──
    'anomaly_consecutive': 0,       # consecutive anomaly windows
    'anomaly_auto_stopped': False,  # True if motor was auto-stopped
    'auto_stop_threshold': 30,      # windows before auto-stop (30 × ~1.28s ≈ 38s)

    # ── Auto-encoder anomaly model ──
    'autoencoder_model': None,
    'use_autoencoder': False,

    # ── Online retraining buffer ──
    'retrain_buffer': [],            # accumulated feature vectors for incremental refit
    'retrain_count': 0,              # number of retrain operations performed
    'retrain_threshold': 100,        # buffer size threshold for auto-retrain
}

ser = None
ser_lock = threading.Lock()

# Rolling raw sample buffers
RAW_WINDOW = WINDOW_SIZE
raw_buffer = []
plot_buffer = []

# ─────────────────────────────────────────────────────────────
#  ML Model Builder
# ─────────────────────────────────────────────────────────────
def build_model(model_type: str = 'ensemble'):
    if model_type == 'gmm':
        return GMMAnomalyDetector()
    if model_type == 'autoencoder':
        return AutoencoderAnomalyDetector()
    return EnsembleAnomalyModel()


# ─────────────────────────────────────────────────────────────
#  Serial Reader Thread
# ─────────────────────────────────────────────────────────────
def _drain_serial(s):
    """Read all available lines from serial in a real OS thread.
    Called via tpool.execute to prevent eventlet greenthreads from
    blocking while pyserial waits on a 1-second timeout read."""
    lines = []
    try:
        waiting = s.in_waiting
        for _ in range(min(waiting, 64)):
            raw = s.readline()
            if raw:
                lines.append(raw)
    except Exception:
        return None
    return lines


def _handle_edge_features(feature_line):
    """Handle pre-extracted feature vectors from ESP32 edge device.
    Feature format: rms,p2p,variance,crest,zcr,band0,...,band7 (13 values)
    Maps into our 28-dim feature space with zeros for missing dims.
    """
    try:
        parts = [float(x) for x in feature_line.split(',')]
        if len(parts) < 13:
            return
        rms, p2p, variance, crest, zcr = parts[0:5]
        bands = parts[5:13]
        feats = [rms, p2p, variance, 0.0, 0.0, crest, 0.0, zcr] + bands + [0.0] * 10
        _process_feature_vector(feats)
    except (ValueError, IndexError):
        pass


def _process_feature_vector(feats):
    """Process a 28-dim feature vector through ML pipeline."""
    global plot_buffer
    if len(feats) < 28:
        return
    if state['current_model_obj'] is not None:
        is_anomaly, score = state['current_model_obj'].predict_score(feats)
        state['anomaly_score'] = float(score)
        state['is_anomaly'] = bool(is_anomaly)
        if is_anomaly:
            state['anomaly_consecutive'] += 1
            if state['anomaly_consecutive'] >= state['auto_stop_threshold']:
                state['auto_stopped'] = True
        else:
            state['anomaly_consecutive'] = max(0, state['anomaly_consecutive'] - 1)
    state['feature_count'] = state.get('feature_count', 0) + 1


def serial_reader():
    global ser, raw_buffer, plot_buffer

    DOWNSAMPLE = 3
    plot_counter = 0
    feature_counter = 0
    FEATURE_WINDOW = WINDOW_SIZE

    while True:
        with ser_lock:
            active_ser = ser

        if active_ser is None or not active_ser.is_open:
            eventlet.sleep(0.5)
            continue

        try:
            raw_lines = eventlet.tpool.execute(_drain_serial, active_ser)
            if raw_lines is None:
                eventlet.sleep(0.1)
                continue
            if not raw_lines:
                eventlet.sleep(0.01)
                continue

            for raw_line in raw_lines:
                line = raw_line.decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                # Lines starting with '#' are ESP32 status/debug messages
                if line.startswith('#'):
                    socketio.emit('esp32_log', {'msg': line[1:].strip()})
                    continue
                # Lines starting with 'F:' are pre-extracted features from ESP32
                if line.startswith('F:'):
                    _handle_edge_features(line[2:])
                    continue
                try:
                    val = float(line)
                except ValueError:
                    continue

                # ── Baseline calibration ──
                if not state['baseline_ready']:
                    state['baseline_samples'].append(val)
                    if len(state['baseline_samples']) >= 80:
                        state['baseline'] = float(np.mean(state['baseline_samples']))
                        state['baseline_ready'] = True
                        state['status'] = 'BASELINE LOCKED ✓ — Ready to train or evaluate'
                        state['status_level'] = 'success'
                        socketio.emit('status_update', {
                            'status': state['status'],
                            'level': state['status_level'],
                            'baseline': state['baseline'],
                        })
                    continue

                baseline = state['baseline']

                # ── Raw buffer for FFT & features ──
                raw_buffer.append(val)
                if len(raw_buffer) > RAW_WINDOW * 2:
                    raw_buffer = raw_buffer[-RAW_WINDOW:]

                # ── Downsampled waveform for oscilloscope ──
                plot_counter += 1
                if plot_counter >= DOWNSAMPLE:
                    plot_buffer.append(round(val - baseline, 4))
                    if len(plot_buffer) > 600:
                        plot_buffer = plot_buffer[-600:]
                    plot_counter = 0

                    state['wf_emit_ctr'] = state.get('wf_emit_ctr', 0) + 1
                    if state['wf_emit_ctr'] >= 2:
                        state['wf_emit_ctr'] = 0
                        socketio.emit('waveform_update', {'waveform': [float(x) for x in plot_buffer[-200:]]})

                # ── Feature extraction every FEATURE_WINDOW samples ──
                feature_counter += 1
                if feature_counter >= FEATURE_WINDOW and len(raw_buffer) >= FEATURE_WINDOW:
                    feature_counter = 0
                    chunk = raw_buffer[-FEATURE_WINDOW:]
                    feats = extract_features(chunk, baseline, SAMPLE_RATE)
                    dom_freq = dominant_frequency(chunk, baseline, SAMPLE_RATE)

                    fft_payload = build_fft_payload(chunk, baseline, SAMPLE_RATE)
                    state['dominant_freq'] = dom_freq

                    # ── Order tracking (RPM normalization) ──
                    rpm = dom_freq * 60.0  # convert Hz to RPM
                    state['rpm'] = round(rpm, 1)
                    arr = np.array(chunk, dtype=np.float64) - baseline
                    win = signal.windows.hann(len(arr))
                    raw_spec = np.abs(rfft(arr * win))
                    raw_freqs = rfftfreq(len(arr), d=1.0 / SAMPLE_RATE)
                    if rpm > 1:
                        state['order_spectrum'] = order_domain(raw_spec, raw_freqs, rpm)

                    # ── Company fingerprinting (first N samples) ──
                    if not state['company_identified']:
                        state['company_fingerprint_buffer'].append(feats)
                        fp_needed = APP_CONFIG.get('companies', {}).get('fingerprint_samples', 512) // FEATURE_WINDOW
                        if len(state['company_fingerprint_buffer']) >= fp_needed:
                            fp_arr = np.array(state['company_fingerprint_buffer'])
                            avg_feat = np.mean(fp_arr, axis=0)
                            idx, name, conf = company_registry.identify_company(avg_feat)
                            state['company_idx'] = idx
                            state['company_name'] = name
                            state['company_confidence'] = conf
                            state['company_identified'] = True
                            state['status'] = f'COMPANY IDENTIFIED: {name} ({conf:.0%} confidence)'
                            state['status_level'] = 'success' if conf > 0.5 else 'warning'
                            socketio.emit('company_identified', {
                                'company': name,
                                'confidence': round(conf, 3),
                                'idx': idx,
                            })
                            # Load company-specific anomaly model if available
                            company_model = company_registry.get_anomaly_model(name)
                            if company_model is not None:
                                state['current_model_obj'] = company_model
                                state['active_model_name'] = f"{name}_model"
                                state['evaluating'] = True
                                state['status'] += f' + loaded {name} anomaly model'
                            socketio.emit('status_update', {
                                'status': state['status'],
                                'level': state['status_level'],
                            })

                    is_anomaly = False
                    score = 0.0

                    if state['training']:
                        state['training_features'].append(feats)
                        elapsed = time.time() - state['train_start']
                        progress = min(elapsed / state['train_duration'], 1.0)
                        if elapsed >= state['train_duration']:
                            _finish_training_internal()
                        else:
                            socketio.emit('train_progress', {
                                'elapsed': round(elapsed, 1),
                                'total': state['train_duration'],
                                'progress': round(progress * 100, 1),
                                'samples': len(state['training_features']),
                            })

                    elif state['evaluating'] and state['current_model_obj'] is not None:
                        m = state['current_model_obj']
                        is_anomaly, score = m.predict_score(feats)
                        state['is_anomaly'] = bool(is_anomaly)
                        state['anomaly_score'] = float(score)

                    # ── Degradation trending (RMS over time) ──
                    rms_val = float(np.sqrt(np.mean(arr ** 2)))
                    now = time.time()
                    state['rms_history'].append((now, rms_val))
                    # Keep last 2 hours of data
                    cutoff = now - 7200
                    state['rms_history'] = [(t, v) for t, v in state['rms_history'] if t > cutoff]
                    # Compute slope over last 30 minutes if enough data
                    recent = [(t, v) for t, v in state['rms_history'] if t > now - 1800]
                    if len(recent) >= 10:
                        t_vals = np.array([r[0] for r in recent])
                        r_vals = np.array([r[1] for r in recent])
                        if np.std(t_vals) > 1:
                            from numpy import polyfit
                            slope, _ = polyfit(t_vals - t_vals[0], r_vals, 1)
                            state['rms_slope'] = float(round(slope * 3600, 6))  # numpy→Python float
                            state['trend_alert'] = bool(state['rms_slope'] > 0.0005)  # numpy→Python bool



                    # Emit real-time data to all clients
                    socketio.emit('sensor_data', {
                        'fft': fft_payload,
                        'dominant_freq': float(round(dom_freq, 2)),
                        'rpm': float(round(rpm, 1)),
                        'is_anomaly': bool(is_anomaly),
                        'anomaly_score': float(round(score * 100, 1)),
                        'rms': float(round(rms_val, 4)),
                        'rms_slope': float(state['rms_slope']),
                        'trend_alert': bool(state['trend_alert']),
                        'evaluating': bool(state['evaluating']),
                        'training': bool(state['training']),
                        'company': str(state['company_name']),
                        'company_identified': bool(state['company_identified']),
                    })

        except Exception as e:
            socketio.emit('status_update', {
                'status': f'Serial error: {e}',
                'level': 'error',
            })
            eventlet.sleep(0.1)

        eventlet.sleep(0.01)


def _finish_training_internal():
    """Called from serial thread when training window is complete."""
    if not state['training_features']:
        state['status'] = 'No data collected — retry training'
        state['status_level'] = 'error'
        state['training'] = False
        return

    pending_name = state.get('pending_model_name', 'model')

    # Train ensemble model (IF + SVM)
    model_obj = EnsembleAnomalyModel()
    model_obj.fit(state['training_features'])

    # Train auto-encoder model
    ae_model = AutoencoderAnomalyDetector()
    ae_model.fit(state['training_features'])
    state['autoencoder_model'] = ae_model
    state['use_autoencoder'] = True

    state['current_model_obj'] = model_obj
    state['active_model_name'] = pending_name
    state['training'] = False
    state['evaluating'] = True

    # Save as per-company model if company is identified
    company = state.get('company_name', 'Unknown')
    if company != 'Unknown':
        company_registry.train_company_model(company, state['training_features'])
        state['status'] = f'COMPANY MODEL "{company}/{pending_name}" TRAINED ✓'
    else:
        state['status'] = f'MODEL "{pending_name}" TRAINED ✓ — Inference active'
    state['status_level'] = 'success'

    socketio.emit('status_update', {
        'status': state['status'],
        'level': state['status_level'],
    })
    socketio.emit('train_complete', {'model_name': pending_name, 'company': company})


# ─────────────────────────────────────────────────────────────
#  REST API
# ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/vui')
def vui_dashboard():
    return render_template('dashboard-vui.html')


@app.route('/api/ports')
def list_ports():
    """List available serial ports."""
    import serial.tools.list_ports
    ports = [{'device': p.device, 'desc': p.description} for p in serial.tools.list_ports.comports()]
    return jsonify(ports)


@app.route('/api/connect', methods=['POST'])
def connect_serial():
    global ser
    data = request.json
    port = data.get('port', state['serial_port'])
    baud = int(data.get('baud', state['baud_rate']))

    # Close existing connection first
    with ser_lock:
        if ser and ser.is_open:
            ser.close()

    # Open serial in a REAL OS thread via tpool so eventlet's patched
    # file I/O doesn't interfere with pyserial's tcsetattr ioctl calls
    # (fixes errno 22 / EINVAL on macOS with monkey_patch active)
    def _open_serial():
        new_ser = serial.Serial(
            port,
            baud,
            timeout=1,          # use 1s timeout — 0.01 triggers EINVAL on macOS
            write_timeout=1,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        new_ser.reset_input_buffer()
        return new_ser

    try:
        new_ser = eventlet.tpool.execute(_open_serial)
        with ser_lock:
            ser = new_ser
        state['serial_connected'] = True
        state['serial_port'] = port
        state['baud_rate'] = baud
        state['baseline_ready'] = False
        state['baseline_samples'] = []
        state['baseline'] = None
        state['company_identified'] = False
        state['company_fingerprint_buffer'] = []
        state['status'] = 'CONNECTED — Calibrating baseline (hold motor still)…'
        state['status_level'] = 'info'
        return jsonify({'ok': True, 'port': port})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400


@app.route('/api/disconnect', methods=['POST'])
def disconnect_serial():
    global ser
    with ser_lock:
        if ser and ser.is_open:
            ser.close()
        ser = None
        state['serial_connected'] = False
        state['training'] = False
        state['evaluating'] = False
        state['status'] = 'DISCONNECTED'
        state['status_level'] = 'info'
    return jsonify({'ok': True})


@app.route('/api/train/start', methods=['POST'])
def start_training():
    data = request.json
    name = (data.get('name') or 'model').strip().replace(' ', '_')
    duration = int(data.get('duration', 20))

    if not state['baseline_ready']:
        return jsonify({'ok': False, 'error': 'Baseline not calibrated yet'}), 400

    state['training'] = True
    state['evaluating'] = False
    state['training_features'] = []
    state['train_start'] = time.time()
    state['train_duration'] = duration
    state['pending_model_name'] = name
    state['status'] = f'TRAINING "{name}" — {duration}s capture…'
    state['status_level'] = 'training'
    socketio.emit('status_update', {'status': state['status'], 'level': state['status_level']})
    return jsonify({'ok': True, 'name': name, 'duration': duration})


@app.route('/api/train/cancel', methods=['POST'])
def cancel_training():
    state['training'] = False
    state['status'] = 'Training cancelled'
    state['status_level'] = 'info'
    return jsonify({'ok': True})


@app.route('/api/model/save', methods=['POST'])
def save_model():
    data = request.json
    name = (data.get('name') or 'model').strip().replace(' ', '_')
    if state['current_model_obj'] is None:
        return jsonify({'ok': False, 'error': 'No model in memory'}), 400

    payload = {
        'model': state['current_model_obj'],
        'baseline': state['baseline'],
        'name': name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'samples': len(state.get('training_features', [])),
    }
    path = os.path.join(MODELS_DIR, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    return jsonify({'ok': True, 'path': path, 'name': name})


@app.route('/api/model/load', methods=['POST'])
def load_model():
    data = request.json
    name = data.get('name', '')
    path = os.path.join(MODELS_DIR, f'{name}.pkl')
    if not os.path.exists(path):
        return jsonify({'ok': False, 'error': 'Model file not found'}), 404
    with open(path, 'rb') as f:
        payload = pickle.load(f)

    state['current_model_obj'] = payload['model']
    state['active_model_name'] = payload['name']
    state['baseline'] = payload['baseline']
    state['baseline_ready'] = True
    state['evaluating'] = True
    state['training'] = False
    state['status'] = f'MODEL "{name}" LOADED ✓ — Inference active'
    state['status_level'] = 'success'
    socketio.emit('status_update', {'status': state['status'], 'level': state['status_level']})
    return jsonify({'ok': True, 'name': name, 'meta': {
        'timestamp': payload.get('timestamp', 'N/A'),
        'samples': payload.get('samples', 0),
        'baseline': round(payload['baseline'], 4),
    }})


@app.route('/api/model/list')
def list_models():
    models = []
    for f in os.listdir(MODELS_DIR):
        if f.endswith('.pkl'):
            path = os.path.join(MODELS_DIR, f)
            try:
                with open(path, 'rb') as fh:
                    p = pickle.load(fh)
                models.append({
                    'name': p.get('name', f[:-4]),
                    'filename': f[:-4],
                    'timestamp': p.get('timestamp', 'N/A'),
                    'samples': p.get('samples', 0),
                    'baseline': round(p.get('baseline', 0), 4),
                })
            except Exception:
                models.append({'name': f[:-4], 'filename': f[:-4], 'timestamp': 'N/A', 'samples': 0, 'baseline': 0})
    return jsonify(models)


@app.route('/api/model/delete', methods=['POST'])
def delete_model():
    data = request.json
    name = data.get('name', '').replace(' ', '_')
    path = os.path.join(MODELS_DIR, f'{name}.pkl')
    if os.path.exists(path):
        os.remove(path)
        return jsonify({'ok': True})
    return jsonify({'ok': False, 'error': 'Not found'}), 404


@app.route('/api/evaluate/stop', methods=['POST'])
def stop_evaluate():
    state['evaluating'] = False
    state['status'] = 'Evaluation stopped'
    state['status_level'] = 'info'
    return jsonify({'ok': True})




@app.route('/api/status')
def get_status():
    return jsonify({
        'connected': state['serial_connected'],
        'baseline_ready': state['baseline_ready'],
        'training': state['training'],
        'evaluating': state['evaluating'],
        'active_model': state['active_model_name'],
        'status': state['status'],
        'status_level': state['status_level'],
        'company_idx': state['company_idx'],
        'company_name': state['company_name'],
        'company_confidence': round(state['company_confidence'], 3),
        'company_identified': state['company_identified'],
        'feature_dim': FEATURE_DIM,
        'available_companies': COMPANY_CLASSES,
        'rpm': state['rpm'],
        'rms_slope': state['rms_slope'],
        'trend_alert': state['trend_alert'],
        'anomaly_auto_stopped': state['anomaly_auto_stopped'],
        'anomaly_consecutive': state['anomaly_consecutive'],
    })


# ─────────────────────────────────────────────────────────────
#  Company Awareness API
# ─────────────────────────────────────────────────────────────
@app.route('/api/company/status')
def get_company_status():
    """Get current company info and available per-company models."""
    available = []
    for f in os.listdir(COMPANY_MODELS_DIR):
        if f.endswith('.pkl'):
            available.append(f[:-4])
    return jsonify({
        'current_company': state['company_name'],
        'identified': state['company_identified'],
        'confidence': round(state['company_confidence'], 3),
        'available_companies': available,
        'registered_companies': company_registry.company_names,
    })


@app.route('/api/company/reidentify', methods=['POST'])
def reidentify_company():
    """Force re-identification of the connected machine."""
    state['company_identified'] = False
    state['company_name'] = 'Unknown'
    state['company_idx'] = 0
    state['company_confidence'] = 0.0
    state['company_fingerprint_buffer'] = []
    state['status'] = 'Re-identifying company…'
    state['status_level'] = 'info'
    socketio.emit('status_update', {'status': state['status'], 'level': state['status_level']})
    return jsonify({'ok': True})


@app.route('/api/company/models')
def list_company_models():
    """List all per-company anomaly models saved."""
    models = []
    for f in os.listdir(COMPANY_MODELS_DIR):
        if f.endswith('.pkl'):
            path = os.path.join(COMPANY_MODELS_DIR, f)
            mtime = os.path.getmtime(path)
            models.append({
                'company': f[:-4],
                'modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime)),
            })
    return jsonify(models)


@app.route('/api/company/classifier/status')
def classifier_status():
    """Check if a company classifier has been trained and loaded."""
    return jsonify({
        'trained': company_registry.company_classifier.trained,
        'num_companies': company_registry.company_classifier.num_companies,
        'company_names': company_registry.company_names,
        'feature_dim': company_registry.company_classifier.feature_dim,
    })


# ─────────────────────────────────────────────────────────────
#  Pretrained Weights API (from ml repo)
# ─────────────────────────────────────────────────────────────
@app.route('/api/pretrained/info')
def pretrained_info():
    """Info about pretrained .pth weights + transfer adapter status."""
    pt_config = APP_CONFIG.get('pretrained', {})
    adapter_cfg = pt_config.get('adapter', {})

    transfer_model_path = os.path.join(MODELS_DIR, 'company_classifier_transfer.pth')
    transfer_available = os.path.exists(transfer_model_path)
    if transfer_available:
        adapter = TransferLearningAdapter.load(transfer_model_path)
        transfer_status = {
            'trained': adapter.trained,
            'classes': adapter.class_names,
        }
    else:
        transfer_status = {'trained': False}

    result = {
        'available': False,
        'files': [],
        'source': pt_config.get('source', ''),
        'input_dim': pt_config.get('input_dim', 39),
        'num_classes': pt_config.get('num_classes', 3),
        'classes': pt_config.get('classes', []),
        'adapter': {
            'enabled': adapter_cfg.get('enabled', True),
            'adapt_dim': adapter_cfg.get('adapt_dim', 28),
            'fallback_to_sklearn': adapter_cfg.get('fallback_to_sklearn', True),
        },
        'transfer_model': transfer_status,
        'note': 'Pretrained weights (39-dim audio) -> TransferLearningAdapter (28-dim vibration) '
                'via Linear(28,39) expansion layer.',
    }
    for key in ['model_best', 'model_quantized']:
        path = pt_config.get(key, '')
        full = os.path.join(BASE_DIR, path)
        if os.path.exists(full):
            size_kb = round(os.path.getsize(full) / 1024, 1)
            result['files'].append({
                'name': key,
                'filename': path,
                'size_kb': size_kb,
            })
            result['available'] = True
    return jsonify(result)


@app.route('/api/pretrained/weights')
def pretrained_weights_summary():
    """Check if pretrained .pth files exist and their metadata."""
    pt_config = APP_CONFIG.get('pretrained', {})
    files_found = []
    for key in ['model_best', 'model_quantized']:
        path = pt_config.get(key, '')
        full = os.path.join(BASE_DIR, path)
        if os.path.exists(full):
            files_found.append({
                'key': key,
                'path': path,
                'size': os.path.getsize(full),
            })
    return jsonify({
        'count': len(files_found),
        'files': files_found,
        'can_use_directly': False,
        'reason': 'Pretrained weights are from audio MFCC pipeline (39-dim). '
                  'Current app uses 28-dim vibration features. Retraining needed.',
    })


# ─────────────────────────────────────────────────────────────
#  Dataset Replay Injection (for cwru_replay.py)
# ─────────────────────────────────────────────────────────────
@app.route('/api/inject', methods=['POST'])
def inject_samples():
    """Direct buffer injection for CWRU replay bridge (dev only)."""
    global raw_buffer, plot_buffer
    try:
        data = request.get_json()
    except Exception:
        return jsonify({'ok': False, 'error': 'invalid JSON'}), 400
    vals = data.get('samples', [])
    if not vals:
        return jsonify({'ok': True, 'injected': 0})

    if not state['baseline_ready']:
        state['baseline_samples'].extend(vals)
        if len(state['baseline_samples']) >= 80:
            state['baseline'] = float(np.mean(state['baseline_samples'][:80]))
            state['baseline_ready'] = True
            state['status'] = 'BASELINE LOCKED (replay) ✓'
            state['status_level'] = 'success'
            socketio.emit('status_update', {
                'status': state['status'],
                'level': state['status_level'],
                'baseline': state['baseline'],
            })

    raw_buffer.extend(vals)
    if len(raw_buffer) > RAW_WINDOW * 4:
        raw_buffer = raw_buffer[-RAW_WINDOW * 2:]

    baseline = state.get('baseline', float(np.mean(vals)))
    plot_buffer.extend([round(v - baseline, 4) for v in vals])
    if len(plot_buffer) > 600:
        plot_buffer = plot_buffer[-600:]

    # Online retraining: accumulate features and auto-retrain at threshold
    if state['baseline_ready'] and len(vals) >= 80:
        feats = extract_features(vals[:80], state['baseline'])
        state['retrain_buffer'].append(feats)
        if len(state['retrain_buffer']) >= state['retrain_threshold']:
            _online_retrain()

    return jsonify({'ok': True, 'injected': len(vals)})


def _online_retrain():
    """Incremental retrain of anomaly model from accumulated buffer."""
    buffer = state['retrain_buffer']
    if len(buffer) < 30:
        return
    X_new = np.array(buffer)
    model = EnsembleAnomalyModel()
    model.fit(X_new)
    state['current_model_obj'] = model
    state['active_model_name'] = 'online_retrained'
    state['retrain_count'] += 1
    state['retrain_buffer'] = state['retrain_buffer'][-50:]  # keep last 50
    state['status'] = f'Online retrain #{state["retrain_count"]} ({len(X_new)} windows)'
    state['status_level'] = 'success'


@app.route('/api/retrain', methods=['POST'])
def trigger_retrain():
    """Manually trigger online retraining from accumulated buffer."""
    before = len(state['retrain_buffer'])
    _online_retrain()
    after = len(state['current_model_obj'].training_data) if hasattr(state['current_model_obj'], 'training_data') else 0
    return jsonify({
        'ok': True,
        'buffer_before': before,
        'model_trained': state['retrain_count'],
        'status': state['status'],
    })


# ─────────────────────────────────────────────────────────────
#  Evaluation & Testing API
# ─────────────────────────────────────────────────────────────
EVAL_REPORT_PATH = os.path.join(BASE_DIR, 'evaluation_report.json')

@app.route('/api/evaluate/report')
def get_eval_report():
    """Return the latest evaluation report (model card)."""
    if os.path.exists(EVAL_REPORT_PATH):
        with open(EVAL_REPORT_PATH, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'No evaluation report found. Run evaluate_pipeline.py first.'}), 404


@app.route('/api/evaluate/run', methods=['POST'])
def run_evaluation():
    """Trigger evaluation pipeline in a subprocess."""
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, 'evaluate_pipeline.py', '--quick'],
            capture_output=True, text=True, timeout=120, cwd=BASE_DIR
        )
        if os.path.exists(EVAL_REPORT_PATH):
            with open(EVAL_REPORT_PATH, 'r') as f:
                report = json.load(f)
            return jsonify({
                'ok': True,
                'verdict': report.get('verdict', 'UNKNOWN'),
                'report': report,
                'log': result.stdout[-2000:] if result.stdout else '',
            })
        else:
            return jsonify({'ok': False, 'error': result.stderr[-1000:]}), 500
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/evaluate/synthetic', methods=['POST'])
def generate_eval_data():
    """Generate synthetic vibration dataset for testing."""
    try:
        from generate_synthetic_data import save_synthetic_dataset
        save_path = save_synthetic_dataset()
        return jsonify({'ok': True, 'path': save_path})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/evaluate/run-tests', methods=['POST'])
def run_unit_tests():
    """Run pytest suite and return results."""
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
            capture_output=True, text=True, timeout=120, cwd=BASE_DIR
        )
        lines = result.stdout.split('\n')
        passed = sum(1 for l in lines if 'PASSED' in l)
        failed = sum(1 for l in lines if 'FAILED' in l)
        return jsonify({
            'ok': result.returncode == 0,
            'passed': passed,
            'failed': failed,
            'summary': lines[-3:] if result.stdout else [],
            'output': result.stdout[-3000:],
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────
#  3D VUI Dashboard API
# ─────────────────────────────────────────────────────────────
@app.route('/api/vibration/current')
def vibration_current():
    """Return latest RMS, FFT array, and spectral bands for 3D VUI."""
    raw = plot_buffer[-128:] if plot_buffer else []
    rms = float(np.sqrt(np.mean(np.square(raw)))) * 3 if raw else 0.0
    return jsonify({
        'rms': round(rms, 4),
        'dominant_freq': round(state['dominant_freq'], 2),
        'anomaly_score': round(state['anomaly_score'] * 100, 1),
        'is_anomaly': state['is_anomaly'],
        'rpm': state['rpm'],
        'rms_slope': state['rms_slope'],
        'trend_alert': state['trend_alert'],
        'fft': [],
        'spectral': [],
    })


@app.route('/api/trend')
def trend_data():
    """Return RMS history for 7-day trend chart."""
    history = state['rms_history']
    if not history:
        return jsonify({'timestamps': [], 'values': [], 'slope': 0})
    return jsonify({
        'timestamps': [int(t) for t, _ in history],
        'values': [round(v, 4) for _, v in history],
        'slope': state['rms_slope'],
    })


@app.route('/api/rul')
def rul_prediction():
    """Predict remaining useful life from latest features."""
    raw = plot_buffer[-128:] if plot_buffer else []
    if len(raw) < 50:
        return jsonify({'rul': 0.5, 'trained': rul_predictor.trained, 'status': 'insufficient_data'})
    baseline = float(np.mean(raw))
    feats = extract_features(raw, baseline)
    pred = rul_predictor.predict(feats)
    return jsonify({
        'rul': round(pred, 4),
        'rul_pct': round(pred * 100, 1),
        'trained': rul_predictor.trained,
        'status': 'active',
    })


@app.route('/api/motor/identify')
def motor_identify():
    """Identify physical motor characteristics from latest vibration data."""
    raw = plot_buffer[-128:] if plot_buffer else []
    if len(raw) < 50:
        return jsonify({'status': 'insufficient_data', 'trained': motor_classifier.trained})
    baseline = float(np.mean(raw))
    feats = extract_features(raw, baseline)
    result = motor_classifier.predict(feats)
    return jsonify({**result, 'trained': motor_classifier.trained, 'status': 'active'})


@app.route('/api/explain')
def explain_prediction():
    """Return SHAP feature importance explanations for latest prediction."""
    feats = None
    raw = plot_buffer[-128:] if plot_buffer else []
    if len(raw) >= 50:
        baseline = float(np.mean(raw))
        feats = extract_features(raw, baseline)
    elif state.get('training_features') and len(state['training_features']) > 0:
        feats = state['training_features'][-1]
    elif state.get('retrain_buffer') and len(state['retrain_buffer']) > 0:
        feats = state['retrain_buffer'][-1]
    if feats is None or not motor_classifier.trained:
        return jsonify({'status': 'unavailable', 'reason': 'Need vibration data and a trained motor classifier'})
    try:
        explanations = motor_classifier.explain(feats, MotorClassifier.feature_names())
        return jsonify({'explanations': explanations, 'status': 'active'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})


# ─────────────────────────────────────────────────────────────
#  Edge / Firmware Endpoints (Phase 4)
# ─────────────────────────────────────────────────────────────
@app.route('/api/config/edge', methods=['GET', 'POST'])
def edge_config():
    """Get or set edge device configuration (feature mode, sampling rate)."""
    if request.method == 'POST':
        data = request.json or {}
        if 'feature_mode' in data:
            state['edge_feature_mode'] = bool(data['feature_mode'])
        if 'sample_rate' in data:
            state['edge_sample_rate'] = int(data['sample_rate'])
        return jsonify({'ok': True, 'feature_mode': state.get('edge_feature_mode', False)})
    return jsonify({
        'feature_mode': state.get('edge_feature_mode', False),
        'sample_rate': state.get('edge_sample_rate', 100),
        'version': '2.0',
    })


@app.route('/api/firmware/upload', methods=['POST'])
def firmware_upload():
    """Upload new firmware binary for OTA update."""
    if 'firmware' not in request.files:
        return jsonify({'ok': False, 'error': 'No firmware file'}), 400
    f = request.files['firmware']
    if f.filename == '':
        return jsonify({'ok': False, 'error': 'Empty filename'}), 400
    firmware_dir = os.path.join(os.path.dirname(__file__), 'firmware')
    os.makedirs(firmware_dir, exist_ok=True)
    save_path = os.path.join(firmware_dir, 'firmware.bin')
    f.save(save_path)
    return jsonify({
        'ok': True,
        'size': os.path.getsize(save_path),
        'path': save_path,
        'note': 'Upload via OTA to ESP32 (send "OTA" over serial first)',
    })


@app.route('/api/firmware/info')
def firmware_info():
    """Return latest firmware info."""
    firmware_dir = os.path.join(os.path.dirname(__file__), 'firmware')
    fw_path = os.path.join(firmware_dir, 'firmware.bin')
    if os.path.exists(fw_path):
        return jsonify({
            'available': True,
            'size': os.path.getsize(fw_path),
            'modified': os.path.getmtime(fw_path),
        })
    return jsonify({'available': False})


@app.route('/api/features/embedding')
def feature_embedding():
    """Return PCA-reduced 3D embedding of recent feature vectors with anomaly labels."""
    from collections import deque
    buf = state['retrain_buffer']
    if len(buf) < 10:
        return jsonify({'points': [], 'status': 'insufficient_data'})
    X = np.array(buf)
    if X.shape[0] < 3 or X.shape[1] < 3:
        return jsonify({'points': [], 'status': 'insufficient_data'})
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_comp = min(3, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(X_scaled)
    scores = []
    if state['current_model_obj'] is not None:
        for i in range(len(X)):
            is_anom, score = state['current_model_obj'].predict_score(X[i])
            scores.append(round(float(score), 4))
    points = [
        {'x': round(float(p[0]), 4), 'y': round(float(p[1]), 4),
         'z': round(float(p[2]), 4) if n_comp > 2 else 0,
         'score': scores[i] if scores else 0.0,
         'anomaly': bool(scores[i] > 0.5) if scores else False}
        for i, p in enumerate(coords)
    ]
    return jsonify({
        'points': points,
        'explained_variance': [round(float(v), 3) for v in pca.explained_variance_ratio_],
        'n_components': n_comp,
        'status': 'active',
    })


# ─────────────────────────────────────────────────────────────
#  Swagger / OpenAPI Docs
# ─────────────────────────────────────────────────────────────
OPENAPI_SPEC = {
    'openapi': '3.0.3',
    'info': {
        'title': 'MotorSense API',
        'version': '1.0.0',
        'description': 'Real-time motor vibration analysis, anomaly detection, RUL prediction, and motor characteristics classification.',
    },
    'servers': [{'url': 'http://localhost:5050', 'description': 'Local development'}],
    'paths': {
        '/': {'get': {'summary': 'Main dashboard HTML', 'responses': {'200': {'description': 'HTML page'}}}},
        '/vui': {'get': {'summary': '3D VUI dashboard HTML', 'responses': {'200': {'description': 'HTML page'}}}},
        '/api/ports': {'get': {'summary': 'List available serial ports', 'responses': {'200': {'description': 'Array of port objects'}}}},
        '/api/connect': {'post': {'summary': 'Connect to serial port', 'requestBody': {'content': {'application/json': {'schema': {'type': 'object', 'properties': {'port': {'type': 'string'}, 'baud': {'type': 'integer'}}}}}}, 'responses': {'200': {'description': 'Connection status'}}}},
        '/api/disconnect': {'post': {'summary': 'Disconnect serial port', 'responses': {'200': {'description': 'Disconnected'}}}},
        '/api/disconnect/force': {'post': {'summary': 'Force disconnect serial port', 'responses': {'200': {'description': 'Disconnected'}}}},
        '/api/status': {'get': {'summary': 'System status with anomaly score', 'responses': {'200': {'description': 'Status object'}}}},
        '/api/fft': {'get': {'summary': 'Latest FFT data', 'responses': {'200': {'description': 'FFT array'}}}},
        '/api/models': {'get': {'summary': 'List saved models', 'responses': {'200': {'description': 'Model list'}}}},
        '/api/company': {'get': {'summary': 'Company classification result', 'responses': {'200': {'description': 'Company info'}}}},
        '/api/vibration/current': {'get': {'summary': 'Latest vibration metrics', 'responses': {'200': {'description': 'RMS, freq, anomaly score'}}}},
        '/api/trend': {'get': {'summary': '7-day RMS trend data', 'responses': {'200': {'description': 'Timestamps and values'}}}},
        '/api/rul': {'get': {'summary': 'RUL prediction from latest features', 'responses': {'200': {'description': 'RUL score 0-1'}}}},
        '/api/motor/identify': {'get': {'summary': 'Motor characteristics (HP, bearing, diameter)', 'responses': {'200': {'description': 'Classification results'}}}},
        '/api/explain': {'get': {'summary': 'SHAP feature importance explanations', 'responses': {'200': {'description': 'Top-5 features per output'}}}},
        '/api/features/embedding': {'get': {'summary': 'PCA 3D embedding of recent features', 'responses': {'200': {'description': '3D points with anomaly labels'}}}},
        '/api/retrain': {'post': {'summary': 'Trigger online model retrain', 'responses': {'200': {'description': 'Retrain result'}}}},
        '/api/config/edge': {'get': {'summary': 'Edge device configuration', 'responses': {'200': {'description': 'Feature mode, sample rate, version'}}}},
        '/api/firmware/info': {'get': {'summary': 'Latest firmware info', 'responses': {'200': {'description': 'Firmware availability and size'}}}},
    },
}


@app.route('/api/openapi.json')
def openapi_spec():
    return jsonify(OPENAPI_SPEC)


@app.route('/api/docs')
def swagger_ui():
    return render_template('swagger.html')


# ─────────────────────────────────────────────────────────────
#  SocketIO Events
# ─────────────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    emit('status_update', {
        'status': state['status'],
        'level': state['status_level'],
    })


# ─────────────────────────────────────────────────────────────
#  Startup
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    reader_thread = threading.Thread(target=serial_reader, daemon=True)
    reader_thread.start()
    print("Motor Analyzer running -> http://127.0.0.1:5050")
    socketio.run(app, host='0.0.0.0', port=5050, debug=False)
