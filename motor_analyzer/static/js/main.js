/* ======================================================
   MotorSense — Frontend Logic
   WebSocket-driven real-time oscilloscope + FFT + ML UI
   ====================================================== */

'use strict';

// ─────────────────────────────────────────────────────────
//  Socket.IO
// ─────────────────────────────────────────────────────────
const socket = io();

// ─────────────────────────────────────────────────────────
//  Chart.js: Waveform (Oscilloscope)
// ─────────────────────────────────────────────────────────
const WAVEFORM_POINTS = 200;
const waveformData = new Array(WAVEFORM_POINTS).fill(0);

const waveformCtx = document.getElementById('waveform-chart').getContext('2d');
const waveformChart = new Chart(waveformCtx, {
  type: 'line',
  data: {
    labels: waveformData.map((_, i) => i),
    datasets: [{
      data: waveformData,
      borderColor: '#00d4ff',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.5,
      borderCapStyle: 'round',
      borderJoinStyle: 'round',
      fill: true,
      backgroundColor: (ctx) => {
        const gradient = ctx.chart.ctx.createLinearGradient(0, 0, 0, ctx.chart.height);
        gradient.addColorStop(0, 'rgba(0,212,255,0.15)');
        gradient.addColorStop(1, 'rgba(0,212,255,0)');
        return gradient;
      },
    }],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 150, easing: 'linear' },
    interaction: { mode: 'none' },
    scales: {
      x: {
        display: false,
      },
      y: {
        display: true,
        grid: {
          color: 'rgba(255,255,255,0.04)',
          drawBorder: false,
        },
        ticks: {
          color: '#475569',
          font: { family: "'JetBrains Mono'", size: 10 },
          maxTicksLimit: 5,
        },
        border: { display: false },
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false },
    },
  },
});

// ─────────────────────────────────────────────────────────
//  Chart.js: FFT Spectrum
// ─────────────────────────────────────────────────────────
const fftCtx = document.getElementById('fft-chart').getContext('2d');
const fftChart = new Chart(fftCtx, {
  type: 'bar',
  data: {
    labels: [],
    datasets: [{
      data: [],
      backgroundColor: (ctx) => {
        const gradient = ctx.chart.ctx.createLinearGradient(0, 0, 0, ctx.chart.height);
        gradient.addColorStop(0, 'rgba(168,85,247,0.8)');
        gradient.addColorStop(1, 'rgba(168,85,247,0.1)');
        return gradient;
      },
      borderColor: 'rgba(168,85,247,0.4)',
      borderWidth: 0,
      borderRadius: 2,
      barPercentage: 1.0,
      categoryPercentage: 1.0,
    }],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { mode: 'none' },
    scales: {
      x: {
        display: true,
        grid: { display: false },
        ticks: {
          color: '#475569',
          font: { family: "'JetBrains Mono'", size: 10 },
          maxTicksLimit: 10,
          callback: (val, idx, ticks) => {
            const label = fftChart.data.labels[idx];
            return label !== undefined ? `${parseFloat(label).toFixed(0)}` : '';
          },
        },
        border: { display: false },
        title: {
          display: true,
          text: 'Frequency (Hz)',
          color: '#475569',
          font: { size: 11, family: "'JetBrains Mono'" },
        },
      },
      y: {
        display: true,
        min: 0, max: 1,
        grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
        ticks: {
          color: '#475569',
          font: { family: "'JetBrains Mono'", size: 10 },
          maxTicksLimit: 4,
          callback: v => `${(v * 100).toFixed(0)}%`,
        },
        border: { display: false },
        title: {
          display: true,
          text: 'Magnitude',
          color: '#475569',
          font: { size: 11, family: "'JetBrains Mono'" },
        },
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(13,18,32,0.95)',
        borderColor: 'rgba(168,85,247,0.3)',
        borderWidth: 1,
        titleColor: '#a855f7',
        bodyColor: '#94a3b8',
        callbacks: {
          title: (items) => `${parseFloat(items[0].label).toFixed(2)} Hz`,
          label: (item) => `Magnitude: ${(item.raw * 100).toFixed(1)}%`,
        },
      },
    },
  },
});

// ─────────────────────────────────────────────────────────
//  DOM References
// ─────────────────────────────────────────────────────────
const $  = (id) => document.getElementById(id);
const el = {
  connDot:         $('conn-dot'),
  connText:        $('conn-text'),
  connPill:        $('conn-pill'),
  modelPill:       $('model-pill'),
  modelPillText:   $('model-pill-text'),
  statusBar:       $('status-bar'),
  statusText:      $('status-text'),
  statusIcon:      $('status-icon'),
  portSelect:      $('port-select'),
  baudSelect:      $('baud-select'),
  connectBtn:      $('connect-btn'),
  disconnectBtn:   $('disconnect-btn'),
  refreshPortsBtn: $('refresh-ports-btn'),
  settingsToggle:  $('settings-toggle-btn'),
  drawer:          $('settings-drawer'),

  trainName:       $('train-name'),
  trainDuration:   $('train-duration'),
  trainDurVal:     $('train-duration-val'),
  trainBtn:        $('train-btn'),
  cancelTrainBtn:  $('cancel-train-btn'),
  trainProgressWrap: $('train-progress-wrap'),
  trainProgressLabel: $('train-progress-label'),
  trainProgressPct: $('train-progress-pct'),
  trainProgressFill: $('train-progress-fill'),
  trainProgressMeta: $('train-progress-meta'),

  saveCard:        $('save-card'),
  saveName:        $('save-name'),
  saveBtn:         $('save-btn'),

  domFreq:         $('dom-freq'),
  anomalyScoreVal: $('anomaly-score-val'),
  freqCard:        $('freq-card'),
  anomalyScoreCard: $('anomaly-score-card'),
  anomalyAlert:    $('anomaly-alert'),

  oscBadge:        $('osc-badge'),
  fftBadge:        $('fft-dominant-badge'),

  evalStartBtn:    $('eval-start-btn'),
  evalStopBtn:     $('eval-stop-btn'),

  modelsList:      $('models-list'),
  refreshModels:   $('refresh-models-btn'),

  logList:         $('log-list'),
  toastContainer:  $('toast-container'),

  // Motor Control
  motor1Btn:       $('motor1-btn'),
  motorStopBtn:    $('motor-stop-btn'),
  motor2Btn:       $('motor2-btn'),
  motorActiveBadge: $('motor-active-badge'),
  motorSpeedSlider: $('motor-speed'),
  motorSpeedVal:   $('motor-speed-val'),

  // Prerequisites
  reqSerial:       $('req-serial'),
  reqBaseline:     $('req-baseline'),

  // Company
  companyBadge:    $('company-badge'),
  companyConf:     $('company-conf'),
  companyDetail:   $('company-detail'),
  companyModelName: $('company-model-name'),
};

// ─────────────────────────────────────────────────────────
//  App State
// ─────────────────────────────────────────────────────────
let appState = {
  connected: false,
  baselineReady: false,
  training: false,
  evaluating: false,
  activeModel: null,
  isAnomaly: false,
  companyName: 'Unknown',
  companyIdentified: false,
  companyConfidence: 0,
};

// ─────────────────────────────────────────────────────────
//  Waveform update helper
// ─────────────────────────────────────────────────────────
function updateWaveform(incoming, isAnomaly) {
  const data = incoming.slice(-WAVEFORM_POINTS);
  while (data.length < WAVEFORM_POINTS) data.unshift(0);

  waveformChart.data.datasets[0].data = data;
  const color = isAnomaly ? '#ef4444' : '#00d4ff';
  waveformChart.data.datasets[0].borderColor = color;
  waveformChart.data.datasets[0].backgroundColor = (ctx) => {
    const g = ctx.chart.ctx.createLinearGradient(0, 0, 0, ctx.chart.height);
    g.addColorStop(0, isAnomaly ? 'rgba(239,68,68,0.2)' : 'rgba(0,212,255,0.15)');
    g.addColorStop(1, 'rgba(0,0,0,0)');
    return g;
  };
  waveformChart.update('none');

  el.oscBadge.textContent = isAnomaly ? '⚠ ANOMALY' : 'LIVE';
  el.oscBadge.className = 'chart-badge' + (isAnomaly ? ' anomaly' : '');
}

function updateFFT(fftPayload) {
  if (!fftPayload || !fftPayload.freqs || fftPayload.freqs.length === 0) return;
  const freqs = fftPayload.freqs.map(f => f.toFixed(2));
  const mags  = fftPayload.magnitudes;

  fftChart.data.labels  = freqs;
  fftChart.data.datasets[0].data = mags;
  fftChart.update('none');

  el.fftBadge.textContent = `${fftPayload.dominant.toFixed(1)} Hz`;
}

// ─────────────────────────────────────────────────────────
//  RMS Trend Chart (7-day localStorage)
// ─────────────────────────────────────────────────────────
const TREND_KEY = 'motorsense_rms_trend'
const TREND_MAX_POINTS = 10080  // 7 days × 1440 min, 1 sample per min

function loadTrendData() {
  try {
    const raw = localStorage.getItem(TREND_KEY)
    if (!raw) return []
    const data = JSON.parse(raw)
    const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000
    return data.filter(p => p.t > cutoff)
  } catch { return [] }
}

function saveTrendPoint(rms) {
  let data = loadTrendData()
  data.push({ t: Date.now(), v: rms })
  // Keep at most one point per 30s to avoid bloat
  if (data.length > 1) {
    const last = data[data.length - 2]
    if (data[data.length - 1].t - last.t < 30000) {
      data[data.length - 2] = data[data.length - 1]
      data.pop()
    }
  }
  if (data.length > TREND_MAX_POINTS) data = data.slice(-TREND_MAX_POINTS)
  localStorage.setItem(TREND_KEY, JSON.stringify(data))
  return data
}

const trendCanvas = document.getElementById('trend-chart')
let trendChart = null
if (trendCanvas) {
  const ctx = trendCanvas.getContext('2d')
  trendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        data: [],
        borderColor: '#00D4AA',
        borderWidth: 1.2,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
        backgroundColor: (ctx) => {
          const g = ctx.chart.ctx.createLinearGradient(0, 0, 0, ctx.chart.height)
          g.addColorStop(0, 'rgba(0,212,170,0.15)')
          g.addColorStop(1, 'rgba(0,212,170,0)')
          return g
        },
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'nearest', intersect: false },
      scales: {
        x: {
          type: 'time',
          time: { unit: 'hour', displayFormats: { hour: 'HH:mm' } },
          display: true,
          grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
          ticks: { color: '#475569', font: { size: 9 }, maxTicksLimit: 6 },
          border: { display: false },
        },
        y: {
          display: true,
          grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
          ticks: { color: '#475569', font: { size: 9 }, maxTicksLimit: 4 },
          border: { display: false },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(13,18,32,0.95)',
          borderColor: 'rgba(0,212,170,0.3)',
          borderWidth: 1,
          bodyColor: '#E8EDF5',
          callbacks: {
            title: (items) => new Date(items[0].parsed.x).toLocaleString(),
            label: (item) => `RMS: ${item.parsed.y.toFixed(4)}`,
          },
        },
      },
    },
  })
}

function updateTrendChart(history, rms, slope) {
  if (!trendChart) return

  const points = history && history.timestamps
    ? history.timestamps.map((t, i) => ({ x: t * 1000, y: history.values[i] }))
    : loadTrendData().map(p => ({ x: p.t, y: p.v }))

  if (points.length === 0) return

  trendChart.data.labels = points.map(p => new Date(p.x))
  trendChart.data.datasets[0].data = points
  trendChart.update('none')

  document.getElementById('trend-slope').textContent = slope
    ? `${(slope * 1000).toFixed(2)} ×10⁻³/hr`
    : '—'
  document.getElementById('trend-current').textContent = rms ? rms.toFixed(4) : '—'
  document.getElementById('trend-count').textContent = points.length

  const badge = document.getElementById('trend-badge')
  if (badge) {
    const trendAlert = window._trendAlert || false
    if (trendAlert) {
      badge.style.display = 'inline'
      badge.textContent = '⚠ RISING'
      badge.style.background = 'rgba(255,107,53,0.2)'
      badge.style.color = '#FF6B35'
    } else {
      badge.style.display = 'inline'
      badge.textContent = '✓ STABLE'
      badge.style.background = 'rgba(0,212,170,0.15)'
      badge.style.color = '#00D4AA'
    }
  }
}

// ─────────────────────────────────────────────────────────
//  Socket Events
// ─────────────────────────────────────────────────────────
socket.on('waveform_update', (d) => {
  updateWaveform(d.waveform, appState.isAnomaly && appState.evaluating);
});

socket.on('sensor_data', (d) => {
  updateFFT(d.fft);

  // Trend chart: save RMS point and update
  if (d.rms !== undefined && d.rms > 0) {
    saveTrendPoint(d.rms)
    window._trendAlert = d.trend_alert || false
  }
  updateTrendChart(null, d.rms, d.rms_slope)

  el.domFreq.textContent = d.dominant_freq;
  el.freqCard.classList.toggle('active', d.dominant_freq > 0);

  const showAnomaly = d.is_anomaly && d.evaluating;
  el.anomalyScoreVal.textContent = d.evaluating ? d.anomaly_score : '—';
  el.anomalyScoreCard.classList.toggle('anomaly', showAnomaly);
  el.anomalyScoreCard.classList.toggle('active', d.evaluating && !showAnomaly);
  el.freqCard.classList.toggle('anomaly', showAnomaly);
  el.anomalyAlert.style.display = showAnomaly ? 'flex' : 'none';

  appState.isAnomaly = showAnomaly;

  // Update company info if available
  if (d.company_identified && d.company) {
    appState.companyIdentified = true;
    appState.companyName = d.company;
    el.companyBadge.textContent = d.company;
    el.companyBadge.className = 'company-badge identified';
  }
});

// Company identification event
socket.on('company_identified', (d) => {
  appState.companyIdentified = true;
  appState.companyName = d.company;
  appState.companyConfidence = d.confidence;
  el.companyBadge.textContent = d.company;
  el.companyBadge.className = 'company-badge identified';
  el.companyConf.textContent = `${(d.confidence * 100).toFixed(0)}% confidence`;
  el.companyDetail.style.display = 'flex';
  el.companyModelName.textContent = d.company + '_model';
  log(`Machine identified as "${d.company}" (${(d.confidence * 100).toFixed(0)}% confidence)`, 'success');
});

socket.on('status_update', (d) => {
  setStatus(d.status, d.level || 'info');
  log(d.status, d.level || 'info');
});

socket.on('train_progress', (d) => {
  el.trainProgressWrap.style.display = 'block';
  el.trainProgressFill.style.width   = `${d.progress}%`;
  el.trainProgressPct.textContent    = `${d.progress.toFixed(0)}%`;
  el.trainProgressMeta.textContent   = `${d.samples} samples  •  ${d.elapsed}s / ${d.total}s`;
  el.trainProgressLabel.textContent  = 'CAPTURING VIBRATION PROFILE…';
});

socket.on('train_complete', (d) => {
  appState.training = false;
  appState.evaluating = true;

  el.trainProgressLabel.textContent = 'TRAINING COMPLETE ✓';
  el.trainProgressPct.textContent   = '100%';
  el.trainProgressFill.style.width  = '100%';

  el.trainBtn.disabled        = false;
  el.cancelTrainBtn.style.display = 'none';
  el.saveCard.style.display   = 'flex';
  el.saveName.value           = d.model_name;

  setEvaluatingUI(true);
  syncButtons();
  toast(`Model "${d.model_name}" trained successfully!`, 'success');
  log(`Training complete: "${d.model_name}"`, 'success');

  // Auto-activate evaluation
  el.evalStartBtn.style.display = 'none';
  el.evalStopBtn.style.display  = 'inline-flex';
});

// ESP32 hardware messages → system log
socket.on('esp32_log', (d) => {
  log(`[ESP32] ${d.msg}`, 'info');
});


// ─────────────────────────────────────────────────────────
//  Status helpers
// ─────────────────────────────────────────────────────────
function setStatus(text, level = 'info') {
  el.statusText.textContent = text;
  el.statusBar.className    = `status-bar ${level}`;
  const icons = { info: '●', success: '✓', error: '✗', training: '◈', warning: '⚠', anomaly: '⚠' };
  el.statusIcon.textContent = icons[level] || '●';
}

function setConnected(yes) {
  appState.connected = yes;
  el.connDot.className  = 'status-dot' + (yes ? ' cyan' : '');
  el.connText.textContent = yes ? 'Connected' : 'Disconnected';
  el.connectBtn.disabled    = yes;
  el.disconnectBtn.disabled = !yes;
  syncButtons();
}

function setEvaluatingUI(yes) {
  appState.evaluating = yes;
  el.evalStartBtn.style.display = yes ? 'none'        : 'inline-flex';
  el.evalStopBtn.style.display  = yes ? 'inline-flex' : 'none';
}

function syncButtons() {
  const ready = appState.connected && appState.baselineReady;
  el.trainBtn.disabled     = !ready || appState.training;
  el.evalStartBtn.disabled = !ready || !appState.activeModel;
  updateReadinessUI();
}

function updateReadinessUI() {
  // Serial row
  if (appState.connected) {
    el.reqSerial.className = 'req-item done';
  } else {
    el.reqSerial.className = 'req-item';
  }

  // Baseline row
  if (appState.baselineReady) {
    el.reqBaseline.className = 'req-item done';
  } else if (appState.connected) {
    el.reqBaseline.className = 'req-item loading';  // connected but still calibrating
  } else {
    el.reqBaseline.className = 'req-item';
  }
}

// ─────────────────────────────────────────────────────────
//  Log
// ─────────────────────────────────────────────────────────
function log(msg, level = 'info') {
  const now = new Date();
  const t = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;
  const div = document.createElement('div');
  div.className = `log-entry ${level}`;
  div.innerHTML = `<span class="log-time">${t}</span><span class="log-msg">${msg}</span>`;
  el.logList.prepend(div);
  if (el.logList.children.length > 50) el.logList.lastChild.remove();
}

// ─────────────────────────────────────────────────────────
//  Toast
// ─────────────────────────────────────────────────────────
function toast(msg, type = 'info') {
  const icons = { success: '✓', error: '✗', info: 'ℹ' };
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.innerHTML = `<span>${icons[type] || '●'}</span><span>${msg}</span>`;
  el.toastContainer.appendChild(t);
  setTimeout(() => {
    t.classList.add('out');
    setTimeout(() => t.remove(), 350);
  }, 3500);
}

// ─────────────────────────────────────────────────────────
//  API helpers
// ─────────────────────────────────────────────────────────
async function api(path, body = null, method = body ? 'POST' : 'GET') {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  return res.json();
}

// ─────────────────────────────────────────────────────────
//  Port List
// ─────────────────────────────────────────────────────────
async function loadPorts() {
  const ports = await api('/api/ports');
  el.portSelect.innerHTML = ports.length
    ? ports.map(p => `<option value="${p.device}">${p.device} — ${p.desc}</option>`).join('')
    : '<option value="">No ports found</option>';
}

// ─────────────────────────────────────────────────────────
//  Models List
// ─────────────────────────────────────────────────────────
async function loadModels() {
  const models = await api('/api/model/list');
  if (!models.length) {
    el.modelsList.innerHTML = '<div class="empty-state">No saved models yet.<br/>Train one to get started.</div>';
    return;
  }

  el.modelsList.innerHTML = models.map(m => `
    <div class="model-item ${appState.activeModel === m.filename ? 'active' : ''}" id="model-item-${m.filename}">
      <div class="model-name">${m.name}</div>
      <div class="model-meta">
        <span>📅 ${m.timestamp}</span>
        <span>📊 ${m.samples} samples</span>
        <span>⚖ baseline ${m.baseline}</span>
      </div>
      <div class="model-actions">
        <button class="btn btn-primary" style="font-size:11px;padding:5px 10px;" onclick="loadModel('${m.filename}')">
          Load &amp; Evaluate
        </button>
        <button class="btn btn-ghost" style="font-size:11px;padding:5px 10px;color:#ef4444;border-color:rgba(239,68,68,0.3);" onclick="deleteModel('${m.filename}')">
          Delete
        </button>
      </div>
    </div>
  `).join('');
}

async function loadModel(filename) {
  const res = await api('/api/model/load', { name: filename });
  if (res.ok) {
    appState.activeModel     = filename;
    appState.baselineReady   = true;
    appState.evaluating      = true;
    el.modelPill.style.display = 'flex';
    el.modelPillText.textContent = filename;
    el.evalStartBtn.disabled = false;
    setEvaluatingUI(true);
    toast(`Model "${filename}" loaded`, 'success');
    log(`Loaded model: ${filename} (baseline=${res.meta.baseline})`, 'success');
    loadModels();
    syncButtons();
  } else {
    toast(res.error || 'Failed to load model', 'error');
  }
}

async function deleteModel(filename) {
  if (!confirm(`Delete model "${filename}"?`)) return;
  const res = await api('/api/model/delete', { name: filename });
  if (res.ok) {
    toast(`Model "${filename}" deleted`, 'info');
    if (appState.activeModel === filename) {
      appState.activeModel = null;
      el.modelPill.style.display = 'none';
      setEvaluatingUI(false);
    }
    loadModels();
    syncButtons();
  }
}

// ─────────────────────────────────────────────────────────
//  Event Listeners
// ─────────────────────────────────────────────────────────

// Settings drawer toggle
el.settingsToggle.addEventListener('click', () => {
  el.drawer.classList.toggle('open');
});

// Close drawer on outside click
document.addEventListener('click', (e) => {
  if (!el.drawer.contains(e.target) && !el.settingsToggle.contains(e.target)) {
    el.drawer.classList.remove('open');
  }
});

// Refresh ports
el.refreshPortsBtn.addEventListener('click', loadPorts);

// Connect
el.connectBtn.addEventListener('click', async () => {
  const port = el.portSelect.value;
  const baud = el.baudSelect.value;
  if (!port) { toast('Select a port first', 'error'); return; }

  const res = await api('/api/connect', { port, baud });
  if (res.ok) {
    setConnected(true);
    el.drawer.classList.remove('open');
    setStatus(`Connected to ${port} @ ${baud} baud — calibrating baseline…`, 'info');
    log(`Connected: ${port} @ ${baud}`, 'success');
    toast(`Connected to ${port}`, 'success');

    // Poll for baseline ready
    const poll = setInterval(async () => {
      const s = await api('/api/status');
      if (s.baseline_ready) {
        appState.baselineReady = true;
        syncButtons();
        clearInterval(poll);
      }
    }, 1000);
  } else {
    toast(res.error || 'Connection failed', 'error');
    log(res.error || 'Connection failed', 'error');
  }
});

// Disconnect
el.disconnectBtn.addEventListener('click', async () => {
  await api('/api/disconnect', {});
  setConnected(false);
  appState.baselineReady = false;
  appState.training = false;
  appState.evaluating = false;
  setStatus('DISCONNECTED', 'info');
  setEvaluatingUI(false);
  toast('Disconnected', 'info');
  log('Disconnected from serial port', 'info');
});

// Train duration slider
el.trainDuration.addEventListener('input', () => {
  el.trainDurVal.textContent = `${el.trainDuration.value}s`;
});

// Start Training
el.trainBtn.addEventListener('click', async () => {
  const name = el.trainName.value.trim() || `model_${Date.now()}`;
  const dur  = parseInt(el.trainDuration.value);

  const res = await api('/api/train/start', { name, duration: dur });
  if (res.ok) {
    appState.training = true;
    appState.evaluating = false;
    setEvaluatingUI(false);
    el.trainBtn.disabled = true;
    el.cancelTrainBtn.style.display = 'inline-flex';
    el.trainProgressWrap.style.display = 'block';
    el.trainProgressFill.style.width = '0%';
    el.trainProgressPct.textContent = '0%';
    el.saveCard.style.display = 'none';
    el.anomalyAlert.style.display = 'none';
    toast(`Training "${name}" for ${dur}s…`, 'info');
    log(`Training started: "${name}" (${dur}s)`, 'training');
  } else {
    toast(res.error || 'Cannot start training', 'error');
  }
});

// Cancel Training
el.cancelTrainBtn.addEventListener('click', async () => {
  await api('/api/train/cancel', {});
  appState.training = false;
  el.trainBtn.disabled = false;
  el.cancelTrainBtn.style.display = 'none';
  el.trainProgressWrap.style.display = 'none';
  toast('Training cancelled', 'info');
  log('Training cancelled', 'warning');
  syncButtons();
});

// Save Model
el.saveBtn.addEventListener('click', async () => {
  const name = el.saveName.value.trim();
  if (!name) { toast('Enter a model name', 'error'); return; }
  const res = await api('/api/model/save', { name });
  if (res.ok) {
    toast(`Model "${name}" saved ✓`, 'success');
    log(`Model saved: ${name}`, 'success');
    appState.activeModel = name;
    el.modelPill.style.display = 'flex';
    el.modelPillText.textContent = name;
    loadModels();
  } else {
    toast(res.error || 'Save failed', 'error');
  }
});

// Evaluate Start
el.evalStartBtn.addEventListener('click', async () => {
  if (!appState.activeModel) { toast('Load a model first', 'error'); return; }
  const res = await api('/api/model/load', { name: appState.activeModel });
  if (res.ok) {
    setEvaluatingUI(true);
    toast('Evaluation started', 'success');
    log('Evaluation started', 'success');
  }
});

// Evaluate Stop
el.evalStopBtn.addEventListener('click', async () => {
  await api('/api/evaluate/stop', {});
  setEvaluatingUI(false);
  el.anomalyAlert.style.display = 'none';
  el.anomalyScoreCard.classList.remove('anomaly', 'active');
  el.freqCard.classList.remove('anomaly');
  toast('Evaluation stopped', 'info');
  log('Evaluation stopped', 'info');
});

// Refresh models
el.refreshModels.addEventListener('click', loadModels);

// ─────────────────────────────────────────────────────────
//  Init
// ─────────────────────────────────────────────────────────
(async () => {
  await loadPorts();
  await loadModels();

  // Check if already connected from server side
  const [status, companyStatus] = await Promise.all([
    api('/api/status'),
    api('/api/company/status'),
  ]);
  if (companyStatus.identified) {
    appState.companyIdentified = true;
    appState.companyName = companyStatus.current_company;
    appState.companyConfidence = companyStatus.confidence;
    el.companyBadge.textContent = companyStatus.current_company;
    el.companyBadge.className = 'company-badge identified';
    el.companyConf.textContent = `${(companyStatus.confidence * 100).toFixed(0)}% confidence`;
    el.companyDetail.style.display = 'flex';
    el.companyModelName.textContent = companyStatus.current_company + '_model';
  }
  if (status.connected) {
    setConnected(true);
    appState.baselineReady = status.baseline_ready;
    appState.evaluating    = status.evaluating;
    appState.activeModel   = status.active_model;
    if (status.active_model) {
      el.modelPill.style.display = 'flex';
      el.modelPillText.textContent = status.active_model;
    }
    if (status.evaluating) setEvaluatingUI(true);
    syncButtons();
  }

  setStatus('IDLE — Connect a serial port to begin', 'info');
  log('MotorSense initialized', 'info');


  // Load XAI on init if data available
  fetchXAI();

  // Load RMS trend from localStorage
  const savedTrend = loadTrendData()
  if (savedTrend.length > 0 && trendChart) {
    const pts = savedTrend.map(p => ({ x: p.t, y: p.v }))
    trendChart.data.labels = pts.map(p => new Date(p.x))
    trendChart.data.datasets[0].data = pts
    trendChart.update('none')
    document.getElementById('trend-count').textContent = pts.length
  }
})();

// ─────────────────────────────────────────────────────────
//  XAI — SHAP Feature Importance
// ─────────────────────────────────────────────────────────
async function fetchXAI() {
  try {
    const res = await api('/api/explain');
    if (!res || res.status !== 'active' || !res.explanations) {
      document.getElementById('xai-card').style.display = 'none';
      return;
    }
    const card = document.getElementById('xai-card');
    const content = document.getElementById('xai-content');
    card.style.display = 'block';

    let html = '';
    for (const [target, feats] of Object.entries(res.explanations)) {
      html += `<div style="margin-bottom:8px;"><strong style="text-transform:capitalize;font-size:12px;">${target}</strong>`;
      const maxAbs = Math.max(...feats.map(f => Math.abs(f.importance)), 0.001);
      html += '<div style="margin-top:4px;">';
      for (const f of feats) {
        const pct = (Math.abs(f.importance) / maxAbs) * 100;
        const barColor = f.importance >= 0 ? 'var(--accent,#00d4aa)' : 'var(--danger,#ef4444)';
        const label = f.importance >= 0 ? 'pushes ↑' : 'pushes ↓';
        html += `<div style="display:flex;align-items:center;gap:6px;margin:2px 0;font-size:11px;">
          <span style="width:80px;text-align:right;flex-shrink:0;color:var(--text-dim,#6B7280);">${f.feature}</span>
          <div style="flex:1;height:14px;background:var(--bg-dark,#0A0E17);border-radius:3px;overflow:hidden;">
            <div style="width:${pct}%;height:100%;background:${barColor};border-radius:3px;transition:width 0.3s;"></div>
          </div>
          <span style="width:60px;font-size:10px;color:var(--text-dim,#6B7280);">${f.importance.toFixed(4)} ${label}</span>
        </div>`;
      }
      html += '</div></div>';
    }
    content.innerHTML = html;
  } catch {
    document.getElementById('xai-card').style.display = 'none';
  }
}

// Refresh XAI on button click
const refreshXaiBtn = document.getElementById('refresh-xai-btn');
if (refreshXaiBtn) {
  refreshXaiBtn.addEventListener('click', fetchXAI);
}

// Auto-refresh XAI on sensor data
socket.on('sensor_data', () => {
  if (document.getElementById('xai-card').style.display !== 'none') {
    fetchXAI();
  }
});
