// Polls the board over plain HTTP -- works on iPhone Safari on the board's
// own hotspot, where the socket connection never came up.
const EMOJI = { standing: '🧍', walking: '🚶', running: '🏃', biking: '🚴' };
const POLL_MS = 200;   // the models update every ~0.2 s
const STALE_MS = 1500; // no good answer this long -> grey out

const connEl = document.getElementById('conn');
const mainEl = document.getElementById('main');
const emojiEl = document.getElementById('emoji');
const gaitEl = document.getElementById('gaitlabel');
const m1El = document.getElementById('m1label');
const agreeEl = document.getElementById('agree');
const stepCountEl = document.getElementById('stepcount');
const cadenceEl = document.getElementById('cadence');
const netNameEl = document.getElementById('netname');
const distWalkedEl = document.getElementById('distwalked');
const stepPmEl = document.getElementById('steppm');
const distPmEl = document.getElementById('distpm');
// 90% of held-out test walks were within these (training/results/
// cv_results.csv and distance_results.csv). See "How accurate is this?".
const STEP_PM_FRAC = 0.106;
const DIST_PM_FRAC = 0.105;
const NET_NAMES = {
  imu_dist: 'IMU + ultrasonic net',
  imu: 'IMU-only net (distance sensor not answering)',
  still: 'not moving',
};
const featuresEl = document.getElementById('features');
const distNowEl = document.getElementById('distnow');
const distMinEl = document.getElementById('distmin');
const sensorWarnEl = document.getElementById('sensorwarn');

// "Alive" = the HC-SR04 answered the ping, whether or not anything was in
// range (dist -1 = no answer, -2 = answered but nothing in range). A working
// sensor is alive ~100% of the time even in a big room, so below this over
// the board's 2 s window the wiring has almost certainly come loose.
const ALIVE_WARN_FRAC = 0.2;
// Recordings below this are suspect -- suggest discarding before saving.
const ALIVE_SAVE_FRAC = 0.8;

let lastOk = 0;

function setConn(text, live) {
  connEl.textContent = text;
  connEl.classList.toggle('live', live);
}

function show(data) {
  const s = data.steps;
  stepCountEl.textContent = Math.round(s.total);
  distWalkedEl.textContent = s.distance_m < 1000
    ? `${s.distance_m.toFixed(1)} m` : `${(s.distance_m / 1000).toFixed(2)} km`;
  // At least ±1 step: a single step at the start or stop is the usual miss.
  stepPmEl.textContent = s.total >= 1 ? `± ${Math.max(1, Math.round(STEP_PM_FRAC * s.total))}` : '';
  distPmEl.textContent = s.distance_m >= 0.1 ? `(± ${(DIST_PM_FRAC * s.distance_m).toFixed(1)} m)` : '';
  cadenceEl.textContent = s.cadence_spm > 0
    ? `${s.cadence_spm} steps/min · ${s.step_len_m.toFixed(2)} m/step` : '0 steps/min';
  netNameEl.textContent = NET_NAMES[s.model] || 'neural net';
  mainEl.style.borderColor = s.cadence_spm > 20 ? 'var(--walking)' : 'var(--muted)';

  // Gait ID rule models from the last project, re-fitted for the new mounting.
  emojiEl.textContent = EMOJI[data.model2] || '';
  gaitEl.textContent = data.model2;
  gaitEl.style.color = `var(--${data.model2})`;
  m1El.textContent = data.model1;
  m1El.style.color = `var(--${data.model1})`;
  agreeEl.textContent = data.model1 === data.model2 ? 'agrees' : 'disagrees';

  const f = data.features;
  distNowEl.textContent = f.dist_now === -1 ? 'not responding'
    : f.dist_now === -2 ? 'nothing in range' : `${f.dist_now.toFixed(1)} cm`;
  distMinEl.textContent = f.dist_min < 0 ? '' : `closest in 2 s: ${f.dist_min.toFixed(1)} cm`;
  sensorWarnEl.hidden = f.dist_alive_frac >= ALIVE_WARN_FRAC;

  featuresEl.innerHTML = Object.entries(data.features)
    .map(([k, v]) => `<tr><td>${k}</td><td>${Number(v).toFixed(2)}</td></tr>`)
    .join('');
}

document.getElementById('resetbtn').onclick = async () => {
  await fetch('api/steps/reset', { method: 'POST', cache: 'no-store' });
  stepCountEl.textContent = '0';
  distWalkedEl.textContent = '0.0 m';
};

// --- Recording ------------------------------------------------------------
// Start -> Stop -> type the steps you counted -> Save (or Discard).
const startBtn = document.getElementById('startbtn');
const stopBtn = document.getElementById('stopbtn');
const saveBtn = document.getElementById('savebtn');
const discardBtn = document.getElementById('discardbtn');
const setupEl = document.getElementById('recsetup');
const stopFormEl = document.getElementById('recstop');
const recStatusEl = document.getElementById('recstatus');
const recListEl = document.getElementById('reclist');
const stepsEl = document.getElementById('steps');
const distanceEl = document.getElementById('distance');
const recWarnEl = document.getElementById('recwarn');

// 'idle' | 'recording' | 'stopping' (stopped on the page, waiting for the count)
let recState = 'idle';

function setRecState(state) {
  recState = state;
  setupEl.hidden = state !== 'idle';
  startBtn.hidden = state !== 'idle';
  stopBtn.hidden = state !== 'recording';
  stopFormEl.hidden = state !== 'stopping';
}

async function api(method, path, params = {}) {
  const res = await fetch(`${path}?${new URLSearchParams(params)}`, { method, cache: 'no-store' });
  return res.json();
}

async function refreshList() {
  try {
    const { recordings } = await api('GET', 'api/recordings');
    recListEl.innerHTML = recordings.slice(0, 8).map((r) => {
      const steps = r.steps_counted >= 0 ? `${r.steps_counted} steps` : 'steps ?';
      // alive_frac is missing on recordings made before it was tracked.
      const warn = r.alive_frac != null && r.alive_frac < ALIVE_SAVE_FRAC
        ? ` &middot; <span style="color:#f28b82">&#9888; sensor ${Math.round(100 * r.alive_frac)}%</span>` : '';
      return `<li>${r.activity} &middot; ${steps} &middot; ${Math.round(r.duration_s)} s${warn}</li>`;
    }).join('');
    if (recState === 'idle') {
      recStatusEl.textContent = `${recordings.length} recordings saved`;
    }
  } catch (err) {
    // board unreachable; the poll loop shows that already
  }
}

startBtn.onclick = async () => {
  const r = await api('POST', 'api/record/start', {
    activity: document.getElementById('activity').value,
    note: document.getElementById('note').value,
  });
  if (!r.ok) { alert(r.error); return; }
  setRecState('recording');
};

// Stop pauses the board's recording right away; Save/Discard then closes it.
stopBtn.onclick = async () => {
  await api('POST', 'api/record/pause');
  setRecState('stopping');
  stepsEl.value = '';
  stepsEl.focus();
};

saveBtn.onclick = async () => {
  if (stepsEl.value === '') { alert('Type how many steps you counted (0 is fine).'); return; }
  const params = { steps: stepsEl.value };
  if (distanceEl.value !== '') params.distance_m = distanceEl.value;
  const r = await api('POST', 'api/record/stop', params);
  if (!r.ok) { alert(r.error); }
  distanceEl.value = '';
  setRecState('idle');
  refreshList();
};

discardBtn.onclick = async () => {
  if (!confirm('Throw this recording away?')) return;
  await api('POST', 'api/record/stop', { keep: false });
  setRecState('idle');
  refreshList();
};

function showRec(rec) {
  if (!rec) return;
  // Recover if the page was reloaded mid-recording.
  if (rec.active && recState === 'idle') setRecState(rec.paused ? 'stopping' : 'recording');
  if (!rec.active && recState !== 'idle') setRecState('idle');
  if (rec.active) {
    const alivePct = Math.round(100 * rec.alive_frac);
    recStatusEl.classList.add('on');
    recStatusEl.textContent = rec.paused
      ? `Stopped at ${rec.elapsed_s.toFixed(1)} s · ${rec.samples} samples · sensor ${alivePct}%. Save or Discard.`
      : `● REC ${rec.elapsed_s.toFixed(1)} s · ${rec.samples} samples · sensor ${alivePct}%`;
    const bad = rec.paused && rec.samples > 0 && rec.alive_frac < ALIVE_SAVE_FRAC;
    recWarnEl.hidden = !bad;
    if (bad) {
      recWarnEl.textContent = `⚠ The distance sensor only answered ${alivePct}% of the time (normal is ~100%). ` +
        'It probably came loose. Discard this one, fix the wiring, and redo it.';
    }
  } else {
    recStatusEl.classList.remove('on');
  }
}

async function poll() {
  try {
    const res = await fetch('api/gait', { cache: 'no-store' });
    const data = await res.json();
    showRec(data.rec);
    if (data.ready) {
      show(data);
      lastOk = Date.now();
      document.body.classList.remove('stale');
      setConn('Live', true);
    } else {
      setConn('Calibrating…', false);
    }
  } catch (err) {
    // fall through to the stale check below
  }
  if (lastOk && Date.now() - lastOk > STALE_MS) {
    document.body.classList.add('stale');
    setConn('No data', false);
  }
  setTimeout(poll, POLL_MS);
}

setRecState('idle');
refreshList();
poll();
