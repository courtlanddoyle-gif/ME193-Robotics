// Polls the board over plain HTTP -- works on iPhone Safari on the board's
// own hotspot, where the socket connection never came up.
const EMOJI = { standing: '🧍', walking: '🚶', running: '🏃', biking: '🚴' };
const POLL_MS = 200;   // the models update every ~0.2 s
const STALE_MS = 1500; // no good answer this long -> grey out

const connEl = document.getElementById('conn');
const mainEl = document.getElementById('main');
const emojiEl = document.getElementById('emoji');
const labelEl = document.getElementById('label');
const m1El = document.getElementById('m1label');
const agreeEl = document.getElementById('agree');
const featuresEl = document.getElementById('features');

let lastOk = 0;

function setConn(text, live) {
  connEl.textContent = text;
  connEl.classList.toggle('live', live);
}

function show(data) {
  emojiEl.textContent = EMOJI[data.model2] || '?';
  labelEl.textContent = data.model2;
  mainEl.style.borderColor = `var(--${data.model2})`;

  m1El.textContent = data.model1;
  m1El.style.color = `var(--${data.model1})`;
  agreeEl.textContent = data.model1 === data.model2 ? 'agrees' : 'disagrees';

  featuresEl.innerHTML = Object.entries(data.features)
    .map(([k, v]) => `<tr><td>${k}</td><td>${Number(v).toFixed(2)}</td></tr>`)
    .join('');
}

async function poll() {
  try {
    const res = await fetch('api/gait', { cache: 'no-store' });
    const data = await res.json();
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

poll();
