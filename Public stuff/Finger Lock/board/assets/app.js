// Polls the board over plain HTTP (same approach as the step counter page).
const COLORS = { locked: '#9a998f', entering: '#f5b400', clear: '#f5b400',
                 unlocked: '#3cc47c', denied: '#e5484d', lockout: '#e5484d' };
const $ = id => document.getElementById(id);

const bars = $('bars');
for (let k = 0; k < 6; k++) {
  bars.insertAdjacentHTML('beforeend',
    `<div class="bar" id="bar${k}"><b>${k}</b><div class="track"><div class="fill"></div></div><span></span></div>`);
}

async function post(path) {
  try { await fetch(path, { method: 'POST' }); } catch (e) { console.log(e); }
}
$('b1').onclick = () => post('/api/button/start');
$('b2').onclick = () => post('/api/button/close');
$('lo').onclick = () => post('/api/latch/open');
$('lc').onclick = () => post('/api/latch/close');
$('ls').onclick = () => post('/api/latch/shorter');
$('ll').onclick = () => post('/api/latch/longer');

async function pollState() {
  try {
    const s = await (await fetch('/api/state')).json();
    $('conn').textContent = 'live';
    $('state').textContent = s.state.toUpperCase();
    $('state').style.color = COLORS[s.state] || '#fff';
    $('msg').textContent = s.message;
    $('leds').classList.toggle('on', s.leds);
    for (let i = 0; i < 3; i++) {
      const el = $('s' + i);
      el.textContent = i < s.entered.length ? s.entered[i] : '';
      el.classList.toggle('now', s.state === 'entering' && i === s.entered.length);
    }
    $('hold').style.width = (s.candidate ? s.hold * 100 : 0) + '%';
    const best = s.probs.indexOf(Math.max(...s.probs));
    s.probs.forEach((p, k) => {
      const bar = $('bar' + k);
      bar.classList.toggle('best', k === best);
      bar.querySelector('.fill').style.width = (p * 100) + '%';
      bar.querySelector('span').textContent = Math.round(p * 100) + '%';
    });
    $('fps').textContent = `${s.fps} frames per second on the UNO Q`;
    $('latch').textContent = s.latch.position + (s.latch.moving ? ' (moving)' : '');
    $('ms').textContent = s.move_ms;
  } catch (e) {
    $('conn').textContent = 'reconnecting…';
  }
  setTimeout(pollState, 200);
}

async function pollImage() {
  try {
    const r = await (await fetch('/api/image')).json();
    if (r.jpg) $('cam').src = 'data:image/jpeg;base64,' + r.jpg;
  } catch (e) {}
  setTimeout(pollImage, 300);
}

pollState();
pollImage();
