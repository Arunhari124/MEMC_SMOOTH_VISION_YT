'use strict';

const DEFAULTS = { mode: 'mc', quality: 'balanced', auto: true, hud: false };
const LABEL = { off: 'Off', dup: 'Duplicate', blend: 'Blend', mc: 'Optical flow' };
const $ = (s) => document.querySelector(s);
let settings = Object.assign({}, DEFAULTS);

function paintSeg(id, value) {
  document.querySelectorAll('#' + id + ' button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.v === value)));
}

function paint() {
  paintSeg('mode', settings.mode);
  paintSeg('quality', settings.quality);
  $('#quality').classList.toggle('muted', settings.mode !== 'mc');
  $('#auto').checked = !!settings.auto;
  $('#hud').checked = !!settings.hud;
}

function save(patch) {
  Object.assign(settings, patch);
  chrome.storage.local.set(patch);
  paint();
}

chrome.storage.local.get(DEFAULTS, (s) => { settings = Object.assign({}, DEFAULTS, s); paint(); });

document.querySelectorAll('.seg').forEach((seg) => seg.addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (b) save({ [seg.id]: b.dataset.v });
}));
$('#auto').addEventListener('change', (e) => save({ auto: e.target.checked }));
$('#hud').addEventListener('change', (e) => save({ hud: e.target.checked }));
$('#test').addEventListener('click', () => chrome.tabs.create({ url: chrome.runtime.getURL('test.html') }));

function setDiag(rows) {
  const dl = $('#diag');
  dl.textContent = '';
  rows.forEach(([k, v]) => {
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = k;
    dd.textContent = v;
    dl.append(dt, dd);
  });
}

const n = (x, d) => (x > 0 ? x.toFixed(d) : '–');

function render(r) {
  const state = $('#state'), note = $('#note');
  if (!r) {
    state.textContent = 'Open a YouTube video to begin';
    state.className = 'idle';
    $('#src').textContent = '–';
    $('#out').textContent = '–';
    note.hidden = true;
    setDiag([]);
    return;
  }

  const bypass = r.bypass || '';
  if (!bypass) {
    state.textContent = 'Smoothing active: ' + LABEL[r.effMode] + (r.effMode === 'mc' ? ', ' + r.quality : '');
    state.className = 'on';
  } else {
    state.textContent = bypass;
    state.className = /^(Off|Already|Measuring|No video|Starting)/.test(bypass) ? 'idle' : 'warn';
  }

  $('#src').textContent = n(r.srcFps, 1);
  // With smoothing bypassed the video plays natively, so what you see is the source rate.
  const shown = bypass ? (/^(Measuring|No video|Starting)/.test(bypass) ? 0 : r.srcFps) : r.outFps;
  $('#out').textContent = n(shown, 0);

  note.hidden = !r.note;
  note.textContent = r.note || '';

  setDiag([
    ['Display refresh', n(r.dispHz, 0) + ' Hz'],
    ['Missed refreshes', (r.late * 100).toFixed(0) + '%'],
    ['Frame prep (GPU)', n(r.prepMs, 1) + ' ms'],
    ['Draw (GPU)', n(r.drawMs, 1) + ' ms'],
    ['Video size', r.res],
    ['Motion search size', r.est],
    ['WebGL2', r.gl ? (r.floatRT ? 'yes, float targets' : 'yes, no float targets') : 'no']
  ]);
}

function poll() {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tab = tabs && tabs[0];
    if (!tab) return render(null);
    chrome.tabs.sendMessage(tab.id, { type: 'sv:stats' }, { frameId: 0 }, (resp) => {
      if (chrome.runtime.lastError || !resp) return render(null);
      render(resp);
    });
  });
}

setInterval(poll, 500);
poll();
