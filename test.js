'use strict';

const DEFAULTS = { mode: 'mc', quality: 'balanced', auto: true, hud: false };
const LABEL = { off: 'Off', dup: 'Duplicate', blend: 'Blend', mc: 'Optical flow' };
const $ = (s) => document.querySelector(s);
const v = $('#v');
let settings = Object.assign({}, DEFAULTS);
let engine = null;

function paint() {
  $('#mode').value = settings.mode;
  $('#quality').value = settings.quality;
  $('#hud').checked = !!settings.hud;
}

function save(patch) {
  Object.assign(settings, patch);
  chrome.storage.local.set(patch);
  if (engine) engine.setOptions(settings);
}

const n = (x, d) => (x > 0 ? x.toFixed(d) : '–');

function update() {
  if (!engine) return;
  const r = engine.getStats();
  const state = $('#state');
  if (r.bypass) {
    state.textContent = r.bypass;
    state.className = /^(Off|Already|Measuring|Starting)/.test(r.bypass) ? 'idle' : 'warn';
  } else {
    state.textContent = 'Smoothing active: ' + LABEL[r.effMode] + (r.effMode === 'mc' ? ', ' + r.quality : '') + (r.note ? '. ' + r.note : '');
    state.className = r.note ? 'warn' : 'on';
  }
  const rows = [
    ['Source frame rate', n(r.srcFps, 1) + ' fps'],
    ['Frames shown', (r.bypass ? '–' : n(r.outFps, 0)) + ' fps'],
    ['Display refresh', n(r.dispHz, 0) + ' Hz'],
    ['Missed refreshes', (r.late * 100).toFixed(0) + '%'],
    ['Frame prep (GPU)', n(r.prepMs, 1) + ' ms'],
    ['Draw (GPU)', n(r.drawMs, 1) + ' ms'],
    ['Video size', r.res],
    ['Motion search size', r.est]
  ];
  const dl = $('#diag');
  dl.textContent = '';
  rows.forEach(([k, val]) => {
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = k;
    dd.textContent = val;
    dl.append(dt, dd);
  });
}

chrome.storage.local.get(DEFAULTS, (s) => {
  settings = Object.assign({}, DEFAULTS, s);
  paint();
  engine = new SmoothEngine(v, settings);
  engine.start();
  setInterval(update, 500);
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== 'local') return;
  for (const k in changes) settings[k] = changes[k].newValue;
  paint();
  if (engine) engine.setOptions(settings);
});

$('#mode').addEventListener('change', (e) => save({ mode: e.target.value }));
$('#quality').addEventListener('change', (e) => save({ quality: e.target.value }));
$('#hud').addEventListener('change', (e) => save({ hud: e.target.checked }));

$('#play').addEventListener('click', () => {
  if (v.paused) v.play(); else v.pause();
});
v.addEventListener('play', () => { $('#play').textContent = 'Pause'; });
v.addEventListener('pause', () => { $('#play').textContent = 'Play'; });

$('#file').addEventListener('change', (e) => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  v.src = URL.createObjectURL(f);
  v.play();
});
