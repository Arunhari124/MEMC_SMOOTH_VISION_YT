(() => {
  'use strict';

  const DEFAULTS = { mode: 'mc', quality: 'balanced', auto: true, hud: false };
  let settings = Object.assign({}, DEFAULTS);
  let lastOn = DEFAULTS.mode;
  let engine = null;
  let timer = 0;

  const alive = () => { try { return !!chrome.runtime.id; } catch (e) { return false; } };

  function findVideo() {
    const all = Array.from(document.querySelectorAll('video.html5-main-video')).filter((v) => v.offsetWidth > 0);
    return all.find((v) => !v.paused) || all[0] || null;
  }

  function ensure() {
    if (!alive()) { // extension was reloaded or removed: clean up this orphaned script
      if (engine) engine.destroy();
      engine = null;
      clearInterval(timer);
      return;
    }
    const v = findVideo();
    if (!v || (engine && engine.video === v)) return;
    if (engine) engine.destroy();
    engine = new SmoothEngine(v, settings);
    engine.start();
  }

  chrome.storage.local.get(DEFAULTS, (s) => {
    settings = Object.assign({}, DEFAULTS, s);
    if (settings.mode !== 'off') lastOn = settings.mode;
    if (engine) engine.setOptions(settings);
    ensure();
  });

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== 'local') return;
    for (const k in changes) settings[k] = changes[k].newValue;
    if (settings.mode !== 'off') lastOn = settings.mode;
    if (engine) engine.setOptions(settings);
  });

  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg && msg.type === 'sv:stats') {
      sendResponse(engine ? engine.getStats() : { bypass: 'No video found on this page', gl: true });
    }
  });

  // Alt+S: quick on/off for A/B comparison
  window.addEventListener('keydown', (e) => {
    if (!e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || e.code !== 'KeyS') return;
    e.preventDefault();
    try { chrome.storage.local.set({ mode: settings.mode === 'off' ? lastOn : 'off' }); } catch (err) { /* orphaned */ }
  }, true);

  timer = setInterval(ensure, 1000); // YouTube is a single-page app; the player element can be swapped at any time
  ensure();
})();
