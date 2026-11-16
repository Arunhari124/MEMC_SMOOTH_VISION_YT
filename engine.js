/* SmoothVision engine
   Real-time frame interpolation (MEMC-style) for a <video>, entirely on the local GPU via WebGL2.
   No network requests, no remote code. */
(function (root) {
  'use strict';

  const QUALITY = {
    low:      { estW: 320, range: 6 },
    balanced: { estW: 480, range: 8 },
    high:     { estW: 640, range: 10 }
  };
  const LABEL = { off: 'Off', dup: 'Duplicate', blend: 'Blend', mc: 'Optical flow' };
  const MAX_PIXELS = 1920 * 1088;
  const CUT = 0.15; // mean luma change between two frames that counts as a scene cut
  const ema = (old, v) => (old ? old * 0.7 + v * 0.3 : v);

  /* ------------------------------------------------------------------ shaders (GLSL ES 3.00) */
  const VS = `#version 300 es
layout(location = 0) in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

  const HEAD = `#version 300 es
precision highp float;
precision highp int;
in vec2 v_uv;
out vec4 o;
`;

  const FS = {
    // video frame -> small luma image (mip-filtered, so no aliasing)
    luma: HEAD + `
uniform sampler2D u_src;
uniform float u_lod;
void main() {
  vec3 c = textureLod(u_src, v_uv, u_lod).rgb;
  float y = dot(c, vec3(0.2126, 0.7152, 0.0722));
  o = vec4(y, y, y, 1.0);
}`,

    // |prev - cur| at 256x128; its top mip level is the mean change (scene-cut detector, no CPU readback)
    diff: HEAD + `
uniform sampler2D u_a;
uniform sampler2D u_b;
void main() {
  float d = abs(textureLod(u_a, v_uv, 0.0).r - textureLod(u_b, v_uv, 0.0).r);
  o = vec4(d, d, d, 1.0);
}`,

    // plain crossfade; also used to show an original frame unchanged (t = 0 or 1)
    blend: HEAD + `
uniform sampler2D u_prev;
uniform sampler2D u_cur;
uniform sampler2D u_diff;
uniform float u_t;
uniform float u_cut;
void main() {
  vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
  float t = u_t;
  if (t < 0.999 && u_cut > 0.0 && textureLod(u_diff, vec2(0.5), 8.0).r > u_cut) t = 0.0;
  o = vec4(mix(textureLod(u_prev, uv, 0.0).rgb, textureLod(u_cur, uv, 0.0).rgb, t), 1.0);
}`,

    // symmetric block matching: one vector per 8x8 block of the small luma image.
    // v = displacement prev -> cur, found by comparing prev(x - v/2) with cur(x + v/2).
    me: HEAD + `
uniform sampler2D u_prev;
uniform sampler2D u_cur;
uniform vec2 u_size;
uniform int u_range;

float cost(vec2 base, vec2 v) {
  float s = 0.0;
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
      vec2 p = base + vec2(float(2 * i + 1), float(2 * j + 1));
      float a = textureLod(u_prev, (p - 0.5 * v) / u_size, 0.0).r;
      float b = textureLod(u_cur, (p + 0.5 * v) / u_size, 0.0).r;
      s += abs(a - b);
    }
  }
  return s * 0.0625;
}

float par(float a, float b, float c) {
  float d = a + c - 2.0 * b;
  return d > 1e-5 ? clamp(0.5 * (a - c) / d, -0.5, 0.5) : 0.0;
}

void main() {
  vec2 base = floor(gl_FragCoord.xy) * 8.0;
  float c0 = cost(base, vec2(0.0));
  vec2 best = vec2(0.0);
  if (c0 > 0.004) {
    float bc = c0;
    for (int dy = -u_range; dy <= u_range; dy++) {
      for (int dx = -u_range; dx <= u_range; dx++) {
        vec2 v = vec2(float(dx), float(dy));
        float c = cost(base, v) + 0.0012 * length(v);
        if (c < bc) { bc = c; best = v; }
      }
    }
    float cc = cost(base, best);
    best += vec2(par(cost(base, best + vec2(-1.0, 0.0)), cc, cost(base, best + vec2(1.0, 0.0))),
                 par(cost(base, best + vec2(0.0, -1.0)), cc, cost(base, best + vec2(0.0, 1.0))));
  }
  o = vec4(best, c0, 1.0);
}`,

    // motion-compensated frame at time t: every pixel tries the vectors of its 4 nearest blocks,
    // keeps the ones whose prev/cur samples agree, and falls back to a crossfade where nothing matches.
    mc: HEAD + `
uniform sampler2D u_prev;
uniform sampler2D u_cur;
uniform sampler2D u_mv;
uniform sampler2D u_diff;
uniform float u_t;
uniform float u_cut;
uniform vec2 u_est;
uniform vec2 u_gs;
void main() {
  vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
  vec3 p0 = textureLod(u_prev, uv, 0.0).rgb;
  if (textureLod(u_diff, vec2(0.5), 8.0).r > u_cut) { o = vec4(p0, 1.0); return; }
  vec3 c0 = textureLod(u_cur, uv, 0.0).rgb;
  vec3 fallback = mix(p0, c0, u_t);

  vec2 g = uv * u_gs - 0.5;
  vec2 gf = floor(g);
  vec2 f = g - gf;
  ivec2 g0 = ivec2(gf);
  ivec2 gmax = ivec2(ceil(u_gs)) - ivec2(1);

  vec3 acc = vec3(0.0);
  float ws = 0.0;
  float es = 0.0;
  for (int j = 0; j < 2; j++) {
    for (int i = 0; i < 2; i++) {
      ivec2 gi = clamp(g0 + ivec2(i, j), ivec2(0), gmax);
      vec2 v = texelFetch(u_mv, gi, 0).xy / u_est;
      float wb = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
      vec3 p = textureLod(u_prev, uv - v * u_t, 0.0).rgb;
      vec3 c = textureLod(u_cur, uv + v * (1.0 - u_t), 0.0).rgb;
      float e = dot(abs(p - c), vec3(0.3333));
      float w = wb * exp(-12.0 * e) + 1e-5;
      acc += w * mix(p, c, u_t);
      ws += w;
      es += w * e;
    }
  }
  float conf = 1.0 - smoothstep(0.06, 0.18, es / ws);
  o = vec4(mix(fallback, acc / ws, conf), 1.0);
}`
  };

  const UNIFORMS = {
    luma:  ['u_src', 'u_lod'],
    diff:  ['u_a', 'u_b'],
    blend: ['u_prev', 'u_cur', 'u_diff', 'u_t', 'u_cut'],
    me:    ['u_prev', 'u_cur', 'u_size', 'u_range'],
    mc:    ['u_prev', 'u_cur', 'u_mv', 'u_diff', 'u_t', 'u_cut', 'u_est', 'u_gs']
  };

  function makeProgram(gl, fsSrc, names) {
    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(s) || 'unknown error';
        gl.deleteShader(s);
        throw new Error(log.trim());
      }
      return s;
    };
    const vs = compile(gl.VERTEX_SHADER, VS);
    const fs = compile(gl.FRAGMENT_SHADER, fsSrc);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: ' + (gl.getProgramInfoLog(prog) || '').trim());
    const u = {};
    for (const n of names) u[n] = gl.getUniformLocation(prog, n);
    return { prog, u };
  }

  /* ------------------------------------------------------------------ timeline (pure logic, no GL)
     Keeps the last two decoded frames and works out, for every display refresh, how far
     between them the picture on screen should be. Output runs one source frame behind the video
     (the unavoidable cost of interpolating between two known frames). */
  class Timeline {
    constructor() { this.reset(); }

    reset() {
      this.gaps = [];
      this.nom = 1 / 30;      // nominal frame interval in media seconds
      this.lastMt = null;
      this.leads = [];
      this.lead = 0;          // ms between a frame's callback and its display time (smallest recently seen)
      this.dropFrames();
    }

    dropFrames() { this.prev = null; this.cur = null; this.off = null; }
    resetClock() { this.off = null; }

    get known() { return this.gaps.length >= 6; }
    get fps() { return 1 / this.nom; }

    // called for every frame callback; returns false for a repeat of the same frame
    observe(mt) {
      const last = this.lastMt;
      this.lastMt = mt;
      if (last === null) return true;
      const g = mt - last;
      if (Math.abs(g) < 1e-6) return false;
      if (g > 0 && g < 0.25) {
        this.gaps.push(g);
        if (this.gaps.length > 15) this.gaps.shift();
        const s = this.gaps.slice().sort((a, b) => a - b);
        this.nom = s[s.length >> 1];
      } else {
        this.dropFrames(); // seek, stall or new video
      }
      return true;
    }

    // make `mt` (media time, s) the newest frame; displayMs/nowMs come from requestVideoFrameCallback
    push(mt, displayMs, nowMs, rate, k) {
      this.prev = this.cur;
      this.cur = { mt, k };
      const L = displayMs - nowMs;
      if (isFinite(L)) {
        this.leads.push(L);
        if (this.leads.length > 30) this.leads.shift();
        this.lead = Math.max(0, Math.min(40, Math.min.apply(null, this.leads)));
      }
      // Map wall clock -> media time. Display times are quantised to vsync, so average them out.
      const e = isFinite(displayMs) ? displayMs / 1000 : nowMs / 1000;
      const c = mt - e * rate;
      if (this.off === null || Math.abs(c - this.off) > 0.08) this.off = c;
      else this.off += 0.04 * (c - this.off);
    }

    // 0 = show prev, 1 = show cur, in between = synthesise
    position(tsMs, rate) {
      const prev = this.prev, cur = this.cur;
      if (!cur || !prev || this.off === null) return 1;
      const span = cur.mt - prev.mt;
      if (!(span > 0)) return 1;
      const x = (tsMs + this.lead) / 1000;
      const target = this.off + x * rate - this.nom;
      const t = (target - prev.mt) / span;
      return t < 0 ? 0 : t > 1 ? 1 : t;
    }
  }

  /* ------------------------------------------------------------------ engine */
  class SmoothEngine {
    constructor(video, opts) {
      this.video = video;
      this.opts = Object.assign({ mode: 'mc', quality: 'balanced', auto: true, hud: false }, opts || {});
      this.tl = new Timeline();

      const c = document.createElement('canvas');
      c.setAttribute('data-smoothvision', '');
      Object.assign(c.style, { position: 'absolute', pointerEvents: 'none', zIndex: '1', display: 'none', background: '#000' });
      this.canvas = c;

      this.gl = null; this.glError = ''; this.floatRT = false; this.lost = false;
      this.progs = {}; this.progErr = {};
      this.F = null; this.L = null; this.fbL = null; this.MV = null; this.fbMV = null; this.D = null; this.fbD = null;
      this.fw = 0; this.fh = 0; this.allocQ = ''; this.srcW = 0; this.srcH = 0;
      this.estW = 0; this.estH = 0; this.mvKey = null;

      this.eff = { mode: 'mc', quality: 'balanced' };
      this.govOff = false; this.note = ''; this.fatal = ''; this.errs = 0;
      this.bypass = 'Starting…'; this._active = false;

      this.refMs = 1000 / 60; this.dispHz = 60; this.skip = 1; this._dts = [];
      this._ts = 0; this._wn = 0; this._wl = 0; this.lateRatio = 0; this._graceUntil = 0;
      this._statT = 0; this.nTick = 0; this.nTick1 = 0; this.nDraw1 = 0; this.outFps = 0;
      this.prepMs = 0; this.drawMs = 0; this._nProc = 0; this._nDraw = 0; this._demandUntil = 0;
      this._lastKey = ''; this._styleSig = ''; this._layoutDirty = true;
      this.destroyed = false; this.started = false;
    }

    /* ---------- lifecycle */
    start() {
      if (this.started) return;
      this.started = true;
      this._getContext();
      this._applyOpts();
      const v = this.video;
      this._onFrameB = this._onFrame.bind(this);
      this._tickB = this._tick.bind(this);
      this._ev = [
        ['seeking',    () => { this.tl.dropFrames(); this._lastKey = ''; }],
        ['emptied',    () => { this.tl.reset(); this.note = ''; this._lastKey = ''; }],
        ['loadstart',  () => { this.tl.reset(); this.note = ''; this._lastKey = ''; }],
        ['ratechange', () => this.tl.resetClock()],
        ['playing',    () => this.tl.resetClock()]
      ];
      this._ev.forEach(([n, f]) => v.addEventListener(n, f));
      this.ro = new ResizeObserver(() => { this._layoutDirty = true; });
      this.ro.observe(v);
      this.canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); this.lost = true; });
      this.canvas.addEventListener('webglcontextrestored', () => { this.lost = false; this._rebuild(); });
      if ('requestVideoFrameCallback' in v) this.vfc = v.requestVideoFrameCallback(this._onFrameB);
      else this.fatal = 'This Chrome is too old (needs 83+)';
      this.raf = requestAnimationFrame(this._tickB);
    }

    destroy() {
      if (this.destroyed) return;
      this.destroyed = true;
      const v = this.video;
      try { if (this.vfc && v.cancelVideoFrameCallback) v.cancelVideoFrameCallback(this.vfc); } catch (e) { /* ignore */ }
      cancelAnimationFrame(this.raf);
      (this._ev || []).forEach(([n, f]) => v.removeEventListener(n, f));
      if (this.ro) this.ro.disconnect();
      try {
        this._free();
        const x = this.gl && this.gl.getExtension('WEBGL_lose_context');
        if (x) x.loseContext();
      } catch (e) { /* ignore */ }
      this.canvas.remove();
      if (this.hudEl) this.hudEl.remove();
    }

    setOptions(o) {
      const old = this.opts;
      const changed = (o.mode !== undefined && o.mode !== old.mode) || (o.quality !== undefined && o.quality !== old.quality);
      Object.assign(this.opts, o);
      if (changed) this._applyOpts();
      this._hud();
    }

    _applyOpts() {
      const o = this.opts;
      this.eff = { mode: o.mode, quality: QUALITY[o.quality] ? o.quality : 'balanced' };
      this.govOff = false;
      this.note = '';
      if (this.eff.mode === 'mc' && this.gl && !this.floatRT) {
        this.eff.mode = 'blend';
        this.note = 'Optical flow needs float render targets on this GPU, so Blend is used.';
      }
      this._graceUntil = performance.now() + 2500;
      this._wn = 0; this._wl = 0; this._lastKey = '';
    }

    /* ---------- WebGL setup */
    _getContext() {
      let gl = null;
      try {
        gl = this.canvas.getContext('webgl2', {
          alpha: false, antialias: false, depth: false, stencil: false,
          powerPreference: 'high-performance', preserveDrawingBuffer: false
        });
      } catch (e) { /* ignore */ }
      if (!gl) { this.glError = 'WebGL2 isn’t available (check chrome://gpu)'; return; }
      this.gl = gl;
      this.floatRT = !!gl.getExtension('EXT_color_buffer_float');
      this._setupGL();
    }

    _setupGL() {
      const gl = this.gl;
      this.vao = gl.createVertexArray();
      gl.bindVertexArray(this.vao);
      const vb = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vb);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
      this.D = this._tex(256, 128, gl.RGBA8, 9, gl.LINEAR_MIPMAP_NEAREST, gl.LINEAR);
      this.fbD = this._fbo(this.D);
    }

    _rebuild() {
      this.progs = {}; this.progErr = {};
      this.fw = 0; this.F = this.L = this.fbL = this.MV = this.fbMV = null;
      this.tl.dropFrames();
      try { this._setupGL(); } catch (e) { this.fatal = 'Could not restore the GPU context'; }
    }

    _tex(w, h, fmt, levels, minF, magF) {
      const gl = this.gl, t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texStorage2D(gl.TEXTURE_2D, levels, fmt, w, h);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, minF);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, magF);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    }

    _fbo(tex) {
      const gl = this.gl, f = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('framebuffer incomplete');
      return f;
    }

    _alloc(w, h) {
      const gl = this.gl;
      this._free();
      const q = QUALITY[this.eff.quality] || QUALITY.balanced;
      this.fw = w; this.fh = h;
      this.estW = Math.min(q.estW, w);
      this.estH = Math.max(16, Math.round(this.estW * h / w));
      this.lod = Math.max(0, Math.log2(w / this.estW));
      const levels = Math.floor(Math.log2(Math.max(w, h))) + 1;
      this.F = [0, 1].map(() => this._tex(w, h, gl.RGBA8, levels, gl.LINEAR_MIPMAP_LINEAR, gl.LINEAR));
      this.L = [0, 1].map(() => this._tex(this.estW, this.estH, gl.RGBA8, 1, gl.LINEAR, gl.LINEAR));
      this.fbL = this.L.map((t) => this._fbo(t));
      if (this.floatRT) {
        try {
          this.MV = this._tex(Math.ceil(this.estW / 8), Math.ceil(this.estH / 8), gl.RGBA16F, 1, gl.NEAREST, gl.NEAREST);
          this.fbMV = this._fbo(this.MV);
        } catch (e) {
          this.floatRT = false; this.MV = null; this.fbMV = null;
          if (this.eff.mode === 'mc') { this.eff.mode = 'blend'; this.note = 'Float render target failed, so Blend is used.'; }
        }
      }
      this.allocQ = this.eff.quality;
      if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
      this.mvKey = null; this._lastKey = ''; this._layoutDirty = true;
    }

    _free() {
      const gl = this.gl;
      if (!gl) return;
      (this.F || []).forEach((t) => gl.deleteTexture(t));
      (this.L || []).forEach((t) => gl.deleteTexture(t));
      (this.fbL || []).forEach((f) => gl.deleteFramebuffer(f));
      if (this.MV) gl.deleteTexture(this.MV);
      if (this.fbMV) gl.deleteFramebuffer(this.fbMV);
      this.F = this.L = this.fbL = this.MV = this.fbMV = null;
      this.fw = 0; this.mvKey = null;
    }

    _prog(name) {
      if (this.progs[name]) return this.progs[name];
      if (this.progErr[name]) throw new Error(this.progErr[name]);
      try {
        this.progs[name] = makeProgram(this.gl, FS[name], UNIFORMS[name]);
        return this.progs[name];
      } catch (e) {
        this.progErr[name] = name + ': ' + e.message;
        throw new Error(this.progErr[name]);
      }
    }

    _run(name, fbo, w, h, texs, setU) {
      const gl = this.gl, p = this._prog(name);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.viewport(0, 0, w, h);
      gl.useProgram(p.prog);
      for (let i = 0; i < texs.length; i++) {
        gl.activeTexture(gl.TEXTURE0 + i);
        gl.bindTexture(gl.TEXTURE_2D, texs[i]);
      }
      setU(p.u);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    _fail(e) {
      const m = String((e && e.message) || e);
      console.warn('[SmoothVision]', m);
      if (/^(me|mc):/.test(m) && this.eff.mode === 'mc') {
        this.eff.mode = 'blend';
        this.note = 'The optical-flow shader failed on this GPU, so Blend is used. ' + m.slice(0, 140);
        this._lastKey = '';
        return;
      }
      if (++this.errs > 4) this.fatal = m.slice(0, 160);
    }

    /* ---------- a new decoded frame arrived */
    _onFrame(now, md) {
      if (this.destroyed) return;
      const v = this.video;
      this.vfc = v.requestVideoFrameCallback(this._onFrameB);
      if (v.readyState < 2) return;
      const mt = md.mediaTime;
      if (!this.tl.observe(mt)) return;
      this.srcW = md.width || v.videoWidth;
      this.srcH = md.height || v.videoHeight;
      if (this.bypass) { this._active = false; return; }
      if (!this._active) { this._active = true; this.tl.dropFrames(); }
      if (!this.gl || this.lost || !this.srcW || !this.srcH) return;
      try {
        if (this.fw !== this.srcW || this.fh !== this.srcH || this.allocQ !== this.eff.quality) {
          this._alloc(this.srcW, this.srcH);
          this.tl.dropFrames();
        }
        const k = this.tl.cur ? 1 - this.tl.cur.k : 0;
        this.tl.push(mt, md.expectedDisplayTime, now, v.playbackRate || 1, k);
        this._process(k);
      } catch (e) { this._fail(e); }
    }

    _wantTiming() { return this.opts.hud || performance.now() < this._demandUntil; }

    _process(k) {
      const gl = this.gl, tl = this.tl, mode = this.eff.mode;
      const sample = this._wantTiming() && (this._nProc++ % 15 === 0);
      const t0 = sample ? performance.now() : 0;
      gl.bindTexture(gl.TEXTURE_2D, this.F[k]);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, this.video);
      if (mode !== 'dup') {
        gl.generateMipmap(gl.TEXTURE_2D);
        this._run('luma', this.fbL[k], this.estW, this.estH, [this.F[k]], (u) => {
          gl.uniform1i(u.u_src, 0); gl.uniform1f(u.u_lod, this.lod);
        });
        if (tl.prev) {
          const pk = tl.prev.k;
          this._run('diff', this.fbD, 256, 128, [this.L[pk], this.L[k]], (u) => {
            gl.uniform1i(u.u_a, 0); gl.uniform1i(u.u_b, 1);
          });
          gl.bindFramebuffer(gl.FRAMEBUFFER, null);
          gl.bindTexture(gl.TEXTURE_2D, this.D);
          gl.generateMipmap(gl.TEXTURE_2D);
          if (mode === 'mc' && this.MV) {
            this._run('me', this.fbMV, Math.ceil(this.estW / 8), Math.ceil(this.estH / 8), [this.L[pk], this.L[k]], (u) => {
              gl.uniform1i(u.u_prev, 0); gl.uniform1i(u.u_cur, 1);
              gl.uniform2f(u.u_size, this.estW, this.estH);
              gl.uniform1i(u.u_range, QUALITY[this.allocQ].range);
            });
            this.mvKey = tl.cur.mt;
          }
        }
      }
      if (sample) { gl.finish(); this.prepMs = ema(this.prepMs, performance.now() - t0); }
    }

    /* ---------- every display refresh */
    _timing(ts) {
      const now = performance.now();
      const dt = ts - this._ts;
      this._ts = ts;
      this.nTick++; this.nTick1++;
      if (dt > 0 && dt < 1000) {
        if (this.bypass) {
          // measure the monitor's refresh only while we put no load on the GPU
          const d = this._dts;
          d.push(dt);
          if (d.length > 120) d.shift();
          if (d.length >= 20) {
            const s = d.slice().sort((a, b) => a - b);
            this.refMs = Math.min(34, Math.max(6.5, s[Math.floor(s.length * 0.15)]));
            this.dispHz = 1000 / this.refMs;
          }
        } else if (!this.video.paused && !document.hidden && now > this._graceUntil) {
          this._wn++;
          if (dt > this.refMs * 1.5) this._wl++;
          if (this._wn >= 120) {
            this.lateRatio = this._wl / this._wn;
            if (this.opts.auto && this.lateRatio > 0.2) this._degrade();
            this._wn = 0; this._wl = 0;
          }
        }
      }
      this.skip = this.dispHz > 75 ? Math.round(this.dispHz / 60) : 1;
      if (now - this._statT >= 1000) {
        const sec = (now - this._statT) / 1000;
        this.outFps = this._statT ? this.nDraw1 / sec : 0;
        this.nDraw1 = 0; this.nTick1 = 0; this._statT = now;
        this._hud();
      }
    }

    _degrade() {
      const e = this.eff;
      const from = LABEL[e.mode] + (e.mode === 'mc' ? ' (' + e.quality + ')' : '');
      let to;
      if (e.mode === 'mc' && e.quality === 'high') { e.quality = 'balanced'; to = 'Optical flow (balanced)'; }
      else if (e.mode === 'mc' && e.quality === 'balanced') { e.quality = 'low'; to = 'Optical flow (low)'; }
      else if (e.mode === 'mc') { e.mode = 'blend'; to = 'Blend'; }
      else if (e.mode === 'blend') { this.govOff = true; to = 'off'; }
      else return;
      this.note = 'The GPU missed ' + Math.round(this.lateRatio * 100) + '% of refreshes, so ' + from + ' was stepped down to ' + to + '.';
      this._graceUntil = performance.now() + 3000;
      this._lastKey = '';
    }

    _evalBypass() {
      const v = this.video, tl = this.tl, rate = v.playbackRate || 1;
      let r = '';
      if (this.opts.mode === 'off') r = 'Off';
      else if (this.fatal) r = this.fatal;
      else if (!this.gl) r = this.glError || 'WebGL2 isn’t available';
      else if (this.lost) r = 'GPU context lost. Reload the page.';
      else if (v.mediaKeys) r = 'Protected (DRM) video can’t be processed';
      else if (this.srcW * this.srcH > MAX_PIXELS) r = 'Above 1080p. Pick 1080p or lower in YouTube.';
      else if (!tl.known || this._dts.length < 20) r = 'Measuring frame rate…';
      else if (tl.fps * rate >= this.dispHz * 0.9) r = 'Already ' + Math.round(tl.fps * rate) + ' fps, nothing to add';
      else if (this.govOff) r = 'Paused because the GPU couldn’t keep up. Pick a lighter mode to retry.';
      this.bypass = r;
    }

    _syncLayout() {
      const v = this.video, c = this.canvas, parent = v.parentNode;
      if (!parent) return;
      if (c.parentNode !== parent) { parent.insertBefore(c, v.nextSibling); this._layoutDirty = true; }
      const sig = v.style.cssText;
      if (sig !== this._styleSig) { this._styleSig = sig; this._layoutDirty = true; }
      if (!this._layoutDirty) return;
      this._layoutDirty = false;
      const s = c.style;
      s.left = v.offsetLeft + 'px';
      s.top = v.offsetTop + 'px';
      s.width = v.offsetWidth + 'px';
      s.height = v.offsetHeight + 'px';
      s.transform = v.style.transform || '';
      s.objectFit = getComputedStyle(v).objectFit || 'contain';
    }

    _show(on) {
      const s = this.canvas.style, d = on ? 'block' : 'none';
      if (s.display === d) return;
      s.display = d;
      if (!on) this._lastKey = '';
    }

    _tick(ts) {
      if (this.destroyed) return;
      this.raf = requestAnimationFrame(this._tickB);
      this._timing(ts);
      this._evalBypass();
      if (this.bypass) { this._show(false); return; }
      const tl = this.tl, v = this.video;
      if (!tl.cur || !this.F || this.lost) return;
      this._syncLayout();
      if (this.skip > 1 && this.nTick % this.skip) return; // 100 Hz+ monitors: render every n-th refresh

      const t = (v.paused || v.seeking || !tl.prev) ? 1 : tl.position(ts, v.playbackRate || 1);
      const mode = this.eff.mode;
      const pass = (!tl.prev || t >= 0.96) ? 'c' : (t <= 0.04 || mode === 'dup') ? 'p' : 'i';
      const key = (tl.prev ? tl.prev.mt : '-') + '|' + tl.cur.mt + '|' + pass + '|' + (pass === 'i' ? Math.round(t * 100) : '') + mode;
      if (key !== this._lastKey) {
        this._lastKey = key;
        const sample = this._wantTiming() && (this._nDraw++ % 40 === 0);
        const t0 = sample ? performance.now() : 0;
        try { this._draw(pass, t); } catch (e) { this._fail(e); return; }
        if (sample) { this.gl.finish(); this.drawMs = ema(this.drawMs, performance.now() - t0); }
        this.nDraw1++;
      }
      this._show(true);
    }

    _draw(pass, t) {
      const gl = this.gl, tl = this.tl, cur = tl.cur, prev = tl.prev || cur, F = this.F;
      const W = this.canvas.width, H = this.canvas.height;
      if (pass === 'i' && this.eff.mode === 'mc' && this.MV && this.mvKey === cur.mt) {
        this._run('mc', null, W, H, [F[prev.k], F[cur.k], this.MV, this.D], (u) => {
          gl.uniform1i(u.u_prev, 0); gl.uniform1i(u.u_cur, 1); gl.uniform1i(u.u_mv, 2); gl.uniform1i(u.u_diff, 3);
          gl.uniform1f(u.u_t, t); gl.uniform1f(u.u_cut, CUT);
          gl.uniform2f(u.u_est, this.estW, this.estH);
          gl.uniform2f(u.u_gs, this.estW / 8, this.estH / 8);
        });
      } else {
        const tt = pass === 'c' ? 1 : pass === 'p' ? 0 : t;
        this._run('blend', null, W, H, [F[prev.k], F[cur.k], this.D], (u) => {
          gl.uniform1i(u.u_prev, 0); gl.uniform1i(u.u_cur, 1); gl.uniform1i(u.u_diff, 2);
          gl.uniform1f(u.u_t, tt); gl.uniform1f(u.u_cut, pass === 'i' ? CUT : 0);
        });
      }
    }

    /* ---------- readouts */
    _hud() {
      if (!this.opts.hud) { if (this.hudEl) this.hudEl.style.display = 'none'; return; }
      const parent = this.video.parentNode;
      if (!parent) return;
      if (!this.hudEl) {
        this.hudEl = document.createElement('div');
        Object.assign(this.hudEl.style, {
          position: 'absolute', left: '8px', top: '8px', zIndex: '2', pointerEvents: 'none',
          font: '12px/1.35 "Segoe UI", system-ui, sans-serif', color: '#67e8f9', background: 'rgba(8,12,16,.78)',
          padding: '4px 8px', borderRadius: '4px', whiteSpace: 'pre'
        });
      }
      if (this.hudEl.parentNode !== parent) parent.appendChild(this.hudEl);
      this.hudEl.style.display = 'block';
      const rate = this.video.playbackRate || 1;
      this.hudEl.textContent = this.bypass
        ? 'SmoothVision: ' + this.bypass
        : 'SmoothVision ' + LABEL[this.eff.mode] + (this.eff.mode === 'mc' ? ' (' + this.eff.quality + ')' : '') +
          '\nsource ' + (this.tl.fps * rate).toFixed(1) + ' fps, shown ' + this.outFps.toFixed(0) + ' fps on a ' + this.dispHz.toFixed(0) + ' Hz display';
    }

    getStats() {
      this._demandUntil = performance.now() + 3000;
      const rate = this.video.playbackRate || 1;
      return {
        bypass: this.bypass, note: this.note, mode: this.opts.mode, effMode: this.eff.mode, quality: this.eff.quality,
        srcFps: this.tl.known ? this.tl.fps * rate : 0, outFps: this.bypass ? 0 : this.outFps, dispHz: this.dispHz,
        late: this.lateRatio, prepMs: this.prepMs, drawMs: this.drawMs,
        res: this.srcW ? this.srcW + '×' + this.srcH : '–', est: this.estW ? this.estW + '×' + this.estH : '–',
        gl: !!this.gl, floatRT: this.floatRT
      };
    }
  }

  root.SmoothEngine = SmoothEngine;
  root.SmoothTimeline = Timeline;
  if (typeof module !== 'undefined' && module.exports) module.exports = { SmoothEngine, Timeline, FS, VS, QUALITY };
})(typeof globalThis !== 'undefined' ? globalThis : this);
