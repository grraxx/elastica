"use strict";

// ------------------------------------------------------------------ state
const S = {
  meta: null, i: 0, field: "q",
  layer: null,          // Float32Array of the current field on the current layer, [j][k]
  qLayer: null,         // q on the current layer (branch in the tooltip)
  snap: { snap: [], smooth: [] },
  image: null,          // offscreen canvas of the coloured layer
  norm: null,
  probe: null,          // node data from /api/node
  pins: [],             // pinned node data, each with .slot
  column: null,         // field along d at the probe
  hover: null,
};
const MAX_PINS = 3;
const FIELD_ORDER = ["q", "energy", "fx", "fy", "moment", "moment_right"];
const SHORT = { q: "q (branch)", energy: "Energy U", fx: "fₓ", fy: "fᵧ", moment: "M₁", moment_right: "M₂" };
const PI_TICKS = [[-Math.PI / 2, "−π/2"], [-Math.PI / 4, "−π/4"], [0, "0"], [Math.PI / 4, "π/4"], [Math.PI / 2, "π/2"]];
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ------------------------------------------------------------------ helpers
function fmt(v, digits = 4) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1000) return Math.round(v).toLocaleString("en-US").replace("-", "−");
  if (a !== 0 && a < 1e-3) return v.toExponential(2).replace("-", "−");
  return Number(v.toPrecision(digits)).toString().replace("-", "−");
}
const fmtPhi = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3);
const branchName = (p) => (p > 0 ? "bent up" : p < 0 ? "bent down" : "S-shape");

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return [0, 2, 4].map((o) => parseInt(h.slice(o, o + 2), 16));
}
function lut(stops) {                         // 256-entry RGB table through the given hex stops
  const rgb = stops.map(hexToRgb), out = new Uint8ClampedArray(256 * 3);
  for (let n = 0; n < 256; n++) {
    const t = (n / 255) * (rgb.length - 1), a = Math.min(Math.floor(t), rgb.length - 2), f = t - a;
    for (let c = 0; c < 3; c++) out[n * 3 + c] = rgb[a][c] + (rgb[a + 1][c] - rgb[a][c]) * f;
  }
  return out;
}
function niceTicks(lo, hi, count = 5) {
  if (!(hi > lo)) { hi = lo + 1; lo -= 1; }
  const raw = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1, r = canvas.getBoundingClientRect();
  const w = Math.max(10, Math.round(r.width)), h = Math.max(10, Math.round(r.height));
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}
function showBanner(text) { const b = $("banner"); b.textContent = text; b.hidden = !text; }
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${(await r.text()).slice(0, 200)}`);
  return r.json();
}
function slotColor(slot) { return css(`--s${slot}`); }

// ------------------------------------------------------------------ colour scales
function makeNorm(field, values) {
  const kind = S.meta.fields[field].scale;
  let lo = Infinity, hi = -Infinity, amax = 0;
  for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; const a = Math.abs(v); if (a > amax) amax = a; }
  const diverging = lut([css("--div-neg-2"), css("--div-neg-1"), css("--div-neg-0"), css("--div-mid"),
                         css("--div-pos-0"), css("--div-pos-1"), css("--div-pos-2")]);
  if (kind === "branch") {
    const m = field === "q" ? 1.15 : amax || 1;
    return { lut: diverging, f: (v) => (v / m + 1) / 2, ticks: niceTicks(-m, m, 4), lo: -m, hi: m,
             caption: field === "q" ? "q = p / a(d)   ·   blue: bent down   ·   red: bent up" : "p   ·   blue: bent down   ·   red: bent up" };
  }
  if (kind === "log") {
    const a = Math.log10(Math.max(lo, 1e-12)), b = Math.log10(hi);
    const ticks = [];
    for (let e = Math.floor(a); e <= Math.ceil(b); e++) for (const m of [1, 2, 5]) {
      const v = m * 10 ** e; if (v >= lo && v <= hi) ticks.push(v);
    }
    return { lut: lut([css("--seq-lo"), css("--seq-mid"), css("--seq-hi")]), f: (v) => (Math.log10(Math.max(v, 1e-12)) - a) / (b - a || 1),
             ticks, lo, hi, caption: `${S.meta.fields[field].label}   ·   log scale, this layer` };
  }
  const c = Math.max(1, amax / 1000), g = (v) => Math.sign(v) * Math.log10(1 + Math.abs(v) / c), G = g(amax) || 1;
  const ticks = [0];
  for (let e = 0; 10 ** e <= amax; e++) { const v = 10 ** e; if (v >= c) ticks.push(v, -v); }
  return { lut: diverging, f: (v) => (g(v) / G + 1) / 2, ticks: ticks.sort((x, y) => x - y), lo: -amax, hi: amax,
           caption: `${S.meta.fields[field].label}   ·   symmetric log scale, this layer` };
}

function buildImage() {
  const [, n1, n2] = S.meta.shape, img = new ImageData(n1, n2), norm = makeNorm(S.field, S.layer);
  for (let j = 0; j < n1; j++) for (let k = 0; k < n2; k++) {
    const t = Math.min(1, Math.max(0, norm.f(S.layer[j * n2 + k]))), n = Math.round(t * 255) * 3;
    const p = ((n2 - 1 - k) * n1 + j) * 4;
    img.data[p] = norm.lut[n]; img.data[p + 1] = norm.lut[n + 1]; img.data[p + 2] = norm.lut[n + 2]; img.data[p + 3] = 255;
  }
  const off = document.createElement("canvas");
  off.width = n1; off.height = n2;
  off.getContext("2d").putImageData(img, 0, 0);
  S.image = off; S.norm = norm;
}

// ------------------------------------------------------------------ the map
const MAP_PAD = { l: 46, r: 10, t: 8, b: 36 };
function mapGeometry(w, h) {
  const size = Math.min(w - MAP_PAD.l - MAP_PAD.r, h - MAP_PAD.t - MAP_PAD.b);
  const p1 = S.meta.phi1, p2 = S.meta.phi2;
  return {
    x0: MAP_PAD.l, y0: MAP_PAD.t, size,
    X: (v) => MAP_PAD.l + ((v - p1[0]) / (p1[p1.length - 1] - p1[0])) * size,
    Y: (v) => MAP_PAD.t + size - ((v - p2[0]) / (p2[p2.length - 1] - p2[0])) * size,
  };
}
function drawMap() {
  const canvas = $("map");
  const { ctx, w, h } = setupCanvas(canvas);
  if (!S.image) return;
  const g = mapGeometry(w, h);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(S.image, g.x0, g.y0, g.size, g.size);
  ctx.strokeStyle = css("--axis"); ctx.lineWidth = 1;
  ctx.strokeRect(g.x0 + 0.5, g.y0 + 0.5, g.size - 1, g.size - 1);
  // axes
  ctx.fillStyle = css("--muted"); ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const [v, label] of PI_TICKS) {
    const x = g.X(v); ctx.fillRect(x, g.y0 + g.size, 1, 4); ctx.fillText(label, x, g.y0 + g.size + 6);
  }
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const [v, label] of PI_TICKS) { const y = g.Y(v); ctx.fillRect(g.x0 - 4, y, 4, 1); ctx.fillText(label, g.x0 - 7, y); }
  ctx.fillStyle = css("--ink-2"); ctx.font = "12px system-ui, sans-serif"; ctx.textAlign = "center";
  ctx.fillText("φ₁", g.x0 + g.size / 2, g.y0 + g.size + 22);
  ctx.save(); ctx.translate(12, g.y0 + g.size / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("φ₂", 0, 0); ctx.restore();
  // snap and smooth-crossing edges
  const dot = Math.max(1.6, g.size / 330);
  if ($("showSmooth").checked) {
    ctx.fillStyle = css("--muted");
    for (const [x, y] of S.snap.smooth) ctx.fillRect(g.X(x) - dot / 2, g.Y(y) - dot / 2, dot, dot);
  }
  if ($("showSnap").checked) {
    ctx.fillStyle = css("--ink");
    for (const [x, y] of S.snap.snap) ctx.fillRect(g.X(x) - dot / 2, g.Y(y) - dot / 2, dot, dot);
  }
  // probe and pins on this layer: a coloured ring with a surface halo
  const rings = [...S.pins.filter((p) => p.i === S.i), ...(S.probe ? [S.probe] : [])];
  for (const p of rings) {
    const x = g.X(p.phi1), y = g.Y(p.phi2);
    ctx.lineWidth = 5; ctx.strokeStyle = css("--surface");
    ctx.beginPath(); ctx.arc(x, y, 7, 0, 2 * Math.PI); ctx.stroke();
    ctx.lineWidth = 2.5; ctx.strokeStyle = slotColor(p.slot);
    ctx.beginPath(); ctx.arc(x, y, 7, 0, 2 * Math.PI); ctx.stroke();
  }
  if (S.hover) {
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1;
    const x = g.X(S.meta.phi1[S.hover.j]), y = g.Y(S.meta.phi2[S.hover.k]), c = Math.max(3, g.size / 315 * 2);
    ctx.strokeRect(x - c, y - c, 2 * c, 2 * c);
  }
}
function drawColorbar() {
  const { ctx, w } = setupCanvas($("colorbar"));
  if (!S.norm) return;
  const x0 = 8, x1 = w - 8, y0 = 4, hBar = 10;
  for (let x = x0; x < x1; x++) {
    const n = Math.round(((x - x0) / (x1 - x0)) * 255) * 3;
    ctx.fillStyle = `rgb(${S.norm.lut[n]},${S.norm.lut[n + 1]},${S.norm.lut[n + 2]})`;
    ctx.fillRect(x, y0, 1, hBar);
  }
  ctx.fillStyle = css("--muted"); ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "top";
  let last = -1e9;
  for (const v of S.norm.ticks) {
    const x = x0 + Math.min(1, Math.max(0, S.norm.f(v))) * (x1 - x0);
    if (x - last < 30) continue;
    ctx.fillRect(x, y0 + hBar, 1, 3); ctx.fillText(fmt(v, 3), x, y0 + hBar + 5); last = x;
  }
  ctx.fillStyle = css("--ink-2"); ctx.textAlign = "left";
  ctx.fillText(S.norm.caption, x0, y0 + hBar + 20);
}
function mapIndexAt(event) {
  const r = $("map").getBoundingClientRect(), g = mapGeometry(r.width, r.height);
  const fx = (event.clientX - r.left - g.x0) / g.size, fy = 1 - (event.clientY - r.top - g.y0) / g.size;
  if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return null;
  const [, n1, n2] = S.meta.shape;
  return { j: Math.min(n1 - 1, Math.round(fx * (n1 - 1))), k: Math.min(n2 - 1, Math.round(fy * (n2 - 1))) };
}

// ------------------------------------------------------------------ line charts
const CHART_PAD = { l: 50, r: 10, t: 8, b: 30 };
function lineChart(canvas, opts) {
  canvas._opts = opts;
  const { ctx, w, h } = setupCanvas(canvas);
  const xs = opts.series.flatMap((s) => s.x), ys = opts.series.flatMap((s) => s.y).filter(Number.isFinite);
  if (!xs.length || !ys.length) {
    ctx.fillStyle = css("--muted"); ctx.font = "12px system-ui"; ctx.textAlign = "center";
    ctx.fillText(opts.empty || "Click the map to probe a point", w / 2, h / 2); return;
  }
  let ylo = Math.min(...ys), yhi = Math.max(...ys);
  if (yhi - ylo < 1e-9) { ylo -= 1; yhi += 1; }
  const pad = (yhi - ylo) * 0.06; ylo -= pad; yhi += pad;
  const xlo = opts.xRange ? opts.xRange[0] : Math.min(...xs), xhi = opts.xRange ? opts.xRange[1] : Math.max(...xs);
  const pw = w - CHART_PAD.l - CHART_PAD.r, ph = h - CHART_PAD.t - CHART_PAD.b;
  const X = (v) => CHART_PAD.l + ((v - xlo) / (xhi - xlo || 1)) * pw, Y = (v) => CHART_PAD.t + ph - ((v - ylo) / (yhi - ylo)) * ph;
  canvas._geom = { X, Y, xlo, xhi, pw, ph };
  ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = css("--muted");
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const v of niceTicks(ylo, yhi, 4)) {
    const y = Math.round(Y(v)) + 0.5;
    ctx.strokeStyle = v === 0 ? css("--axis") : css("--grid"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(CHART_PAD.l, y); ctx.lineTo(CHART_PAD.l + pw, y); ctx.stroke();
    ctx.fillText(fmt(v, 3), CHART_PAD.l - 6, y);
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const xt = opts.xTicks || niceTicks(xlo, xhi, 5).map((v) => [v, fmt(v, 3)]);
  for (const [v, label] of xt) { if (v < xlo - 1e-9 || v > xhi + 1e-9) continue; ctx.fillRect(X(v), CHART_PAD.t + ph, 1, 4); ctx.fillText(label, X(v), CHART_PAD.t + ph + 6); }
  ctx.strokeStyle = css("--axis"); ctx.beginPath(); ctx.moveTo(CHART_PAD.l, CHART_PAD.t + ph + 0.5); ctx.lineTo(CHART_PAD.l + pw, CHART_PAD.t + ph + 0.5); ctx.stroke();
  if (opts.xLabel) { ctx.fillStyle = css("--ink-2"); ctx.textAlign = "right"; ctx.fillText(opts.xLabel, CHART_PAD.l + pw, CHART_PAD.t + ph + 18); }
  for (const s of opts.series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2; ctx.lineJoin = "round"; ctx.beginPath();
    let pen = false;
    s.x.forEach((x, n) => {
      const y = s.y[n];
      if (!Number.isFinite(y)) { pen = false; return; }
      if (!pen) { ctx.moveTo(X(x), Y(y)); pen = true; } else ctx.lineTo(X(x), Y(y));
    });
    ctx.stroke();
  }
  for (const m of opts.markers || []) {
    ctx.fillStyle = css("--surface"); ctx.beginPath(); ctx.arc(X(m.x), Y(m.y), 6, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = m.color; ctx.beginPath(); ctx.arc(X(m.x), Y(m.y), 4.5, 0, 2 * Math.PI); ctx.fill();
  }
  if (canvas._hoverX !== undefined && canvas._hoverX !== null) {
    ctx.strokeStyle = css("--axis"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(Math.round(X(canvas._hoverX)) + 0.5, CHART_PAD.t); ctx.lineTo(Math.round(X(canvas._hoverX)) + 0.5, CHART_PAD.t + ph); ctx.stroke();
  }
}
function attachCrosshair(canvas) {             // nearest-x tooltip listing every series
  const tip = document.createElement("div");
  tip.className = "tooltip"; tip.hidden = true;
  canvas.parentElement.appendChild(tip);
  canvas.addEventListener("pointermove", (e) => {
    const o = canvas._opts, g = canvas._geom;
    if (!o || !g || !o.series.length || !o.series[0].x.length) return;
    const r = canvas.getBoundingClientRect(), px = e.clientX - r.left;
    const xv = g.xlo + ((px - CHART_PAD.l) / g.pw) * (g.xhi - g.xlo);
    const xsRef = o.series[0].x;
    let n = 0, best = Infinity;
    xsRef.forEach((x, m) => { const dd = Math.abs(x - xv); if (dd < best) { best = dd; n = m; } });
    canvas._hoverX = xsRef[n];
    lineChart(canvas, o);
    tip.replaceChildren();
    const head = document.createElement("div"); head.className = "sec";
    head.textContent = `${o.xName || "x"} = ${o.xFormat ? o.xFormat(xsRef[n]) : fmt(xsRef[n])}`;
    tip.appendChild(head);
    for (const s of o.series) {
      const row = document.createElement("div"); row.className = "row";
      const sw = document.createElement("span"); sw.className = "swatch"; sw.style.background = s.color;
      const v = document.createElement("strong"); v.textContent = fmt(s.y[n]);
      const name = document.createElement("span"); name.className = "sec"; name.textContent = s.name || "";
      row.append(sw, v, name); tip.appendChild(row);
    }
    tip.hidden = false;
    const fr = canvas.parentElement.getBoundingClientRect();
    tip.style.left = `${Math.min(e.clientX - fr.left + 12, fr.width - tip.offsetWidth - 4)}px`;
    tip.style.top = `${e.clientY - fr.top + 12}px`;
  });
  canvas.addEventListener("pointerleave", () => { tip.hidden = true; canvas._hoverX = null; if (canvas._opts) lineChart(canvas, canvas._opts); });
}

// ------------------------------------------------------------------ the rod
function drawRod() {
  const canvas = $("rod");
  const { ctx, w, h } = setupCanvas(canvas);
  const rods = [...(S.probe ? [S.probe] : []), ...S.pins];
  canvas._rods = null;
  if (!rods.length) {
    ctx.fillStyle = css("--muted"); ctx.font = "13px system-ui"; ctx.textAlign = "center";
    ctx.fillText("Click a point on the map to see the rod", w / 2, h / 2); return;
  }
  const pad = { l: 46, r: 12, t: 12, b: 28 };
  let xlo = 0, xhi = 1, ylo = -0.05, yhi = 0.05;
  for (const r of rods) for (let n = 0; n < r.x.length; n++) {
    xlo = Math.min(xlo, r.x[n]); xhi = Math.max(xhi, r.x[n]); ylo = Math.min(ylo, r.y[n]); yhi = Math.max(yhi, r.y[n]);
  }
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  const scale = Math.min(pw / (xhi - xlo), ph / ((yhi - ylo) * 1.1));
  const cx = pad.l + pw / 2 - ((xlo + xhi) / 2) * scale, cy = pad.t + ph / 2 + ((ylo + yhi) / 2) * scale;
  const X = (v) => cx + v * scale, Y = (v) => cy - v * scale;
  ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = css("--muted"); ctx.lineWidth = 1;
  const yMin = (cy - (pad.t + ph)) / scale, yMax = (cy - pad.t) / scale, xMin = (pad.l - cx) / scale, xMax = (pad.l + pw - cx) / scale;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const v of niceTicks(yMin, yMax, 4)) {
    const y = Math.round(Y(v)) + 0.5; ctx.strokeStyle = v === 0 ? css("--axis") : css("--grid");
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke(); ctx.fillText(fmt(v, 3), pad.l - 6, y);
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const v of niceTicks(xMin, xMax, 6)) { const x = X(v); if (x < pad.l || x > pad.l + pw) continue; ctx.fillRect(x, pad.t + ph, 1, 4); ctx.fillText(fmt(v, 3), x, pad.t + ph + 6); }
  ctx.fillStyle = css("--ink-2"); ctx.textAlign = "right"; ctx.fillText("x", pad.l + pw, pad.t + ph + 14);
  ctx.textAlign = "left"; ctx.fillText("y", pad.l + 4, pad.t);
  for (const r of [...rods].reverse()) {
    ctx.strokeStyle = slotColor(r.slot); ctx.lineWidth = r === S.probe ? 2.5 : 2; ctx.lineJoin = "round";
    ctx.beginPath(); r.x.forEach((x, n) => (n ? ctx.lineTo(X(x), Y(r.y[n])) : ctx.moveTo(X(x), Y(r.y[n])))); ctx.stroke();
    for (const [x, y, ang] of [[0, 0, r.phi1], [r.x[r.x.length - 1], r.y[r.y.length - 1], r.phi2]]) {
      ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(x), Y(y), 3.5, 0, 2 * Math.PI); ctx.fill();
      ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1; ctx.beginPath();     // clamp direction
      ctx.moveTo(X(x) - 9 * Math.cos(ang), Y(y) + 9 * Math.sin(ang)); ctx.lineTo(X(x) + 9 * Math.cos(ang), Y(y) - 9 * Math.sin(ang)); ctx.stroke();
    }
  }
  if (canvas._hover) {
    const { r, n } = canvas._hover;
    ctx.fillStyle = css("--surface"); ctx.beginPath(); ctx.arc(X(r.x[n]), Y(r.y[n]), 6, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = slotColor(r.slot); ctx.beginPath(); ctx.arc(X(r.x[n]), Y(r.y[n]), 4.5, 0, 2 * Math.PI); ctx.fill();
  }
  canvas._rods = { rods, X, Y };
}
function attachRodHover() {
  const canvas = $("rod"), tip = document.createElement("div");
  tip.className = "tooltip"; tip.hidden = true; canvas.parentElement.style.position = "relative"; canvas.parentElement.appendChild(tip);
  canvas.addEventListener("pointermove", (e) => {
    const g = canvas._rods; if (!g) return;
    const b = canvas.getBoundingClientRect(), px = e.clientX - b.left, py = e.clientY - b.top;
    let best = { d: 400 };
    for (const r of g.rods) for (let n = 0; n < r.x.length; n++) {
      const dd = (g.X(r.x[n]) - px) ** 2 + (g.Y(r.y[n]) - py) ** 2;
      if (dd < best.d) best = { d: dd, r, n };
    }
    canvas._hover = best.r ? best : null; drawRod();
    if (!best.r) { tip.hidden = true; return; }
    const { r, n } = best;
    tip.replaceChildren();
    const head = document.createElement("div"); head.className = "row";
    const sw = document.createElement("span"); sw.className = "swatch"; sw.style.background = slotColor(r.slot);
    const name = document.createElement("span"); name.className = "sec"; name.textContent = `d ${r.d.toFixed(2)}, φ (${fmtPhi(r.phi1)}, ${fmtPhi(r.phi2)})`;
    head.append(sw, name); tip.appendChild(head);
    for (const [k, v] of [["s", r.s[n]], ["θ", r.theta[n]], ["θ′", r.curvature[n]]]) {
      const row = document.createElement("div"); row.className = "row";
      const strong = document.createElement("strong"); strong.textContent = fmt(v);
      const lab = document.createElement("span"); lab.className = "sec"; lab.textContent = k;
      row.append(strong, lab); tip.appendChild(row);
    }
    tip.hidden = false;
    tip.style.left = `${Math.min(px + 14, b.width - tip.offsetWidth - 4)}px`; tip.style.top = `${py + 14}px`;
  });
  canvas.addEventListener("pointerleave", () => { tip.hidden = true; canvas._hover = null; drawRod(); });
}
function rodLegend() {
  const box = $("rodLegend"); box.replaceChildren();
  const rods = [...(S.probe ? [S.probe] : []), ...S.pins];
  if (rods.length < 2) return;
  for (const r of rods) {
    const item = document.createElement("span");
    const sw = document.createElement("span"); sw.className = "swatch"; sw.style.background = slotColor(r.slot);
    item.append(sw, document.createTextNode(`${r === S.probe ? "Probe" : "Pin"}: d ${r.d.toFixed(2)}, φ (${fmtPhi(r.phi1)}, ${fmtPhi(r.phi2)}), U ${fmt(r.energy, 3)}`));
    box.appendChild(item);
  }
}

// ------------------------------------------------------------------ panels
function renderStats() {
  const box = $("stats"); box.replaceChildren();
  const p = S.probe;
  $("probeWhere").textContent = p ? `d = ${p.d.toFixed(2)}, φ = (${fmtPhi(p.phi1)}, ${fmtPhi(p.phi2)}) · node ${p.node.toLocaleString("en-US")}${p.waited ? " · was deferred" : ""}` : "Click the map";
  const tiles = p ? [
    ["Branch", branchName(p.p), `p = ${fmt(p.p)}`],
    ["Energy U", fmt(p.energy), "½∫θ′² ds"],
    ["Axial force fₓ", fmt(p.fx), p.fx > 0 ? "compression" : "tension"],
    ["Transverse fᵧ", fmt(p.fy), "end reaction"],
    ["|f|", fmt(Math.hypot(p.fx, p.fy)), "resultant"],
    ["Moment M₁", fmt(p.moment), "θ′(0), left clamp"],
    ["Moment M₂", fmt(p.moment_right), "θ′(L), right clamp"],
    ["q", fmt(p.q, 3), "p / a(d)"],
  ] : [];
  for (const [k, v, n] of tiles) {
    const t = document.createElement("div"); t.className = "stat";
    const a = document.createElement("div"); a.className = "k"; a.textContent = k;
    const b = document.createElement("div"); b.className = "v"; b.textContent = v;
    const c = document.createElement("div"); c.className = "n"; c.textContent = n;
    t.append(a, b, c); box.appendChild(t);
  }
  $("pinBtn").disabled = !p || S.pins.length >= MAX_PINS;
  $("csvBtn").disabled = !p;
}
function renderRodCharts() {
  const rods = [...(S.probe ? [S.probe] : []), ...S.pins];
  const series = (key) => rods.map((r) => ({ x: r.s, y: r[key], color: slotColor(r.slot), width: r === S.probe ? 2 : 1.5,
                                             name: r === S.probe ? "probe" : "pin" }));
  lineChart($("thetaChart"), { series: series("theta"), xLabel: "s", xName: "s" });
  lineChart($("kappaChart"), { series: series("curvature"), xLabel: "s", xName: "s" });
}
function renderProfiles() {
  const label = SHORT[S.field];
  $("profileField").textContent = S.meta.fields[S.field].label;
  const p = S.probe, color = slotColor(1), [, n1, n2] = S.meta.shape;
  if (!p || p.i !== S.i || !S.layer) {
    for (const id of ["profPhi1", "profPhi2", "profD"]) lineChart($(id), { series: [] });
    return;
  }
  $("capPhi1").textContent = `${label} along φ₁ (d = ${p.d.toFixed(2)}, φ₂ = ${fmtPhi(p.phi2)})`;
  $("capPhi2").textContent = `${label} along φ₂ (d = ${p.d.toFixed(2)}, φ₁ = ${fmtPhi(p.phi1)})`;
  $("capD").textContent = `${label} along d (φ = (${fmtPhi(p.phi1)}, ${fmtPhi(p.phi2)}))`;
  const row = Array.from({ length: n1 }, (_, j) => S.layer[j * n2 + p.k]);
  const col = Array.from({ length: n2 }, (_, k) => S.layer[p.j * n2 + k]);
  const piFmt = (v) => fmtPhi(v);
  const value = S.layer[p.j * n2 + p.k];
  lineChart($("profPhi1"), { series: [{ x: S.meta.phi1, y: row, color, name: label }], xTicks: PI_TICKS, xName: "φ₁", xFormat: piFmt,
                             markers: [{ x: p.phi1, y: value, color }] });
  lineChart($("profPhi2"), { series: [{ x: S.meta.phi2, y: col, color, name: label }], xTicks: PI_TICKS, xName: "φ₂", xFormat: piFmt,
                             markers: [{ x: p.phi2, y: value, color }] });
  if (S.column) lineChart($("profD"), { series: [{ x: S.meta.d, y: S.column, color, name: label }], xName: "d", xFormat: (v) => v.toFixed(2),
                                        markers: [{ x: p.d, y: value, color }] });
}
function renderTable() {
  const body = $("pointsTable").querySelector("tbody"); body.replaceChildren();
  const rows = [...(S.probe ? [S.probe] : []), ...S.pins];
  for (const r of rows) {
    const tr = document.createElement("tr");
    const key = document.createElement("td"); const sw = document.createElement("span"); sw.className = "swatch"; sw.style.background = slotColor(r.slot);
    key.append(sw, document.createTextNode(r === S.probe ? " Probe" : " Pin")); tr.appendChild(key);
    for (const v of [r.d.toFixed(2), fmtPhi(r.phi1), fmtPhi(r.phi2), branchName(r.p), fmt(r.energy), fmt(r.fx), fmt(r.fy), fmt(r.moment), fmt(r.moment_right), fmt(r.q, 3)]) {
      const td = document.createElement("td"); td.textContent = v; tr.appendChild(td);
    }
    const act = document.createElement("td");
    if (r !== S.probe) {
      const b = document.createElement("button"); b.type = "button"; b.textContent = "Remove"; b.setAttribute("aria-label", "Remove pinned point");
      b.addEventListener("click", () => { S.pins = S.pins.filter((x) => x !== r); renderAll(); });
      act.appendChild(b);
    }
    tr.appendChild(act); body.appendChild(tr);
  }
}
function renderAll() {
  drawMap(); drawColorbar(); drawRod(); rodLegend(); renderStats(); renderRodCharts(); renderProfiles(); renderTable(); writeHash();
}

// ------------------------------------------------------------------ data flow
async function loadLayer() {
  const i = S.i, field = S.field;
  $("map").style.opacity = 0.6;
  try {
    const [buf, qbuf, snap] = await Promise.all([
      fetch(`/api/layer?i=${i}&field=${field}`).then((r) => { if (!r.ok) throw new Error(`layer ${r.status}`); return r.arrayBuffer(); }),
      field === "q" ? null : fetch(`/api/layer?i=${i}&field=q`).then((r) => r.arrayBuffer()),
      getJSON(`/api/snap?i=${i}`),
    ]);
    if (i !== S.i || field !== S.field) return;          // a newer request superseded this one
    S.layer = new Float32Array(buf); S.qLayer = qbuf ? new Float32Array(qbuf) : S.layer; S.snap = snap;
    buildImage();
    let up = 0; for (const v of S.qLayer) if (v > 0) up++;
    $("mapTitle").textContent = `${S.meta.fields[S.field].label}`;
    $("mapSub").textContent = `d = ${S.meta.d[i].toFixed(2)} · layer ${i + 1} of ${S.meta.d.length} · bent up ${Math.round((100 * up) / S.qLayer.length)} %`;
    $("layerValue").textContent = `d = ${S.meta.d[i].toFixed(2)}`;
    showBanner("");
  } catch (err) {
    showBanner(`Could not load the layer: ${err.message}. Is the server still running?`);
  } finally {
    $("map").style.opacity = 1;
  }
  if (S.probe && S.probe.i !== S.i) await probe(S.probe.j, S.probe.k, false);
  else renderAll();
}
let probeTicket = 0;
async function probe(j, k, pinAfter) {
  const ticket = ++probeTicket;              // only the latest click may land
  try {
    const [node, column] = await Promise.all([getJSON(`/api/node?i=${S.i}&j=${j}&k=${k}`),
                                              getJSON(`/api/column?j=${j}&k=${k}&field=${S.field}`)]);
    if (ticket !== probeTicket) return;
    node.slot = 1;
    if (pinAfter && S.probe) pin();
    S.probe = node;
    S.column = column;
    showBanner("");
  } catch (err) {
    showBanner(`Could not load the point: ${err.message}`);
  }
  renderAll();
}
function pin() {
  if (!S.probe || S.pins.length >= MAX_PINS) return;
  if (S.pins.some((p) => p.node === S.probe.node)) return;
  const used = new Set(S.pins.map((p) => p.slot));
  const slot = [2, 3, 4].find((s) => !used.has(s));
  S.pins.push({ ...S.probe, slot });
}
function setLayer(i) {
  i = Math.max(0, Math.min(S.meta.d.length - 1, i));
  if (i === S.i && S.layer) return;
  S.i = i; $("layerSlider").value = i; $("layerValue").textContent = `d = ${S.meta.d[i].toFixed(2)}`;
  loadLayer();
}
async function setField(field) {
  S.field = field;
  for (const b of $("fieldButtons").children) b.setAttribute("aria-pressed", b.dataset.field === field);
  if (S.probe) S.column = await getJSON(`/api/column?j=${S.probe.j}&k=${S.probe.k}&field=${field}`).catch(() => null);
  loadLayer();
}

// ------------------------------------------------------------------ URL state
function writeHash() {
  const parts = [`d=${S.meta.d[S.i].toFixed(2)}`, `field=${S.field}`];
  if (S.probe) parts.push(`phi1=${S.probe.phi1.toFixed(4)}`, `phi2=${S.probe.phi2.toFixed(4)}`);
  history.replaceState(null, "", `#${parts.join("&")}`);
}
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1)), nearest = (arr, v) => arr.reduce((b, x, n) => (Math.abs(x - v) < Math.abs(arr[b] - v) ? n : b), 0);
  const out = {};
  if (h.has("d")) out.i = nearest(S.meta.d, +h.get("d"));
  if (h.has("field") && FIELD_ORDER.includes(h.get("field"))) out.field = h.get("field");
  if (h.has("phi1") && h.has("phi2")) out.jk = [nearest(S.meta.phi1, +h.get("phi1")), nearest(S.meta.phi2, +h.get("phi2"))];
  return out;
}

// ------------------------------------------------------------------ events
function wire() {
  for (const f of FIELD_ORDER) {
    const b = document.createElement("button"); b.type = "button"; b.dataset.field = f; b.textContent = SHORT[f];
    b.title = S.meta.fields[f].label; b.setAttribute("aria-pressed", f === S.field);
    b.addEventListener("click", () => setField(f)); $("fieldButtons").appendChild(b);
  }
  $("layerSlider").max = S.meta.d.length - 1;
  $("layerSlider").addEventListener("input", (e) => setLayer(+e.target.value));
  $("prevLayer").addEventListener("click", () => setLayer(S.i - 1));
  $("nextLayer").addEventListener("click", () => setLayer(S.i + 1));
  for (const id of ["showSnap", "showSmooth"]) $(id).addEventListener("change", drawMap);
  const map = $("map"), tip = $("mapTip");
  map.addEventListener("pointermove", (e) => {
    const at = mapIndexAt(e);
    if (!at || !S.layer) { tip.hidden = true; S.hover = null; drawMap(); return; }
    S.hover = at; drawMap();
    const n2 = S.meta.shape[2], v = S.layer[at.j * n2 + at.k], q = S.qLayer[at.j * n2 + at.k];
    tip.replaceChildren();
    const strong = document.createElement("strong"); strong.textContent = `${fmt(v)}`;
    const name = document.createElement("span"); name.className = "sec"; name.textContent = ` ${SHORT[S.field]}`;
    const l1 = document.createElement("div"); l1.append(strong, name);
    const l2 = document.createElement("div"); l2.className = "sec"; l2.textContent = `φ₁ ${fmtPhi(S.meta.phi1[at.j])}, φ₂ ${fmtPhi(S.meta.phi2[at.k])} · ${branchName(q)}`;
    tip.append(l1, l2); tip.hidden = false;
    const r = map.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    tip.style.left = `${Math.min(x + 14, r.width - tip.offsetWidth - 4)}px`; tip.style.top = `${Math.max(4, y - tip.offsetHeight - 10)}px`;
  });
  map.addEventListener("pointerleave", () => { tip.hidden = true; S.hover = null; drawMap(); });
  map.addEventListener("click", (e) => { const at = mapIndexAt(e); if (at) { map.focus(); probe(at.j, at.k, e.shiftKey); } });
  map.addEventListener("keydown", (e) => {
    const step = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[e.key];
    if (step) {
      e.preventDefault();
      const [, n1, n2] = S.meta.shape, p = S.probe || { j: (n1 / 2) | 0, k: (n2 / 2) | 0 };
      probe(Math.max(0, Math.min(n1 - 1, p.j + step[0])), Math.max(0, Math.min(n2 - 1, p.k + step[1])), false);
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && e.target.type !== "checkbox") return;   // the slider has its own keys
    if (e.key === "PageUp" || e.key === "]") { e.preventDefault(); setLayer(S.i + 1); }
    else if (e.key === "PageDown" || e.key === "[") { e.preventDefault(); setLayer(S.i - 1); }
    else if (e.key === "p" || e.key === "P") { pin(); renderAll(); }
  });
  $("pinBtn").addEventListener("click", () => { pin(); renderAll(); });
  $("csvBtn").addEventListener("click", () => {
    const p = S.probe; if (!p) return;
    const lines = [`# d=${p.d} phi1=${p.phi1} phi2=${p.phi2} fx=${p.fx} fy=${p.fy} energy=${p.energy} M1=${p.moment} M2=${p.moment_right} p=${p.p}`,
                   "s,theta,x,y,curvature", ...p.s.map((s, n) => [s, p.theta[n], p.x[n], p.y[n], p.curvature[n]].join(","))];
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
    a.download = `rod_d${p.d.toFixed(2)}_phi1_${p.phi1.toFixed(3)}_phi2_${p.phi2.toFixed(3)}.csv`;
    a.click(); URL.revokeObjectURL(a.href);
  });
  $("themeBtn").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme === "dark" ||
      (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", document.documentElement.dataset.theme); } catch (_) { /* storage unavailable */ }
    themeChanged();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", themeChanged);
  for (const id of ["thetaChart", "kappaChart", "profPhi1", "profPhi2", "profD"]) attachCrosshair($(id));
  attachRodHover();
  new ResizeObserver(() => renderAll()).observe(document.querySelector(".app"));
}
function themeChanged() {
  const dark = document.documentElement.dataset.theme === "dark" ||
    (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  $("themeBtn").textContent = dark ? "Light mode" : "Dark mode";
  if (S.layer) buildImage();
  renderAll();
}

async function init() {
  try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (_) { /* storage unavailable */ }
  try {
    S.meta = await getJSON("/api/meta");
  } catch (err) {
    showBanner(`Could not reach the server: ${err.message}`); return;
  }
  const n = S.meta.shape.reduce((a, b) => a * b, 1);
  $("runInfo").textContent = `Run “${S.meta.run}” · ${n.toLocaleString("en-US")} equilibria · d ${S.meta.d[0].toFixed(2)}–${S.meta.d[S.meta.d.length - 1].toFixed(2)} · φ ±π/2`;
  $("jumpTol").textContent = S.meta.jump_tol;
  const h = readHash();
  if (h.field) S.field = h.field;
  if (h.i !== undefined) S.i = h.i;
  wire();                                   // sets the slider's max first
  $("layerSlider").value = S.i;
  themeChanged();
  await loadLayer();
  if (h.jk) await probe(h.jk[0], h.jk[1], false);
}
init();
