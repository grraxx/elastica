// 3D iso-surface viewer. Loaded as a module after app.js; uses its globals (S, $, css, fmt,
// fmtPhi, lut, makeNorm, slotColor, goTo) and is driven by app.js through window.viewer3d.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const ISO_SHORT = { energy: "U", fx: "fₓ", fy: "fᵧ", moment: "M₁", moment_right: "M₂", q: "q", d: "d" };
const PRESETS = {                     // the levels of postprocess.py
  energy: [2, 4, 8, 16, 32, 64],
  fx: [-1000, -100, -10, 0, 20, 40, 60, 80],
  fy: [-20, -10, -5, 0, 5, 10, 20],
  moment: [-12, -8, -4, 0, 4, 8, 12],
  moment_right: [-12, -8, -4, 0, 4, 8, 12],
};
const VIRIDIS = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"];     // d, as in the deck figures
const EXTRA = ["#8f5bd6", "#c2417a", "#4a9fb5", "#7b8a3a"];
const MAX_KEPT = 7;

const V = {
  started: false, field: "energy", level: 8, color: "d", stride: 2, zs: 8, clip: "",
  active: null,          // the surface the slider drives
  kept: [],              // surfaces kept with "Keep"
  snap: null,            // the p = 0 surface: snap triangles, then smooth crossings
  planes: [], abort: null, timer: 0, frame: 0,
};
let renderer, scene, camera, controls, world, box, layerPlane, layerEdge, markers, raycaster;

// ------------------------------------------------------------------ helpers
const palette = (slot) => (slot < 4 ? slotColor(slot + 1) : EXTRA[slot - 4]);
const describe = (s) => `${ISO_SHORT[s.field]} = ${fmt(s.level)}`;
const nearest = (arr, v) => arr.reduce((b, x, n) => (Math.abs(x - v) < Math.abs(arr[b] - v) ? n : b), 0);
const toLinear = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
function linearLut(table) { return Float32Array.from(table, toLinear); }
function niceLevel(v) {
  if (Math.abs(v) < 0.05) return 0;
  return Number(v.toPrecision(Math.abs(v) < 10 ? 2 : 3));
}
function surfaces() { return [...(V.active ? [V.active] : []), ...V.kept]; }
function status(text) { const el = $("isoStatus"); el.textContent = text; el.hidden = !text; }

// Slider position <-> level: log for the energy, symmetric log (scale 1) for the rest.
function sliderMap(field) {
  const { min, max } = S.meta.fields[field];
  if (field === "energy") {
    const a = Math.log10(min), b = Math.log10(max);
    return { to: (t) => 10 ** (a + t * (b - a)), from: (v) => (Math.log10(Math.max(v, min)) - a) / (b - a) };
  }
  const g = (v) => Math.sign(v) * Math.log10(1 + Math.abs(v)), gi = (u) => Math.sign(u) * (10 ** Math.abs(u) - 1);
  const a = g(min), b = g(max);
  return { to: (t) => gi(a + t * (b - a)), from: (v) => (g(v) - a) / (b - a) };
}

// ------------------------------------------------------------------ scene
function buildScene() {
  const canvas = $("iso3d");
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.localClippingEnabled = true;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(32, 1, 0.01, 200);
  camera.up.set(0, 0, 1);                         // d is vertical
  controls = new OrbitControls(camera, canvas);
  controls.addEventListener("change", requestRender);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a8a, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.position.set(1, 2, 3);
  camera.add(sun);                                 // light from over the viewer's shoulder
  scene.add(camera);

  // Surfaces live in raw (phi1, phi2, d); the world group stretches d by zs about mid-d.
  world = new THREE.Group();
  scene.add(world);
  const { phi1: p1, phi2: p2, d } = S.meta;
  const bx = new THREE.BoxGeometry(p1.at(-1) - p1[0], p2.at(-1) - p2[0], d.at(-1) - d[0]);
  box = new THREE.LineSegments(new THREE.EdgesGeometry(bx), new THREE.LineBasicMaterial());
  box.position.set((p1[0] + p1.at(-1)) / 2, (p2[0] + p2.at(-1)) / 2, (d[0] + d.at(-1)) / 2);
  world.add(box);
  const pg = new THREE.PlaneGeometry(p1.at(-1) - p1[0], p2.at(-1) - p2[0]);
  layerPlane = new THREE.Mesh(pg, new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.07, depthWrite: false, side: THREE.DoubleSide }));
  layerEdge = new THREE.LineSegments(new THREE.EdgesGeometry(pg), new THREE.LineBasicMaterial());
  for (const o of [layerPlane, layerEdge]) { o.position.x = box.position.x; o.position.y = box.position.y; world.add(o); }
  markers = new THREE.Group();
  scene.add(markers);
  raycaster = new THREE.Raycaster();
  setZs(V.zs);
  resetView();
  new ResizeObserver(resize).observe($("isoWrap"));
  resize();
}
function setZs(zs) {
  V.zs = zs;
  const d = S.meta.d, mid = (d[0] + d.at(-1)) / 2;
  world.scale.set(1, 1, zs);
  world.position.set(0, 0, -mid * zs);
  world.updateMatrixWorld(true);
  $("isoZsValue").textContent = String(zs);
  sync();
}
const toWorld = (phi1, phi2, d) => world.localToWorld(new THREE.Vector3(phi1, phi2, d));
function resetView() {
  const az = THREE.MathUtils.degToRad(-50), el = THREE.MathUtils.degToRad(24), r = 9.6;
  camera.position.set(r * Math.cos(el) * Math.cos(az), r * Math.cos(el) * Math.sin(az), r * Math.sin(el));
  controls.target.set(0, 0, 0);
  controls.update();
  requestRender();
}
function resize() {
  const r = $("isoWrap").getBoundingClientRect();
  if (!r.width || !r.height) return;
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
  requestRender();
}
function requestRender() {
  if (V.frame) return;
  V.frame = requestAnimationFrame(() => { V.frame = 0; renderer.render(scene, camera); drawLabels(); });
}

// Tick labels: HTML spans at projected points on the box edges nearest the viewer.
const labelPool = [];
function drawLabels() {
  const box3 = $("isoLabels"), r = box3.getBoundingClientRect(), { phi1: p1, phi2: p2, d } = S.meta;
  const items = [], cam = world.worldToLocal(camera.position.clone());
  const yEdge = cam.y < 0 ? p2[0] : p2.at(-1), xEdge = cam.x < 0 ? p1[0] : p1.at(-1);
  const out = 0.22, dy = Math.sign(yEdge) * out, dx = Math.sign(xEdge) * out;
  // d along the vertical edge that projects furthest left.
  const corners = [[p1[0], p2[0]], [p1[0], p2.at(-1)], [p1.at(-1), p2[0]], [p1.at(-1), p2.at(-1)]];
  const screenX = ([x, y]) => toWorld(x, y, (d[0] + d.at(-1)) / 2).project(camera).x;
  const [cx, cy] = corners.reduce((a, b) => (screenX(b) < screenX(a) ? b : a));
  const ticks = [[p1[0], "−π/2"], [0, "0"], [p1.at(-1), "π/2"]];
  for (const [v, t] of ticks) {               // skip the tick under the d labels
    if (!(v === cx && yEdge === cy)) items.push([v, yEdge + dy, d[0], t]);
    if (!(v === cy && xEdge === cx)) items.push([xEdge + dx, v, d[0], t]);
  }
  items.push([0, yEdge + 2.2 * dy, d[0], "φ₁", true], [xEdge + 2.2 * dx, 0, d[0], "φ₂", true]);
  const len = Math.hypot(cx, cy), ox = (cx / len) * out, oy = (cy / len) * out;
  const dTicks = [d[0]];
  for (let v = Math.ceil(d[0] * 20 + 1e-6) / 20; v < d.at(-1) - 0.02; v += 0.05) if (v > d[0] + 0.02) dTicks.push(v);
  dTicks.push(d.at(-1));
  for (const v of dTicks) items.push([cx + ox, cy + oy, v, v.toFixed(2)]);
  items.push([cx + 2.4 * ox, cy + 2.4 * oy, d.at(-1) + 0.03, "d", true]);
  items.forEach(([x, y, z, text, axis], n) => {
    let el = labelPool[n];
    if (!el) { el = document.createElement("span"); box3.appendChild(el); labelPool[n] = el; }
    const p = toWorld(x, y, z).project(camera);
    el.textContent = text; el.className = axis ? "axis" : "";
    el.style.left = `${((p.x + 1) / 2) * r.width}px`; el.style.top = `${((1 - p.y) / 2) * r.height}px`;
    el.hidden = p.z > 1;
  });
  for (let n = items.length; n < labelPool.length; n++) labelPool[n].hidden = true;
}

// ------------------------------------------------------------------ surfaces
function parse(buf) {
  const [nv, nt, nSnap] = new Uint32Array(buf, 0, 4);
  return { nv, nt, nSnap, xyz: new Float32Array(buf, 16, nv * 3), col: new Float32Array(buf, 16 + 12 * nv, nv),
           tri: new Uint32Array(buf, 16 + 16 * nv, nt * 3) };
}
const colorField = () => (V.color === "level" ? "d" : V.color);
async function fetchSurface(field, level, signal) {
  const url = `/api/iso?field=${field}&level=${level}&color=${colorField()}&stride=${V.stride}`;
  const r = await fetch(url, { signal });
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 200)}`);
  return { field, level, color: colorField(), stride: V.stride, ...parse(await r.arrayBuffer()) };
}
function makeMesh(s) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(s.xyz, 3));
  g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(s.nv * 3), 3));
  g.setIndex(new THREE.BufferAttribute(s.tri, 1));
  g.computeVertexNormals();
  const m = new THREE.MeshPhongMaterial({ vertexColors: true, side: THREE.DoubleSide, shininess: 30,
                                          specular: 0x1a1a1a, clippingPlanes: V.planes });
  s.mesh = new THREE.Mesh(g, m);
  s.mesh.userData.surface = s;
  world.add(s.mesh);
}
function dropMesh(s) {
  if (!s || !s.mesh) return;
  world.remove(s.mesh); s.mesh.geometry.dispose(); s.mesh.material.dispose(); s.mesh = null;
}

// Colour every surface from the shared scale; returns the scale for the colour bar.
function recolor() {
  let norm = null;
  const all = surfaces();
  if (V.color === "d") {
    const d = S.meta.d;
    norm = { lut: lut(VIRIDIS), f: (v) => (v - d[0]) / (d.at(-1) - d[0]), ticks: [0.7, 0.75, 0.8, 0.85, 0.9, 0.95].filter((v) => v >= d[0] && v <= d.at(-1)),
             caption: "Colour: d (end-to-end distance)" };
  } else if (V.color !== "level") {
    let lo = Infinity, hi = -Infinity;
    for (const s of all) for (const v of s.col) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (lo <= hi) {
      norm = makeNorm(V.color, V.color === "energy" ? [Math.max(lo, 1e-6), hi] : [lo, hi]);
      norm.caption = `Colour: ${S.meta.fields[V.color].label} on the surfaces shown`;
    }
  }
  const linear = norm ? linearLut(norm.lut) : null;
  if (V.active && V.active.mesh) {               // a kept copy of the slider's surface shows instead
    V.active.mesh.visible = !V.kept.some((s) => s.field === V.active.field && s.level === V.active.level);
  }
  for (const s of all) {
    if (!s.mesh) continue;
    const c = s.mesh.geometry.getAttribute("color"), a = c.array;
    if (!norm) {
      const k = new THREE.Color(palette(s.slot));                 // sRGB in, linear out
      for (let n = 0; n < s.nv; n++) { a[3 * n] = k.r; a[3 * n + 1] = k.g; a[3 * n + 2] = k.b; }
    } else {
      for (let n = 0; n < s.nv; n++) {
        const m = Math.round(Math.min(1, Math.max(0, norm.f(s.col[n]))) * 255) * 3;
        a[3 * n] = linear[m]; a[3 * n + 1] = linear[m + 1]; a[3 * n + 2] = linear[m + 2];
      }
    }
    c.needsUpdate = true;
  }
  drawColorbar3d(norm);
  renderChips();
  requestRender();
}
function drawColorbar3d(norm) {
  const canvas = $("isoColorbar");
  canvas.hidden = !norm;
  if (!norm) return;
  const { ctx, w } = setupCanvas(canvas), x0 = 8, x1 = w - 8, y0 = 4, hBar = 10;
  for (let x = x0; x < x1; x++) {
    const n = Math.round(((x - x0) / (x1 - x0)) * 255) * 3;
    ctx.fillStyle = `rgb(${norm.lut[n]},${norm.lut[n + 1]},${norm.lut[n + 2]})`;
    ctx.fillRect(x, y0, 1, hBar);
  }
  ctx.fillStyle = css("--muted"); ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "top";
  let last = -1e9;
  for (const v of norm.ticks) {
    const x = x0 + Math.min(1, Math.max(0, norm.f(v))) * (x1 - x0);
    if (x - last < 30) continue;
    ctx.fillRect(x, y0 + hBar, 1, 3); ctx.fillText(fmt(v, 3), x, y0 + hBar + 5); last = x;
  }
  ctx.fillStyle = css("--ink-2"); ctx.textAlign = "left";
  ctx.fillText(norm.caption, x0, y0 + hBar + 20);
}
function renderChips() {
  const box3 = $("isoKept"); box3.replaceChildren();
  for (const s of surfaces()) {
    const chip = document.createElement(s === V.active ? "span" : "button");
    chip.className = "chip";
    const sw = document.createElement("span"); sw.className = "swatch"; sw.style.background = palette(s.slot);
    chip.append(sw, document.createTextNode(`${describe(s)}${s === V.active ? " (slider)" : ""}`));
    if (s !== V.active) {
      const x = document.createElement("span"); x.className = "x"; x.textContent = "×"; chip.appendChild(x);
      chip.type = "button"; chip.title = "Remove this surface";
      chip.addEventListener("click", () => { dropMesh(s); V.kept = V.kept.filter((k) => k !== s); recolor(); });
    }
    box3.appendChild(chip);
  }
}

// The slider's surface: newest request wins.
async function loadActive() {
  if (V.abort) V.abort.abort();
  const ctl = new AbortController(), { field, level } = V;
  V.abort = ctl;
  status(`Computing ${ISO_SHORT[field]} = ${fmt(level)}…`);
  const t0 = performance.now();
  try {
    const s = await fetchSurface(field, level, ctl.signal);
    dropMesh(V.active);
    s.slot = 0;
    V.active = s;
    makeMesh(s);
    recolor();
    status(s.nt ? "" : `No ${describe(s)} level set in the box`);
    $("isoInfo").textContent = `${describe(s)}: ${s.nt.toLocaleString("en-US")} triangles · ${Math.round(performance.now() - t0)} ms`;
  } catch (err) {
    if (err.name !== "AbortError") status(`Could not compute the surface: ${err.message}`);
  }
}
async function keep(field, level) {
  if (V.kept.some((s) => s.field === field && s.level === level)) return;
  if (V.kept.length >= MAX_KEPT) { status(`At most ${MAX_KEPT} kept surfaces; remove one first`); return; }
  const used = new Set(V.kept.map((s) => s.slot));
  const slot = [1, 2, 3, 4, 5, 6, 7].find((n) => !used.has(n));
  let s;
  if (V.active && V.active.field === field && V.active.level === level && V.active.color === colorField()) {
    s = V.active; V.active = null;            // hand the mesh over; the slider gets a fresh copy
  } else {
    try { s = await fetchSurface(field, level); makeMesh(s); } catch (err) { status(`Could not compute the surface: ${err.message}`); return; }
  }
  s.slot = slot;
  V.kept.push(s);
  recolor();
  if (!V.active) loadActive();
}
async function refetchAll() {            // after a change of colour field or grid stride
  const old = V.kept;
  V.kept = [];
  for (const s of old) dropMesh(s);
  status("Recomputing…");
  const fresh = await Promise.all(old.map((s) => fetchSurface(s.field, s.level).then((n) => ({ ...n, slot: s.slot })).catch(() => null)));
  for (const s of fresh) if (s) { makeMesh(s); V.kept.push(s); }
  await loadActive();
}
async function loadSnap() {
  try {
    const r = await fetch(`/api/snapsurface?stride=${V.stride}`);
    if (!r.ok) throw new Error(String(r.status));
    const s = parse(await r.arrayBuffer());
    if (V.snap) { world.remove(V.snap.mesh); V.snap.mesh.geometry.dispose(); }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(s.xyz, 3));
    g.setIndex(new THREE.BufferAttribute(s.tri, 1));
    g.computeVertexNormals();
    g.addGroup(0, 3 * s.nSnap, 0);
    g.addGroup(3 * s.nSnap, 3 * (s.nt - s.nSnap), 1);
    const mats = V.snap ? V.snap.mesh.material : [0.3, 0.16].map((opacity) => new THREE.MeshLambertMaterial({
      transparent: true, opacity, depthWrite: false, side: THREE.DoubleSide, clippingPlanes: V.planes }));
    s.mesh = new THREE.Mesh(g, mats);
    s.mesh.renderOrder = 1;
    s.mesh.userData.snap = s;
    V.snap = s;
    world.add(s.mesh);
    theme();
  } catch (err) {
    status(`Could not load the snap surface: ${err.message}`);
  }
}

// ------------------------------------------------------------------ app state -> scene
function sync() {
  if (!V.started) return;
  const d = S.meta.d[S.i];
  layerPlane.position.z = layerEdge.position.z = d;
  layerPlane.visible = layerEdge.visible = $("isoPlane").checked;
  // Clipping planes in world coordinates: a point is kept where n . p + c >= 0.
  const z = toWorld(0, 0, d).z, eps = 1e-4;
  const plane = {
    above: [new THREE.Vector3(0, 0, -1), z + eps], below: [new THREE.Vector3(0, 0, 1), -z + eps],
    mirror: [new THREE.Vector3(-1, -1, 0).normalize(), 1e-6], phi1: [new THREE.Vector3(-1, 0, 0), 1e-6],
  }[V.clip];
  V.planes.length = 0;
  if (plane) V.planes.push(new THREE.Plane(plane[0], plane[1]));
  if (V.snap) {
    V.snap.mesh.material[0].visible = $("isoSnap").checked;
    V.snap.mesh.material[1].visible = $("isoSmooth").checked;
  }
  // Probe and pins: spheres drawn on top, so they show inside the surfaces too.
  for (const m of [...markers.children]) { markers.remove(m); m.geometry.dispose(); m.material.dispose(); }
  for (const p of [...(S.probe ? [S.probe] : []), ...S.pins]) {
    const ball = new THREE.Mesh(new THREE.SphereGeometry(p === S.probe ? 0.055 : 0.045, 20, 14),
                                new THREE.MeshBasicMaterial({ color: slotColor(p.slot), depthTest: false }));
    ball.position.copy(toWorld(p.phi1, p.phi2, p.d));
    ball.renderOrder = 3;
    markers.add(ball);
  }
  requestRender();
}
function theme() {
  if (!V.started) return;
  renderer.setClearColor(new THREE.Color(css("--page")));
  box.material.color.set(css("--axis"));
  layerPlane.material.color.set(css("--ink"));
  layerEdge.material.color.set(css("--accent"));
  if (V.snap) {
    V.snap.mesh.material[0].color.set(css("--critical"));
    V.snap.mesh.material[1].color.set(css("--muted"));
  }
  recolor();
  sync();
}

// ------------------------------------------------------------------ picking
function pick(event) {
  const r = $("iso3d").getBoundingClientRect();
  const ndc = new THREE.Vector2(((event.clientX - r.left) / r.width) * 2 - 1, -((event.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const targets = surfaces().map((s) => s.mesh).filter((m) => m && m.visible);
  if (V.snap && $("isoSnap").checked) targets.push(V.snap.mesh);
  for (const hit of raycaster.intersectObjects(targets, false)) {
    if (V.planes.some((pl) => pl.distanceToPoint(hit.point) < 0)) continue;       // cut away
    const snapHit = hit.object.userData.snap;
    if (snapHit && hit.faceIndex >= snapHit.nSnap && !$("isoSmooth").checked) continue;
    const local = world.worldToLocal(hit.point.clone());
    return { hit, local, surface: hit.object.userData.surface, snap: snapHit };
  }
  return null;
}
function nodeAt(local) {
  return { i: nearest(S.meta.d, local.z), j: nearest(S.meta.phi1, local.x), k: nearest(S.meta.phi2, local.y) };
}
function valueAt(s, hit) {                       // the colour field, barycentric on the hit triangle
  const { a, b, c } = hit.face, P = s.xyz, vec = (n) => new THREE.Vector3(P[3 * n], P[3 * n + 1], P[3 * n + 2]);
  const bary = new THREE.Triangle(vec(a), vec(b), vec(c)).getBarycoord(world.worldToLocal(hit.point.clone()), new THREE.Vector3());
  return bary.x * s.col[a] + bary.y * s.col[b] + bary.z * s.col[c];
}
let hoverEvent = null, hoverFrame = 0;
function hover() {
  hoverFrame = 0;
  const e = hoverEvent, tip = $("isoTip");
  const found = e && pick(e);
  if (!found) { tip.hidden = true; return; }
  const { hit, local, surface, snap } = found, node = nodeAt(local), m = S.meta;
  tip.replaceChildren();
  const add = (text, cls) => { const el = document.createElement("div"); if (cls) el.className = cls; el.textContent = text; tip.appendChild(el); return el; };
  const head = add("");
  const strong = document.createElement("strong");
  strong.textContent = surface ? describe(surface) : hit.faceIndex < snap.nSnap ? "Snap surface (p = 0, jump)" : "p = 0, smooth crossing";
  head.appendChild(strong);
  add(`φ₁ ${fmtPhi(local.x)}, φ₂ ${fmtPhi(local.y)}, d ${local.z.toFixed(3)}`, "sec");
  if (surface && V.color !== "level" && V.color !== surface.field && V.color !== "d") {
    add(`${ISO_SHORT[V.color]} ≈ ${fmt(valueAt(surface, hit))}`);
  }
  add(`Click: probe d ${m.d[node.i].toFixed(2)}, φ (${fmtPhi(m.phi1[node.j])}, ${fmtPhi(m.phi2[node.k])})`, "sec");
  tip.hidden = false;
  const r = $("isoWrap").getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  tip.style.left = `${Math.min(x + 14, r.width - tip.offsetWidth - 4)}px`;
  tip.style.top = `${Math.max(4, y - tip.offsetHeight - 10)}px`;
}

// ------------------------------------------------------------------ controls
function setField(field) {
  V.field = field;
  for (const b of $("isoFieldButtons").children) b.setAttribute("aria-pressed", b.dataset.field === field);
  const presets = PRESETS[field].filter((v) => v > S.meta.fields[field].min && v < S.meta.fields[field].max);
  const box3 = $("isoPresets"); box3.replaceChildren();
  for (const v of presets) {
    const b = document.createElement("button"); b.type = "button"; b.className = "chip"; b.textContent = fmt(v);
    b.addEventListener("click", () => setLevel(v, true)); box3.appendChild(b);
  }
  const all = document.createElement("button"); all.type = "button"; all.className = "chip"; all.textContent = "Keep all";
  all.title = "Keep a surface at every preset level";
  all.addEventListener("click", async () => { for (const v of presets) await keep(field, v); });
  box3.appendChild(all);
  setLevel(presets.includes(V.level) ? V.level : presets[Math.floor(presets.length / 2)], true);
}
function setLevel(level, now) {
  V.level = level;
  $("isoLevel").value = level;
  $("isoSlider").value = Math.round(sliderMap(V.field).from(level) * 1000);
  clearTimeout(V.timer);
  if (now) loadActive(); else V.timer = setTimeout(loadActive, 140);
}
function wire() {
  for (const f of S.meta.iso_fields) {
    const b = document.createElement("button"); b.type = "button"; b.dataset.field = f; b.textContent = ISO_SHORT[f];
    b.title = S.meta.fields[f].label; b.addEventListener("click", () => setField(f));
    $("isoFieldButtons").appendChild(b);
  }
  $("isoSlider").addEventListener("input", (e) => {
    const v = niceLevel(sliderMap(V.field).to(+e.target.value / 1000));
    V.level = v; $("isoLevel").value = v;
    clearTimeout(V.timer); V.timer = setTimeout(loadActive, 140);
  });
  $("isoLevel").addEventListener("change", (e) => { const v = parseFloat(e.target.value); if (Number.isFinite(v)) setLevel(v, true); });
  $("isoKeep").addEventListener("click", () => keep(V.field, V.level));
  $("isoColor").addEventListener("change", (e) => {
    const before = colorField();
    V.color = e.target.value;
    if (colorField() !== before) refetchAll(); else recolor();
  });
  $("isoStride").addEventListener("change", (e) => { V.stride = +e.target.value; loadSnap(); refetchAll(); });
  $("isoClip").addEventListener("change", (e) => { V.clip = e.target.value; sync(); });
  $("isoZs").addEventListener("input", (e) => setZs(+e.target.value));
  for (const id of ["isoSnap", "isoSmooth", "isoPlane"]) $(id).addEventListener("change", sync);
  $("isoReset").addEventListener("click", resetView);

  const canvas = $("iso3d");
  let down = null;
  canvas.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY }; canvas.classList.add("dragging"); $("isoTip").hidden = true; });
  canvas.addEventListener("pointerup", (e) => {
    canvas.classList.remove("dragging");
    const click = down && e.button === 0 && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5;
    down = null;
    if (!click) return;
    const found = pick(e);
    if (found) { const n = nodeAt(found.local); goTo(n.i, n.j, n.k, e.shiftKey); }
  });
  canvas.addEventListener("pointermove", (e) => {
    if (down) return;
    hoverEvent = e;
    if (!hoverFrame) hoverFrame = requestAnimationFrame(hover);
  });
  canvas.addEventListener("pointerleave", () => { hoverEvent = null; $("isoTip").hidden = true; });
}

function init() {
  if (V.started || !S.meta) return;
  V.started = true;
  buildScene();
  wire();
  theme();
  loadSnap();
  setField(V.field);
}

window.viewer3d = { init, sync, theme };
if (S.meta) init();
