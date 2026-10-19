// 3D height map of the current d layer. Loaded as a module after app.js; uses its globals (S, $, css,
// fmt, fmtPhi, branchName, makeNorm, slotColor, probe, renderAll, SHORT, FIELD_ORDER, isDensity)
// and is driven by app.js through window.heightmap. The surface is coloured by the 2D map's own
// image (S.image), so colours match the heat map; the height is a chosen field on the same layer.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const SCALE_WORD = { log: "log scale, this layer", symlog: "symmetric log scale, this layer",
                     branch: "linear", count: "log scale, all layers" };
const H = {
  started: false, zs: 0.5,
  key: "",               // layer + height field the surface was built for
  values: null,          // height field on the layer, [j][k]
  norm: null,            // its scale; heights are norm.f(v) in [0, 1]
  image: null, snap: null, fetching: "", frame: 0,
};
let renderer, scene, camera, controls, world, surface, box, zeroPlane, snapDots, smoothDots, markers, raycaster;

const heightField = () => S.heightField || S.field;
const span = () => [S.meta.phi1[0], S.meta.phi1.at(-1), S.meta.phi2[0], S.meta.phi2.at(-1)];
const toWorld = (x, y, t) => world.localToWorld(new THREE.Vector3(x, y, t));
const heightOf = (v) => { const t = H.norm.f(v); return Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0; };

// ------------------------------------------------------------------ scene
function buildScene() {
  const canvas = $("map3d");
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(32, 1, 0.01, 200);
  camera.up.set(0, 0, 1);                         // the height is vertical
  controls = new OrbitControls(camera, canvas);
  controls.addEventListener("change", requestRender);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a8a, 1.9));
  const sun = new THREE.DirectionalLight(0xffffff, 1.1);
  sun.position.set(1, 2, 3);
  camera.add(sun);
  scene.add(camera);

  // The surface lives in (phi1, phi2, t), t in [0, 1]; the world group stretches t by zs * pi.
  world = new THREE.Group();
  scene.add(world);
  const [x0, x1, y0, y1] = span(), [, n1, n2] = S.meta.shape;
  const g = new THREE.BufferGeometry(), pos = new Float32Array(n1 * n2 * 3), uv = new Float32Array(n1 * n2 * 2);
  for (let j = 0; j < n1; j++) for (let k = 0; k < n2; k++) {
    const n = j * n2 + k;                         // the layout of S.layer
    pos[3 * n] = S.meta.phi1[j]; pos[3 * n + 1] = S.meta.phi2[k];
    uv[2 * n] = (j + 0.5) / n1; uv[2 * n + 1] = (k + 0.5) / n2;      // texel centres of S.image
  }
  const tri = new Uint32Array((n1 - 1) * (n2 - 1) * 6);
  let m = 0;
  for (let j = 0; j < n1 - 1; j++) for (let k = 0; k < n2 - 1; k++) {
    const a = j * n2 + k, b = a + n2;
    tri.set([a, b, b + 1, a, b + 1, a + 1], m); m += 6;
  }
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  g.setIndex(new THREE.BufferAttribute(tri, 1));
  surface = new THREE.Mesh(g, new THREE.MeshPhongMaterial({ side: THREE.DoubleSide, shininess: 20, specular: 0x111111 }));
  world.add(surface);

  const bx = new THREE.BoxGeometry(x1 - x0, y1 - y0, 1);
  box = new THREE.LineSegments(new THREE.EdgesGeometry(bx), new THREE.LineBasicMaterial());
  box.position.set((x0 + x1) / 2, (y0 + y1) / 2, 0.5);
  world.add(box);
  const pg = new THREE.PlaneGeometry(x1 - x0, y1 - y0);
  zeroPlane = new THREE.Mesh(pg, new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.08, depthWrite: false, side: THREE.DoubleSide }));
  zeroPlane.position.set(box.position.x, box.position.y, 0.5);
  world.add(zeroPlane);
  const dots = () => new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 2.5, sizeAttenuation: false }));
  snapDots = dots(); smoothDots = dots();
  world.add(snapDots, smoothDots);
  markers = new THREE.Group();
  scene.add(markers);
  raycaster = new THREE.Raycaster();
  setZs(H.zs);
  resetView();
  new ResizeObserver(resize).observe($("map3d").parentElement);
  resize();
}
function setZs(zs) {
  H.zs = zs;
  world.scale.set(1, 1, zs * Math.PI);
  world.updateMatrixWorld(true);
  $("heightZsValue").textContent = zs.toFixed(2);
  if (H.values) { placeMarkers(); requestRender(); }
}
function resetView() {
  const az = THREE.MathUtils.degToRad(-60), el = THREE.MathUtils.degToRad(30), r = 9.6;
  camera.position.set(r * Math.cos(el) * Math.cos(az), r * Math.cos(el) * Math.sin(az), r * Math.sin(el));
  controls.target.set(0, 0, H.zs * Math.PI / 2);
  controls.update();
  requestRender();
}
function resize() {
  const r = $("map3d").getBoundingClientRect();
  if (!r.width || !r.height) return;
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
  requestRender();
}
function requestRender() {
  if (H.frame || !H.started) return;
  H.frame = requestAnimationFrame(() => { H.frame = 0; renderer.render(scene, camera); drawLabels(); });
}

// Tick labels: HTML spans at projected points on the box edges nearest the viewer.
const labelPool = [];
function drawLabels() {
  const host = $("mapLabels"), r = host.getBoundingClientRect(), [x0, x1, y0, y1] = span();
  const items = [], cam = world.worldToLocal(camera.position.clone());
  const yEdge = cam.y < 0 ? y0 : y1, xEdge = cam.x < 0 ? x0 : x1;
  const out = 0.22, dy = Math.sign(yEdge) * out, dx = Math.sign(xEdge) * out;
  const corners = [[x0, y0], [x0, y1], [x1, y0], [x1, y1]];
  const screenX = ([x, y]) => toWorld(x, y, 0.5).project(camera).x;
  const [cx, cy] = corners.reduce((a, b) => (screenX(b) < screenX(a) ? b : a));
  for (const [v, t] of [[x0, "−π/2"], [0, "0"], [x1, "π/2"]]) {
    if (!(v === cx && yEdge === cy)) items.push([v, yEdge + dy, 0, t]);
    if (!(v === cy && xEdge === cx)) items.push([xEdge + dx, v, 0, t]);
  }
  items.push([0, yEdge + 2.2 * dy, 0, "φ₁", true], [xEdge + 2.2 * dx, 0, 0, "φ₂", true]);
  const len = Math.hypot(cx, cy), ox = (cx / len) * out, oy = (cy / len) * out;
  if (H.norm) {                                  // height ticks, thinned on screen
    let last = -1e9;
    const lo = Math.min(...H.range), hi = Math.max(...H.range);
    for (const v of [...H.norm.ticks].sort((a, b) => a - b)) {
      if (v < lo || v > hi) continue;
      const t = heightOf(v), py = (1 - toWorld(cx, cy, t).project(camera).y) / 2 * r.height;
      if (Math.abs(py - last) < 16) continue;
      items.push([cx + ox, cy + oy, t, fmt(v, 3)]); last = py;
    }
    items.push([cx, cy, 1.12, SHORT[heightField()] || heightField(), true]);      // above the tick column
  }
  items.forEach(([x, y, z, text, axis], n) => {
    let el = labelPool[n];
    if (!el) { el = document.createElement("span"); host.appendChild(el); labelPool[n] = el; }
    const p = toWorld(x, y, z).project(camera);
    el.textContent = text; el.className = axis ? "axis" : "";
    const half = el.offsetWidth / 2 + 2;           // keep labels inside the canvas
    el.style.left = `${Math.min(r.width - half, Math.max(half, ((p.x + 1) / 2) * r.width))}px`;
    el.style.top = `${((1 - p.y) / 2) * r.height}px`;
    el.hidden = p.z > 1;
  });
  for (let n = items.length; n < labelPool.length; n++) labelPool[n].hidden = true;
}

// ------------------------------------------------------------------ data -> surface
async function heightValues(field, i) {
  if (field === S.field) return S.layer;
  if (field === "q" && S.qLayer) return S.qLayer;
  const r = await fetch(`/api/layer?i=${i}&field=${field}`);
  if (!r.ok) throw new Error(`layer ${r.status}`);
  return new Float32Array(await r.arrayBuffer());
}
function setHeights(values, field) {
  H.values = values;
  H.norm = makeNorm(field, values);
  let lo = Infinity, hi = -Infinity;
  for (const v of values) if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  H.range = [lo, hi];
  const pos = surface.geometry.getAttribute("position"), a = pos.array;
  for (let n = 0; n < values.length; n++) a[3 * n + 2] = heightOf(values[n]);
  pos.needsUpdate = true;
  surface.geometry.computeVertexNormals();
  surface.geometry.computeBoundingSphere();
  const kind = S.meta.fields[field].scale, zero = kind === "branch" || kind === "symlog";
  zeroPlane.visible = zero;
  if (zero) zeroPlane.position.z = heightOf(0);
  $("heightInfo").textContent = `Height: ${S.meta.fields[field].label}   ·   ${SCALE_WORD[kind]}` + (zero ? "   ·   plane: zero" : "");
  H.snap = null;                                 // the dots sit on the surface: re-place them
}
function setTexture() {
  H.image = S.image;
  const old = surface.material.map, tex = new THREE.CanvasTexture(S.image);
  tex.magFilter = tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.colorSpace = THREE.SRGBColorSpace;
  surface.material.map = tex;
  surface.material.needsUpdate = true;
  if (old) old.dispose();
}
// Height at any (phi1, phi2), bilinear on the grid: an edge midpoint gets the mean of its ends.
function heightAt(x, y) {
  const { phi1: p1, phi2: p2, shape: [, n1, n2] } = S.meta;
  const u = ((x - p1[0]) / (p1.at(-1) - p1[0])) * (n1 - 1), v = ((y - p2[0]) / (p2.at(-1) - p2[0])) * (n2 - 1);
  const j = Math.min(n1 - 2, Math.max(0, Math.floor(u))), k = Math.min(n2 - 2, Math.max(0, Math.floor(v)));
  const fu = u - j, fv = v - k, h = (jj, kk) => heightOf(H.values[jj * n2 + kk]);
  return (1 - fu) * ((1 - fv) * h(j, k) + fv * h(j, k + 1)) + fu * ((1 - fv) * h(j + 1, k) + fv * h(j + 1, k + 1));
}
function placeDots() {
  H.snap = S.snap;
  for (const [obj, list] of [[snapDots, S.snap.snap], [smoothDots, S.snap.smooth]]) {
    const a = new Float32Array(list.length * 3);
    list.forEach(([x, y], n) => { a[3 * n] = x; a[3 * n + 1] = y; a[3 * n + 2] = heightAt(x, y) + 0.004; });
    obj.geometry.dispose();
    obj.geometry = new THREE.BufferGeometry();
    obj.geometry.setAttribute("position", new THREE.BufferAttribute(a, 3));
  }
}
function placeMarkers() {
  for (const m of [...markers.children]) { markers.remove(m); m.geometry.dispose(); m.material.dispose(); }
  if (!H.values) return;
  const n2 = S.meta.shape[2];
  for (const p of [...S.pins.filter((q) => q.i === S.i), ...(S.probe && S.probe.i === S.i ? [S.probe] : [])]) {
    const ball = new THREE.Mesh(new THREE.SphereGeometry(p === S.probe ? 0.06 : 0.05, 20, 14),
                                new THREE.MeshBasicMaterial({ color: slotColor(p.slot), depthTest: false }));
    ball.position.copy(toWorld(p.phi1, p.phi2, heightOf(H.values[p.j * n2 + p.k])));
    ball.renderOrder = 3;
    markers.add(ball);
  }
}

// Called from renderAll(): cheap unless the layer, the height field or the colours changed.
async function update() {
  showView();
  if (S.view !== "3d" || !S.meta || !S.layer || !S.image) return;
  if (!H.started) { H.started = true; buildScene(); theme(); }
  const field = heightField(), key = `${S.i}:${field}:${S.field}`;
  if (key !== H.key && H.fetching !== key) {
    H.fetching = key;
    try {
      const values = await heightValues(field, S.i);
      if (H.fetching !== key) return;            // a newer layer or field superseded this one
      H.key = key;
      setHeights(values, field);
    } catch (err) {
      showBanner(`Could not load the height layer: ${err.message}`);
    } finally {
      if (H.fetching === key) H.fetching = "";
    }
  }
  if (!H.values || H.key !== key) return;
  if (H.image !== S.image) setTexture();
  if (H.snap !== S.snap) placeDots();
  snapDots.visible = $("showSnap").checked;
  smoothDots.visible = $("showSmooth").checked;
  placeMarkers();
  requestRender();
}
function theme() {
  if (!H.started) return;
  renderer.setClearColor(new THREE.Color(css("--surface")));
  box.material.color.set(css("--axis"));
  zeroPlane.material.color.set(css("--ink"));
  snapDots.material.color.set(css("--ink"));
  smoothDots.material.color.set(css("--muted"));
  placeMarkers();
  requestRender();
}

// ------------------------------------------------------------------ view switch and picking
function showView() {
  const three = S.view === "3d";
  for (const b of $("viewButtons").children) b.setAttribute("aria-pressed", b.dataset.view === S.view);
  $("map").hidden = three;
  $("map3d").hidden = $("mapLabels").hidden = !three;
  for (const id of ["heightControl", "heightZsControl", "heightReset", "heightInfo"]) $(id).hidden = !three;
  $("mapHint").hidden = three; $("mapHint3d").hidden = !three;
}
function pick(event) {
  const r = $("map3d").getBoundingClientRect();
  const ndc = new THREE.Vector2(((event.clientX - r.left) / r.width) * 2 - 1, -((event.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const hit = raycaster.intersectObject(surface, false)[0];
  if (!hit) return null;
  const local = world.worldToLocal(hit.point.clone()), [, n1, n2] = S.meta.shape, [x0, x1, y0, y1] = span();
  const j = Math.round(((local.x - x0) / (x1 - x0)) * (n1 - 1)), k = Math.round(((local.y - y0) / (y1 - y0)) * (n2 - 1));
  return { j: Math.min(n1 - 1, Math.max(0, j)), k: Math.min(n2 - 1, Math.max(0, k)) };
}
let hoverEvent = null, hoverFrame = 0;
function hover() {
  hoverFrame = 0;
  const e = hoverEvent, tip = $("mapTip"), at = e && H.values && pick(e);
  if (!at) { tip.hidden = true; return; }
  const n2 = S.meta.shape[2], n = at.j * n2 + at.k, field = heightField();
  tip.replaceChildren();
  const line = (value, name, cls) => {
    const el = document.createElement("div"); if (cls) el.className = cls;
    const strong = document.createElement("strong"); strong.textContent = value;
    const sec = document.createElement("span"); sec.className = "sec"; sec.textContent = ` ${name}`;
    el.append(strong, sec); tip.appendChild(el);
  };
  line(isDensity(S.field) ? `${fmt(S.layer[n], 3)} points` : fmt(S.layer[n]), `${SHORT[S.field]} (colour)`);
  if (field !== S.field) line(isDensity(field) ? `${fmt(H.values[n], 3)} points` : fmt(H.values[n]), `${SHORT[field]} (height)`);
  const where = document.createElement("div"); where.className = "sec";
  where.textContent = `φ₁ ${fmtPhi(S.meta.phi1[at.j])}, φ₂ ${fmtPhi(S.meta.phi2[at.k])} · ${branchName(S.qLayer[n])}`;
  tip.appendChild(where);
  tip.hidden = false;
  const r = $("map3d").getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  tip.style.left = `${Math.min(x + 14, r.width - tip.offsetWidth - 4)}px`;
  tip.style.top = `${Math.max(4, y - tip.offsetHeight - 10)}px`;
}
function wire() {
  const select = $("heightField");
  const same = document.createElement("option"); same.value = ""; same.textContent = "same as colour";
  select.appendChild(same);
  const fields = [...FIELD_ORDER.filter((f) => f !== "density"), ...Object.keys(S.meta.fields).filter(isDensity)];
  for (const f of fields) {
    if (!S.meta.fields[f]) continue;
    const o = document.createElement("option"); o.value = f; o.textContent = SHORT[f]; select.appendChild(o);
  }
  select.value = S.heightField || "";
  select.addEventListener("change", () => { S.heightField = select.value; renderAll(); });
  for (const b of $("viewButtons").children) {
    b.disabled = false;
    b.addEventListener("click", () => {          // unhide first: drawMap() sizes the canvas from its box
      S.view = b.dataset.view; $("mapTip").hidden = true; showView(); renderAll();
    });
  }
  $("heightZs").addEventListener("input", (e) => { if (H.started) setZs(+e.target.value); else H.zs = +e.target.value; });
  $("heightReset").addEventListener("click", () => { if (H.started) resetView(); });

  const canvas = $("map3d");
  let down = null;
  canvas.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY }; canvas.classList.add("dragging"); $("mapTip").hidden = true; });
  canvas.addEventListener("pointerup", (e) => {
    canvas.classList.remove("dragging");
    const click = down && e.button === 0 && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5;
    down = null;
    if (!click) return;
    const at = pick(e);
    if (at) { canvas.focus(); probe(at.j, at.k, e.shiftKey); }
  });
  canvas.addEventListener("pointermove", (e) => {
    if (down) return;
    hoverEvent = e;
    if (!hoverFrame) hoverFrame = requestAnimationFrame(hover);
  });
  canvas.addEventListener("pointerleave", () => { hoverEvent = null; $("mapTip").hidden = true; });
  canvas.addEventListener("keydown", (e) => {          // the same probe keys as the 2D map
    const step = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[e.key];
    if (!step) return;
    e.preventDefault();
    const [, n1, n2] = S.meta.shape, p = S.probe || { j: (n1 / 2) | 0, k: (n2 / 2) | 0 };
    probe(Math.max(0, Math.min(n1 - 1, p.j + step[0])), Math.max(0, Math.min(n2 - 1, p.k + step[1])), false);
  });
}

let wired = false;
function init() {
  if (wired || !S.meta) return;
  wired = true;
  wire();
  update();
}

window.heightmap = { init, update, theme };
if (S.meta) init();
