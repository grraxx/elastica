"use strict";
// Run console: a form for each program's parameters, the runs in data/runs, and their logs.
(() => {
  const $ = (id) => document.getElementById(id);
  const C = { info: null, program: "solve", runs: [], selected: null, logOffset: 0, logState: null,
              timer: null, logTimer: null };
  const POST_HEADERS = { "Content-Type": "application/json", "X-Elastica-Console": "1" };
  const ACTIVE = new Set(["starting", "running", "stopping"]);

  function el(tag, props = {}, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === "class") e.className = v; else if (k === "text") e.textContent = v;
      else if (k in e) e[k] = v; else e.setAttribute(k, v);
    }
    for (const kid of kids) if (kid != null) e.append(kid);
    return e;
  }
  function banner(text) { const b = $("consoleBanner"); b.textContent = text || ""; b.hidden = !text; }
  async function api(url, body) {
    const r = await fetch(url, body === undefined ? {} : { method: "POST", headers: POST_HEADERS, body: JSON.stringify(body) });
    const data = await r.json().catch(() => ({ error: `${r.status}` }));
    if (!r.ok) throw new Error(data.error || `${r.status}`);
    return data;
  }
  const fmtTime = (t) => (t ? new Date(t * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "–");
  function fmtDuration(s) {
    if (!(s >= 0)) return "–";
    if (s < 90) return `${Math.round(s)} s`;
    if (s < 5400) return `${Math.round(s / 60)} min`;
    return `${(s / 3600).toFixed(1)} h`;
  }
  const params = () => C.info.programs[C.program].groups.flatMap((g) => g.params);
  // Angles: shown as multiples of π where they are one, and typed as numbers or like −π/2, 3π/8, pi/4.
  function fmtAngle(v) {
    for (const den of [1, 2, 3, 4, 6, 8, 12]) {
      const k = (v * den) / Math.PI;
      if (Math.abs(k - Math.round(k)) < 1e-12 && Math.round(k) !== 0) {
        const n = Math.round(k), sign = n < 0 ? "−" : "", a = Math.abs(n);
        return `${sign}${a === 1 ? "" : a}π${den === 1 ? "" : "/" + den}`;
      }
    }
    return String(v);
  }
  function parseAngle(text) {
    const t = text.trim().replace(/−/g, "-").replace(/pi/gi, "π").replace(/\s+/g, "");
    if (!t.includes("π")) return t === "" ? NaN : Number(t);
    const m = t.match(/^([+-]?)(\d*\.?\d*)\*?π(?:\/(\d*\.?\d+))?$/);
    if (!m) return NaN;
    const k = m[2] === "" ? 1 : Number(m[2]), den = m[3] ? Number(m[3]) : 1;
    return (m[1] === "-" ? -1 : 1) * k * Math.PI / den;
  }

  // ---------------------------------------------------------------- the form
  function input(p) {
    const id = `param-${p.key}`, help = p.help ? el("small", { text: p.help }) : null;
    if (p.type === "bool") {
      const box = el("input", { type: "checkbox", id, checked: !!p.default });
      box.dataset.key = p.key;
      return el("label", { class: "param bool", title: p.help || "" }, box, el("span", { text: p.label }));
    }
    let field;
    if (p.type === "range") {
      const make = (v, end) => p.angle
        ? el("input", { type: "text", value: fmtAngle(v), "aria-label": `${p.label} ${end}` })
        : el("input", { type: "number", step: "any", value: v, "aria-label": `${p.label} ${end}` });
      const a = make(p.default[0], "from"), b = make(p.default[1], "to");
      field = el("span", { class: "pair-in", id }, a, el("span", { text: "to" }), b);
    } else if (p.type === "run") {
      field = el("select", { id });
      const runs = C.info.source_runs.length ? C.info.source_runs : [p.default];
      for (const r of runs) field.append(el("option", { value: r, text: r }));
      field.value = runs.includes(p.default) ? p.default : runs[0];
    } else if (p.type === "stages") {
      field = el("span", { class: "checks", id });
      for (const c of p.choices) {
        const box = el("input", { type: "checkbox", value: c, checked: p.default.includes(c) });
        field.append(el("label", { class: "check" }, box, c));
      }
    } else if (p.type === "choice") {
      field = el("select", { id });
      for (const c of p.choices) field.append(el("option", { value: c, text: c }));
      field.value = p.default;
    } else if (p.type === "file") {
      field = el("input", { type: "text", id, value: p.default, pattern: "[A-Za-z0-9_.\\-]*" });
    } else {
      field = el("input", { type: "number", id, value: p.default, step: p.type === "int" ? "1" : "any" });
      if (p.min !== undefined) field.min = p.min;
      if (p.max !== undefined) field.max = p.max;
    }
    const wrap = el("label", { class: "param", title: p.help || "" }, el("span", { text: p.label }), field, help);
    wrap.dataset.key = p.key;
    return wrap;
  }
  function buildForm() {
    const prog = C.info.programs[C.program];
    $("programAbout").textContent = prog.about;
    const form = $("paramForm"); form.replaceChildren();
    for (const g of prog.groups) {
      const fs = el("fieldset", {}, el("legend", { text: g.title }));
      const grid = el("div", { class: "params" });
      for (const p of g.params) grid.append(input(p));
      fs.append(grid); form.append(fs);
    }
    for (const b of $("programTabs").children) b.setAttribute("aria-pressed", b.dataset.program === C.program);
    update();
  }
  function readParam(p) {
    const wrap = $("paramForm").querySelector(`[data-key="${CSS.escape(p.key)}"]`);
    if (p.type === "bool") return (wrap.matches("input") ? wrap : wrap.querySelector("input")).checked;
    if (p.type === "range") return [...wrap.querySelectorAll("input")].map((i) => (p.angle ? parseAngle(i.value) : Number(i.value)));
    if (p.type === "stages") return [...wrap.querySelectorAll("input:checked")].map((i) => i.value);
    const field = wrap.querySelector("input, select");
    if (p.type === "int" || p.type === "float") return field.value === "" ? NaN : Number(field.value);
    return field.value;
  }
  function writeParam(p, v) {
    const wrap = $("paramForm").querySelector(`[data-key="${CSS.escape(p.key)}"]`);
    if (!wrap || v === undefined) return;
    if (p.type === "bool") (wrap.matches("input") ? wrap : wrap.querySelector("input")).checked = !!v;
    else if (p.type === "range") wrap.querySelectorAll("input").forEach((i, n) => { i.value = p.angle ? fmtAngle(v[n]) : v[n]; });
    else if (p.type === "stages") wrap.querySelectorAll("input").forEach((i) => { i.checked = v.includes(i.value); });
    else wrap.querySelector("input, select").value = v;
  }
  function values() {
    const out = {};
    for (const p of params()) out[p.key] = readParam(p);
    return out;
  }
  function update() {                      // mark edited fields, estimate the run
    const v = values();
    for (const i of $("paramForm").querySelectorAll(".pair-in input[type=text]")) i.classList.toggle("bad", !Number.isFinite(parseAngle(i.value)));
    for (const p of params()) {
      const wrap = $("paramForm").querySelector(`.param[data-key="${CSS.escape(p.key)}"]`) ||
                   $("paramForm").querySelector(`[data-key="${CSS.escape(p.key)}"]`).closest(".param");
      wrap.classList.toggle("changed", JSON.stringify(v[p.key]) !== JSON.stringify(p.default));
    }
    let est = "";
    if (C.program === "solve") {
      let nodes = 1;
      for (const a of ["d", "phi1", "phi2"]) {
        const [lo, hi] = v[`${a}_bounds`], step = v[`grid_steps.${a}`];
        nodes *= Math.round((hi - lo) / step) + 1;
      }
      if (Number.isFinite(nodes) && nodes > 0) {
        const solves = v.solve_phi_symmetry ? nodes / 2 : nodes;
        const sec = solves * 0.011 * (v.beam_elements / 200);
        est = `${nodes.toLocaleString("en-US")} grid nodes, ~${Math.round(solves).toLocaleString("en-US")} solves: roughly ${fmtDuration(sec)} on one core (production averaged 11 ms per solve)`;
      }
    } else if (C.program === "refine") {
      est = v.plan_only ? "plan only: branches, then counts and a time estimate" : `stages: ${v.stages.join(" → ") || "none"}; ${v.workers} workers`;
    }
    $("runEstimate").textContent = est;
    $("runCommand").textContent = `→ data/runs/<date>_<time>_${C.program}${$("runLabel").value.trim() ? "_" + $("runLabel").value.trim().replace(/[^A-Za-z0-9_-]+/g, "-").slice(0, 40) : ""}/`;
  }

  async function run() {
    banner("");
    const btn = $("runBtn"); btn.disabled = true;
    try {
      const { id } = await api("/api/runs", { program: C.program, label: $("runLabel").value, params: values() });
      C.selected = id; C.logOffset = 0; $("runLog").textContent = "";
      await refresh();
    } catch (err) {
      banner(`Not started: ${err.message}`);
    } finally {
      btn.disabled = !C.info.runs_enabled;
    }
  }

  // ---------------------------------------------------------------- runs and logs
  function renderRuns() {
    const body = $("runsTable").querySelector("tbody"); body.replaceChildren();
    if (!C.runs.length) {
      body.append(el("tr", {}, el("td", { colSpan: 6, class: "muted", text: "No runs yet." })));
      return;
    }
    const now = Date.now() / 1000;
    for (const r of C.runs) {
      const end = ACTIVE.has(r.state) ? now : (r.finished || r.updated);
      const acts = el("td", { class: "acts" });
      const button = (text, fn, title) => {
        const b = el("button", { type: "button", text, title });
        b.addEventListener("click", (e) => { e.stopPropagation(); fn(); });
        acts.append(b);
      };
      if (ACTIVE.has(r.state) && C.info.runs_enabled) button(r.state === "stopping" ? "Kill" : "Stop", () => act("stop", r.id),
        r.state === "stopping" ? "Stop again: kill the step at once" : "Interrupt: the solver saves a checkpoint");
      if (["stopped", "failed", "lost"].includes(r.state) && C.info.runs_enabled) button("Resume", () => act("resume", r.id), "Continue in the same folder from the step that did not finish");
      button("Load", () => load(r.id), "Copy this run's parameters into the form");
      const name = el("td", { title: r.path }, r.label || r.id.replace(/_[a-z]+(_.*)?$/, ""));
      const tr = el("tr", { class: r.id === C.selected ? "selected" : "" }, name,
        el("td", { text: C.info.programs[r.program] ? C.info.programs[r.program].label : r.program }),
        el("td", {}, el("span", { class: `state ${r.state}${r.warning ? " warn" : ""}`, title: r.warning || "",
                                  text: r.state + (ACTIVE.has(r.state) && r.steps > 1 ? ` ${(r.step || 0) + 1}/${r.steps}` : "") + (r.warning ? " ⚠" : "") })),
        el("td", { text: fmtTime(r.started || r.created) }),
        el("td", { text: fmtDuration(end - (r.started || r.created)) }), acts);
      tr.addEventListener("click", () => select(r.id));
      body.append(tr);
    }
  }
  async function act(what, id) {
    banner("");
    try { await api(`/api/runs/${what}`, { id }); } catch (err) { banner(err.message); }
    setTimeout(refresh, 400);
  }
  async function load(id) {
    try {
      const job = await api(`/api/runs/job?id=${encodeURIComponent(id)}`);
      if (C.info.programs[job.program]) { C.program = job.program; buildForm(); }
      for (const p of params()) writeParam(p, job.params[p.key]);
      $("runLabel").value = job.label ? `${job.label} (again)` : "";
      update();
      banner("");
      $("programTabs").scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) { banner(err.message); }
  }
  function select(id) {
    if (C.selected !== id) { C.selected = id; C.logOffset = 0; $("runLog").textContent = ""; }
    renderRuns(); pollLog();
  }
  function hint(r) {
    if (!r) return "";
    if (r.program === "solve" && r.outputs.includes("solution_space_results.npz"))
      return `Explore it: python postprocessing/web/server.py ${r.path} --port 8766`;
    if (r.program === "refine" && r.outputs.includes("nn_dataset.npz"))
      return `Density maps from it: python postprocessing/web/server.py --dataset ${r.path}/nn_dataset.npz`;
    return `Folder: ${r.path}`;
  }
  async function pollLog() {
    clearTimeout(C.logTimer);
    if (!C.selected) return;
    const id = C.selected;
    try {
      const d = await api(`/api/runs/log?id=${encodeURIComponent(id)}&offset=${C.logOffset}`);
      if (id !== C.selected) return;
      const pre = $("runLog");
      if (d.offset < C.logOffset || (C.logOffset === 0 && d.offset - d.text.length > 0)) pre.textContent = d.offset - d.text.length > 0 ? "… (earlier lines skipped)\n" : "";
      pre.textContent += d.text.replace(/\r(?!\n)/g, "\n");
      C.logOffset = d.offset;
      if ($("logFollow").checked) pre.scrollTop = pre.scrollHeight;
      const r = C.runs.find((x) => x.id === id);
      $("logTitle").textContent = r && r.label ? r.label : id;
      $("logInfo").textContent = `${d.state}${d.warning ? ` (${d.warning})` : ""}${d.steps > 1 ? ` · step ${Math.min((d.step || 0) + 1, d.steps)} of ${d.steps}` : ""} · ${(d.size / 1024).toFixed(0)} kB`;
      $("logHint").textContent = hint(r);
      if (ACTIVE.has(d.state)) C.logTimer = setTimeout(pollLog, 1500);
    } catch (err) {
      $("logInfo").textContent = err.message;
    }
  }
  async function refresh() {
    clearTimeout(C.timer);
    try {
      C.runs = await api("/api/runs");
      renderRuns();
      if (C.selected) pollLog();
    } catch (err) { /* the server may be restarting */ }
    C.timer = setTimeout(refresh, C.runs.some((r) => ACTIVE.has(r.state)) ? 3000 : 15000);
  }

  async function init() {
    try { C.info = await api("/api/console"); } catch (err) { banner(`The run console is unavailable: ${err.message}`); return; }
    for (const [key, prog] of Object.entries(C.info.programs)) {
      const b = el("button", { type: "button", text: prog.label, title: prog.script });
      b.dataset.program = key;
      b.addEventListener("click", () => { C.program = key; buildForm(); });
      $("programTabs").append(b);
    }
    $("paramForm").addEventListener("input", update);
    $("paramForm").addEventListener("change", update);
    $("runLabel").addEventListener("input", update);
    $("paramForm").addEventListener("submit", (e) => e.preventDefault());
    $("runBtn").addEventListener("click", run);
    $("resetBtn").addEventListener("click", buildForm);
    $("logFollow").addEventListener("change", () => { if ($("logFollow").checked) $("runLog").scrollTop = $("runLog").scrollHeight; });
    $("runBtn").disabled = !C.info.runs_enabled;
    if (!C.info.runs_enabled) banner("Runs are disabled because the server is not on localhost; start it with --allow-remote-runs to enable them. Existing runs and logs are still shown.");
    buildForm();
    await refresh();
    const active = C.runs.find((r) => ACTIVE.has(r.state));
    if (active || C.runs.length) select((active || C.runs[0]).id);
  }
  init();
})();
