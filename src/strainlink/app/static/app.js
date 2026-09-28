/* StrainLink app logic. Talks to the Python engine through window.pywebview.api. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const api = () => window.pywebview.api;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (v, d = 2, unit = "") => (v === null || v === undefined || Number.isNaN(v)) ? "—" : `${Number(v).toFixed(d)}${unit ? " " + unit : ""}`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const ROLES = [["time", "Time"], ["force", "Actuator force"], ["displacement", "Wall push (actuator)"], ["lvdt", "LVDT (horizontal)"],
  ["uplift", "Wall uplift"], ["pressure", "Pressure cell"], ["ignore", "Ignore"]];
const UNITS = { time: ["s", "ms", "min"], force: ["kN", "N", "MN", "lbf"], displacement: ["mm", "m", "cm", "in"], lvdt: ["mm", "m", "cm", "in"],
  uplift: ["mm", "m", "cm", "in"], pressure: ["kPa", "Pa", "MPa", "psi", "bar"], ignore: ["-"] };
const FACTORS = { s: 1, ms: 1e-3, min: 60, kN: 1, N: 1e-3, MN: 1e3, lbf: 4.44822e-3, mm: 1, m: 1e3, cm: 10, in: 25.4, kPa: 1, Pa: 1e-3, MPa: 1e3, psi: 6.89476, bar: 100, "-": 1 };
const BAND_COLOURS = { primary: "--band-primary", radial: "--band-radial", intermediate: "--band-intermediate", "wall-interface": "--band-wall", unattached: "--band-other" };
const BAND_NAMES = { primary: "Band-1 (primary slip surface)", radial: "Radial band (from wall top)", intermediate: "Intermediate band", "wall-interface": "Wall-interface shear zone", unattached: "Other band" };
const STAGES = ["Loading", "Measuring soil movement (DIC)", "Finding shear bands", "Reading sensors", "Synchronising clocks", "Engineering results", "Correlation engine", "Saving results"];

const state = {
  step: 0, photos: null, photosFolder: "", interval: null, sensors: null, sensorPath: "", calPath: "",
  channels: [], cal: { points: [], toe: null, tool: "scale" }, projectPath: "", results: null, frame: 0, playing: null, frameCache: {},
};

function toast(message) {
  const t = $("#toast"); t.textContent = message; t.classList.remove("hidden");
  clearTimeout(toast.timer); toast.timer = setTimeout(() => t.classList.add("hidden"), 3200);
}
function show(view, crumb = "") {
  $$(".view").forEach((v) => v.classList.add("hidden"));
  $(`#view-${view}`).classList.remove("hidden");
  $("#crumbs").textContent = crumb;
  window.scrollTo(0, 0);
}

/* ---------------- home ---------------- */
async function loadHome() {
  const info = await api().info();
  $("#version").textContent = `v${info.version}`;
  const list = $("#recent-list");
  list.innerHTML = info.recent.length ? "" : '<li class="muted">Nothing yet — try the demo.</li>';
  info.recent.forEach((r) => {
    const li = document.createElement("li");
    li.innerHTML = `<button><span>${esc(r.name)}</span><span class="path">${esc(r.path)}</span></button>`;
    li.querySelector("button").onclick = () => openProject(r.path);
    list.appendChild(li);
  });
  show("home");
}
$(".brand").onclick = () => { stopPlay(); loadHome(); };
$("#btn-new").onclick = () => startWizard();
$("#btn-open").onclick = async () => { const p = await api().choose("project"); if (p) openProject(p); };
$$("#demo-choice button").forEach((b) => b.onclick = () => { $$("#demo-choice button").forEach((x) => x.classList.remove("on")); b.classList.add("on"); });
$("#btn-demo").onclick = async () => {
  const preset = $("#demo-choice .on").dataset.preset;
  const started = await api().start_demo(preset);
  if (!started.ok) return toast(started.error);
  runProgress(`Demo: ${preset} sand`, (res) => openResults(res.project));
};

async function openProject(path) {
  const r = await api().load_project(path);
  if (!r.ok) return toast(r.error);
  if (r.has_results) openResults(path);
  else { toast("This project has not been analysed yet."); state.projectPath = path; }
}

/* ---------------- wizard ---------------- */
const STEP_COUNT = 4;
function startWizard() {
  Object.assign(state, { step: 0, photos: null, photosFolder: "", interval: null, sensors: null, sensorPath: "", calPath: "",
    channels: [], cal: { points: [], toe: null, tool: "scale" }, testFolder: "" });
  $("#found").classList.add("hidden"); $("#dropzone").classList.remove("hidden", "compact");
  $("#proj-name").value = ""; $("#scale-direct").value = "";
  show("wizard", "New analysis"); goStep(0);
}
function goStep(n) {
  state.step = n;
  $$(".step").forEach((s) => s.classList.toggle("hidden", +s.dataset.step !== n));
  $$("#steps li").forEach((li) => { const i = +li.dataset.step; li.classList.toggle("on", i === n); li.classList.toggle("done", i < n && stepValid(i)); });
  $("#btn-back").style.visibility = n === 0 ? "hidden" : "visible";
  $("#btn-next").classList.toggle("hidden", n === STEP_COUNT - 1);
  if (n === 1) drawCalibration();
  if (n === 2) fillTestDetails();
  if (n === 3) fillReview();
  updateNav();
}
function photosTimed() {
  return !!state.photos && (!["file-time", "index"].includes(state.photos.time_source) || !!state.interval || syncMode() === "push-step");
}
function stepValid(i) {
  if (i === 0) return photosTimed() && !!state.sensors && ["time", "force", "displacement"].every((r) => state.channels.some((c) => c.role === r));
  if (i === 1) return !!mmPerPx() && !!state.cal.toe && +$("#wall-h").value > 0;
  if (i === 2) return +$("#gamma").value > 0 && +$("#wall-w").value > 0;
  return [0, 1, 2].every(stepValid);
}
function updateNav() {
  const ok = stepValid(state.step);
  $("#btn-next").disabled = !ok;
  let hint = "";
  if (!ok && state.step === 0) hint = !state.photos ? "Choose or drop your test folder." : !photosTimed() ? "Tell StrainLink how often the photos were taken."
    : !state.sensors ? "Choose the data-logger file." : "Mark the time, force and wall-push columns.";
  if (!ok && state.step === 1) hint = !mmPerPx() ? "Click two points a known distance apart (or type the scale)." : "Now click the wall toe: the bottom corner of the wall face.";
  if (!ok && state.step === 2) hint = "Enter the unit weight and the wall width.";
  $("#step-hint").textContent = hint;
}
$$("#steps li").forEach((li) => li.onclick = () => { const i = +li.dataset.step; if (i <= state.step || [...Array(i).keys()].every(stepValid)) goStep(i); });
$("#btn-back").onclick = () => goStep(Math.max(0, state.step - 1));
$("#btn-next").onclick = () => { if (stepValid(state.step)) goStep(state.step + 1); };
document.addEventListener("input", () => { if (!$("#view-wizard").classList.contains("hidden")) updateNav(); });

/* step 0: test files */
$("#pick-folder").onclick = async () => { const f = await api().choose("folder", state.testFolder); if (f) await scanFolder(f); };
async function scanFolder(folder) {
  $("#pick-folder").textContent = "Looking inside…";
  const r = await api().scan_test_folder(folder);
  $("#pick-folder").textContent = "Choose test folder…";
  if (!r.ok) return toast(r.error);
  if (r.project && r.project !== state.projectPath) {
    const loaded = await api().load_project(r.project);
    if (loaded.ok && loaded.has_results && confirm("This folder already has a StrainLink analysis. Open its results?")) return openResults(r.project);
  }
  state.testFolder = r.folder;
  $("#found").classList.remove("hidden"); $("#dropzone").classList.add("compact");
  if (r.photos) await inspectPhotos(r.photos); else markFound("photos", false, "no photos found — choose the photo folder");
  if (r.logger) await loadSensors(r.logger); else markFound("logger", false, "not found — choose the logger export");
  setCalibration(r.calibration);
  updateNav();
}
function markFound(which, ok, text, optional = false) {
  const el = $(`#found-${which}`);
  el.querySelector(".st").className = `st ${ok ? "ok" : optional ? "opt" : "no"}`;
  el.querySelector(".st").textContent = ok ? "✓" : optional ? "–" : "!";
  el.querySelector(".detail").textContent = text;
}
function setCalibration(path) {
  state.calPath = path || "";
  markFound("cal", !!path, path ? path.split("/").pop() : "none found — forces will include base friction", true);
}
$$("[data-change]").forEach((b) => b.onclick = async () => {
  const kind = b.dataset.change;
  if (kind === "photos") { const f = await api().choose("folder", state.testFolder); if (f) await inspectPhotos(f); }
  if (kind === "logger") { const f = await api().choose("sensor", state.testFolder); if (f) await loadSensors(f); }
  if (kind === "cal") { const f = await api().choose("calibration", state.testFolder); if (f) setCalibration(f); }
  updateNav();
});
async function inspectPhotos(folder) {
  const r = await api().inspect_photos(folder, state.interval);
  if (!r.ok) { markFound("photos", false, r.error); return; }
  state.photos = r; state.photosFolder = folder; state.cal = { points: [], toe: null, tool: "scale" };
  const timed = { exif: "timed by the camera", interval: `one every ${state.interval} s`, "file-time": "no camera time stamps", index: "no time information" }[r.time_source];
  markFound("photos", true, `${r.count} photos · ${r.width} × ${r.height} px · ${timed}`);
  $("#thumb-first").src = r.first; $("#thumb-last").src = r.last;
  $("#cap-first").textContent = `First photo (${r.first_name})`; $("#cap-last").textContent = `Last photo (${r.last_name})`;
  $("#thumbs").classList.remove("hidden");
  $("#interval-box").classList.toggle("hidden", !["file-time", "index"].includes(r.time_source) || !!state.interval);
  updateNav();
}
$("#interval").onchange = () => { state.interval = +$("#interval").value || null; if (state.photosFolder) inspectPhotos(state.photosFolder); };
async function loadSensors(path) {
  const r = await api().inspect_sensors(path);
  if (!r.ok) { markFound("logger", false, r.error); return; }
  state.sensors = r; state.sensorPath = path;
  state.channels = r.channels.map((c) => ({ ...c, unit: normaliseUnit(c.unit, c.role) }));
  r.unmapped.forEach((col) => state.channels.push({ column: col, role: "ignore", unit: "-", factor: 1, name: col, confidence: 0 }));
  markFound("logger", true, `${path.split("/").pop()} · ${r.rows} rows · ${r.columns.length} columns`);
  renderMapping(); updateNav();
}
// Drag and drop a folder (or files) anywhere on the window.
document.addEventListener("dragover", (e) => { e.preventDefault(); $("#dropzone").classList.add("over"); });
document.addEventListener("dragleave", () => $("#dropzone").classList.remove("over"));
document.addEventListener("drop", async (e) => {
  e.preventDefault(); $("#dropzone").classList.remove("over");
  const file = e.dataTransfer.files[0]; const path = file && (file.pywebviewFullPath || file.path);
  if (!path) return toast("Could not read the dropped item — please use “Choose test folder…”.");
  if ($("#view-wizard").classList.contains("hidden")) startWizard();
  if (state.step !== 0) goStep(0);
  await scanFolder(path);
});

function normaliseUnit(unit, role) {
  const list = UNITS[role] || ["-"]; const hit = list.find((u) => u.toLowerCase() === String(unit).toLowerCase());
  return hit || list[0];
}
function renderMapping() {
  const body = $("#mapping-body"); body.innerHTML = "";
  state.channels.forEach((c, i) => {
    const col = state.sensors.columns.indexOf(c.column);
    const vals = state.sensors.preview.map((row) => row[col]).filter((v) => v !== null).slice(0, 3).map((v) => +(+v).toPrecision(4)).join(", ");
    const conf = c.confidence >= 0.8 ? "hi" : c.confidence >= 0.5 ? "mid" : "lo";
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${esc(c.column)}</td><td class="vals">${esc(vals)}</td>
      <td><select data-i="${i}" class="role">${ROLES.map(([k, v]) => `<option value="${k}" ${k === c.role ? "selected" : ""}>${v}</option>`).join("")}</select></td>
      <td><input type="text" data-i="${i}" class="cname" value="${esc(c.name)}"></td>
      <td><select data-i="${i}" class="unit">${(UNITS[c.role] || ["-"]).map((u) => `<option ${u === c.unit ? "selected" : ""}>${u}</option>`).join("")}</select></td>
      <td title="How sure StrainLink is about this guess"><span class="conf ${conf}"></span></td>`;
    body.appendChild(tr);
  });
  $$(".role", body).forEach((s) => s.onchange = () => { const c = state.channels[+s.dataset.i]; c.role = s.value; c.unit = normaliseUnit(c.unit, c.role); c.confidence = 1; renderMapping(); updateNav(); });
  $$(".unit", body).forEach((s) => s.onchange = () => { state.channels[+s.dataset.i].unit = s.value; });
  $$(".cname", body).forEach((s) => s.onchange = () => { state.channels[+s.dataset.i].name = s.value.trim() || state.channels[+s.dataset.i].column; });
  const missing = ["time", "force", "displacement"].filter((r) => !state.channels.some((c) => c.role === r));
  const w = $("#mapping-warn");
  w.classList.toggle("hidden", !missing.length);
  w.textContent = missing.length ? `Please mark which column is: ${missing.map((m) => ROLES.find((r) => r[0] === m)[1]).join(", ")} (open “Check the logger columns”).` : "";
  const cells = state.channels.filter((c) => c.role === "pressure").length;
  $("#mapping-summary").innerHTML = missing.length ? "Check the logger columns — <b>something needs your help</b>"
    : `Logger columns recognised: time, force, wall push${cells ? `, ${cells} pressure cells` : ""} <span class="muted">(check or change)</span>`;
  if (missing.length) $("#mapping-box").open = true;
}

/* step 1: scale and wall */
$$(".tool").forEach((b) => b.onclick = () => { $$(".tool").forEach((x) => x.classList.remove("on")); b.classList.add("on"); state.cal.tool = b.dataset.tool; });
const calImage = new Image();
function drawCalibration() {
  if (!state.photos) return;
  const canvas = $("#cal-canvas");
  const draw = () => {
    canvas.width = calImage.naturalWidth; canvas.height = calImage.naturalHeight;
    const g = canvas.getContext("2d"); g.drawImage(calImage, 0, 0);
    const s = state.photos.preview_scale;
    const accent = css("--accent") || "#2f6fed";
    const u = Math.max(canvas.width / 700, 1);          // line/label size relative to the photo
    // Labels on dark rounded tags stay readable on any speckle.
    const tag = (text, x, y, colour) => {
      g.font = `600 ${Math.round(13 * u)}px -apple-system, sans-serif`;
      const w = g.measureText(text).width + 12 * u, h = 20 * u;
      g.fillStyle = "rgba(10,12,18,.82)"; g.beginPath(); g.roundRect(x, y - h / 2, w, h, 6 * u); g.fill();
      g.fillStyle = colour; g.fillText(text, x + 6 * u, y + 4.5 * u);
    };
    const dot = (p, colour) => { g.fillStyle = colour; g.strokeStyle = "#fff"; g.lineWidth = 2 * u; g.beginPath(); g.arc(p[0], p[1], 6 * u, 0, 7); g.fill(); g.stroke(); };
    const mmpp = mmPerPx();
    if (state.cal.toe && mmpp) {
      const [tx, ty] = state.cal.toe; const hpx = (+$("#wall-h").value / mmpp) * s;
      // Dim everything outside the measurement region.
      g.fillStyle = "rgba(0,0,0,.45)";
      g.fillRect(0, 0, canvas.width, Math.max(ty - hpx, 0)); g.fillRect(0, Math.max(ty - hpx, 0), tx, canvas.height);
      g.strokeStyle = "rgba(255,255,255,.9)"; g.lineWidth = 1.5 * u; g.setLineDash([8 * u, 6 * u]);
      g.strokeRect(tx, ty - hpx, canvas.width - tx - 1, canvas.height - (ty - hpx) - 1); g.setLineDash([]);
      tag("measurement region", tx + 10 * u, ty - hpx + 16 * u, "#fff");
      g.strokeStyle = "#e0452b"; g.lineWidth = 4 * u; g.beginPath(); g.moveTo(tx, ty); g.lineTo(tx, ty - hpx); g.stroke();
      tag("wall face", tx + 10 * u, ty - hpx / 2, "#ff8a70");
    }
    if (state.cal.points.length === 2) {
      const [a, b] = state.cal.points; g.strokeStyle = accent; g.lineWidth = 3 * u;
      g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke();
      tag(`${$("#scale-mm").value} mm`, (a[0] + b[0]) / 2 - 20 * u, (a[1] + b[1]) / 2 - 16 * u, "#9db8ff");
    }
    state.cal.points.forEach((p) => dot(p, accent));
    if (state.cal.toe) { dot(state.cal.toe, "#e0452b"); tag("toe", state.cal.toe[0] + 10 * u, state.cal.toe[1] + 14 * u, "#ff8a70"); }
    const facts = [];
    if (mmpp) facts.push(`<span class="fact good">Scale <b>${mmpp.toFixed(4)}</b> mm/px</span>`);
    else facts.push('<span class="fact">Scale: click two points</span>');
    facts.push(state.cal.toe ? `<span class="fact good">Wall toe at <b>${(state.cal.toe[0] / s).toFixed(0)}, ${(state.cal.toe[1] / s).toFixed(0)}</b> px</span>` : '<span class="fact">Wall toe: not set</span>');
    $("#cal-facts").innerHTML = facts.join("");
    updateNav();
  };
  if (calImage.src !== state.photos.first) { calImage.onload = draw; calImage.src = state.photos.first; } else draw();
  drawCalibration.redraw = draw;
}
function mmPerPx() {
  const direct = +$("#scale-direct").value;
  if (direct > 0) return direct;
  if (state.cal.points.length !== 2) return null;
  const [a, b] = state.cal.points; const s = state.photos.preview_scale;
  const px = Math.hypot(a[0] - b[0], a[1] - b[1]) / s;
  const mm = +$("#scale-mm").value;
  return px > 0 && mm > 0 ? mm / px : null;
}
$("#cal-canvas").onclick = (e) => {
  const c = e.currentTarget; const r = c.getBoundingClientRect();
  const p = [(e.clientX - r.left) * c.width / r.width, (e.clientY - r.top) * c.height / r.height];
  if (state.cal.tool === "scale") {
    state.cal.points = state.cal.points.length >= 2 ? [p] : [...state.cal.points, p];
    if (state.cal.points.length === 2) { $$(".tool").forEach((x) => x.classList.toggle("on", x.dataset.tool === "toe")); state.cal.tool = "toe"; }
  } else state.cal.toe = p;
  drawCalibration.redraw && drawCalibration.redraw();
};
["#scale-mm", "#wall-h", "#scale-direct"].forEach((s) => $(s).oninput = () => drawCalibration.redraw && drawCalibration.redraw());

/* step 2: test details */
async function fillTestDetails() {
  const d = await api().defaults();
  const body = $("#cells-body"); const existing = {};
  $$("input", body).forEach((i) => existing[i.dataset.name] = i.value);
  body.innerHTML = "";
  const cells = state.channels.filter((c) => c.role === "pressure");
  if (!cells.length) body.innerHTML = '<tr><td colspan="2" class="muted">No pressure cells in the logger file.</td></tr>';
  cells.forEach((c) => {
    const value = existing[c.name] || (d.cell_depths_mm || {})[c.name] || "";
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${esc(c.name)}</td><td><input type="number" step="any" data-name="${esc(c.name)}" value="${value}" placeholder="mm"></td>`;
    body.appendChild(tr);
  });
  if (!$("#gamma").value && d.unit_weight_kn_m3) $("#gamma").value = d.unit_weight_kn_m3;
  if (d.wall_width_mm) $("#wall-w").value = $("#wall-w").value || d.wall_width_mm;
  if (!$("#wall-weight").value && d.wall_weight_kn) $("#wall-weight").value = d.wall_weight_kn;
  if (!$("#proj-name").value && state.testFolder) $("#proj-name").value = state.testFolder.split("/").filter(Boolean).pop();
  updateNav();
}
function syncMode() { const r = $('input[name="sync"]:checked'); return r ? r.value : "dic"; }

/* step 3: review and run */
function buildProject() {
  const s = state.photos.preview_scale; const mmpp = mmPerPx();
  const cellDepths = {}; $$("#cells-body input").forEach((i) => { if (i.value) cellDepths[i.dataset.name] = +i.value; });
  const project = {
    name: $("#proj-name").value || "Untitled test",
    images: { folder: state.photosFolder, ...(state.interval ? { interval_s: state.interval } : {}) },
    sensors: { file: state.sensorPath, ...(state.calPath ? { calibration_file: state.calPath } : {}),
      channels: state.channels.map((c) => ({ column: c.column, role: c.role, unit: c.unit, factor: FACTORS[c.unit] || 1, name: c.name, confidence: 1 })) },
    calibration: { mm_per_px: mmpp, toe_px: [state.cal.toe[0] / s, state.cal.toe[1] / s], wall_height_mm: +$("#wall-h").value },
    test: { unit_weight_kn_m3: +$("#gamma").value, wall_width_mm: +$("#wall-w").value, cell_depths_mm: cellDepths,
      ...($("#wall-weight").value ? { wall_weight_kn: +$("#wall-weight").value } : {}) },
    sync: { mode: syncMode(), ...(syncMode() === "offset" ? { offset_s: +$("#sync-offset").value } : {}), ...(syncMode() === "push-step" ? { push_step_mm: +$("#sync-step").value } : {}) },
  };
  return project;
}
function fillReview() {
  const items = [["Test", $("#proj-name").value || null], ["Photos", state.photos ? `${state.photos.count} photos` : null], ["Data-logger file", state.sensorPath ? state.sensorPath.split("/").pop() : null],
    ["Calibration run", state.calPath ? state.calPath.split("/").pop() : { warn: "none — forces will include base friction" }], ["Scale", mmPerPx() ? `${mmPerPx().toFixed(4)} mm/px` : null],
    ["Wall toe", state.cal.toe ? "set" : null], ["Unit weight", $("#gamma").value ? `${$("#gamma").value} kN/m³` : null], ["Clock matching", { dic: "automatic", offset: "known offset", "push-step": "photo every fixed push" }[syncMode()]]];
  $("#review-list").innerHTML = items.map(([k, v]) => {
    const cls = !v ? "missing" : v.warn ? "caution" : "ok";
    return `<div class="item"><span>${k}</span><span class="${cls}">${!v ? "missing" : esc(v.warn || v)}</span></div>`;
  }).join("");
  if (!state.savePath) state.savePath = `${state.testFolder || state.photosFolder}/StrainLink analysis/project.json`;
  $("#save-path-label").textContent = state.savePath.replace(/\/project\.json$/, "");
  $("#btn-run").disabled = !stepValid(3);
}
$("#pick-save").onclick = async () => { const p = await api().choose("folder", state.testFolder); if (p) { state.savePath = `${p}/project.json`; fillReview(); } };
$("#btn-run").onclick = async () => {
  const project = buildProject();
  api().save_defaults({ unit_weight_kn_m3: project.test.unit_weight_kn_m3, wall_width_mm: project.test.wall_width_mm,
    wall_weight_kn: project.test.wall_weight_kn, cell_depths_mm: project.test.cell_depths_mm, wall_height_mm: project.calibration.wall_height_mm });
  const saved = await api().save_project(project, state.savePath);
  if (!saved.ok) return toast(saved.error);
  state.projectPath = saved.path;
  const started = await api().run_analysis(saved.path);
  if (!started.ok) return toast(started.error);
  runProgress($("#proj-name").value, () => openResults(saved.path));
};

/* ---------------- running ---------------- */
function runProgress(title, onDone) {
  show("running", title); $("#run-title").textContent = title;
  $("#run-error").classList.add("hidden"); $("#run-back").classList.add("hidden");
  $("#run-stages").innerHTML = STAGES.map((s) => `<li>${s}</li>`).join("");
  const t0 = Date.now();
  const timer = setInterval(async () => {
    const p = await api().progress();
    const idx = STAGES.indexOf(p.stage);
    $$("#run-stages li").forEach((li, i) => { li.classList.toggle("done", idx > i || (!p.running && !p.error)); li.classList.toggle("now", i === idx && p.running); });
    const overall = idx >= 0 ? (idx + p.fraction) / STAGES.length : p.fraction * 0.3;
    $("#run-bar").style.width = `${Math.round(100 * Math.min(overall, 1))}%`;
    $("#run-message").textContent = `${p.stage}${p.message ? " — " + p.message : ""}`;
    $("#run-elapsed").textContent = `${Math.round((Date.now() - t0) / 1000)} s`;
    if (!p.running) {
      clearInterval(timer);
      if (p.error) {
        $("#run-error").textContent = p.error; $("#run-error").classList.remove("hidden");
        $("#run-back").classList.remove("hidden"); $("#run-back").onclick = () => history.length && loadHome();
      } else onDone(p.result);
    }
  }, 500);
}

/* ---------------- results ---------------- */
async function openResults(path) {
  const r = await api().results(path);
  if (!r.ok) return toast(r.error);
  state.projectPath = path; state.results = r; state.frameCache = {}; state.frame = 0;
  const s = r.summary;
  show("results", s.project);
  $("#res-title").textContent = s.project;
  $("#res-meta").textContent = `${s.frames} photos · analysed ${s.finished} · ${Math.round(s.seconds)} s · StrainLink ${s.version}`;
  $("#res-synthetic").classList.toggle("hidden", !s.synthetic);
  renderCards(s); renderFindings(s); renderReview(s); renderAccuracy(path);
  const n = s.frames - 1;
  [$("#scrub-overview"), $("#scrub-soil")].forEach((el) => { el.max = n; el.value = n; });
  state.frame = n;
  selectTab("overview");
}
function lastValue(arr) { for (let i = arr.length - 1; i >= 0; i--) if (arr[i] !== null && arr[i] !== undefined) return arr[i]; return null; }
function renderCards(s) {
  const m = s.mechanics, p = s.bands.primary;
  const flagged = (text) => `<span class="flag" title="${esc(text)}">⚑</span>`;
  const cards = [
    ["Peak passive force", fmt(m.passive.peak, 2, "kN"), `± ${fmt(m.passive.peak_sigma, 3)} kN`],
    ["Push at peak δp", fmt(m.passive.push_at_peak_mm, 1, "mm"), `${m.passive.plateau ? "plateau · " : ""}range ${fmt(m.passive.push_range_mm[0], 1)}–${fmt(m.passive.push_range_mm[1], 1)} mm`, m.passive.plateau ? "The force plateaus, so the peak position is spread over a range." : ""],
    ["Kp", fmt(m.kp, 2), `± ${fmt(m.kp_sigma, 3)}`],
    ["Band-1 appears at", fmt(p.onset_mm, 1, "mm"), "wall push"],
    ["Band-1 reaches surface", fmt(p.surface_reached_mm, 1, "mm"), "wall push"],
    ["Lift-off angle", fmt(lastValue(p.lift_off_deg), 1, "°"), `± ${fmt(lastValue(p.lift_off_sigma_deg), 1)}° · end of test`],
    ["Impacted length", fmt(lastValue(p.length_mm), 0, "mm"), "end of test"],
    ["Wedge area", fmt((lastValue(p.area_mm2) || NaN) / 1e4, 2, "×10⁴ mm²"), "end of test"],
    ["Surface heave", fmt(lastValue(p.heave_mm), 1, "mm"), "end of test"],
    ["Slip surface shape", (lastValue(p.shape) || "—").replace("?", ""), (lastValue(p.shape) || "").endsWith("?") ? "ambiguous — see Review" : "end of test", (lastValue(p.shape) || "").endsWith("?") ? "Shape is ambiguous" : ""],
    ["δmob (wall friction)", fmt(m.delta_mob_deg, 1, "°"), "at uplift onset"],
    ["Cell thrust vs actuator", m.thrust_mismatch_fraction === null ? "—" : fmt(100 * m.thrust_mismatch_fraction, 1, "%"), "difference at peak"],
  ];
  $("#res-cards").innerHTML = cards.map(([k, v, sub, flag]) => `<div class="card">${flag ? flagged(flag) : ""}<div class="k">${k}</div><div class="v">${esc(v)}</div><div class="s">${esc(sub)}</div></div>`).join("");
}
function renderFindings(s) {
  const badge = { strong: "strong", moderate: "moderate", weak: "weak", "not supported": "not" };
  const list = s.correlation.findings;
  $("#findings-list").innerHTML = (list.length ? list : [{ strength: "not supported", statement: "No findings could be computed for this test.", evidence: {}, caveats: [], method: "" }])
    .map((f) => `<div class="finding"><span class="badge ${badge[f.strength]}">${esc(f.strength)}</span>
      <div class="st">${esc(f.statement)}</div>
      ${f.method ? `<details><summary>Evidence and method</summary><p class="muted small">${esc(f.method)}</p>
      <table>${Object.entries(f.evidence).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("")}</table>
      ${f.caveats.map((c) => `<div class="caveat">⚠ ${esc(c)}</div>`).join("")}</details>` : ""}</div>`).join("")
    + (s.correlation.warnings || []).map((w) => `<div class="flagitem">${esc(w)}</div>`).join("");
}
function renderReview(s) {
  $("#review-count").textContent = s.flags.length || "";
  $("#review-count").classList.toggle("hidden", !s.flags.length);
  $("#review-flags").innerHTML = s.flags.length ? s.flags.map((f) => `<div class="flagitem">⚑ <span>${esc(f)}</span></div>`).join("")
    : '<p class="muted">Nothing needs your review: all results are within the validated tolerances.</p>';
}
async function renderAccuracy(path) {
  const r = await api().accuracy(path);
  $("#tab-accuracy").classList.toggle("hidden", !r.ok);
  if (!r.ok) return;
  const all = r.passed === r.total;
  $("#acc-banner").className = `acc-banner ${all ? "good" : "warn"}`;
  $("#acc-banner").innerHTML = `<b>${r.passed} of ${r.total}</b> checks within tolerance${all ? " — every automatic result matches the known answer." : " — see the ✗ rows below."}`;
  const groups = [...new Set(r.checks.map((c) => c.group))];
  $("#acc-table").innerHTML = groups.map((g) => {
    const rows = r.checks.filter((c) => c.group === g).map((c) => `<tr class="${c.ok ? "" : "bad"}">
      <td class="mark">${c.ok ? "✓" : "✗"}</td><td>${esc(c.item)}${c.note ? `<div class="muted small">${esc(c.note)}</div>` : ""}</td>
      <td>${esc(c.found)}</td><td>${esc(c.true)}</td><td>${esc(c.error)}</td><td class="muted">${esc(c.tolerance)}</td></tr>`).join("");
    const ok = r.checks.filter((c) => c.group === g && c.ok).length, n = r.checks.filter((c) => c.group === g).length;
    return `<h3>${esc(g)} <span class="muted small">${ok}/${n}</span></h3><table class="acc"><thead><tr><th></th><th>What</th><th>StrainLink found</th><th>True answer</th><th>Error</th><th>Tolerance</th></tr></thead><tbody>${rows}</tbody></table>`;
  }).join("");
}
$$("#res-tabs button").forEach((b) => b.onclick = () => selectTab(b.dataset.tab));
function selectTab(tab) {
  $$("#res-tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
  $$("#view-results .tab").forEach((t) => t.classList.toggle("hidden", t.dataset.tab !== tab));
  if (tab !== "soil") stopPlay();
  // Draw after the tab is laid out: a chart drawn into a hidden box gets zero width.
  requestAnimationFrame(() => {
    if (tab === "overview") drawOverview();
    if (tab === "soil") { drawBandPlots(); showFrame(state.frame); }
    fitCharts();
  });
}
function fitCharts() {
  $$("#view-results .tab:not(.hidden) .plot").forEach((el) => { if (el.data) Plotly.Plots.resize(el); });
}
window.addEventListener("resize", fitCharts);
function plotLayout(xTitle, yTitle) {
  return { margin: { l: 56, r: 16, t: 26, b: 44 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "-apple-system, sans-serif", size: 12, color: css("--ink") },
    xaxis: { title: xTitle, gridcolor: css("--line"), zeroline: false }, yaxis: { title: yTitle, gridcolor: css("--line"), zeroline: false },
    showlegend: true, legend: { orientation: "h", y: 1.12, font: { size: 11 } }, shapes: [] };
}
const PLOT_CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"] };
function cursorShape(x) { return { type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color: css("--accent"), width: 1.5 } }; }
function shortEvent(e) {
  if (e.key === "band-1.onset") return "band-1 appears";
  if (e.key === "band-1.surface") return "band-1 at surface";
  if (e.key.endsWith(".onset")) return `${e.subject} appears`;
  return e.label;
}
function drawOverview() {
  const s = state.results.summary, pp = state.results.per_photo;
  const push = s.frame_push_mm[state.frame];
  const force = { x: s.mechanics.push_mm, y: s.mechanics.passive_kn, mode: "lines", name: "Passive force", line: { color: css("--ink"), width: 1.6 } };
  const events = s.correlation.events.filter((e) => e.source === "soil" && !e.key.startsWith("band-1.reaches"));
  const lf = plotLayout("Wall push (mm)", "Passive force (kN)");
  lf.shapes = [cursorShape(push), ...events.map((e) => ({ type: "line", x0: e.push_mm, x1: e.push_mm, yref: "paper", y0: 0, y1: 1, line: { color: css("--band-primary"), width: 1, dash: "dot" } }))];
  // Short horizontal tags, staggered in height so neighbouring events never overlap.
  lf.annotations = events.map((e, i) => ({ x: e.push_mm, yref: "paper", y: 0.97 - 0.09 * (i % 4), text: shortEvent(e), showarrow: false,
    xanchor: "left", xshift: 3, font: { size: 10.5, color: css("--band-primary") }, bgcolor: css("--panel"), opacity: 0.95 }));
  Plotly.react("plot-force", [force], lf, PLOT_CONFIG);
  const cells = s.mechanics.cells.map((c) => c.name).filter((n) => pp[n]);
  const traces = cells.map((n) => ({ x: pp.push_mm, y: pp[n], mode: "lines", name: n }));
  const lc = plotLayout("Wall push (mm)", "Earth pressure (kPa)"); lc.shapes = [cursorShape(push)];
  const reach = s.correlation.events.filter((e) => e.key.startsWith("band-1.reaches"));
  lc.shapes.push(...reach.map((e) => ({ type: "line", x0: e.push_mm, x1: e.push_mm, yref: "paper", y0: 0, y1: 0.08, line: { color: css("--band-primary"), width: 2 } })));
  Plotly.react("plot-cells", traces.length ? traces : [{ x: [], y: [] }], lc, PLOT_CONFIG);
  $("#scrub-overview-label").textContent = `${state.frame + 1} / ${s.frames} · push ${fmt(push, 1, "mm")}`;
}
$("#scrub-overview").oninput = (e) => { state.frame = +e.target.value; $("#scrub-soil").value = state.frame; drawOverview(); };

function drawBandPlots() {
  const p = state.results.summary.bands.primary, x = p.delta_mm, push = x[state.frame];
  const mk = (id, y, title, err) => {
    const l = plotLayout("Wall push (mm)", title); l.showlegend = false; l.shapes = [cursorShape(push)]; l.margin.t = 10;
    const trace = { x, y, mode: "lines+markers", marker: { size: 4 }, line: { color: css("--band-primary") } };
    if (err) trace.error_y = { type: "data", array: err.map((v) => (v === null ? 0 : 2 * v)), visible: true, thickness: 1, width: 0, color: css("--muted") };
    Plotly.react(id, [trace], l, PLOT_CONFIG);
  };
  mk("plot-len", p.developed_length_mm, "Band-1 length (mm)");
  mk("plot-lift", p.lift_off_deg, "Lift-off angle (°)", p.lift_off_sigma_deg);
  mk("plot-area", p.area_mm2.map((v) => (v === null ? null : v / 1e4)), "Wedge area (×10⁴ mm²)");
}
async function showFrame(k) {
  state.frame = k; $("#scrub-soil").value = k; $("#scrub-overview").value = k;
  const showStrain = $("#show-strain").checked; const key = `${k}:${showStrain}`;
  $("#frame-loading").classList.remove("hidden");
  const f = state.frameCache[key] || (state.frameCache[key] = await api().frame(state.projectPath, k, showStrain));
  if (state.frame !== k) return;
  $("#frame-loading").classList.add("hidden");
  $("#frame-img").src = f.image;
  const svg = $("#frame-svg"); svg.setAttribute("viewBox", `0 0 ${f.width} ${f.height}`); svg.setAttribute("preserveAspectRatio", "none");
  svg.innerHTML = f.bands.map((b) => `<polyline points="${b.points.map((p) => p.join(",")).join(" ")}" fill="none" stroke="${css(BAND_COLOURS[b.label])}" stroke-width="3" stroke-linecap="round" opacity=".95"><title>${esc(b.name)} — ${esc(BAND_NAMES[b.label])}</title></polyline>
    <text x="${b.points[b.points.length - 1][0] + 6}" y="${b.points[b.points.length - 1][1]}" fill="${css(BAND_COLOURS[b.label])}" font-size="15" font-weight="700" paint-order="stroke" stroke="#000" stroke-width="3">${esc(b.name)}</text>`).join("");
  const s = state.results.summary, p = s.bands.primary;
  $("#frame-label").textContent = `Push ${fmt(f.push_mm, 1, "mm")} · photo ${k + 1}/${s.frames}`;
  const metric = (name, v) => `<div class="m"><span>${name}</span><span>${v}</span></div>`;
  $("#frame-metrics").innerHTML = [
    metric("Bands found", f.bands.length ? f.bands.map((b) => b.name).join(", ") : "none yet"),
    metric("Band-1 length", fmt(p.developed_length_mm[k], 0, "mm")),
    metric("Lift-off angle", p.lift_off_deg[k] === null ? "—" : `${fmt(p.lift_off_deg[k], 1)}° ± ${fmt(p.lift_off_sigma_deg[k], 1)}°`),
    metric("Shape", (p.shape[k] || "—").replace("?", " (ambiguous)")),
    metric("Impacted length", fmt(p.length_mm[k], 0, "mm")),
    metric("Wedge area", fmt(p.area_mm2[k] === null ? null : p.area_mm2[k] / 1e4, 2, "×10⁴ mm²")),
    metric("Surface heave", fmt(p.heave_mm[k], 1, "mm")),
    metric("Apparent band width", fmt(p.width_mm[k], 0, "mm")),
  ].join("");
  const labels = [...new Set(f.bands.map((b) => b.label))];
  $("#legend").innerHTML = (labels.length ? labels : ["primary"]).map((l) => `<div><i style="background:${css(BAND_COLOURS[l])}"></i>${BAND_NAMES[l]}</div>`).join("")
    + (showStrain ? `<div>Strain overlay: red = high shear strain (up to ${fmt(f.gamma_max, 2)})</div>` : "");
  drawBandPlots();
}
$("#scrub-soil").oninput = (e) => showFrame(+e.target.value);
$("#show-strain").onchange = () => showFrame(state.frame);
function stopPlay() { if (state.playing) { clearInterval(state.playing); state.playing = null; $("#play").textContent = "▶"; } }
$("#play").onclick = () => {
  if (state.playing) return stopPlay();
  $("#play").textContent = "❚❚";
  if (state.frame >= state.results.summary.frames - 1) state.frame = -1;
  state.playing = setInterval(() => {
    if (state.frame >= state.results.summary.frames - 1) return stopPlay();
    showFrame(state.frame + 1);
  }, 450);
};
$("#btn-report").onclick = async () => { const r = await api().open_report(state.projectPath); if (!r.ok) toast(r.error); };
$("#btn-finder").onclick = () => api().reveal(state.projectPath);

window.addEventListener("pywebviewready", loadHome);

// Development mode (plain browser via devserver.py): the same API over HTTP.
if (!window.pywebview && location.protocol.startsWith("http")) {
  window.pywebview = { api: new Proxy({}, { get: (_, name) => (...args) =>
    fetch(`/api/${name}`, { method: "POST", body: JSON.stringify(args) }).then((r) => r.json()) }) };
  setTimeout(() => window.dispatchEvent(new Event("pywebviewready")), 0);
}
