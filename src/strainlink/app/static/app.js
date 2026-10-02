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
const STAGES_GENERAL = ["Loading", "Measuring movement (DIC)", "Strain and localisation zones", "Reading sensors", "Matching clocks", "Correlation engine", "Saving results"];
// Any-test mode: zones are coloured by what they do (crack opens, shear band slides).
const MODE_COLOURS = { opening: "#e8590c", shear: "#7048e8", mixed: "#1098ad", "": "#868e96" };
const MODE_NAMES = { opening: "Crack (the sides open)", shear: "Shear band (the sides slide)", mixed: "Mixed-mode zone", "": "Zone" };
const MODE_SHORT = { opening: "crack", shear: "shear band", mixed: "mixed", "": "zone" };
function isGeneral() { return state.kind === "general"; }

const state = {
  kind: "wall", step: 0, photos: null, photosFolder: "", video: null, interval: null, sensors: null, sensorPath: "", calPath: "",
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
  // A way back from every screen; hidden on the start page and while an analysis runs.
  $("#nav-back").classList.toggle("hidden", view === "home" || view === "running");
  state.view = view;
  window.scrollTo(0, 0);
}
function goHome() { stopPlay(); document.body.classList.toggle("kind-general", false); loadHome(); }
$("#nav-back").onclick = goHome;
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && (state.view === "results" || state.view === "wizard") && !e.target.closest("input, select, textarea")) goHome();
});

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
$(".brand").onclick = goHome;
$("#btn-new").onclick = () => startWizard();
$("#btn-open").onclick = async () => { const p = await api().choose("project"); if (p) openProject(p); };
$$("#demo-choice button").forEach((b) => b.onclick = () => { $$("#demo-choice button").forEach((x) => x.classList.remove("on")); b.classList.add("on"); });
$("#btn-demo").onclick = async () => {
  const preset = $("#demo-choice .on").dataset.preset;
  const started = await api().start_demo(preset);
  if (!started.ok) return toast(started.error);
  const crack = preset === "crack";
  runProgress(crack ? "Demo: a plate that cracks" : `Demo: ${preset} sand`, (res) => openResults(res.project), crack ? ["Creating the demo test (synthetic photos and logs)", ...STAGES_GENERAL] : STAGES);
};

async function openProject(path) {
  const r = await api().load_project(path);
  if (!r.ok) return toast(r.error);
  if (r.has_results) openResults(path);
  else { toast("This project has not been analysed yet."); state.projectPath = path; }
}

/* ---------------- wizard ---------------- */
// What is being tested decides which steps and results apply (one switch for the whole page).
function setKind(kind) {
  state.kind = kind;
  document.body.classList.toggle("kind-general", kind === "general");
  $$("#kind-choice button").forEach((b) => b.classList.toggle("on", b.dataset.kind === kind));
  $('#steps li[data-step="1"]').textContent = kind === "general" ? "Scale" : "Scale & wall";
  $('#steps li[data-step="2"]').textContent = kind === "general" ? "Name & timing" : "Test details";
  if (state.sensors) renderMapping();
  if (state.photosFolder && kind === "general") loadArea();
  updateNav();
}
$$("#kind-choice button").forEach((b) => b.onclick = () => setKind(b.dataset.kind));
const STEP_COUNT = 4;
function startWizard() {
  Object.assign(state, { step: 0, photos: null, photosFolder: "", video: null, interval: null, sensors: null, sensorPath: "", calPath: "",
    channels: [], cal: { points: [], toe: null, tool: "scale" }, testFolder: "" });
  $("#found").classList.add("hidden"); $("#dropzone").classList.remove("hidden", "compact");
  $("#proj-name").value = ""; $("#scale-direct").value = "";
  show("wizard", "New analysis"); goStep(0); setKind(state.kind || "wall");
}
function goStep(n) {
  state.step = n;
  $$(".step").forEach((s) => s.classList.toggle("hidden", +s.dataset.step !== n));
  $$("#steps li").forEach((li) => { const i = +li.dataset.step; li.classList.toggle("on", i === n); li.classList.toggle("done", i < n && stepValid(i)); });
  $("#btn-back").style.visibility = n === 0 ? "hidden" : "visible";
  $("#btn-next").classList.toggle("hidden", n === STEP_COUNT - 1);
  if (n === 1) { drawCalibration(); if (isGeneral() && state.areaFor !== state.photosFolder) { state.areaFor = state.photosFolder; loadArea(); } }
  if (n === 2) fillTestDetails();
  if (n === 3) fillReview();
  updateNav();
}
function photosTimed() {
  return !!state.photos && (!["file-time", "index"].includes(state.photos.time_source) || !!state.interval || syncMode() === "push-step");
}
function stepValid(i) {
  if (isGeneral()) {
    // Any test: photos are enough; a logger (optional) needs a time column and timed photos.
    if (i === 0) return !!state.photos && (!state.sensors || (photosTimed() && state.channels.some((c) => c.role === "time")));
    if (i === 1) return !!mmPerPx();
    if (i === 2) return true;
    return [0, 1, 2].every(stepValid);
  }
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
    : isGeneral() ? "Mark the time column of the logger file." : !state.sensors ? "Choose the data-logger file." : "Mark the time, force and wall-push columns.";
  if (!ok && state.step === 1) hint = !mmPerPx() ? "Click two points a known distance apart (or type the scale)." : "Now click the wall toe: the bottom corner of the wall face.";
  if (ok && state.step === 0 && isGeneral() && !state.sensors) hint = "No logger file: StrainLink will measure the object only (add one to correlate with sensors).";
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
  if (r.video) await useVideo(r.video);
  else if (r.photos) await inspectPhotos(r.photos);
  else markFound("photos", false, "no photos or video found — choose the photo folder or the video");
  $("#mapping-box").classList.toggle("hidden", !r.logger);
  if (r.logger) await loadSensors(r.logger);
  else if (isGeneral()) markFound("logger", false, "none — optional: add one to correlate the object with your sensor records", true);
  else markFound("logger", false, "not found — choose the logger export");
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
  if (kind === "photos") { const f = await api().choose("folder", state.testFolder); if (f) { state.video = null; $("#video-box").classList.add("hidden"); await inspectPhotos(f); } }
  if (kind === "video") { const f = await api().choose("video", state.testFolder); if (f) await useVideo(f); }
  if (kind === "logger") { const f = await api().choose("sensor", state.testFolder); if (f) await loadSensors(f); }
  if (kind === "cal") { const f = await api().choose("calibration", state.testFolder); if (f) setCalibration(f); }
  updateNav();
});
// A video is turned into a folder of timed frames once; after that it is handled like photos.
async function useVideo(path, everyS = null) {
  markFound("photos", true, `Reading the video ${path.split("/").pop()}…`);
  const r = await api().extract_video(path, everyS);
  if (!r.ok) { markFound("photos", false, r.error); return; }
  state.video = { path, every_s: r.every_s, name: r.name };
  await inspectPhotos(r.folder);
  markFound("photos", true, `${r.count} frames from ${r.name} (${fmt(r.duration_s / 60, 1)} min at ${fmt(r.fps, 0)} fps) · ${state.photos.width} × ${state.photos.height} px`);
  $("#video-every").value = +r.every_s.toPrecision(3);
  $("#video-note").textContent = `automatic: every ${+r.default_every_s.toPrecision(3)} s`;
  $("#video-box").classList.remove("hidden");
  $("#cap-first").textContent = "First frame"; $("#cap-last").textContent = "Last frame";
}
$("#video-every").onchange = () => { const v = +$("#video-every").value; if (state.video && v > 0) useVideo(state.video.path, v); };
async function inspectPhotos(folder) {
  const r = await api().inspect_photos(folder, state.interval);
  if (!r.ok) { markFound("photos", false, r.error); return; }
  state.photos = r; state.photosFolder = folder; state.cal = { points: [], toe: null, tool: "scale" };
  const timed = { video: "timed by the video", exif: "timed by the camera", interval: `one every ${state.interval} s`, "file-time": "no camera time stamps", index: "no time information" }[r.time_source];
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
  state.sensors = r; state.sensorPath = path; $("#mapping-box").classList.remove("hidden");
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
  const missing = (isGeneral() ? ["time"] : ["time", "force", "displacement"]).filter((r) => !state.channels.some((c) => c.role === r));
  const w = $("#mapping-warn");
  w.classList.toggle("hidden", !missing.length);
  w.textContent = missing.length ? `Please mark which column is: ${missing.map((m) => ROLES.find((r) => r[0] === m)[1]).join(", ")} (open “Check the logger columns”).` : "";
  const cells = state.channels.filter((c) => c.role === "pressure").length;
  $("#mapping-summary").innerHTML = missing.length ? "Check the logger columns — <b>something needs your help</b>"
    : isGeneral() ? `Logger columns: time + ${state.channels.filter((c) => c.role !== "time").length} other(s), all used for correlation <span class="muted">(check or change)</span>`
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
    if (state.cal.box) {
      const [x0, y0, x1, y1] = state.cal.box;
      g.strokeStyle = "#fff"; g.lineWidth = 2 * u; g.setLineDash([8 * u, 6 * u]);
      g.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0)); g.setLineDash([]);
      tag("measured area limited to this box", Math.min(x0, x1) + 8 * u, Math.min(y0, y1) + 16 * u, "#fff");
    }
    state.cal.points.forEach((p) => dot(p, accent));
    if (state.cal.toe) { dot(state.cal.toe, "#e0452b"); tag("toe", state.cal.toe[0] + 10 * u, state.cal.toe[1] + 14 * u, "#ff8a70"); }
    const facts = [];
    if (mmpp) facts.push(`<span class="fact good">Scale <b>${mmpp.toFixed(4)}</b> mm/px</span>`);
    else facts.push('<span class="fact">Scale: click two points</span>');
    if (!isGeneral()) facts.push(state.cal.toe ? `<span class="fact good">Wall toe at <b>${(state.cal.toe[0] / s).toFixed(0)}, ${(state.cal.toe[1] / s).toFixed(0)}</b> px</span>` : '<span class="fact">Wall toe: not set</span>');
    $("#cal-facts").innerHTML = facts.join("");
    updateNav();
  };
  if (calImage.src !== state.photos.first) { calImage.onload = draw; calImage.src = state.photos.first; } else draw();
  drawCalibration.redraw = draw;
}
// Any test: show the area StrainLink will measure (found automatically on the first photo).
async function loadArea() {
  $("#area-note").textContent = "Finding the object…"; $("#area-img").removeAttribute("src");
  const r = await api().preview_area(state.photosFolder, roiPx());
  if (!r.ok) { $("#area-note").textContent = r.error; return; }
  $("#area-img").src = r.image;
  $("#area-note").textContent = `${Math.round(100 * r.share)}% of the photo is measured. If the outline misses part of the object, check its speckle pattern and focus.`;
}
// The optional box that limits the measured area, in photo pixels (the canvas shows a scaled preview).
function roiPx() {
  const b = state.cal.box; if (!b) return null;
  const s = state.photos.preview_scale;
  return [Math.min(b[0], b[2]) / s, Math.min(b[1], b[3]) / s, Math.max(b[0], b[2]) / s, Math.max(b[1], b[3]) / s].map(Math.round);
}
function canvasPoint(e) {
  const c = $("#cal-canvas"); const r = c.getBoundingClientRect();
  return [(e.clientX - r.left) * c.width / r.width, (e.clientY - r.top) * c.height / r.height];
}
$("#cal-canvas").onmousedown = (e) => {
  if (state.cal.tool !== "area") return;
  const p = canvasPoint(e); state.cal.dragging = [...p, ...p];
};
$("#cal-canvas").onmousemove = (e) => {
  if (!state.cal.dragging) return;
  const p = canvasPoint(e); state.cal.dragging[2] = p[0]; state.cal.dragging[3] = p[1];
  state.cal.box = state.cal.dragging; drawCalibration.redraw && drawCalibration.redraw();
};
document.addEventListener("mouseup", () => {
  if (!state.cal.dragging) return;
  const b = state.cal.dragging; state.cal.dragging = null;
  state.cal.box = Math.abs(b[2] - b[0]) > 10 && Math.abs(b[3] - b[1]) > 10 ? b : null;
  $("#clear-area").classList.toggle("hidden", !state.cal.box);
  drawCalibration.redraw && drawCalibration.redraw(); loadArea();
});
$("#use-markers").onchange = () => $("#marker-options").classList.toggle("hidden", !$("#use-markers").checked);
$("#clear-area").onclick = () => { state.cal.box = null; $("#clear-area").classList.add("hidden"); drawCalibration.redraw && drawCalibration.redraw(); loadArea(); };
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
  if (state.cal.tool === "area") return;       // the box is drawn by dragging
  const c = e.currentTarget; const r = c.getBoundingClientRect();
  const p = [(e.clientX - r.left) * c.width / r.width, (e.clientY - r.top) * c.height / r.height];
  if (state.cal.tool === "scale") {
    state.cal.points = state.cal.points.length >= 2 ? [p] : [...state.cal.points, p];
    if (state.cal.points.length === 2 && !isGeneral()) { $$(".tool").forEach((x) => x.classList.toggle("on", x.dataset.tool === "toe")); state.cal.tool = "toe"; }
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
  if (isGeneral()) {
    return {
      kind: "general", name: $("#proj-name").value || "Untitled test",
      images: { folder: state.photosFolder, ...(state.interval ? { interval_s: state.interval } : {}),
                ...(state.video ? { video: state.video.path, video_every_s: state.video.every_s } : {}) },
      calibration: { mm_per_px: mmpp, ...(roiPx() ? { roi_px: roiPx() } : {}) },
      ...($("#use-markers").checked ? { markers: { polarity: $("#marker-polarity").value === "bright" ? "bright" : "dark",
        ...($("#marker-polarity").value === "dark-colour" ? { channel: "brightness" } : {}),
        diameter_px: [0.7 * +$("#marker-size").value, 1.6 * +$("#marker-size").value] } } : {}),
      ...(state.sensorPath ? { sensors: { file: state.sensorPath,
        channels: state.channels.map((c) => ({ column: c.column, role: c.role === "ignore" ? "ignore" : c.role, unit: c.unit, factor: FACTORS[c.unit] || 1, name: c.name, confidence: 1 })) } } : {}),
      sync: syncMode() === "offset" ? { mode: "offset", offset_s: +$("#sync-offset").value } : { mode: "auto" },
    };
  }
  const cellDepths = {}; $$("#cells-body input").forEach((i) => { if (i.value) cellDepths[i.dataset.name] = +i.value; });
  const project = {
    name: $("#proj-name").value || "Untitled test",
    images: { folder: state.photosFolder, ...(state.interval ? { interval_s: state.interval } : {}),
              ...(state.video ? { video: state.video.path, video_every_s: state.video.every_s } : {}) },
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
  const items = isGeneral() ? [["Test", $("#proj-name").value || "Untitled test"], ["Kind", "Any object or test"],
    ["Photos", state.photos ? `${state.photos.count} ${state.video ? "frames from the video" : "photos"}` : null],
    ["Data-logger file", state.sensorPath ? state.sensorPath.split("/").pop() : { warn: "none — the object is measured, but there is nothing to correlate with" }],
    ["Scale", mmPerPx() ? `${mmPerPx().toFixed(4)} mm/px` : null], ["Measured area", roiPx() ? "found automatically, limited to your box" : "found automatically"],
    ["Accuracy check", $("#use-markers").checked ? `markers (${$("#marker-polarity").value}, ~${$("#marker-size").value} px)` : { warn: "no markers — accuracy on this test can't be measured directly" }],
    ["Clock matching", state.sensorPath ? (syncMode() === "offset" ? "known offset" : "automatic") : "not needed"]] : [["Test", $("#proj-name").value || null], ["Photos", state.photos ? `${state.photos.count} photos` : null], ["Data-logger file", state.sensorPath ? state.sensorPath.split("/").pop() : null],
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
  // Wall tests: remember the soil and wall details for the next test (any-object tests have none).
  if (project.test) api().save_defaults({ unit_weight_kn_m3: project.test.unit_weight_kn_m3, wall_width_mm: project.test.wall_width_mm,
    wall_weight_kn: project.test.wall_weight_kn, cell_depths_mm: project.test.cell_depths_mm, wall_height_mm: project.calibration.wall_height_mm });
  const saved = await api().save_project(project, state.savePath);
  if (!saved.ok) return toast(saved.error);
  state.projectPath = saved.path;
  const started = await api().run_analysis(saved.path);
  if (!started.ok) return toast(started.error);
  runProgress($("#proj-name").value, () => openResults(saved.path), isGeneral() ? STAGES_GENERAL : STAGES);
};

/* ---------------- running ---------------- */
function runProgress(title, onDone, stages = STAGES) {
  show("running", title); $("#run-title").textContent = title;
  $("#run-error").classList.add("hidden"); $("#run-back").classList.add("hidden");
  $("#run-stages").innerHTML = stages.map((s) => `<li>${s}</li>`).join("");
  const t0 = Date.now();
  const timer = setInterval(async () => {
    const p = await api().progress();
    const idx = stages.indexOf(p.stage);
    $$("#run-stages li").forEach((li, i) => { li.classList.toggle("done", idx > i || (!p.running && !p.error)); li.classList.toggle("now", i === idx && p.running); });
    const overall = idx >= 0 ? (idx + p.fraction) / stages.length : p.fraction * 0.3;
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

// Wall push at which band-1 broke through, with its range when the breakthrough isn't confirmed.
function surfaceRange(p) {
  const r = p.surface_reached_range_mm;
  return r && r[1] - r[0] > 1.01 ? `range ${fmt(r[0], 1)}–${fmt(r[1], 1)} mm (not yet confirmed)` : "wall push";
}
/* ---------------- results ---------------- */
async function openResults(path) {
  const r = await api().results(path);
  if (!r.ok) return toast(r.error);
  stopPlay(); state.projectPath = path; state.results = r; state.frameCache = {}; state.frameQueue = null; state.frame = 0;
  const s = r.summary;
  show("results", s.project);
  $("#res-title").textContent = s.project;
  $("#res-meta").textContent = `${s.frames} photos · analysed ${s.finished} · ${Math.round(s.seconds)} s · StrainLink ${s.version}`;
  $("#res-synthetic").classList.toggle("hidden", !s.synthetic);
  state.kind = s.kind === "general" ? "general" : "wall";
  document.body.classList.toggle("kind-general", isGeneral());
  if (isGeneral()) { renderGeneralCards(s); fillPickers(s, r.per_photo); } else renderCards(s);
  renderFindings(s); renderReview(s); renderAccuracy(path);
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
    ["Band-1 reaches surface", fmt(p.surface_reached_mm, 1, "mm"), surfaceRange(p)],
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
  const badge = { strong: "strong", moderate: "moderate", weak: "weak", "not supported": "not", info: "info" };
  const list = s.correlation.findings;
  // No findings is information, not a failure: say why, in a neutral card.
  $("#findings-list").innerHTML = (list.length ? list : [{ strength: "info", statement: "Nothing to link yet: there are no zones or sensor events in this test to compare.", evidence: {}, caveats: [], method: "" }])
    .map((f) => `<div class="finding"><span class="badge ${badge[f.strength]}">${esc(f.strength)}</span>
      <div class="st">${esc(f.statement)}</div>
      ${f.method ? `<details><summary>Evidence and method</summary><p class="muted small">${esc(f.method)}</p>
      <table>${Object.entries(f.evidence).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("")}</table>
      ${f.caveats.map((c) => `<div class="caveat">⚠ ${esc(c)}</div>`).join("")}</details>` : ""}</div>`).join("")
    + (s.correlation.warnings || []).map((w) => `<div class="noteitem">ℹ︎ ${esc(w)}</div>`).join("");
}
function renderReview(s) {
  $("#review-count").textContent = s.flags.length || "";
  $("#review-count").classList.toggle("hidden", !s.flags.length);
  $("#review-flags").innerHTML = s.flags.length ? s.flags.map((f) => `<div class="flagitem">⚑ <span>${esc(f)}</span></div>`).join("")
    : '<p class="muted">Nothing needs your review: all results are within the validated tolerances.</p>';
}
function renderMarkerCheck(s) {
  const m = s.markers, box = $("#marker-check");
  box.classList.toggle("hidden", !m);
  if (!m) return;
  const note = (i) => m.in_zones.includes(i) ? "inside a crack/band (reported separately)" : (m.disagreeing || []).includes(i) ? "⚑ disagrees strongly: probably mis-tracked (jumped to a nearby dark spot) — check it" : "";
  const rows = (m.per_marker_rms_px || []).map((v, i) => `<tr class="${(m.disagreeing || []).includes(i) ? "flagged" : ""}"><td>M${i + 1}</td><td>${fmt(m.diameters_px[i], 1)} px</td><td>${v === null ? "not compared" : fmt(v, 3) + " px"}</td><td class="muted">${note(i)}</td></tr>`).join("");
  box.innerHTML = `<h3>Check against the markers on this test</h3>
    <div class="acc-banner ${m.used && m.median_px <= 0.2 ? "good" : "warn"}">${m.used ? `DIC and the markers agree to <b>${fmt(m.median_px, 3)} px</b> (median, ${fmt(m.median_mm * 1000, 1)} µm) over ${m.used} markers; RMS over all ${fmt(m.rms_px, 3)} px${m.disagreeing.length ? `; ${m.disagreeing.length} marker(s) disagree strongly (listed below)` : ""}.` : "No markers could be compared."}</div>
    <p class="muted small">Each marker is located from its own dark (bright) spot, independently of DIC. The markers themselves are accurate to about 0.06 px, so agreement at that level means DIC is at least that accurate on this test.</p>
    <table class="acc"><thead><tr><th>Marker</th><th>Size</th><th>DIC − marker (RMS)</th><th></th></tr></thead><tbody>${rows}</tbody></table>`;
}
async function renderAccuracy(path) {
  const s = state.results.summary;
  renderMarkerCheck(s);
  const r = await api().accuracy(path);
  $("#tab-accuracy").classList.toggle("hidden", !r.ok && !s.markers);
  ["#acc-banner", "#acc-intro", "#acc-table"].forEach((q) => $(q).classList.toggle("hidden", !r.ok));
  if (!r.ok) return;
  const all = r.passed === r.total;
  $("#acc-banner").className = `acc-banner ${all ? "good" : r.silent ? "warn" : "flag"}`;
  const parts = [];
  if (r.flagged) parts.push(`${r.flagged} outside but flagged for your review (⚑)`);
  if (r.silent) parts.push(`${r.silent} silent error${r.silent > 1 ? "s" : ""} (✗: outside tolerance and not flagged)`);
  $("#acc-banner").innerHTML = `<b>${r.passed} of ${r.total}</b> checks within tolerance`
    + (all ? " — every automatic result matches the known answer." : ` · ${parts.join(" · ")}`);
  const groups = [...new Set(r.checks.map((c) => c.group))];
  $("#acc-table").innerHTML = groups.map((g) => {
    const rows = r.checks.filter((c) => c.group === g).map((c) => `<tr class="${c.ok ? "" : c.flagged ? "flagged" : "bad"}">
      <td class="mark" title="${c.ok ? "within tolerance" : c.flagged ? "outside tolerance, flagged for review" : "outside tolerance, not flagged"}">${c.ok ? "✓" : c.flagged ? "⚑" : "✗"}</td><td>${esc(c.item)}${c.note ? `<div class="muted small">${esc(c.note)}</div>` : ""}</td>
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
  if (isGeneral()) return drawOverviewGeneral();
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

/* ---------------- results: any object or test ---------------- */
function axisText(s, v) { return `${fmt(v, 3)} ${s.axis.unit}`; }
function renderGeneralCards(s) {
  const pp = state.results.per_photo, zones = s.zones || [];
  const zoneList = zones.map((z) => `${z.name} (${MODE_SHORT[z.mode || ""]}${z.onset === null ? ", seen briefly" : ""})`).join(", ");
  const cards = [
    ["Photos analysed", String(s.frames), `${s.area.points} measurement points`],
    ["Object size", `${fmt(s.area.size_mm[0], 1)} × ${fmt(s.area.size_mm[1], 1)} mm`, "measured area"],
    ["Strain εxx (end)", fmt(100 * (lastValue(pp["strain εxx"] || []) ?? NaN), 3, "%"), "typical, over the object"],
    ["Strain εzz (end)", fmt(100 * (lastValue(pp["strain εzz"] || []) ?? NaN), 3, "%"), "typical, over the object"],
    ["Peak shear strain", fmt(lastValue(pp["peak shear strain γmax"] || []), 4), "γmax, end of test"],
    ["Localisation zones", String(zones.length), zoneList || "none found"],
    // Only zones seen in consecutive photos (a confirmed onset) get their own card.
    ...zones.filter((z) => z.onset !== null).slice(0, 3).map((z) => [`${z.name} appears at`, axisText(s, z.onset), `${s.axis.name}${z.through !== null ? " · runs across the object" : ""}`]),
  ];
  if (s.markers) {
    const m = s.markers;
    cards.push(m.used ? ["DIC vs markers", `${fmt(m.median_px, 3)} px`, `median · ${fmt(m.median_mm * 1000, 1)} µm · ${m.used} markers${m.disagreeing.length ? ` · ${m.disagreeing.length} disagree` : ""}`,
      m.disagreeing.length ? "Some markers disagree strongly: see the Accuracy tab" : ""]
      : ["DIC vs markers", "—", `${m.found} markers found`, "No markers could be compared: see Review"]);
  }
  if (s.sync && s.sync.clock_offset_s !== undefined) {
    cards.push(["Clock match", `${fmt(s.sync.clock_offset_s, 2)} s`, `± ${fmt(s.sync.offset_sigma_s, 2)} s · ${s.sync.source === "dic" ? "automatic" : s.sync.source}`,
      s.flags.some((f) => f.includes("matched to the logger")) ? "The clock match is uncertain" : ""]);
  }
  const flagged = (text) => `<span class="flag" title="${esc(text)}">⚑</span>`;
  $("#res-cards").innerHTML = cards.map(([k, v, sub, flag]) => `<div class="card">${flag ? flagged(flag) : ""}<div class="k">${k}</div><div class="v">${esc(v)}</div><div class="s">${esc(sub)}</div></div>`).join("");
}
function fillPickers(s, pp) {
  const sensors = (s.sensor_columns || []).filter((c) => pp[c]);
  const objects = (s.object_columns || []).filter((c) => pp[c]);
  // Without a logger the left chart shows a second measure of the object instead of an empty plot.
  const left = sensors.length ? sensors : objects;
  $("#pick-sensor").innerHTML = left.map((c) => `<option>${esc(c)}</option>`).join("");
  $("#pick-sensor").previousSibling.textContent = sensors.length ? "Sensor " : "Measure ";
  if (!sensors.length && objects.includes("peak shear strain γmax")) $("#pick-sensor").value = "peak shear strain γmax";
  $("#pick-object").innerHTML = objects.map((c) => `<option>${esc(c)}</option>`).join("");
  const preferred = objects.find((c) => /opening|slip/.test(c)) || objects.find((c) => c === "peak shear strain γmax");
  if (preferred) $("#pick-object").value = preferred;
  $("#pick-sensor").onchange = $("#pick-object").onchange = drawOverview;
}
function eventLines(events) {
  return events.map((e) => ({ type: "line", x0: e.push_mm, x1: e.push_mm, yref: "paper", y0: 0, y1: 1,
    line: { color: e.source === "sensor" ? css("--muted") : MODE_COLOURS.opening, width: 1, dash: "dot" } }));
}
function eventTags(events) {
  return events.map((e, i) => ({ x: e.push_mm, yref: "paper", y: 0.97 - 0.09 * (i % 4), text: e.label, showarrow: false, xanchor: "left", xshift: 3,
    font: { size: 10.5, color: e.source === "sensor" ? css("--muted") : MODE_COLOURS.opening }, bgcolor: css("--panel"), opacity: 0.95 }));
}
function drawOverviewGeneral() {
  const s = state.results.summary, pp = state.results.per_photo, x = pp[s.axis.label], at = x[state.frame];
  const sensor = $("#pick-sensor").value, object = $("#pick-object").value;
  const objectEvents = s.correlation.events.filter((e) => e.source === "object");
  const sensorEvents = s.correlation.events.filter((e) => e.source === "sensor" && (!sensor || e.subject === sensor));
  const leftPercent = /^(strain|shear strain)/.test(sensor);
  const ls = plotLayout(s.axis.label, sensor + (leftPercent ? " (%)" : "")); ls.showlegend = false;
  ls.shapes = [cursorShape(at), ...eventLines([...sensorEvents, ...objectEvents])]; ls.annotations = eventTags([...sensorEvents, ...objectEvents]);
  const ly = (pp[sensor] || []).map((v) => (v === null ? null : leftPercent ? 100 * v : v));
  Plotly.react("plot-force", [{ x, y: ly, mode: "lines", line: { color: css("--ink"), width: 1.6 } }], ls, PLOT_CONFIG);
  const percent = /^(strain|shear strain)/.test(object);
  const lo = plotLayout(s.axis.label, object + (percent ? " (%)" : "")); lo.showlegend = false;
  lo.shapes = [cursorShape(at), ...eventLines(objectEvents)];
  const y = (pp[object] || []).map((v) => (v === null ? null : percent ? 100 * v : v));
  Plotly.react("plot-cells", [{ x, y, mode: "lines+markers", marker: { size: 4 }, line: { color: MODE_COLOURS.opening } }], lo, PLOT_CONFIG);
  $("#scrub-overview-label").textContent = `${state.frame + 1} / ${s.frames} · ${s.axis.name} ${axisText(s, at)}`;
}
function drawZonePlots() {
  const s = state.results.summary, pp = state.results.per_photo, x = pp[s.axis.label], at = x[state.frame];
  const zones = s.zones || [];
  const mk = (id, traces, title) => {
    const l = plotLayout(s.axis.label, title); l.shapes = [cursorShape(at)]; l.margin.t = 10; l.showlegend = traces.length > 1;
    Plotly.react(id, traces.length ? traces : [{ x: [], y: [] }], l, PLOT_CONFIG);
  };
  const series = (suffix) => zones.map((z) => { const col = Object.keys(pp).find((c) => c.startsWith(`${z.name} `) && c.includes(suffix)); return col && { name: z.name, col, mode: z.mode }; }).filter(Boolean);
  mk("plot-len", series("length").map((t) => ({ x, y: pp[t.col], name: t.name, mode: "lines+markers", marker: { size: 4 }, line: { color: MODE_COLOURS[t.mode || ""] } })), "Zone length (mm)");
  mk("plot-lift", [...series("opening"), ...series("slip")].map((t) => ({ x, y: pp[t.col], name: `${t.name} ${t.col.includes("slip") ? "slip" : "opening"}`, mode: "lines+markers", marker: { size: 4 }, line: { color: MODE_COLOURS[t.mode || ""] } })), "Opening / slip (mm)");
  mk("plot-area", ["strain εxx", "strain εzz"].filter((c) => pp[c]).map((c) => ({ x, y: pp[c].map((v) => (v === null ? null : 100 * v)), name: c, mode: "lines" })), "Strain of the object (%)");
}
function frameSideGeneral(f, k, showStrain) {
  const s = state.results.summary, pp = state.results.per_photo;
  $("#frame-label").textContent = `${s.axis.name} ${axisText(s, f.push_mm)} · photo ${k + 1}/${s.frames}`;
  const metric = (name, v) => `<div class="m"><span>${name}</span><span>${v}</span></div>`;
  const zoneRows = f.bands.map((b) => {
    const opening = pp[`${b.name} opening (mm)`], slip = pp[`${b.name} slip (mm)`];
    const jump = opening ? `opening ${fmt(opening[k], 3, "mm")}` : slip ? `slip ${fmt(slip[k], 3, "mm")}` : "";
    return metric(`${b.name} — ${MODE_SHORT[b.mode || ""]}`, jump);
  });
  $("#frame-metrics").innerHTML = [
    metric("Zones", f.bands.length ? f.bands.map((b) => b.name).join(", ") : "none yet"), ...zoneRows,
    metric("Strain εxx", fmt(100 * (pp["strain εxx"] || [])[k], 3, "%")),
    metric("Strain εzz", fmt(100 * (pp["strain εzz"] || [])[k], 3, "%")),
    metric("Peak shear strain", fmt((pp["peak shear strain γmax"] || [])[k], 4)),
    ...(s.sensor_columns || []).slice(0, 4).map((c) => metric(c, fmt(pp[c] ? pp[c][k] : null, 3))),
  ].join("");
  const modes = [...new Set(f.bands.map((b) => b.mode || ""))];
  $("#legend").innerHTML = (modes.length ? modes : ["opening", "shear"]).map((m) => `<div><i style="background:${MODE_COLOURS[m]}"></i>${MODE_NAMES[m]}</div>`).join("")
    + (showStrain ? `<div>Strain overlay: red = high shear strain (up to ${fmt(f.gamma_max, 3)})</div>` : "");
  drawBandPlots();
}
$("#scrub-overview").oninput = (e) => { state.frame = +e.target.value; $("#scrub-soil").value = state.frame; drawOverview(); };

function drawBandPlots() {
  if (isGeneral()) return drawZonePlots();
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
// Photos are fetched through one queue: requests never overlap (they used to
// pile up during playback and freeze the viewer), and a failed request is
// dropped from the cache so it can be retried.
function fetchFrame(k, showStrain) {
  const key = `${k}:${showStrain}`;
  if (!state.frameCache[key]) {
    const request = (state.frameQueue || Promise.resolve()).catch(() => {}).then(() => api().frame(state.projectPath, k, showStrain));
    state.frameQueue = request;
    state.frameCache[key] = request.catch((e) => { delete state.frameCache[key]; throw e; });
  }
  return state.frameCache[key];
}
async function showFrame(k) {
  state.frame = k; $("#scrub-soil").value = k; $("#scrub-overview").value = k;
  const showStrain = $("#show-strain").checked;
  $("#frame-loading").classList.remove("hidden");
  let f;
  try { f = await fetchFrame(k, showStrain); } catch (e) {
    if (state.frame === k) { $("#frame-loading").classList.add("hidden"); stopPlay(); toast(`Could not show photo ${k + 1}. Try again.`); }
    return false;
  }
  if (state.frame !== k) return false;
  $("#frame-loading").classList.add("hidden");
  $("#frame-img").src = f.image;
  const svg = $("#frame-svg"); svg.setAttribute("viewBox", `0 0 ${f.width} ${f.height}`); svg.setAttribute("preserveAspectRatio", "none");
  const colour = (b) => b.label === "zone" ? MODE_COLOURS[b.mode || ""] : css(BAND_COLOURS[b.label]);
  const title = (b) => b.label === "zone" ? MODE_NAMES[b.mode || ""] : BAND_NAMES[b.label];
  svg.innerHTML = f.bands.map((b) => `<polyline points="${b.points.map((p) => p.join(",")).join(" ")}" fill="none" stroke="${colour(b)}" stroke-width="3" stroke-linecap="round" opacity=".95"><title>${esc(b.name)} — ${esc(title(b))}</title></polyline>
    <text x="${b.points[b.points.length - 1][0] + 6}" y="${b.points[b.points.length - 1][1]}" fill="${colour(b)}" font-size="15" font-weight="700" paint-order="stroke" stroke="#000" stroke-width="3">${esc(b.name)}</text>`).join("");
  if (isGeneral()) { frameSideGeneral(f, k, showStrain); return true; }
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
  return true;
}
$("#scrub-soil").oninput = (e) => showFrame(+e.target.value);
$("#show-strain").onchange = () => showFrame(state.frame);
// Playback shows each photo for at least PLAY_STEP_MS, waiting for it to
// arrive, and pre-loads the next one meanwhile. Dragging the slider while
// playing continues from the new position.
const PLAY_STEP_MS = 400;
function stopPlay() { state.playing = null; $("#play").textContent = "▶"; }
$("#play").onclick = async () => {
  if (state.playing) return stopPlay();
  const run = {}; state.playing = run;
  $("#play").textContent = "❚❚";
  const last = state.results.summary.frames - 1;
  if (state.frame >= last) await showFrame(0);
  while (state.playing === run && state.frame < last) {
    const next = state.frame + 1, showStrain = $("#show-strain").checked;
    const started = performance.now();
    const showing = showFrame(next);   // queued first, so it is not delayed by the pre-load
    if (next + 1 <= last) fetchFrame(next + 1, showStrain).catch(() => {});
    const shown = await showing;
    if (state.playing !== run) return;
    if (!shown && state.frame === next) return stopPlay();
    await new Promise((r) => setTimeout(r, Math.max(0, PLAY_STEP_MS - (performance.now() - started))));
  }
  if (state.playing === run) stopPlay();
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
