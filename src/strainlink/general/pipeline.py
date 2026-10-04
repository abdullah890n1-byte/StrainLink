"""The complete analysis for any test: photos/video (+ an optional logger) -> fields, zones, findings.

Project file (kind "general"):

    {"kind": "general", "name": ...,
     "images": {"folder": ..., "interval_s": optional, "video": optional},
     "calibration": {"mm_per_px": s, "roi_px": [x0, y0, x1, y1] optional},
     "sensors": {"file": ..., "channels": [...]} optional,
     "sync": {"mode": "auto" | "offset", "offset_s": s} optional,
     "dic": {...} optional}

Results are written like the retaining-wall pipeline's (summary.json,
fields.npz, bands.json, per_photo.csv), with summary["kind"] == "general".

Coordinates: mm, x to the right, z up, origin at the photo's bottom-left.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .. import __version__, correlate
from ..bands import DetectionSettings, detect_bands
from ..bands.track import BandTrack, _secondary_tracks
from ..calib import WallFrame
from ..dic import DICSettings, Grid, track_sequence
from ..dic.mask import MIN_FILL as MIN_SUBSET_FILL, SubsetMask
from ..io.images import load_sequence
from ..io.sensors import Channel
from ..pipeline import (LazyImages, Progress, ProjectError, _clean, _json_default, _load_sensor_files, _resolve,
                        corrected_images, load_project)
from ..strain import compute_strain, fill_displacement_gaps, measures
from .events import MODE_NAMES, channel_events, zone_events
from .findings import formation_order, zone_growth
from .region import object_mask
from .zones import kinematics
from ..markers import ColourBrightness, check_dic

# Further strain measures stored per photo (strain.measures; y there is z here).
_MEASURE_KEYS = {"eyy_eng": "ezz_eng", "almansi_yy": "almansi_zz", "almansi_xy": "almansi_xz"}
MEASURES = ("exx_eng", "ezz_eng", "shear_eng", "almansi_xx", "almansi_zz", "almansi_xz", "von_mises", "rotation_deg")
STAGES = ("Loading", "Measuring movement (DIC)", "Strain and localisation zones", "Reading sensors",
          "Matching clocks", "Correlation engine", "Saving results")
# DIC defaults for any object: 31 px subsets on a 5 px grid, searching ±10 px
# around the previous photo's motion.
DEFAULT_SUBSET_RADIUS_PX = 15
DEFAULT_STEP_PX = 5
DEFAULT_SEARCH_RADIUS_PX = 10
DEFAULT_STRAIN_WINDOW = 5
EDGE_AWARE_STRAIN_ORDER = 2
# Zone detection scaled to the object: widest zone and shortest zone as
# fractions of the object's smaller dimension.
MAX_ZONE_WIDTH_FRACTION = 0.25
MIN_ZONE_LENGTH_FRACTION = 0.1
# A zone "runs across the object" when both ends are within this many zone
# widths of the object's outline.
THROUGH_WIDTHS = 2.0
# Share of the object at each end used for the virtual extensometers.
EXTENSOMETER_END_FRACTION = 0.05
# Logger records are thinned to at most this many samples for event detection.
MAX_EVENT_SAMPLES = 5000
# DIC vs markers above this (px RMS) is flagged for review.
MARKER_WARNING_PX = 0.2
# Clock offsets searched (as a fraction of the logger duration) and the fit's resolution.
OFFSET_SEARCH_STEPS = 400
# The typical strain needs at least this many points outside the zones (else all points are used).
MIN_INTACT_POINTS = 25
# Offsets whose fit is statistically as good as the best one: residual within this 95% bound for
# one fitted parameter (chi-squared, 1 degree of freedom) of the best mean squared residual.
OFFSET_CHI2_95 = 3.84
# Clock matching from the load: the load must follow the intact material's strain at least this
# well (R² at the best offset) for the pair to be used.
MIN_SYNC_R2 = 0.9
# A pairing needs this many photos; a zone counts as formed once it is seen in this many photos in a row.
MIN_SYNC_PHOTOS = 5
PERSISTENT_ZONE_PHOTOS = 3


def image_frame(mm_per_px: float, shape: tuple[int, int]) -> WallFrame:
    """Photo pixels <-> mm (x right, z up, origin bottom-left); WallFrame is used as a plain image frame."""
    return WallFrame(mm_per_px=mm_per_px, toe_col=0.0, toe_row=float(shape[0]), wall_height_mm=shape[0] * mm_per_px)


def run(project_path: str | Path, progress: Progress | None = None) -> dict:
    started = time.perf_counter()
    report = progress or (lambda stage, fraction, message: None)
    project = load_project(project_path)
    out_dir = Path(project["_dir"]) / "results"
    out_dir.mkdir(exist_ok=True)

    # 1. Photos, scale and the object's area.
    report(STAGES[0], 0.0, "Reading photos")
    try:
        mm_per_px = float(project["calibration"]["mm_per_px"])
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectError("The scale is not set. Complete the Scale step.") from error
    images_cfg = project.get("images", {})
    try:
        sequence = load_sequence(_resolve(project, images_cfg.get("folder", "images")), images_cfg.get("interval_s"))
    except FileNotFoundError as error:
        raise ProjectError(f"No photos found: {error}. Choose the photos or video in the first step.") from error
    if len(sequence) < 2:
        raise ProjectError("At least two photos are needed (the first must show the object before loading).")
    images = corrected_images(project, LazyImages(sequence.paths))
    frame = image_frame(mm_per_px, images.shape)
    roi = project["calibration"].get("roi_px")
    try:
        mask = object_mask(images[0], tuple(int(v) for v in roi) if roi else None)
    except ValueError as error:
        raise ProjectError(f"{error}.") from error

    # 2. DIC over the object only.
    dic_cfg = project.get("dic", {})
    settings = DICSettings(subset_radius=int(dic_cfg.get("subset_radius_px", DEFAULT_SUBSET_RADIUS_PX)),
                           step=int(dic_cfg.get("step_px", DEFAULT_STEP_PX)),
                           search_radius=int(dic_cfg.get("search_radius_px", DEFAULT_SEARCH_RADIUS_PX)),
                           reference_mode="incremental", update_every=1)
    rows, cols = np.nonzero(mask)
    # Edge-aware subsets let points reach the object's own outline; only the photo's border keeps
    # them a subset radius away (the integer search's template must fit inside the photo).
    r = settings.subset_radius
    box = (max(int(cols.min()), r), max(int(rows.min()), r),
           min(int(cols.max()) + 1, mask.shape[1] - r), min(int(rows.max()) + 1, mask.shape[0] - r))
    grid = Grid.over_roi(box, settings.step, 0)
    # Edge-aware subsets use only the object's pixels, so points reach right up to holes and
    # edges; each needs a fair share of its subset on the object.
    on_object = mask[np.clip(grid.y.astype(int), 0, mask.shape[0] - 1), np.clip(grid.x.astype(int), 0, mask.shape[1] - 1)]
    fill = SubsetMask(mask, grid.x.ravel(), grid.y.ravel(), settings.subset_radius).fill().reshape(grid.shape)
    inside = on_object & (fill >= MIN_SUBSET_FILL)
    if inside.sum() < 25:
        raise ProjectError("The object is too small for DIC at this subset size. Use closer photos or a smaller subset.")
    grid = Grid(x=np.where(inside, grid.x, np.nan), y=np.where(inside, grid.y, np.nan))
    tracked = track_sequence(images, grid, settings,
                             progress=lambda k, n: report(STAGES[1], k / n, f"Photo {k + 1} of {n + 1}"),
                             object_mask=mask)
    gx = np.arange(grid.shape[1]) * settings.step + box[0]
    gy = np.arange(grid.shape[0]) * settings.step + box[1]
    x_mm, z_mm = frame.to_wall(*np.meshgrid(gx, gy))
    spacing = settings.step * mm_per_px

    # 3. Strain and localisation zones, photo by photo.
    n = len(images)
    size_mm = min(np.ptp(x_mm[inside.reshape(grid.shape)]), np.ptp(z_mm[inside.reshape(grid.shape)]))
    det = DetectionSettings(**{"min_width_mm": 2.0 * spacing, "max_width_mm": MAX_ZONE_WIDTH_FRACTION * size_mm,
                               "min_length_mm": MIN_ZONE_LENGTH_FRACTION * size_mm, "attach_tolerance_mm": 0.0,
                               **project.get("detection", {})})
    stored = ("ux", "uz", "gamma", "exx", "ezz", "exz", "e1", "e2") + MEASURES
    fields = {k: np.full((n,) + grid.shape, np.nan) for k in stored}
    per_frame, kin = [], []
    for k in range(n):
        report(STAGES[2], k / n, f"Photo {k + 1} of {n}")
        ux, uz = frame.displacement_to_wall(tracked.field(k, "u"), tracked.field(k, "v"))
        fields["ux"][k], fields["uz"][k] = ux, uz
        if k == 0:
            for name in stored[2:]:
                fields[name][k] = np.where(inside.reshape(grid.shape), 0.0, np.nan)
            per_frame.append(detect_bands(np.zeros(grid.shape), x_mm, z_mm, None, det))
            kin.append({})
            continue
        lost = (tracked.valid[1:k + 1].any(axis=0) & ~tracked.valid[k]).reshape(grid.shape)
        ux_f, uz_f, refilled = fill_displacement_gaps(ux, uz, lost)
        # Edge-aware gradients (order 2): strain right up to holes and edges is not smoothed inwards.
        strain = compute_strain(ux_f, uz_f, spacing, -spacing, int(dic_cfg.get("strain_window", DEFAULT_STRAIN_WINDOW)),
                                order=EDGE_AWARE_STRAIN_ORDER)
        fields["gamma"][k], fields["e1"][k], fields["e2"][k] = strain.gamma_max, strain.e1, strain.e2
        for name, values in measures(strain).items():
            fields[_MEASURE_KEYS.get(name, name)][k] = values
        fields["exx"][k], fields["ezz"][k], fields["exz"][k] = strain.exx, strain.eyy, strain.exy
        found = detect_bands(strain.gamma_max, x_mm, z_mm, None, det, uz=uz_f, reconstructed=refilled)
        per_frame.append(found)
        kin.append({id(b): kinematics(b, x_mm, z_mm, ux_f, uz_f) for b in found.bands})

    # 4. Timing and the common axis.
    report(STAGES[3], 0.0, "Reading the logger")
    table = None
    if project.get("sensors", {}).get("file"):
        table, _ = _load_sensor_files(project)
        if table.time_s is None:
            raise ProjectError(f"No time column found in {table.path.name}. Assign it in the first step.")
        _all_columns(table, confirmed="channels" in project["sensors"])
    report(STAGES[4], 0.0, "Matching photos to the logger clock")
    ext = _extensometers(fields["ux"], fields["uz"], x_mm, z_mm)
    object_series = _object_series(fields, ext, per_frame, x_mm, z_mm)
    timing = _timing(project, sequence.times_s, table, ext, object_series, _first_zone_photo(per_frame))
    timing["sync"]["photo_time_source"] = sequence.time_source
    axis_values = timing["axis"]
    tracks = _secondary_tracks(per_frame, axis_values)
    modes, series, crossing = _zone_summaries(tracks, per_frame, kin, x_mm, z_mm, inside.reshape(grid.shape), n, spacing)

    # Markers (optional): an independent check of DIC on this very test.
    marker_result = None
    if project.get("markers"):
        report(STAGES[2], 1.0, "Checking DIC against the markers")
        cfg = project["markers"]
        zone_mask = _zone_mask(per_frame[-1], x_mm, z_mm)
        marker_images = (corrected_images(project, ColourBrightness(sequence.paths)) if cfg.get("channel") == "brightness"
                         else images)
        marker_result, marker_tracks = check_dic(
            marker_images, mask, gx.astype(float), gy.astype(float),
            fields["ux"] / mm_per_px, -fields["uz"] / mm_per_px, zone_mask, mm_per_px,
            cfg.get("polarity", "dark"), tuple(cfg.get("diameter_px", (6.0, 40.0))))
        marker_result["channel"] = cfg.get("channel", "grey")
        _write_markers(out_dir, marker_tracks)

    # 5. Per-photo table and correlation.
    report(STAGES[5], 0.0, "Finding and testing relationships")
    frames = pd.DataFrame({"photo": np.arange(1, n + 1), "time_s": sequence.times_s, timing["axis_label"]: axis_values})
    obj_cols = []
    for label, values in object_series.items():
        frames[label] = values
        obj_cols.append(label)
    for name, values in series.items():
        frames[name] = values
        obj_cols.append(name)
    sensor_cols, s_events = [], []
    if table is not None:
        for channel in table.channels:
            if channel.role in ("time", "ignore"):
                continue
            values = table.values(channel)
            frames[channel.name] = np.interp(timing["logger_time_at_frames"], table.time_s, values, left=np.nan, right=np.nan)
            sensor_cols.append(channel.name)
            if channel.name != timing.get("axis_channel"):
                s_events += channel_events(channel.name, *_on_axis(timing, table, values), timing["event_sigma"])
    frame_sigma = 0.5 * float(np.nanmedian(np.abs(np.diff(axis_values)))) if n > 1 else 0.0
    o_events = zone_events(tracks, modes, crossing, max(frame_sigma, timing["axis_sigma"]))
    axis_words = (timing["axis_name"], timing["axis_unit"])
    findings = correlate.coincidences(s_events, o_events, axis_words)
    # The clock may have been matched with one of the channels: say so on its timing findings.
    matched = timing["sync"].get("matched_channel")
    if matched and matched != timing.get("axis_channel"):
        for finding in findings:
            if finding.title.startswith(matched):
                finding.caveats.append(f"The photos were matched to the logger clock by assuming {matched} follows the "
                                       f"strain of the intact material, so this timing is not fully independent of it.")
    # Findings about the object itself (these need no logger).
    zone_list = [{"name": t.name, "mode": modes.get(t.name, ""), "onset": t.onset_mm} for t in tracks]
    findings += formation_order(o_events, axis_words)
    findings += zone_growth(zone_list, series, axis_values, axis_words)
    warnings = []
    # The loading axis itself is left out of screening: everything the object does changes with it.
    screened = [c for c in sensor_cols if c != timing.get("axis_channel")]
    if len(frames) >= correlate.MIN_FRAMES_FOR_SCREENING and screened:
        findings += correlate.screen(frames, screened, obj_cols)
    elif sensor_cols:
        warnings.append(f"Only {len(frames)} photos: continuous screening needs at least "
                        f"{correlate.MIN_FRAMES_FOR_SCREENING}.")
    if table is None:
        warnings.append("No logger file: the findings describe the object itself. Add the logger file to link it to loads and sensors.")

    # 6. Save.
    report(STAGES[6], 0.0, "Writing results")
    # Points whose subset reaches the object's outline are measured with a partial (edge-aware)
    # subset and a one-sided strain fit: unbiased but noisier. Probes there say so.
    outline_px = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    near_edge = outline_px[np.clip(np.round(gy).astype(int), 0, mask.shape[0] - 1)[:, None],
                           np.clip(np.round(gx).astype(int), 0, mask.shape[1] - 1)[None, :]] < settings.subset_radius
    np.savez_compressed(out_dir / "fields.npz", x_mm=x_mm, z_mm=z_mm, push_mm=axis_values,
                        valid=tracked.valid.reshape((n,) + grid.shape), near_edge=near_edge, **fields)
    (out_dir / "bands.json").write_text(json.dumps(_clean([
        [{"name": _track_name(tracks, k, b), "label": "zone", "mode": kin[k][id(b)].mode if id(b) in kin[k] else "",
          "points": b.points.round(3), "width_mm": b.width_mm, "flags": b.flags} for b in fb.bands]
        for k, fb in enumerate(per_frame)]), default=_json_default))
    frames.to_csv(out_dir / "per_photo.csv", index=False)
    summary = _clean({
        "kind": "general", "project": project.get("name", Path(project_path).stem),
        "synthetic": bool(project.get("synthetic", False)), "version": __version__,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": time.perf_counter() - started, "frames": n,
        "frame_push_mm": axis_values, "axis": {"name": timing["axis_name"], "unit": timing["axis_unit"],
                                              "label": timing["axis_label"]},
        "frame": {"mm_per_px": mm_per_px, "height_px": images.shape[0], "width_px": images.shape[1]},
        "area": {"points": int(inside.sum()), "size_mm": [float(np.ptp(x_mm[inside.reshape(grid.shape)])),
                                                           float(np.ptp(z_mm[inside.reshape(grid.shape)]))]},
        "sync": timing["sync"],
        "markers": marker_result,
        "zones": [{"name": t.name, "mode": modes.get(t.name, ""), "onset": t.onset_mm, "through": crossing.get(t.name),
                   "push_mm": t.delta_mm, "length_mm": [b.length_mm for b in t.bands],
                   "width_mm": [b.width_mm for b in t.bands]} for t in tracks],
        "object_columns": obj_cols, "sensor_columns": sorted(sensor_cols, key=lambda c: c == timing.get("axis_channel")),
        "correlation": {"findings": [f.as_dict() for f in correlate.ranked(findings)],
                        "events": [asdict(e) for e in s_events + o_events], "warnings": warnings},
        "flags": _flags(per_frame, timing, sequence.time_source, sequence.paths) + _marker_flags(marker_result),
    })
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default))
    report(STAGES[6], 1.0, "Done")
    return summary


# --- stages ------------------------------------------------------------------

def _all_columns(table, confirmed: bool) -> None:
    """Any test: every logger column takes part, under its own name.

    The automatic guesser names channels for retaining-wall tests ("Force",
    "PT3"); for any other test the file's own headers are clearer, and
    columns it did not recognise (temperature, strain gauges, ...) are
    still records worth correlating. A mapping the user confirmed is kept.
    """
    if confirmed:
        return
    for channel in table.channels:
        if channel.role != "time":
            channel.name = channel.column
    for column in table.unmapped:
        values = pd.to_numeric(table.frame[column], errors="coerce")
        if values.notna().mean() > 0.9:
            table.channels.append(Channel(column=column, role="other", unit="", factor=1.0, name=column, confidence=0.0))
    table.unmapped = []


def _extensometers(ux: np.ndarray, uz: np.ndarray, x: np.ndarray, z: np.ndarray) -> dict[str, np.ndarray]:
    """Virtual extensometers across the whole object: change in width (x) and height (z), mm."""
    out = {}
    for label, u, coord in (("stretch x (mm)", ux, x), ("stretch z (mm)", uz, z)):
        values = []
        for k in range(len(u)):
            ok = np.isfinite(u[k])
            if ok.sum() < 10:
                values.append(np.nan)
                continue
            lo, hi = np.quantile(coord[ok], [EXTENSOMETER_END_FRACTION, 1 - EXTENSOMETER_END_FRACTION])
            values.append(float(np.mean(u[k][ok & (coord >= hi)]) - np.mean(u[k][ok & (coord <= lo)])))
        out[label] = np.array(values)
    return out


def _first_zone_photo(per_frame) -> int | None:
    """First photo from which a zone is seen in several photos in a row (None: no lasting zone)."""
    present = [len(fb.bands) > 0 for fb in per_frame]
    for k in range(len(present) - PERSISTENT_ZONE_PHOTOS + 1):
        if all(present[k:k + PERSISTENT_ZONE_PHOTOS]):
            return k
    return None


def _timing(project: dict, frame_times: np.ndarray, table, ext: dict, object_series: dict,
            first_zone: int | None) -> dict:
    """Common axis for events and charts, and each photo's time on the logger clock.

    The clock offset is fitted from pairs of signals that physics ties together, and the pair
    that fixes it best is used:

    - the object's end-to-end stretch and the logger's displacement, over the photos before a
      zone forms (afterwards the zone takes part of the movement and they are no longer
      proportional, which biased the clock);
    - the strain of the intact material and the logger's load (a specimen loaded in series
      carries the load through its intact material): a load drop fixes the clock sharply even
      when the loading rate never changes.
    """
    base = {"axis": frame_times - frame_times[0], "axis_name": "time", "axis_unit": "s", "axis_label": "time (s)",
            "axis_sigma": 0.0, "event_sigma": 0.0, "logger_time_at_frames": frame_times - frame_times[0],
            "offset_s": 0.0, "sync": {"source": "photos only"}}
    if table is None:
        return base
    cfg = project.get("sync", {})
    disp = next(iter(table.by_role("displacement")), None)
    force = next(iter(table.by_role("force")), None)
    mode = cfg.get("mode", "auto" if (disp is not None or force is not None) else "offset")
    logger_t = table.time_s
    step = float(np.nanmedian(np.diff(logger_t))) if len(logger_t) > 1 else 0.0
    sync_extra: dict = {}
    if mode == "offset" or (disp is None and force is None):
        offset, sigma, source = float(cfg.get("offset_s", 0.0)), 0.0, "user" if "offset_s" in cfg else "assumed"
    else:
        photo_t = frame_times - frame_times[0]
        candidates = []
        if disp is not None:
            name, dic = max(ext.items(), key=lambda kv: np.nanmax(np.abs(kv[1])) if np.isfinite(kv[1]).any() else 0.0)
            if first_zone is not None and first_zone >= MIN_SYNC_PHOTOS:
                dic = np.where(np.arange(len(dic)) < first_zone, dic, np.nan)
            candidates.append({"object": name, "channel": disp.name, "eligible": True,
                               **_fit_offset(photo_t, dic, logger_t, table.values(disp))})
        if force is not None:
            name = max(("strain εxx", "strain εzz"),
                       key=lambda k: np.nanmax(object_series[k]) - np.nanmin(object_series[k]) if np.isfinite(object_series[k]).any() else 0.0)
            fit = _fit_offset(photo_t, object_series[name], logger_t, table.values(force))
            candidates.append({"object": name, "channel": force.name, "eligible": bool(fit["r2"] >= MIN_SYNC_R2), **fit})
        usable = [c for c in candidates if c["eligible"] and np.isfinite(c["sigma"])]
        chosen = min(usable, key=lambda c: c["sigma"]) if usable else candidates[0]
        offset, sigma, source = chosen["offset"], chosen["sigma"], "dic"
        sync_extra = {"offset_range_s": chosen["range"], "open_ended": chosen["open_ended"],
                      "matched_object": chosen["object"], "matched_channel": chosen["channel"],
                      "candidates": [{k: c[k] for k in ("object", "channel", "offset", "sigma", "range", "open_ended", "r2", "eligible")}
                                     for c in candidates]}
        frame_gap = float(np.median(np.diff(photo_t))) if len(photo_t) > 1 else 0.0
        sharp = [c for c in usable if c["sigma"] <= 0.5 * frame_gap]
        if len(sharp) == 2 and abs(sharp[0]["offset"] - sharp[1]["offset"]) > 2.0 * np.hypot(sharp[0]["sigma"], sharp[1]["sigma"]):
            sync_extra["conflict"] = (f"Two ways of matching the clock disagree: {sharp[0]['channel']} gives "
                                      f"{sharp[0]['offset']:.2f} s, {sharp[1]['channel']} gives {sharp[1]['offset']:.2f} s.")
    at_frames = frame_times - frame_times[0] + offset
    sync = {"source": source, "clock_offset_s": offset, "offset_sigma_s": sigma, "logger_step_s": step, **sync_extra}
    if disp is not None:
        axis = np.interp(at_frames, logger_t, table.values(disp), left=np.nan, right=np.nan)
        slope = np.nanmedian(np.abs(np.gradient(table.values(disp), logger_t))) if len(logger_t) > 2 else 0.0
        unit = disp.unit or "mm"
        label = disp.name if f"({unit.lower()})" in disp.name.lower() else f"{disp.name} ({unit})"
        return {**base, "axis": axis, "axis_name": disp.name, "axis_unit": unit, "axis_label": label, "axis_sigma": float(slope * sigma),
                "event_sigma": float(max(slope * max(step, sigma), 1e-9)), "logger_time_at_frames": at_frames,
                "offset_s": offset, "axis_channel": disp.name, "sync": sync}
    return {**base, "axis": at_frames - offset, "axis_sigma": sigma, "event_sigma": max(step, sigma),
            "logger_time_at_frames": at_frames, "offset_s": offset, "sync": sync}


def _fit_offset(frame_t: np.ndarray, dic: np.ndarray, logger_t: np.ndarray, channel: np.ndarray) -> dict:
    """Logger time of the first photo, from a DIC signal = a * (logger channel - its value at the first photo).

    DIC measures change since the first photo, so the logger's channel is
    taken relative to its value at the first photo too (no free
    intercept). Scale-free (a is fitted), so it works whatever the
    specimen's stiffness. Timing information comes only from changes in the
    signal's rate (start, stops, drops): on a steady ramp any clock shift
    fits equally well.

    The estimate is the centre of the range of offsets that fit
    statistically as well as the best one (95%): inside that range the
    best fit's position is noise. The uncertainty is the half-width of that
    range, or the disagreement between the two halves of the test where it
    exceeds what each half can determine by itself (a half with no timing
    information says nothing either way).

    Returns offset, sigma, range [lowest, highest], open_ended (the range
    runs into the end of the search: the photos cannot fix the clock on
    that side) and r2 (how well the two signals match at that offset).
    """
    ok = np.isfinite(dic)
    if ok.sum() < MIN_SYNC_PHOTOS or np.ptp(dic[ok]) == 0:
        return {"offset": 0.0, "sigma": float("nan"), "range": [float("nan"), float("nan")], "open_ended": True, "r2": 0.0}
    duration = float(logger_t[-1] - logger_t[0])

    def cost(offset: float, sel: np.ndarray = ok) -> float:
        d = np.interp(frame_t[sel] + offset, logger_t, channel, left=np.nan, right=np.nan)
        d0 = np.interp(frame_t[0] + offset, logger_t, channel, left=np.nan, right=np.nan)
        rel = d - d0
        good = np.isfinite(rel)
        if good.sum() < max(MIN_SYNC_PHOTOS, 0.8 * sel.sum()) or not np.any(rel[good]):
            return np.inf
        y = dic[sel][good]
        a = float(rel[good] @ y / (rel[good] @ rel[good]))
        return float(np.mean((y - a * rel[good]) ** 2))

    grid = np.linspace(logger_t[0] - frame_t[0], logger_t[-1] - frame_t[-1], OFFSET_SEARCH_STEPS)
    if grid[-1] <= grid[0]:
        grid = np.linspace(-duration, duration, OFFSET_SEARCH_STEPS)
    grid_step = float(grid[1] - grid[0]) if len(grid) > 1 else 1.0

    def fit(sel: np.ndarray) -> tuple[float, float, float, bool] | None:
        """Centre, lowest and highest of the equally good offsets, and whether that range is open-ended."""
        costs = np.array([cost(o, sel) for o in grid])
        finite = np.isfinite(costs)
        if not finite.any():
            return None
        n = int(sel.sum())
        # Equally good (95%): the usual bound for one fitted parameter, plus the fit's own sampling
        # noise. Every offset reads different (noisy) logger samples, so the mean squared residual
        # itself scatters by about sqrt(2/n) of its value from one offset to the next; without that
        # allowance a noise dip on a flat curve looked like a sharp minimum.
        allowance = OFFSET_CHI2_95 / max(n - 2, 1) + 2.0 * np.sqrt(2.0 / n)
        equal_mask = finite & (costs <= float(np.min(costs[finite])) * (1.0 + allowance) + 1e-15)
        equal, usable = grid[equal_mask], np.flatnonzero(finite)
        open_ended = bool(equal_mask[usable[0]] or equal_mask[usable[-1]])
        return 0.5 * float(equal.min() + equal.max()), float(equal.min()), float(equal.max()), open_ended

    whole = fit(ok)
    if whole is None:
        return {"offset": 0.0, "sigma": float("nan"), "range": [float("nan"), float("nan")], "open_ended": True, "r2": 0.0}
    offset, low, high, open_ended = whole
    idx = np.flatnonzero(ok)
    first, second = np.zeros_like(ok), np.zeros_like(ok)
    first[idx[: len(idx) // 2]], second[idx[len(idx) // 2:]] = True, True
    halves = [h for h in (fit(first) if first.sum() >= MIN_SYNC_PHOTOS else None,
                          fit(second) if second.sum() >= MIN_SYNC_PHOTOS else None) if h is not None]
    disagreement = 0.0
    if len(halves) == 2:
        own = max(0.5 * (halves[0][2] - halves[0][1]), 0.5 * (halves[1][2] - halves[1][1]))
        disagreement = max(0.5 * abs(halves[0][0] - halves[1][0]) - own, 0.0)
    frame_step = float(np.median(np.diff(frame_t))) if len(frame_t) > 1 else 0.0
    variance = float(np.var(dic[ok]))
    final = cost(offset)
    r2 = 1.0 - final / variance if variance > 0 and np.isfinite(final) else 0.0
    sigma = max(0.5 * (high - low), disagreement, 0.5 * grid_step, 0.1 * frame_step)
    return {"offset": float(offset), "sigma": float(sigma), "range": [low, high], "open_ended": open_ended, "r2": float(r2)}


def _on_axis(timing: dict, table, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A logger channel placed on the common axis (thinned for event detection).

    The axis is the loading displacement channel if there is one, else the
    time since the first photo.
    """
    stride = max(len(values) // MAX_EVENT_SAMPLES, 1)
    v = values[::stride]
    if "axis_channel" in timing:
        axis_channel = next(c for c in table.channels if c.name == timing["axis_channel"])
        return table.values(axis_channel)[::stride], v
    return table.time_s[::stride] - timing["offset_s"], v


def _zone_summaries(tracks: list[BandTrack], per_frame, kin, x, z, inside, n, spacing):
    """Per zone: its dominant mode, series per photo, and when it first runs across the object."""
    # Padded: the edge of the grid is outline too (the object may fill its whole box).
    outline_distance = ndimage.distance_transform_edt(np.pad(inside, 1))[1:-1, 1:-1] * spacing
    x0, dx, z0, dz = x[0, 0], x[0, 1] - x[0, 0], z[0, 0], z[1, 0] - z[0, 0]
    modes, series, crossing = {}, {}, {}
    for t in tracks:
        found = [kin[k][id(b)] for k, b in zip(t.frames, t.bands) if id(b) in kin[k]]
        if found:
            votes = [f.mode for f in found[-max(len(found) // 2, 1):]]
            modes[t.name] = max(set(votes), key=votes.count)
        length = np.full(n, np.nan)
        opening = np.full(n, np.nan)
        slip = np.full(n, np.nan)
        for k, b in zip(t.frames, t.bands):
            length[k] = b.length_mm
            if id(b) in kin[k]:
                opening[k], slip[k] = kin[k][id(b)].opening_mm, kin[k][id(b)].slip_mm
            ends = b.points[[0, -1]]
            r = np.clip(np.round((ends[:, 1] - z0) / dz).astype(int), 0, inside.shape[0] - 1)
            c = np.clip(np.round((ends[:, 0] - x0) / dx).astype(int), 0, inside.shape[1] - 1)
            if t.name not in crossing and np.all(outline_distance[r, c] <= THROUGH_WIDTHS * b.width_mm):
                crossing[t.name] = t.delta_mm[t.frames.index(k)]
        series[f"{t.name} length (mm)"] = length
        series[f"{t.name} {'opening' if modes.get(t.name) != 'shear' else 'slip'} (mm)"] = (
            opening if modes.get(t.name) != "shear" else slip)
    return modes, series, crossing


def _object_series(fields: dict, ext: dict, per_frame, x: np.ndarray, z: np.ndarray) -> dict[str, np.ndarray]:
    """Whole-object measures per photo: typical strain (a virtual strain gauge), extensometers, peak strains."""
    def over_object(name: str, how) -> np.ndarray:
        return np.array([how(f[np.isfinite(f)]) if np.isfinite(f).any() else np.nan for f in fields[name]])

    # The typical strain is the strain of the intact material: each photo's zones are left out
    # (a crack's jump, spread by the strain window, covered over a quarter of a small plate and
    # pulled even a median up by 5%), and the median resists anything left.
    intact = [~_zone_mask(fb, x, z) for fb in per_frame]

    def typical(name: str) -> np.ndarray:
        out = []
        for f, keep in zip(fields[name], intact):
            sel = np.isfinite(f) & keep
            sel = sel if sel.sum() >= MIN_INTACT_POINTS else np.isfinite(f)
            out.append(float(np.median(f[sel])) if sel.any() else np.nan)
        return np.array(out)

    return {
        "strain εxx": typical("exx"),
        "strain εzz": typical("ezz"),
        "shear strain εxz": typical("exz"),
        "peak shear strain γmax": over_object("gamma", lambda v: np.percentile(v, 99)),
        "peak major strain ε1": over_object("e1", lambda v: np.percentile(v, 99)),
        **ext,
        "zones": np.array([len(fb.bands) for fb in per_frame], dtype=float),
    }


def _zone_mask(frame_bands, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Grid points within one zone width of a zone's centreline (in one photo)."""
    mask = np.zeros(x.shape, dtype=bool)
    points = np.column_stack([x.ravel(), z.ravel()])
    for band in frame_bands.bands:
        from scipy.spatial import cKDTree
        distance = cKDTree(band.points).query(points)[0].reshape(x.shape)
        mask |= distance <= band.width_mm
    return mask


def _write_markers(out_dir: Path, tracks) -> None:
    rows = {"photo": np.arange(1, tracks.x.shape[0] + 1)}
    for m in range(tracks.count):
        rows[f"M{m + 1} x (px)"], rows[f"M{m + 1} y (px)"] = tracks.x[:, m], tracks.y[:, m]
    pd.DataFrame(rows).to_csv(out_dir / "markers.csv", index=False)


def _marker_flags(result: dict | None) -> list[str]:
    if not result:
        return []
    if result["found"] == 0:
        return ["Markers were expected but none were found: check their size (px) and whether they are darker or brighter "
                "than the object."]
    flags = []
    if result.get("used", 0) and result["median_px"] > MARKER_WARNING_PX:
        flags.append(f"DIC differs from the markers by {result['median_px']:.2f} px (median, above {MARKER_WARNING_PX} px): "
                     f"check focus, lighting, the speckle pattern and the photo spacing.")
    if result.get("disagreeing"):
        names = ", ".join(f"M{i + 1}" for i in result["disagreeing"])
        flags.append(f"Marker(s) {names} disagree strongly with DIC: probably mis-tracked (jumped to a nearby dark spot, "
                     f"or a natural marker rolled). Check them in the Accuracy tab.")
    return flags


def _track_name(tracks: list[BandTrack], k: int, band) -> str:
    for t in tracks:
        for kk, b in zip(t.frames, t.bands):
            if kk == k and b is band:
                return t.name
    return ""


def _flags(per_frame, timing: dict, time_source: str, photo_paths=()) -> list[str]:
    flags = []
    # Lossy compression biases sub-pixel matching (strong JPEG: ~1% strain bias in testing).
    if time_source == "video":
        flags.append("The frames come from a compressed video: expect slightly lower accuracy than from photos "
                     "(record at the highest quality, or take photos as TIFF/PNG for the most precise strain).")
    elif any(Path(p).suffix.lower() in (".jpg", ".jpeg") for p in photo_paths):
        flags.append("The photos are JPEG (compressed): at strong compression strain can be biased by ~1%. "
                     "Prefer TIFF/PNG or the camera's highest quality setting.")
    if time_source == "file-time":
        flags.append("Photo times come from file modification times (least reliable). Prefer camera times, a video, "
                     "or the photo interval.")
    sync = timing["sync"]
    if sync.get("source") == "assumed":
        flags.append("The logger has no displacement channel and no clock offset was given: the first photo is assumed "
                     "to be at logger time 0. Enter the offset if it is known.")
    if sync.get("source") == "dic" and np.isfinite(sync.get("offset_sigma_s", np.nan)):
        step = sync.get("logger_step_s", 0.0)
        frame_gap = float(np.nanmedian(np.diff(timing["logger_time_at_frames"]))) if len(timing["logger_time_at_frames"]) > 1 else 0.0
        if sync.get("conflict"):
            flags.append(sync["conflict"] + " Check which logger column is the load and which the displacement, "
                         "or enter the clock offset.")
        if sync.get("open_ended"):
            lo, hi = sync["offset_range_s"]
            flags.append(f"Photos could not be matched to the logger clock from the images: the loading rate does not "
                         f"change while the photos are taken, so every clock offset from {lo:.1f} s to {hi:.1f} s fits "
                         f"equally well (used: {sync['clock_offset_s']:.1f} s). Findings that depend on timing are "
                         f"uncertain; enter the clock offset if it is known.")
        elif frame_gap and sync["offset_sigma_s"] > 0.5 * frame_gap:
            flags.append(f"Photos are matched to the logger only within ±{sync['offset_sigma_s']:.2f} s "
                         f"(photos are {frame_gap:.2f} s apart).")
    review = sum(1 for fb in per_frame if any(b.flags for b in fb.bands))
    if review:
        flags.append(f"{review} photo(s) have zone measurements flagged for review.")
    return flags
