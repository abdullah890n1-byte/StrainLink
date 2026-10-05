"""Bridge between the app's interface (JavaScript) and the analysis engine.

Every public method of `Api` is callable from the page as
`window.pywebview.api.<method>(...)` and returns JSON-friendly data. Long
jobs (demo generation, analysis) run in a background thread; the page
polls `progress()` so the window never freezes.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import traceback
import webbrowser
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .. import __version__, pipeline, report
from ..calib import WallFrame
from ..dic import load_grayscale
from ..bands.ridges import fill_missing
from ..fields import QUANTITIES, FieldStore
from ..io.images import load_sequence
from ..io.sensors import load_sensors
from ..io import video as video_io
from ..synth.demo import DemoSpec, make_demo

APP_DIR = Path.home() / "Library" / "Application Support" / "StrainLink"
RECENT_FILE = APP_DIR / "recent.json"
DEFAULTS_FILE = APP_DIR / "defaults.json"
# Names that mark a calibration (no-soil pushover) run rather than the test log.
CALIBRATION_WORDS = ("calib", "friction", "empty", "pushover", "no_soil", "nosoil")
SENSOR_SUFFIXES = {".csv", ".txt", ".tsv", ".xlsx", ".xls"}
# How deep to look for photos and logs inside a chosen test folder.
SCAN_DEPTH = 2
MAX_RECENT = 8
# Photos are sent to the page downscaled to this width (px) for speed.
PREVIEW_WIDTH = 900
# Strain overlay colour map endpoints (transparent -> strong magenta-red).
OVERLAY_ALPHA = 0.75
# Samples along a line profile.
LINE_SAMPLES = 120
# Fixed-point iterations to find where a displayed pixel was in the first
# photo (X = x - u(X)); the displacement varies slowly enough that a few suffice.
INVERSE_MAP_ITERATIONS = 4
# The overlay is computed every this many display pixels and then upsampled.
OVERLAY_STEP_PX = 2


def _png_data_url(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _to_8bit(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [0.5, 99.5])
    return np.clip((image - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)


def _downscale(image: np.ndarray) -> tuple[np.ndarray, float]:
    scale = min(1.0, PREVIEW_WIDTH / image.shape[1])
    if scale >= 1.0:
        return image, 1.0
    size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    return np.asarray(Image.fromarray(image).resize(size, Image.LANCZOS)), scale


class _Analysis:
    """Everything the photo viewer and the inspection tools need from one analysis, loaded once."""

    def __init__(self, project_path: str, store: FieldStore, bands: list, photo_paths: list, wall: WallFrame):
        self.project_path = project_path
        self.wall = wall
        self.bands = bands
        self.photo_paths = photo_paths
        self.store = store
        x, z = store.x, store.z
        # The DIC grid is regular in the first photo: index = (position - origin) / step.
        self.x0, self.dx = float(x[0, 0]), float(x[0, 1] - x[0, 0])
        self.z0, self.dz = float(z[0, 0]), float(z[1, 0] - z[0, 0])
        self.ux, self.uz = store.data["ux"], store.data["uz"]
        self.push_mm = store.data["push_mm"]
        self._scales: dict[str, float] = {}
        self.undistort = None          # lens correction for the displayed photos (set by load)
        self.compressed = False        # JPEG photos or video frames: a larger accuracy allowance (set by load)

    @classmethod
    def load(cls, project_path: str) -> "_Analysis":
        project = pipeline.load_project(project_path)
        folder = Path(project_path).parent
        bands = json.loads((folder / "results" / "bands.json").read_text())
        cal = project["calibration"]
        if project.get("kind") == "general":
            from ..general.pipeline import image_frame
            shape = json.loads((folder / "results" / "summary.json").read_text())["frame"]
            wall = image_frame(float(cal["mm_per_px"]), (int(shape["height_px"]), int(shape["width_px"])))
        else:
            wall = WallFrame(float(cal["mm_per_px"]), float(cal["toe_px"][0]), float(cal["toe_px"][1]),
                             float(cal["wall_height_mm"]))
        seq = load_sequence(folder / project["images"].get("folder", "images"), project["images"].get("interval_s"))
        analysis = cls(project_path, FieldStore(folder / "results"), bands, list(seq.paths), wall)
        # The fields are in lens-corrected coordinates, so the photos are shown corrected too.
        analysis.undistort = pipeline.lens_corrector(project)
        analysis.compressed = seq.time_source == "video" or any(
            Path(p).suffix.lower() in pipeline.COMPRESSED_SUFFIXES for p in seq.paths)
        return analysis

    def _index(self, x, z) -> np.ndarray:
        return np.stack([(np.ravel(z) - self.z0) / self.dz, (np.ravel(x) - self.x0) / self.dx])

    def displacement_at(self, k: int, x, z) -> tuple[np.ndarray, np.ndarray]:
        """Displacement (mm) of photo k at reference positions (mm); gaps filled from neighbours."""
        idx = self._index(x, z)
        ux = ndimage.map_coordinates(fill_missing(self.ux[k]), idx, order=1, mode="nearest")
        uz = ndimage.map_coordinates(fill_missing(self.uz[k]), idx, order=1, mode="nearest")
        return ux.reshape(np.shape(x)), uz.reshape(np.shape(x))

    def reference_of(self, k: int, x_now: np.ndarray, z_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Where material now at (x_now, z_now) in photo k was in the first photo (fixed-point inverse)."""
        ux, uz = fill_missing(self.ux[k]), fill_missing(self.uz[k])
        x_ref, z_ref = x_now, z_now
        for _ in range(INVERSE_MAP_ITERATIONS):
            idx = self._index(x_ref, z_ref)
            x_ref = x_now - ndimage.map_coordinates(ux, idx, order=1, mode="nearest").reshape(np.shape(x_now))
            z_ref = z_now - ndimage.map_coordinates(uz, idx, order=1, mode="nearest").reshape(np.shape(x_now))
        return x_ref, z_ref

    def value_at(self, key: str, k: int, x_ref, z_ref) -> np.ndarray:
        """A quantity at reference positions (NaN where it was not measured)."""
        field = self.store.get(key, k)
        idx = self._index(x_ref, z_ref)
        measured = ndimage.map_coordinates(np.isfinite(field).astype(float), idx, order=0, mode="constant", cval=0.0)
        values = ndimage.map_coordinates(np.nan_to_num(field), idx, order=1, mode="constant", cval=0.0)
        return np.where(measured > 0.5, values, np.nan).reshape(np.shape(x_ref))

    def uncertainty_at(self, key: str, k: int, x_ref, z_ref) -> np.ndarray:
        """95% uncertainty of a quantity at reference positions (the nearest grid point's value)."""
        field = self.store.uncertainty(key, k, self.compressed)
        idx = self._index(x_ref, z_ref)
        known = np.isfinite(field)
        if not known.any():
            return np.full(np.shape(x_ref), np.nan)
        filled = np.where(known, field, np.nanmedian(field))
        return ndimage.map_coordinates(filled, idx, order=0, mode="nearest").reshape(np.shape(x_ref))

    def typical_uncertainty(self, key: str, k: int) -> float | None:
        """Typical 95% uncertainty of a quantity over the measured area of photo k."""
        field = self.store.uncertainty(key, k, self.compressed)
        return float(np.nanmedian(field)) if np.isfinite(field).any() else None

    def near_edge(self, x_ref, z_ref) -> np.ndarray:
        """True where the DIC subset reaches the object's outline (a partial, edge-aware subset: noisier)."""
        mask = self.store.data.get("near_edge")
        if mask is None:
            return np.zeros(np.shape(x_ref), dtype=bool)
        idx = self._index(x_ref, z_ref)
        return ndimage.map_coordinates(mask.astype(float), idx, order=0, mode="nearest").reshape(np.shape(x_ref)) > 0.5

    def scale(self, key: str) -> float:
        """Colour scale of a quantity: its 99th percentile magnitude in the last photo."""
        if key not in self._scales:
            last = np.abs(self.store.get(key, self.store.frames - 1))
            self._scales[key] = float(np.nanpercentile(last, 99)) if np.isfinite(last).any() else 1.0
            self._scales[key] = self._scales[key] or 1.0
        return self._scales[key]

    def overlay_level(self, k: int, shape: tuple[int, int], scale: float, key: str = "gamma") -> np.ndarray:
        """Quantity level at every display pixel of photo k, -1..1 (signed) or 0..1, drawn where the material is now."""
        rows, cols = np.mgrid[0:shape[0]:OVERLAY_STEP_PX, 0:shape[1]:OVERLAY_STEP_PX].astype(float)
        x_now, z_now = self.wall.to_wall(cols / scale, rows / scale)
        x_ref, z_ref = self.reference_of(k, x_now, z_now)
        values = self.value_at(key, k, x_ref, z_ref)
        level = np.nan_to_num(np.clip(values / self.scale(key), -1, 1))
        full = np.repeat(np.repeat(level, OVERLAY_STEP_PX, axis=0), OVERLAY_STEP_PX, axis=1)[:shape[0], :shape[1]]
        return np.clip(ndimage.uniform_filter(full, OVERLAY_STEP_PX), -1.0, 1.0)


class Api:
    def __init__(self):
        self._window = None
        self._job: dict = {"running": False, "stage": "", "fraction": 0.0, "message": "", "error": None, "result": None}
        self._lock = threading.Lock()
        self._frame_cache: dict = {}
        self._frame_lock = threading.Lock()
        self._viewed: _Analysis | None = None
        APP_DIR.mkdir(parents=True, exist_ok=True)

    def attach(self, window) -> None:
        self._window = window

    # --- general -------------------------------------------------------------
    def info(self) -> dict:
        return {"version": __version__, "recent": self._recent()}

    def _recent(self) -> list[dict]:
        try:
            items = json.loads(RECENT_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        return [i for i in items if Path(i["path"]).exists()]

    def _remember(self, project_path: str, name: str) -> None:
        items = [i for i in self._recent() if i["path"] != project_path]
        items.insert(0, {"path": project_path, "name": name})
        RECENT_FILE.write_text(json.dumps(items[:MAX_RECENT], indent=1))

    def defaults(self) -> dict:
        """Test details used last time (unit weight, wall size, cell depths), to prefill the form."""
        try:
            return json.loads(DEFAULTS_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def save_defaults(self, values: dict) -> dict:
        DEFAULTS_FILE.write_text(json.dumps(values, indent=1))
        return {"ok": True}

    def scan_test_folder(self, folder: str) -> dict:
        """Find the photos, the logger export and the calibration run inside a test folder."""
        root = Path(folder)
        if root.is_file() and video_io.is_video(root):
            return {"ok": True, "folder": str(root.parent), "photos": None, "photo_count": 0, "video": str(root),
                    **self._scan_logger(root.parent)}
        if not root.is_dir():
            return {"ok": False, "error": "That is not a folder or a video."}
        from ..io.images import IMAGE_SUFFIXES
        candidates = [root] + [p for p in root.rglob("*") if p.is_dir() and len(p.relative_to(root).parts) <= SCAN_DEPTH
                               and not p.name.startswith(".") and p.name != "results"]
        counts = {d: sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES and not f.name.startswith("."))
                  for d in candidates}
        photos = max(counts, key=counts.get) if counts and max(counts.values()) >= 2 else None
        # A video (the largest, if several) is used unless the photos are a real photo folder:
        # frames StrainLink extracted earlier from that video don't count as photos.
        videos = [f for d in candidates for f in d.iterdir() if f.is_file() and video_io.is_video(f)]
        video = max(videos, key=lambda f: f.stat().st_size) if videos else None
        if photos and video_io.read_frame_times(photos) is not None:
            photos = None
        project = next((p for p in (root / "project.json", root / "StrainLink analysis" / "project.json") if p.exists()), None)
        return {"ok": True, "folder": str(root), "photos": str(photos) if photos else None,
                "photo_count": counts.get(photos, 0) if photos else 0, "video": str(video) if video and not photos else None,
                "project": str(project) if project else None, **self._scan_logger(root, candidates)}

    def _scan_logger(self, root: Path, candidates: list | None = None) -> dict:
        """The logger export and the calibration run among the test's files."""
        candidates = candidates or [root] + [p for p in root.rglob("*") if p.is_dir() and not p.name.startswith(".")
                                             and len(p.relative_to(root).parts) <= SCAN_DEPTH and p.name != "results"]
        files = [f for d in candidates for f in d.iterdir()
                 if f.is_file() and f.suffix.lower() in SENSOR_SUFFIXES and not f.name.startswith(".")]
        calibration = next((f for f in files if any(w in f.name.lower() for w in CALIBRATION_WORDS)), None)
        logger, best = None, -1.0
        for f in files:
            if f == calibration:
                continue
            try:
                table = load_sensors(f)
            except (OSError, ValueError):
                continue
            roles = {c.role for c in table.channels}
            # A logger export has a time column and at least one other numeric column
            # (a README or a one-line text file is not a logger).
            if "time" not in roles or len(table.channels) + len(table.unmapped) < 2:
                continue
            score = sum(r in roles for r in ("time", "force", "displacement")) + 0.1 * len(table.channels)
            if score > best:
                logger, best = f, score
        return {"logger": str(logger) if logger else None, "calibration": str(calibration) if calibration else None}

    def extract_video(self, path: str, every_s: float | None = None) -> dict:
        """Frames of a test video at the given spacing (default: automatic), ready to analyse like photos."""
        try:
            folder, info, spacing = video_io.extract(path, every_s)
        except (OSError, ValueError) as error:
            return {"ok": False, "error": str(error)}
        return {"ok": True, "folder": str(folder), "every_s": spacing, "fps": info.fps, "duration_s": info.duration_s,
                "default_every_s": video_io.default_spacing_s(info), "name": Path(path).name,
                "count": len(list(folder.glob("frame_*.png")))}

    def choose(self, kind: str, start: str = "") -> str | None:
        """Native dialog. kind: folder | sensor | calibration | project | video | save-project."""
        import webview
        types = {
            "sensor": ("Logger exports (*.csv;*.txt;*.tsv;*.xlsx;*.xls)", "All files (*.*)"),
            "calibration": ("Calibration run (*.csv;*.txt;*.xlsx)", "All files (*.*)"),
            "image": ("Photos (*.tif;*.tiff;*.png;*.jpg;*.jpeg;*.bmp)", "All files (*.*)"),
            "project": ("StrainLink project (*.json)",),
            "video": ("Videos (*.mp4;*.mov;*.m4v;*.avi;*.mkv;*.mts;*.wmv)", "All files (*.*)"),
        }
        if kind == "folder":
            result = self._window.create_file_dialog(webview.FOLDER_DIALOG, directory=start or str(Path.home()))
        elif kind == "save-project":
            result = self._window.create_file_dialog(webview.SAVE_DIALOG, directory=start or str(Path.home()),
                                                     save_filename="project.json")
        else:
            result = self._window.create_file_dialog(webview.OPEN_DIALOG, directory=start or str(Path.home()),
                                                     file_types=types.get(kind, ("All files (*.*)",)))
        if not result:
            return None
        return result if isinstance(result, str) else result[0]

    # --- wizard steps -----------------------------------------------------------
    def inspect_photos(self, folder: str, interval_s: float | None = None) -> dict:
        try:
            seq = load_sequence(folder, interval_s)
        except (FileNotFoundError, OSError) as error:
            return {"ok": False, "error": str(error)}
        first = load_grayscale(seq.paths[0])
        preview, scale = _downscale(_to_8bit(first))
        last, _ = _downscale(_to_8bit(load_grayscale(seq.paths[-1])))
        return {"ok": True, "count": len(seq), "time_source": seq.time_source,
                "duration_s": float(seq.times_s[-1]), "width": int(first.shape[1]), "height": int(first.shape[0]),
                "first": _png_data_url(preview), "last": _png_data_url(last), "preview_scale": scale,
                "first_name": seq.paths[0].name, "last_name": seq.paths[-1].name}

    def calibrate_lens(self, path: str, square_mm: float | None = None) -> dict:
        """Lens model from a photo of a checkerboard (found automatically); also the scale if the square size is given."""
        from ..lens import GOOD_FIT_PX, calibrate
        try:
            model = calibrate([load_grayscale(path)], float(square_mm) if square_mm else None)
        except (OSError, ValueError) as error:
            return {"ok": False, "error": str(error)}
        k1, k2 = (float(v) for v in model.distortion.ravel()[:2])
        w, h = model.image_size
        # How far the lens moves a photo corner: what the correction removes.
        import cv2
        corner = np.array([[[0.0, 0.0]]], np.float32)
        moved = float(np.hypot(*(cv2.undistortPoints(corner, model.camera_matrix, model.distortion,
                                                     P=model.camera_matrix).reshape(2) - corner.reshape(2))))
        return {"ok": True, "lens": model.to_json(), "rms_px": model.rms_px, "k1": k1, "k2": k2,
                "board": list(model.board), "corner_shift_px": moved, "mm_per_px": model.mm_per_px,
                "good": model.rms_px <= GOOD_FIT_PX, "size": [w, h]}

    def preview_area(self, folder: str, roi_px: list | None = None) -> dict:
        """First photo with the automatically found object area outlined (any-test mode).

        roi_px [x0, y0, x1, y1] (photo pixels) limits the search, if the user drew a box.
        """
        from ..general.region import object_mask
        try:
            seq = load_sequence(folder)
            first = load_grayscale(seq.paths[0])
            mask = object_mask(first, tuple(int(v) for v in roi_px) if roi_px else None)
        except (FileNotFoundError, OSError, ValueError) as error:
            return {"ok": False, "error": str(error)}
        preview, scale = _downscale(_to_8bit(first))
        small = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(preview.shape[::-1], Image.NEAREST)) > 0
        edge = small & ~ndimage.binary_erosion(small, iterations=2)
        rgb = np.repeat(preview[..., None], 3, axis=2).astype(float)
        rgb[~small] *= 0.35                         # dim what is not measured
        rgb[edge] = (47, 111, 237)
        share = float(mask.mean())
        return {"ok": True, "image": _png_data_url(rgb.astype(np.uint8)), "share": share}

    def inspect_sensors(self, path: str) -> dict:
        try:
            table = load_sensors(path)
        except (OSError, ValueError) as error:
            return {"ok": False, "error": f"Could not read this file: {error}"}
        preview = table.frame.head(6).round(4).astype(object).where(table.frame.head(6).notna(), None)
        return {"ok": True, "rows": len(table.frame), "columns": list(table.frame.columns),
                "preview": preview.values.tolist(), "channels": [asdict(c) for c in table.channels],
                "unmapped": table.unmapped}

    def save_project(self, project: dict, path: str) -> dict:
        path_obj = Path(path)
        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            clean = {k: v for k, v in project.items() if not k.startswith("_")}
            path_obj.write_text(json.dumps(clean, indent=2))
        except OSError as error:
            return {"ok": False, "error": str(error)}
        self._remember(str(path_obj), project.get("name", path_obj.parent.name))
        return {"ok": True, "path": str(path_obj)}

    def load_project(self, path: str) -> dict:
        try:
            project = pipeline.load_project(path)
        except pipeline.ProjectError as error:
            return {"ok": False, "error": str(error)}
        has_results = (Path(path).parent / "results" / "summary.json").exists()
        self._remember(path, project.get("name", Path(path).parent.name))
        return {"ok": True, "project": {k: v for k, v in project.items() if not k.startswith("_")},
                "has_results": has_results, "path": path}

    # --- background jobs ------------------------------------------------------
    def progress(self) -> dict:
        with self._lock:
            return dict(self._job)

    def _start(self, target, *args) -> dict:
        with self._lock:
            if self._job["running"]:
                return {"ok": False, "error": "Another job is already running."}
            self._job.update(running=True, stage="Starting", fraction=0.0, message="", error=None, result=None)
        threading.Thread(target=self._guard, args=(target, *args), daemon=True).start()
        return {"ok": True}

    def _guard(self, target, *args) -> None:
        try:
            result = target(*args)
            with self._lock:
                self._job.update(running=False, result=result, fraction=1.0)
        except pipeline.ProjectError as error:
            with self._lock:
                self._job.update(running=False, error=str(error))
        except Exception as error:    # unexpected: show it rather than fail silently
            with self._lock:
                self._job.update(running=False, error=f"Unexpected problem: {error}",
                                 detail=traceback.format_exc())

    def _report(self, stage: str, fraction: float, message: str) -> None:
        with self._lock:
            self._job.update(stage=stage, fraction=float(fraction), message=message)

    def _forget_views(self) -> None:
        self._frame_cache.clear()
        self._viewed = None

    def run_analysis(self, project_path: str) -> dict:
        self._forget_views()
        return self._start(self._run, project_path)

    def _run(self, project_path: str) -> dict:
        pipeline.run(project_path, progress=self._report)
        self._report("Writing the report", 1.0, "")
        report.build(Path(project_path).parent / "results")
        return {"project": project_path}

    def start_demo(self, preset: str) -> dict:
        self._forget_views()
        return self._start(self._demo, preset)

    def _demo(self, preset: str) -> dict:
        folder = APP_DIR / "demos" / f"SYNTH_{preset}"
        project_path = folder / "project.json"
        if not (folder / "results" / "summary.json").exists():
            if not project_path.exists() and preset == "crack":
                from ..synth.general import GeneralSpec, make_general_test
                self._report("Creating the demo test (synthetic photos and logs)", 0.0, "Rendering a plate that cracks")
                make_general_test(folder, GeneralSpec(mode="opening", zone_angle_deg=70.0))
            elif not project_path.exists():
                make_demo(folder, DemoSpec(preset=preset),
                          progress=lambda f, m: self._report("Creating the demo test (synthetic photos and logs)", f, m))
            pipeline.run(project_path, progress=self._report)
            report.build(folder / "results")
        self._remember(str(project_path), json.loads(project_path.read_text()).get("name", preset))
        return {"project": str(project_path)}

    # --- results ----------------------------------------------------------------
    def results(self, project_path: str) -> dict:
        folder = Path(project_path).parent / "results"
        try:
            summary = json.loads((folder / "summary.json").read_text())
        except OSError:
            return {"ok": False, "error": "This analysis has not been run yet."}
        import pandas as pd
        per_photo = pd.read_csv(folder / "per_photo.csv")
        per_photo = per_photo.astype(object).where(per_photo.notna(), None).to_dict(orient="list")
        if "accuracy" not in summary:
            # Analysed before accuracy statements existed: work it out from the saved fields, once.
            try:
                a = self._analysis(project_path)
                project = pipeline.load_project(project_path)
                summary["accuracy"] = pipeline._clean(pipeline.measurement_accuracy(
                    folder, project, a.photo_paths, "video" if a.compressed else ""))
                (folder / "summary.json").write_text(json.dumps(summary, indent=1, default=pipeline._json_default))
            except (OSError, KeyError, ValueError):
                pass
        return {"ok": True, "summary": summary, "per_photo": per_photo}

    def frame(self, project_path: str, k: int, show_strain: bool = True, quantity: str = "gamma") -> dict:
        """Photo k with a quantity's overlay and the band centrelines in display pixels."""
        key = (project_path, int(k), bool(show_strain), quantity)
        if key in self._frame_cache:
            return self._frame_cache[key]
        # One frame at a time: overlapping requests (fast scrubbing, playback)
        # would otherwise compete for the CPU and all arrive late.
        with self._frame_lock:
            if key in self._frame_cache:
                return self._frame_cache[key]
            result = self._render_frame(self._analysis(project_path), int(k), bool(show_strain), quantity)
            self._frame_cache[key] = result
            return result

    def quantities(self) -> list[dict]:
        """What the viewer and the inspection tools can show."""
        return [{"key": q.key, "label": q.label, "unit": q.unit, "signed": q.signed, "percent": q.percent}
                for q in QUANTITIES.values()]

    def _analysis(self, project_path: str) -> _Analysis:
        """Results of one analysis, loaded once and kept while it is being viewed."""
        if self._viewed is None or self._viewed.project_path != project_path:
            self._viewed = _Analysis.load(project_path)
        return self._viewed

    @staticmethod
    def _render_frame(a: "_Analysis", k: int, show_strain: bool, quantity: str = "gamma") -> dict:
        raw = load_grayscale(a.photo_paths[k])
        photo = _to_8bit(a.undistort(raw) if a.undistort else raw)
        display, scale = _downscale(photo)
        rgb = np.repeat(display[..., None], 3, axis=2).astype(float)
        q = QUANTITIES.get(quantity, QUANTITIES["gamma"])
        if show_strain and k > 0:
            level = a.overlay_level(k, display.shape, scale, q.key)
            # Positive: red; negative: blue; strength by magnitude.
            red = np.array([235.0, 60.0, 40.0])
            blue = np.array([40.0, 100.0, 235.0])
            colour = np.where(level[..., None] >= 0, red, blue)
            alpha = (OVERLAY_ALPHA * np.abs(level) ** 0.8)[..., None]
            rgb = rgb * (1 - alpha) + colour * alpha
        # Bands: reference positions (mm) -> current positions -> display pixels.
        disp_bands = []
        for b in a.bands[k]:
            pts = np.array(b["points"], dtype=float)
            if len(pts) < 2:
                continue
            bux, buz = a.displacement_at(k, pts[:, 0], pts[:, 1])
            col, row = a.wall.to_image(pts[:, 0] + bux, pts[:, 1] + buz)
            disp_bands.append({"name": b["name"], "label": b["label"], "mode": b.get("mode", ""),
                               "width_mm": b["width_mm"], "flags": b["flags"],
                               "points": np.column_stack([col * scale, row * scale]).round(1).tolist()})
        return {"image": _png_data_url(np.clip(rgb, 0, 255).astype(np.uint8)), "width": display.shape[1],
                "height": display.shape[0], "bands": disp_bands, "push_mm": float(a.push_mm[k]),
                "gamma_max": a.scale("gamma"), "quantity": q.key, "quantity_label": q.label,
                "quantity_scale": a.scale(q.key), "signed": q.signed, "percent": q.percent, "unit": q.unit,
                "display_scale": scale, "accuracy_u95": a.typical_uncertainty(q.key, k)}

    # --- inspection tools (Vic-2D style) -----------------------------------------
    def probe(self, project_path: str, kind: str, points: list, k: int, quantity: str = "exx",
              display_scale: float = 1.0) -> dict:
        """Point probe, line profile or virtual extensometer at points clicked on photo k (display px).

        Clicked positions are mapped back to the material points in the first photo,
        so a probe follows the same material through the test (Lagrangian).
        """
        a = self._analysis(project_path)
        q = QUANTITIES.get(quantity, QUANTITIES["exx"])
        cols = np.array([p[0] for p in points], float) / display_scale
        rows = np.array([p[1] for p in points], float) / display_scale
        x_now, z_now = a.wall.to_wall(cols, rows)
        x_ref, z_ref = a.reference_of(int(k), x_now, z_now)
        axis = [float(v) for v in a.push_mm]
        frames = range(a.store.frames)
        clean = lambda v: [None if not np.isfinite(x) else float(x) for x in np.ravel(v)]  # noqa: E731
        if kind == "point":
            # Like a small strain gauge: the mean over the 3 x 3 grid points around the spot
            # (a single point carries the full measurement noise, ~3x more in testing).
            ox, oz = np.meshgrid(np.array([-1, 0, 1]) * a.dx, np.array([-1, 0, 1]) * a.dz)
            gx, gz = x_ref[0] + ox.ravel(), z_ref[0] + oz.ravel()
            series = [float(np.nanmean(a.value_at(q.key, j, gx, gz))) for j in frames]
            near_edge = a.near_edge(x_ref[:1], z_ref[:1])[0]
            u = [a.displacement_at(j, x_ref[:1], z_ref[:1]) for j in frames]
            # Neighbouring grid points share their noise (overlapping subsets), so the 3 x 3 mean
            # keeps the uncertainty of one point.
            u95 = [float(a.uncertainty_at(q.key, j, x_ref[:1], z_ref[:1])[0]) for j in frames]
            return {"ok": True, "kind": kind, "axis": axis, "quantity": q.label, "unit": q.unit, "percent": q.percent,
                    "values": clean(series), "u95": clean(u95), "ux": clean([v[0][0] for v in u]), "uz": clean([v[1][0] for v in u]),
                    "ux_u95": clean([a.uncertainty_at("ux", j, x_ref[:1], z_ref[:1])[0] for j in frames]),
                    "uz_u95": clean([a.uncertainty_at("uz", j, x_ref[:1], z_ref[:1])[0] for j in frames]),
                    "point_mm": [float(x_ref[0]), float(z_ref[0])], "near_edge": bool(near_edge)}
        if kind == "line":
            t = np.linspace(0, 1, LINE_SAMPLES)
            xs = x_ref[0] + t * (x_ref[1] - x_ref[0])
            zs = z_ref[0] + t * (z_ref[1] - z_ref[0])
            distance = t * float(np.hypot(x_ref[1] - x_ref[0], z_ref[1] - z_ref[0]))
            return {"ok": True, "kind": kind, "distance_mm": clean(distance), "quantity": q.label, "unit": q.unit,
                    "percent": q.percent, "values": clean(a.value_at(q.key, int(k), xs, zs)), "photo": int(k),
                    "u95": clean(a.uncertainty_at(q.key, int(k), xs, zs)),
                    "near_edge": [bool(v) for v in a.near_edge(xs, zs)]}
        if kind == "extensometer":
            lengths = []
            for j in frames:
                ux, uz = a.displacement_at(j, x_ref[:2], z_ref[:2])
                lengths.append(float(np.hypot(x_ref[1] + ux[1] - x_ref[0] - ux[0], z_ref[1] + uz[1] - z_ref[0] - uz[0])))
            l0 = float(np.hypot(x_ref[1] - x_ref[0], z_ref[1] - z_ref[0]))
            # The change in length is the difference of two displacements: their uncertainties add
            # in quadrature (each end: the larger of its two components).
            ends = [np.maximum(a.uncertainty_at("ux", j, x_ref[:2], z_ref[:2]), a.uncertainty_at("uz", j, x_ref[:2], z_ref[:2]))
                    for j in frames]
            length_u95 = np.array([float(np.hypot(e[0], e[1])) for e in ends])
            return {"ok": True, "kind": kind, "axis": axis, "gauge_mm": l0, "length_mm": clean(lengths),
                    "strain": clean(np.array(lengths) / l0 - 1.0), "length_u95_mm": clean(length_u95),
                    "strain_u95": clean(length_u95 / l0 if l0 > 0 else length_u95 * np.nan)}
        return {"ok": False, "error": f"unknown tool {kind}"}

    def export_fields(self, project_path: str, fmt: str = "csv") -> dict:
        """Every field of every photo, as CSV files or one MATLAB file, next to the results."""
        from ..export import export
        try:
            path = export(Path(project_path).parent / "results", fmt)
        except OSError as error:
            return {"ok": False, "error": str(error)}
        self.reveal(str(path))
        return {"ok": True, "path": str(path)}

    def save_probe(self, project_path: str, name: str, table: dict) -> dict:
        """A probe's numbers as a CSV file in results/probes/."""
        import pandas as pd
        folder = Path(project_path).parent / "results" / "probes"
        folder.mkdir(exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in name).strip() or "probe"
        path = folder / f"{safe}.csv"
        pd.DataFrame(table).to_csv(path, index=False)
        self.reveal(str(path))
        return {"ok": True, "path": str(path)}

    def accuracy(self, project_path: str) -> dict:
        """Known-answer comparison for synthetic tests (those with a truth.json)."""
        from ..validation.known_answers import compare
        try:
            return {"ok": True, **compare(Path(project_path).parent)}
        except FileNotFoundError:
            return {"ok": False}

    def open_report(self, project_path: str) -> dict:
        path = Path(project_path).parent / "results" / "report.html"
        if not path.exists():
            try:
                report.build(path.parent)
            except OSError as error:
                return {"ok": False, "error": str(error)}
        webbrowser.open(path.as_uri())
        return {"ok": True, "path": str(path)}

    def reveal(self, path: str) -> dict:
        import subprocess
        subprocess.run(["open", "-R", path], check=False)
        return {"ok": True}
