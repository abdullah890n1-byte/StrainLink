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
from scipy.interpolate import griddata

from .. import __version__, pipeline, report
from ..calib import WallFrame
from ..dic import load_grayscale
from ..io.images import load_sequence
from ..io.sensors import load_sensors
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


class Api:
    def __init__(self):
        self._window = None
        self._job: dict = {"running": False, "stage": "", "fraction": 0.0, "message": "", "error": None, "result": None}
        self._lock = threading.Lock()
        self._frame_cache: dict = {}
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
        if not root.is_dir():
            return {"ok": False, "error": "That is not a folder."}
        from ..io.images import IMAGE_SUFFIXES
        candidates = [root] + [p for p in root.rglob("*") if p.is_dir() and len(p.relative_to(root).parts) <= SCAN_DEPTH
                               and not p.name.startswith(".") and p.name != "results"]
        counts = {d: sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES and not f.name.startswith("."))
                  for d in candidates}
        photos = max(counts, key=counts.get) if counts and max(counts.values()) >= 2 else None
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
            score = sum(r in roles for r in ("time", "force", "displacement")) + 0.1 * len(table.channels)
            if score > best:
                logger, best = f, score
        project = next((p for p in (root / "project.json", root / "StrainLink analysis" / "project.json") if p.exists()), None)
        return {"ok": True, "folder": str(root), "photos": str(photos) if photos else None,
                "photo_count": counts.get(photos, 0) if photos else 0,
                "logger": str(logger) if logger else None, "calibration": str(calibration) if calibration else None,
                "project": str(project) if project else None}

    def choose(self, kind: str, start: str = "") -> str | None:
        """Native dialog. kind: folder | sensor | calibration | project | save-project."""
        import webview
        types = {
            "sensor": ("Logger exports (*.csv;*.txt;*.tsv;*.xlsx;*.xls)", "All files (*.*)"),
            "calibration": ("Calibration run (*.csv;*.txt;*.xlsx)", "All files (*.*)"),
            "project": ("StrainLink project (*.json)",),
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

    def run_analysis(self, project_path: str) -> dict:
        self._frame_cache.clear()
        return self._start(self._run, project_path)

    def _run(self, project_path: str) -> dict:
        pipeline.run(project_path, progress=self._report)
        self._report("Writing the report", 1.0, "")
        report.build(Path(project_path).parent / "results")
        return {"project": project_path}

    def start_demo(self, preset: str) -> dict:
        self._frame_cache.clear()
        return self._start(self._demo, preset)

    def _demo(self, preset: str) -> dict:
        folder = APP_DIR / "demos" / f"SYNTH_{preset}"
        project_path = folder / "project.json"
        if not (folder / "results" / "summary.json").exists():
            if not project_path.exists():
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
        return {"ok": True, "summary": summary, "per_photo": per_photo}

    def frame(self, project_path: str, k: int, show_strain: bool = True) -> dict:
        """Photo k with the strain overlay and the band centrelines in display pixels."""
        key = (project_path, int(k), bool(show_strain))
        if key in self._frame_cache:
            return self._frame_cache[key]
        project = pipeline.load_project(project_path)
        folder = Path(project_path).parent
        fields = np.load(folder / "results" / "fields.npz")
        bands = json.loads((folder / "results" / "bands.json").read_text())[int(k)]
        cal = project["calibration"]
        frame = WallFrame(float(cal["mm_per_px"]), float(cal["toe_px"][0]), float(cal["toe_px"][1]),
                          float(cal["wall_height_mm"]))
        seq = load_sequence(folder / project["images"].get("folder", "images"), project["images"].get("interval_s"))
        photo = _to_8bit(load_grayscale(seq.paths[int(k)]))
        display, scale = _downscale(photo)
        rgb = np.repeat(display[..., None], 3, axis=2).astype(float)

        x, z = fields["x_mm"], fields["z_mm"]
        ux, uz, gamma = fields["ux"][k], fields["uz"][k], fields["gamma"][k]
        cur_col, cur_row = frame.to_image(x + np.nan_to_num(ux), z + np.nan_to_num(uz))
        gmax = float(np.nanpercentile(fields["gamma"][-1], 99)) if np.isfinite(fields["gamma"][-1]).any() else 1.0
        if show_strain and np.isfinite(gamma).any():
            ok = np.isfinite(gamma) & np.isfinite(ux)
            rows, cols = np.mgrid[0:display.shape[0], 0:display.shape[1]]
            grid_g = griddata((cur_row[ok] * scale, cur_col[ok] * scale), gamma[ok], (rows, cols), method="linear")
            level = np.clip(np.nan_to_num(grid_g / max(gmax, 1e-9)), 0, 1)
            colour = np.stack([235 * np.ones_like(level), 60 + 80 * (1 - level), 40 + 60 * (1 - level)], axis=-1)
            alpha = (OVERLAY_ALPHA * level ** 0.8)[..., None]
            rgb = rgb * (1 - alpha) + colour * alpha
        # Bands: reference positions (mm) -> current positions -> display pixels.
        disp_bands = []
        valid_pts = np.isfinite(ux) & np.isfinite(uz)
        for b in bands:
            pts = np.array(b["points"], dtype=float)
            if len(pts) < 2:
                continue
            bux = griddata((x[valid_pts], z[valid_pts]), ux[valid_pts], pts, method="linear")
            buz = griddata((x[valid_pts], z[valid_pts]), uz[valid_pts], pts, method="linear")
            col, row = frame.to_image(pts[:, 0] + np.nan_to_num(bux), pts[:, 1] + np.nan_to_num(buz))
            disp_bands.append({"name": b["name"], "label": b["label"], "width_mm": b["width_mm"], "flags": b["flags"],
                               "points": np.column_stack([col * scale, row * scale]).round(1).tolist()})
        result = {"image": _png_data_url(np.clip(rgb, 0, 255).astype(np.uint8)), "width": display.shape[1],
                  "height": display.shape[0], "bands": disp_bands, "push_mm": float(fields["push_mm"][k]),
                  "gamma_max": gmax}
        self._frame_cache[key] = result
        return result

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
