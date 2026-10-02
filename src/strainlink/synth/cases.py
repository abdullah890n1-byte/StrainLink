"""Standard DIC tasks with exact answers (SYNTHETIC DATA): the jobs engineers use Vic-2D for.

Each case is a speckled specimen deformed by an analytical displacement
field u(X, Y) (px, image axes, X right, Y down) over a few photos,
optionally with a hole, image-quality effects (noise, blur, lighting drift,
low contrast, compression) and a file format. The true displacement and
strain are known everywhere, so StrainLink's fields can be scored exactly.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage

from .speckle import SpeckleTexture, _pull_back

Field = Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]]   # (X, Y, load 0..1) -> (u, v) px
Region = Callable[[np.ndarray, np.ndarray], np.ndarray]                             # (X, Y) -> bool (on the specimen)

BACKGROUND = 30.0
PIXEL_SAMPLES = 2


@dataclass
class Case:
    name: str
    title: str
    field: Field
    shape: tuple[int, int] = (400, 600)          # rows, cols
    photos: int = 8
    mm_per_px: float = 0.05
    region: Region | None = None                 # specimen outline (None: the whole photo)
    noise_grey: float = 1.0
    contrast: tuple[float, float] = (40.0, 210.0)
    blur_px: float = 0.0                         # Gaussian blur (defocus)
    lighting_drift: float = 0.0                  # brightness changes by this fraction from first to last photo
    jpeg_quality: int | None = None
    file_format: str = "tif8"                    # tif8 | tif16 | png | jpg | bmp
    grain_px: float = 2.0
    seed: int = 1
    meta: dict = field(default_factory=dict)     # analytical parameters for the task checks


def render_case(case: Case, folder: Path) -> Path:
    """Write the photos and a project file (any-object mode); return the project path."""
    folder = Path(folder)
    (folder / "photos").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(case.seed)
    texture = SpeckleTexture.granular(case.shape, grain_radius_px=case.grain_px, seed=case.seed)
    rows, cols = case.shape
    sub = (np.arange(PIXEL_SAMPLES) + 0.5) / PIXEL_SAMPLES - 0.5
    gy, gx = np.meshgrid((np.arange(rows)[:, None] + sub[None, :]).ravel(),
                         (np.arange(cols)[:, None] + sub[None, :]).ravel(), indexing="ij")
    lo, hi = case.contrast
    for k in range(case.photos):
        load = k / (case.photos - 1)
        ref_x, ref_y = (gx, gy) if k == 0 else _pull_back(gx, gy, lambda X, Y: case.field(X, Y, load))
        value = lo + (hi - lo) * texture.evaluate(ref_x, ref_y)
        if case.region is not None:
            value = np.where(case.region(ref_x, ref_y), value, BACKGROUND)
        image = value.reshape(rows, PIXEL_SAMPLES, cols, PIXEL_SAMPLES).mean(axis=(1, 3))
        image = image * (1.0 + case.lighting_drift * load)
        if case.blur_px:
            image = ndimage.gaussian_filter(image, case.blur_px)
        image = image + rng.normal(0.0, case.noise_grey, image.shape)
        _write(folder / "photos", k, image, case)
    project = {"kind": "general", "name": f"{case.title} (SYNTHETIC DATA)", "synthetic": True,
               "images": {"folder": "photos", "interval_s": 1.0}, "calibration": {"mm_per_px": case.mm_per_px}}
    (folder / "project.json").write_text(json.dumps(project, indent=2))
    (folder / "case.json").write_text(json.dumps({"name": case.name, "title": case.title, "meta": case.meta,
                                                  "photos": case.photos, "shape": case.shape,
                                                  "mm_per_px": case.mm_per_px}, indent=1))
    return folder / "project.json"


def _write(folder: Path, k: int, image: np.ndarray, case: Case) -> None:
    fmt = case.file_format
    if fmt == "tif16":
        tifffile.imwrite(folder / f"img_{k:03d}.tif", np.clip(np.round(image * 257.0), 0, 65535).astype(np.uint16))
        return
    eight = np.clip(np.round(image), 0, 255).astype(np.uint8)
    if case.jpeg_quality or fmt == "jpg":
        buffer = io.BytesIO()
        Image.fromarray(eight).save(buffer, format="JPEG", quality=case.jpeg_quality or 90)
        if fmt == "jpg":
            (folder / f"img_{k:03d}.jpg").write_bytes(buffer.getvalue())
            return
        eight = np.asarray(Image.open(io.BytesIO(buffer.getvalue())))
    suffix = {"tif8": "tif", "png": "png", "bmp": "bmp"}.get(fmt, "tif")
    if suffix == "tif":
        tifffile.imwrite(folder / f"img_{k:03d}.tif", eight)
    else:
        Image.fromarray(eight).save(folder / f"img_{k:03d}.{suffix}")


# --- analytical fields (X, Y px; load 0..1) ---------------------------------------------

def translation(du: float, dv: float) -> Field:
    return lambda X, Y, t: (np.full_like(X, du * t), np.full_like(Y, dv * t))


def rotation(degrees: float, centre: tuple[float, float]) -> Field:
    def f(X, Y, t):
        a = np.radians(degrees * t)
        x, y = X - centre[0], Y - centre[1]
        return x * np.cos(a) - y * np.sin(a) - x, x * np.sin(a) + y * np.cos(a) - y
    return f


def uniaxial(strain: float, poisson: float, centre: tuple[float, float]) -> Field:
    """Engineering strain `strain` along X, -poisson*strain along Y (tension or compression)."""
    return lambda X, Y, t: (strain * t * (X - centre[0]), -poisson * strain * t * (Y - centre[1]))


def simple_shear(gamma: float, centre: tuple[float, float]) -> Field:
    return lambda X, Y, t: (gamma * t * (Y - centre[1]), np.zeros_like(Y))


def isotropic(strain: float, centre: tuple[float, float]) -> Field:
    return lambda X, Y, t: (strain * t * (X - centre[0]), strain * t * (Y - centre[1]))


def bending(curvature_per_px: float, neutral_row: float, centre_col: float) -> Field:
    """Pure bending (plane sections, small rotations): u = -k (Y - y0)(X - x0)... with axial strain = k * (Y - y0)."""
    def f(X, Y, t):
        k = curvature_per_px * t
        y = Y - neutral_row
        x = X - centre_col
        return k * x * y, -0.5 * k * x**2
    return f


def plate_with_hole(stress_over_e: float, poisson: float, radius_px: float, centre: tuple[float, float]) -> Field:
    """Kirsch solution (plane stress), uniaxial remote stress along X, as displacement (px)."""
    nu = poisson
    kappa = (3 - nu) / (1 + nu)
    mu_over_e = 1.0 / (2 * (1 + nu))

    def f(X, Y, t):
        x, y = X - centre[0], -(Y - centre[1])          # y up for the polar formulas
        r = np.maximum(np.hypot(x, y), 1e-9)
        th = np.arctan2(y, x)
        a, s = radius_px, stress_over_e * t
        # Kirsch displacements, scaled by s = sigma/E: sigma/(4 mu) with mu = E/(2(1+nu)). Checked: far field exx = s,
        # eyy = -nu s; hoop strain at the hole edge 90 deg from the load = 3 s (the concentration factor).
        ur = s / (4 * mu_over_e) * (r * ((kappa - 1) / 2 + np.cos(2 * th)) + a**2 / r * (1 + (1 + kappa) * np.cos(2 * th))
                                    - a**4 / r**3 * np.cos(2 * th))
        ut = s / (4 * mu_over_e) * ((1 - kappa) * a**2 / r - r - a**4 / r**3) * np.sin(2 * th)
        ux = ur * np.cos(th) - ut * np.sin(th)
        uy = ur * np.sin(th) + ut * np.cos(th)
        return ux, -uy
    return f


def large_tension(strain: float, poisson_like: float, centre: tuple[float, float]) -> Field:
    """Large uniaxial stretch (rubber-like): stretch λ = 1 + strain along X, λ^-ν across."""
    def f(X, Y, t):
        lam = 1 + strain * t
        return (lam - 1) * (X - centre[0]), (lam ** (-poisson_like) - 1) * (Y - centre[1])
    return f


def true_strain(field_fn: Field, X: np.ndarray, Y: np.ndarray, load: float, h: float = 0.25) -> dict[str, np.ndarray]:
    """Green-Lagrange strain of the analytical field at reference points (image axes; exy sign for y down)."""
    def grad(i):
        up = field_fn(X + h, Y, load)[i] - field_fn(X - h, Y, load)[i]
        vp = field_fn(X, Y + h, load)[i] - field_fn(X, Y - h, load)[i]
        return up / (2 * h), vp / (2 * h)
    dudx, dudy = grad(0)
    dvdx, dvdy = grad(1)
    exx = dudx + 0.5 * (dudx**2 + dvdx**2)
    eyy = dvdy + 0.5 * (dudy**2 + dvdy**2)
    exy = 0.5 * (dudy + dvdx + dudx * dudy + dvdx * dvdy)
    return {"exx": exx, "eyy": eyy, "exy": exy}
