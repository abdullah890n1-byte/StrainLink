"""Training and test scenes for the failure-zone model, with exact answers (SYNTHETIC DATA).

A scene is a speckled specimen photographed before and after loading. The
movement between the two photos is built from:

- a smooth background deformation (stretch, shear, rotation, bending, and
  gentle uneven strain like a heterogeneous material),
- optionally a hole (with the Kirsch stress concentration around it),
- zero, one or two localisation zones: a crack (the sides move apart), a
  shear band (they slide) or a mix, entering from an edge and either
  crossing the specimen or ending at a tip where the jump fades to zero.

The background parts are the distractors: strain that is high but is not
a failure zone. The zones' jumps range from far below to far above what a
person would see in a strain map, so the model learns faint zones too.

Every scene knows its truth: which points lie in a zone ("zone corridor"),
which are too close to a zone to call either way ("don't care"), and which
are intact material.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..synth.speckle import SpeckleTexture, _pull_back

PIXEL_SAMPLES = 2
BACKGROUND_GREY = 30.0
# A point is "in the zone" within this distance of its centre line (px): about half a DIC subset,
# or the zone's own half-width when that is larger. Up to DONT_CARE_FACTOR times that it is not scored.
CORRIDOR_PX = 12.0
DONT_CARE_FACTOR = 2.0
# A zone point is labelled "zone" only where the zone's strain is at least this many times the
# measurement noise; fainter stretches are physically invisible in the measurement and are not scored.
DETECTABLE_SNR = 3.0


@dataclass
class Zone:
    start: np.ndarray        # entry point (px, reference photo), outside or on the specimen's edge
    direction: np.ndarray    # unit vector along the zone
    length: float            # px from the start to the tip
    jump: np.ndarray         # (u, v) px: the far side's movement relative to the near side
    width: float             # px over which the jump happens
    taper: float             # px over which the jump fades out before the tip
    mode: str                # "opening", "shear" or "mixed"

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]])

    def along_and_side(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = x - self.start[0], y - self.start[1]
        return dx * self.direction[0] + dy * self.direction[1], dx * self.normal[0] + dy * self.normal[1]

    def amount(self, along: np.ndarray) -> np.ndarray:
        """Share of the full jump at each position along the zone (0 beyond the tip)."""
        return np.clip((self.length - along) / max(self.taper, 1e-9), 0.0, 1.0)

    def displacement(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        along, side = self.along_and_side(x, y)
        step = 0.5 * (1.0 + np.tanh(2.0 * side / self.width)) - 0.5
        share = step * self.amount(along)
        return self.jump[0] * share, self.jump[1] * share


@dataclass
class Scene:
    shape: tuple[int, int]                     # rows, cols
    background: dict                            # parameters of the smooth deformation
    zones: list[Zone] = field(default_factory=list)
    hole: tuple[float, float, float] | None = None    # (col, row, radius) px
    noise_grey: float = 1.0
    blur_px: float = 0.0
    contrast: tuple[float, float] = (40.0, 210.0)
    grain_px: float = 2.0
    jpeg_quality: int | None = None             # saved as JPEG (compression artefacts), or lossless
    seed: int = 0

    def displacement(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = self.background
        cx, cy = self.shape[1] / 2, self.shape[0] / 2
        X, Y = x - cx, y - cy
        rot = np.radians(b["rotation_deg"])
        u = b["exx"] * X + b["exy"] * Y + (np.cos(rot) - 1) * X - np.sin(rot) * Y
        v = b["exy"] * X + b["eyy"] * Y + np.sin(rot) * X + (np.cos(rot) - 1) * Y
        k = b["curvature"]                      # bending about the horizontal centre line
        u = u + k * X * Y
        v = v - 0.5 * k * X**2
        for amp, kx, ky, phase in b["waves"]:    # gentle uneven strain (a heterogeneous material)
            u = u + amp * np.sin(kx * X + ky * Y + phase)
            v = v + amp * np.cos(kx * X - ky * Y + phase)
        if self.hole is not None:
            u_h, v_h = _kirsch(x, y, *self.hole, b["hole_strain"])
            u, v = u + u_h, v + v_h
        for zone in self.zones:
            u_z, v_z = zone.displacement(x, y)
            u, v = u + u_z, v + v_z
        return u, v

    def on_specimen(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.hole is None:
            return np.ones(np.shape(x), dtype=bool)
        hx, hy, r = self.hole
        return np.hypot(x - hx, y - hy) > r

    def truth(self, x: np.ndarray, y: np.ndarray, faint_below: float | None = None, subset_px: float = 31.0) -> np.ndarray:
        """Per reference point: 1 in a zone, 0 intact, -1 don't care (near a zone, or where its jump is ~0).

        faint_below: zone points whose strain (jump spread over the zone width, or over the DIC subset when
        the zone is narrower) is below this are "don't care" too: no measurement could show them.
        """
        label = np.zeros(np.shape(x), dtype=np.int8)
        for zone in self.zones:
            along, side = zone.along_and_side(x, y)
            corridor = max(CORRIDOR_PX, zone.width)
            reach = zone.amount(along)
            inside = (np.abs(side) <= corridor) & (reach > 0.25)
            near = (np.abs(side) <= DONT_CARE_FACTOR * corridor) & (along <= zone.length + DONT_CARE_FACTOR * corridor)
            label = np.where(near & (label == 0), -1, label)
            if faint_below is not None:
                strain = np.hypot(*zone.jump) * reach / max(zone.width, subset_px)
                faint = inside & (strain < faint_below)
                label = np.where(faint & (label != 1), -1, label)
                inside = inside & ~faint
            label = np.where(inside, 1, label)
        return label

    def jump_size(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Size of the zone's jump (px) at each point inside a zone corridor, 0 elsewhere."""
        size = np.zeros(np.shape(x))
        for zone in self.zones:
            along, side = zone.along_and_side(x, y)
            inside = np.abs(side) <= max(CORRIDOR_PX, zone.width)
            size = np.maximum(size, np.where(inside, np.hypot(*zone.jump) * zone.amount(along), 0.0))
        return size

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        """The photos before and after loading (float grey levels)."""
        from scipy import ndimage
        rows, cols = self.shape
        rng = np.random.default_rng(self.seed)
        texture = SpeckleTexture.granular(self.shape, grain_radius_px=self.grain_px, seed=self.seed)
        sub = (np.arange(PIXEL_SAMPLES) + 0.5) / PIXEL_SAMPLES - 0.5
        gy, gx = np.meshgrid((np.arange(rows)[:, None] + sub[None, :]).ravel(),
                             (np.arange(cols)[:, None] + sub[None, :]).ravel(), indexing="ij")
        photos = []
        lo, hi = self.contrast
        for loaded in (False, True):
            ref_x, ref_y = _pull_back(gx, gy, self.displacement) if loaded else (gx, gy)
            value = lo + (hi - lo) * texture.evaluate(ref_x, ref_y)
            value = np.where(self.on_specimen(ref_x, ref_y), value, BACKGROUND_GREY)
            image = value.reshape(rows, PIXEL_SAMPLES, cols, PIXEL_SAMPLES).mean(axis=(1, 3))
            if self.blur_px:
                image = ndimage.gaussian_filter(image, self.blur_px)
            image = image + rng.normal(0.0, self.noise_grey, image.shape)
            if self.jpeg_quality is not None:
                import cv2
                ok, data = cv2.imencode(".jpg", np.clip(np.round(image), 0, 255).astype(np.uint8),
                                        [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_quality)])
                image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE).astype(float)
            photos.append(image)
        return photos[0], photos[1]


def _kirsch(x, y, hx, hy, radius, strain, nu=0.3):
    """Displacement (px) around a hole under remote strain along x (plane stress, Kirsch)."""
    kappa = (3 - nu) / (1 + nu)
    mu_over_e = 1.0 / (2 * (1 + nu))
    X, Yup = x - hx, -(y - hy)
    # Inside the hole there is no material (the photo shows background there); hold the field at its
    # edge value so it stays finite and the photo can be rendered.
    r = np.maximum(np.hypot(X, Yup), radius)
    th = np.arctan2(Yup, X)
    s, a = strain, radius
    ur = s / (4 * mu_over_e) * (a**2 / r * (1 + (1 + kappa) * np.cos(2 * th)) - a**4 / r**3 * np.cos(2 * th))
    ut = s / (4 * mu_over_e) * ((1 - kappa) * a**2 / r - a**4 / r**3) * np.sin(2 * th)
    # Only the disturbance caused by the hole: the remote strain is part of the background.
    ux = ur * np.cos(th) - ut * np.sin(th)
    uy = ur * np.sin(th) + ut * np.cos(th)
    return ux, -uy


def random_scene(seed: int, shape: tuple[int, int] = (300, 400), zone_chance: float = 0.75) -> Scene:
    """A random scene: the distribution the model is trained (and first tested) on."""
    rng = np.random.default_rng(seed)
    rows, cols = shape
    # Mostly small strain; sometimes large (a soft material, or a late photo of a long test).
    reach = 0.03 if rng.random() < 0.15 else 0.006
    background = {
        "exx": rng.uniform(-reach, reach), "eyy": rng.uniform(-reach, reach), "exy": rng.uniform(-reach / 2, reach / 2),
        "rotation_deg": rng.uniform(-1.0, 1.0),
        "curvature": rng.choice([0.0, rng.uniform(-6e-5, 6e-5)], p=[0.6, 0.4]),
        "waves": [(rng.uniform(0.02, 0.25) * rng.choice([0, 1], p=[0.5, 0.5]), rng.uniform(0.004, 0.02) * rng.choice([-1, 1]),
                   rng.uniform(0.004, 0.02) * rng.choice([-1, 1]), rng.uniform(0, 2 * np.pi)) for _ in range(2)],
        "hole_strain": rng.uniform(0.001, 0.006),
    }
    hole = None
    if rng.random() < 0.2:
        r = rng.uniform(18, 45)
        hole = (rng.uniform(cols * 0.3, cols * 0.7), rng.uniform(rows * 0.3, rows * 0.7), r)
    zones = []
    count = 0 if rng.random() > zone_chance else (1 if rng.random() < 0.8 else 2)
    for _ in range(count):
        angle = rng.uniform(0, np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        # Enter from a random point on the photo's border, heading inwards.
        centre = np.array([cols / 2, rows / 2]) + rng.uniform(-0.25, 0.25, 2) * np.array([cols, rows])
        half_span = np.hypot(cols, rows)
        start = centre - direction * half_span * 0.5
        crossing = rng.random() < 0.55
        length = half_span * (1.2 if crossing else rng.uniform(0.35, 0.7))
        mode = rng.choice(["opening", "shear", "mixed"], p=[0.4, 0.4, 0.2])
        normal = np.array([-direction[1], direction[0]])
        if mode == "opening":
            jump_dir = normal
        elif mode == "shear":
            jump_dir = direction
        else:
            mix = rng.uniform(0.3, 0.7)
            jump_dir = np.sqrt(1 - mix) * normal + np.sqrt(mix) * direction
        size = float(np.exp(rng.uniform(np.log(0.003), np.log(3.0))))          # px, log-uniform
        zones.append(Zone(start=start, direction=direction, length=length, jump=jump_dir * size,
                          width=float(np.exp(rng.uniform(np.log(1.0), np.log(40.0)))),
                          taper=float(rng.uniform(15, 80)), mode=str(mode)))
    # Image quality from a good lab camera to a compressed video of natural soil texture.
    return Scene(shape=shape, background=background, zones=zones, hole=hole,
                 noise_grey=float(np.exp(rng.uniform(np.log(0.5), np.log(12.0)))),
                 blur_px=float(rng.choice([0.0, rng.uniform(0.3, 2.0)], p=[0.5, 0.5])),
                 contrast=(float(rng.uniform(20, 90)), float(rng.uniform(120, 230))),
                 grain_px=float(rng.uniform(1.3, 4.0)),
                 jpeg_quality=int(rng.integers(30, 96)) if rng.random() < 0.35 else None, seed=seed)
