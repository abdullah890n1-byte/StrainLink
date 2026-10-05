"""The shareable HTML report for any test (kind "general")."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import __version__
from ..report import INK, STYLE, _num, _png, _style_axes, accuracy_section, findings_section
from .events import MODE_NAMES

MODE_COLOURS = {"opening": "#e8590c", "shear": "#7048e8", "mixed": "#1098ad", "": INK}


def _last(values):
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return finite[-1] if finite else None


def _series_figure(per_photo: pd.DataFrame, axis_label: str, columns: list[str], title: str) -> str | None:
    columns = [c for c in columns if c in per_photo and per_photo[c].notna().sum() > 1]
    if not columns:
        return None
    fig, axes = plt.subplots(len(columns), 1, figsize=(8.5, 1.6 * len(columns) + 0.6), sharex=True, squeeze=False)
    for ax, column in zip(axes[:, 0], columns):
        ax.plot(per_photo[axis_label], per_photo[column], color=INK, lw=1.4)
        ax.set_ylabel(column, fontsize=8)
        _style_axes(ax)
    axes[-1, 0].set_xlabel(axis_label, fontsize=9)
    axes[0, 0].set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    return _png(fig)


def _strain_maps(results_dir: Path, summary: dict) -> list[tuple[str, str]]:
    fields = np.load(results_dir / "fields.npz")
    bands = json.loads((results_dir / "bands.json").read_text())
    axis = fields["push_mm"]
    x, z, gamma = fields["x_mm"], fields["z_mm"], fields["gamma"]
    unit = summary["axis"]["unit"]
    moments = [(e["label"], int(np.nanargmin(np.abs(axis - e["push_mm"]))))
               for e in summary["correlation"]["events"] if e["key"].endswith((".onset", ".peak", ".drop"))]
    moments.append(("End of test", len(axis) - 1))
    vmax = float(np.nanpercentile(gamma[-1], 99)) if np.isfinite(gamma[-1]).any() else 1.0
    seen, figures = set(), []
    for title, k in moments:
        if k in seen:
            continue
        seen.add(k)
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        ax.pcolormesh(x, z, gamma[k], cmap="magma_r", vmin=0, vmax=vmax, shading="auto")
        for b in bands[k]:
            pts = np.array(b["points"], dtype=float)
            colour = MODE_COLOURS.get(b.get("mode", ""), INK)
            ax.plot(pts[:, 0], pts[:, 1], color=colour, lw=1.6)
            ax.text(pts[-1, 0], pts[-1, 1], " " + b["name"], fontsize=7, color=colour)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)", fontsize=8)
        ax.set_ylabel("z (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
        figures.append((f"{title} — photo {k + 1} ({axis[k]:.3g} {unit})", _png(fig)))
    return figures


def build_general(results_dir: Path, summary: dict) -> Path:
    per_photo = pd.read_csv(results_dir / "per_photo.csv")
    axis = summary["axis"]
    zones = summary.get("zones", [])
    cards = [
        ("Photos analysed", str(summary["frames"]), f"{summary['area']['points']} measurement points"),
        ("Object size", f"{summary['area']['size_mm'][0]:.1f} × {summary['area']['size_mm'][1]:.1f} mm", "measured area"),
        ("Peak shear strain γmax", _num(_last(per_photo.get("peak shear strain γmax", [])), 4), "end of test (99th percentile)"),
        ("Localisation zones", str(len(zones)),
         ", ".join(f"{z['name']} ({MODE_NAMES.get(z['mode'], 'zone')})" for z in zones) or "none found"),
    ]
    for z in zones[:3]:
        cards.append((f"{z['name']} appears at", _num(z["onset"], 3, axis["unit"]), axis["name"]))
    sync = summary["sync"]
    if "clock_offset_s" in sync:
        cards.append(("Clock offset (logger − photos)", _num(sync["clock_offset_s"], 2, "s"),
                      f"± {_num(sync.get('offset_sigma_s'), 2)} s ({sync['source']})"))
    card_html = "".join(f'<div class="card"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div>'
                        f'<div class="s">{html.escape(s)}</div></div>' for k, v, s in cards)
    flags_html = "".join(f"<li>{html.escape(f)}</li>" for f in summary["flags"]) or "<li>Nothing flagged.</li>"
    sensors_png = _series_figure(per_photo, axis["label"], summary.get("sensor_columns", []), "Sensor records at each photo")
    object_png = _series_figure(per_photo, axis["label"],
                                ["strain εxx", "strain εzz", "peak shear strain γmax", "peak major strain ε1"]
                                + [c for c in summary.get("object_columns", []) if c.startswith("Z")],
                                "What the object does")
    maps = "".join(f'<figure><img src="data:image/png;base64,{png}"><figcaption>{html.escape(t)}</figcaption></figure>'
                   for t, png in _strain_maps(results_dir, summary))
    synthetic = ('<div class="synthetic">SYNTHETIC DATA — generated test with known answers, not a real experiment.</div>'
                 if summary.get("synthetic") else "")
    img = lambda png: f'<img src="data:image/png;base64,{png}">' if png else ""  # noqa: E731
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{html.escape(summary['project'])} — StrainLink report</title>
<style>
{STYLE}
</style></head><body><main>
<h1>{html.escape(summary['project'])}</h1>
<div class="meta">StrainLink {html.escape(summary['version'])} · any-test mode · analysed {html.escape(summary['finished'])} · {summary['frames']} photos · {summary['seconds']:.0f} s</div>
{synthetic}
<h2>Key results</h2><div class="cards">{card_html}</div>
{accuracy_section(summary)}
<h2>Needs your review</h2><ul class="flags">{flags_html}</ul>
<h2>Findings (correlation engine)</h2>
<p class="meta">Graded by the strength of the evidence. Every statement is computed from the data; open “Evidence and method” for the numbers.</p>
{findings_section(summary["correlation"]["findings"])}
<h2>Over the test</h2>{img(object_png)}{img(sensors_png)}
<h2>Shear strain and localisation zones at key moments</h2>
<p class="meta">Zone colours: orange = crack (the sides open), violet = shear band (the sides slide), teal = mixed.</p>
<div class="grid">{maps}</div>
<h2>Methods</h2>
<p class="meta">The object is found automatically as the textured, in-focus area. Displacements by digital image correlation (IC-GN, ZNSSD,
incremental Lagrangian tracking); lost-correlation gaps refilled from the displacement on both sides; strains: Green–Lagrange components and
Hencky principal strains, γmax = ε1 − ε2; localisation zones by multi-scale ridge detection on γmax and tracked through the test; a zone is a
crack or a shear band according to the displacement jump across it (opening vs slip); the object's strain is the median over the measured area.
Photos are matched to the logger by fitting the DIC stretch of the object to the logger displacement. Findings: event timing coincidences and
screening of frame-to-frame changes with an exact circular-shift permutation test and false-discovery control.</p>
<div class="foot">Generated by StrainLink {html.escape(__version__)}. Flags mark results that need an engineer's check.</div>
</main></body></html>"""
    out = results_dir / "report.html"
    out.write_text(page)
    return out
