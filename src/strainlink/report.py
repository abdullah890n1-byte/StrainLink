"""Shareable HTML report of an analysis.

One self-contained file (figures embedded as images, no internet needed)
that can be emailed, printed or attached to a paper. Figures are static on
purpose: the report is a record; interactive exploration happens in the app.
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import __version__  # noqa: E402

# Colours (consistent with the app).
INK = "#1d2330"
MUTED = "#6b7385"
ACCENT = "#2f6fed"
BAND_COLOURS = {"primary": "#e0452b", "radial": "#f59f00", "intermediate": "#12b886",
                "wall-interface": "#868e96", "unattached": "#7950f2"}
STRENGTH_COLOURS = {"strong": "#2b8a3e", "moderate": "#1971c2", "weak": "#868e96", "not supported": "#c92a2a"}
FIG_DPI = 110




# Page style shared by every StrainLink report.
STYLE = f""":root{{--ink:{INK};--muted:{MUTED};--accent:{ACCENT};--line:#e6e8ef;--bg:#ffffff;--soft:#f6f7fb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:32px 28px 64px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:17px;margin:34px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}}
.meta{{color:var(--muted);font-size:12.5px}} .synthetic{{margin:14px 0;padding:10px 14px;border-radius:10px;background:#fff4e6;color:#a44a00;font-weight:600;font-size:13px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}}
.card{{background:var(--soft);border-radius:12px;padding:12px 14px}} .card .k{{color:var(--muted);font-size:12px}} .card .v{{font-size:20px;font-weight:650;margin-top:2px}} .card .s{{color:var(--muted);font-size:11.5px}}
ul.flags li{{margin:4px 0}} .finding{{border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:10px 0}}
.badge{{display:inline-block;color:#fff;font-size:11px;font-weight:650;border-radius:20px;padding:1px 9px;text-transform:uppercase;letter-spacing:.03em}}
.ft{{margin:6px 0 2px;font-size:14.5px}} details summary{{cursor:pointer;color:var(--accent);font-size:12.5px}} .method{{color:var(--muted);font-size:12.5px}}
table{{border-collapse:collapse;font-size:12.5px;margin:6px 0}} td{{padding:3px 12px 3px 0;border-bottom:1px solid var(--line)}} .caveats{{color:#a44a00;font-size:12.5px}}
img{{max-width:100%}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}} figure{{margin:0}} figcaption{{color:var(--muted);font-size:12px;text-align:center}}
.foot{{margin-top:40px;color:var(--muted);font-size:12px}} @media print{{main{{padding:0}} details{{display:block}} details>summary{{display:none}}}}
"""


def findings_section(findings: list[dict]) -> str:
    """The graded findings with their evidence, as HTML."""
    out = ""
    for f in findings:
        colour = STRENGTH_COLOURS[f["strength"]]
        evidence = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in f["evidence"].items())
        caveats = "".join(f"<li>{html.escape(c)}</li>" for c in f["caveats"])
        out += (f'<div class="finding"><span class="badge" style="background:{colour}">{html.escape(f["strength"])}</span>'
                f'<div class="ft">{html.escape(f["statement"])}</div>'
                f'<details><summary>Evidence and method</summary><p class="method">{html.escape(f["method"])}</p>'
                f'<table>{evidence}</table><ul class="caveats">{caveats}</ul></details></div>')
    return out or '<p class="meta">No findings could be computed for this test.</p>'


def _surface_range(p: dict) -> str:
    lo, hi = p.get("surface_reached_range_mm") or (None, None)
    if lo is None or hi is None or not (hi - lo > 1.01):
        return "wall push"
    return f"range {lo:.1f}–{hi:.1f} mm (not yet confirmed)"

def _png(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def _num(value, digits=2, unit="") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}{(' ' + unit) if unit else ''}"


def _style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.tick_params(colors=MUTED)


def _force_figure(summary: dict) -> str:
    m = summary["mechanics"]
    push = np.array(m["push_mm"], dtype=float)
    force = np.array(m["passive_kn"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.plot(push, force, color=INK, lw=1.4)
    ax.set_xlabel("Wall push (mm)")
    ax.set_ylabel("Passive force (kN)")
    marks = [e for e in summary["correlation"]["events"] if e["source"] == "soil" and not e["key"].startswith("band-1.reaches")]
    for e in marks:
        ax.axvline(e["push_mm"], color=ACCENT, lw=0.8, ls="--")
        ax.text(e["push_mm"], ax.get_ylim()[1] * 0.98, " " + e["label"], rotation=90, va="top", fontsize=7, color=ACCENT)
    p = m["passive"]
    ax.plot([p["push_at_peak_mm"]], [p["peak"]], "o", color="#e0452b")
    _style_axes(ax)
    return _png(fig)


def _cells_figure(summary: dict, per_photo: dict) -> str | None:
    cells = [c for c in summary["mechanics"]["cells"]]
    if not cells:
        return None
    push = np.array(per_photo["push_mm"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    colours = plt.cm.viridis(np.linspace(0.1, 0.85, len(cells)))
    reach = {e["subject"]: e["push_mm"] for e in summary["correlation"]["events"] if e["key"].startswith("band-1.reaches")}
    for colour, c in zip(colours, cells):
        if c["name"] in per_photo:
            ax.plot(push, per_photo[c["name"]], color=colour, lw=1.3, label=f"{c['name']} ({c['depth_mm']:.0f} mm deep)")
        if c["name"] in reach:
            ax.axvline(reach[c["name"]], color=colour, lw=0.8, ls=":")
    ax.set_xlabel("Wall push (mm)")
    ax.set_ylabel("Earth pressure (kPa)")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title("Dotted lines: band-1's tip reaches the cell's level", fontsize=8, color=MUTED)
    _style_axes(ax)
    return _png(fig)


def _band_figure(summary: dict) -> str:
    p = summary["bands"]["primary"]
    push = np.array(p["delta_mm"], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    series = [("developed_length_mm", "Band-1 length (mm)"), ("lift_off_deg", "Lift-off angle (°)"),
              ("area_mm2", "Wedge area (mm²)")]
    for ax, (key, label) in zip(axes, series):
        values = np.array([np.nan if v is None else v for v in p[key]], dtype=float)
        ax.plot(push, values, "o-", ms=2.5, color=INK, lw=1.1)
        if key == "lift_off_deg":
            sigma = np.array([np.nan if v is None else v for v in p["lift_off_sigma_deg"]], dtype=float)
            ax.fill_between(push, values - 2 * sigma, values + 2 * sigma, color=ACCENT, alpha=0.15, lw=0)
        ax.set_xlabel("Wall push (mm)")
        ax.set_title(label, fontsize=9)
        _style_axes(ax)
    fig.tight_layout()
    return _png(fig)


def _strain_figures(results_dir: Path, summary: dict) -> list[tuple[str, str]]:
    fields = np.load(results_dir / "fields.npz")
    bands = json.loads((results_dir / "bands.json").read_text())
    push = fields["push_mm"]
    x, z, gamma = fields["x_mm"], fields["z_mm"], fields["gamma"]
    keys = {"band-1.onset": "Band-1 appears", "band-1.surface": "Band-1 reaches the surface",
            "force.peak": "Peak force", "force.plateau": "Force plateau"}
    moments = []
    for e in summary["correlation"]["events"]:
        if e["key"] in keys:
            moments.append((keys[e["key"]], int(np.argmin(np.abs(push - e["push_mm"])))))
    moments.append(("End of test", len(push) - 1))
    seen, figures = set(), []
    vmax = float(np.nanpercentile(gamma[-1], 99)) if np.isfinite(gamma[-1]).any() else 1.0
    for title, k in moments:
        if k in seen:
            continue
        seen.add(k)
        fig, ax = plt.subplots(figsize=(4.8, 2.9))
        ax.pcolormesh(x, z, gamma[k], cmap="magma_r", vmin=0, vmax=vmax, shading="auto")
        for b in bands[k]:
            pts = np.array(b["points"], dtype=float)
            ax.plot(pts[:, 0], pts[:, 1], color=BAND_COLOURS.get(b["label"], INK), lw=1.6)
            ax.text(pts[-1, 0], pts[-1, 1], " " + b["name"], fontsize=7, color=BAND_COLOURS.get(b["label"], INK))
        ax.set_aspect("equal")
        ax.set_xlabel("x from wall (mm)", fontsize=8)
        ax.set_ylabel("z above toe (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
        figures.append((f"{title} — push {push[k]:.1f} mm", _png(fig)))
    return figures


def build(results_dir: str | Path) -> Path:
    """Write results/report.html from a finished analysis and return its path."""
    results_dir = Path(results_dir)
    summary = json.loads((results_dir / "summary.json").read_text())
    if summary.get("kind") == "general":
        from .general.report import build_general
        return build_general(results_dir, summary)
    import pandas as pd
    per_photo = pd.read_csv(results_dir / "per_photo.csv").to_dict(orient="list")
    m, p = summary["mechanics"], summary["bands"]["primary"]
    last = lambda key: next((v for v in reversed(p[key]) if v is not None), None)  # noqa: E731

    cards = [
        ("Peak passive force", f"{_num(m['passive']['peak'], 2, 'kN')}", f"± {_num(m['passive']['peak_sigma'], 3)}"),
        ("Push at peak δp", f"{_num(m['passive']['push_at_peak_mm'], 1, 'mm')}",
         ("plateau " if m["passive"]["plateau"] else "") + f"range {_num(m['passive']['push_range_mm'][0], 1)}–{_num(m['passive']['push_range_mm'][1], 1)} mm"),
        ("Kp", _num(m["kp"], 2), f"± {_num(m['kp_sigma'], 3)}"),
        ("Band-1 appears at", _num(p["onset_mm"], 1, "mm"), "wall push"),
        ("Band-1 reaches surface at", _num(p["surface_reached_mm"], 1, "mm"), _surface_range(p)),
        ("Lift-off angle (end)", _num(last("lift_off_deg"), 1, "°"), f"± {_num(last('lift_off_sigma_deg'), 1)}°"),
        ("Impacted length (end)", _num(last("length_mm"), 0, "mm"), ""),
        ("Wedge area (end)", _num((last("area_mm2") or float('nan')) / 1e4, 2, "×10⁴ mm²"), ""),
        ("Surface heave (end)", _num(last("heave_mm"), 1, "mm"), ""),
        ("Mobilised wall friction δmob", _num(m["delta_mob_deg"], 1, "°"), "at uplift onset"),
        ("Pressure-cell thrust vs actuator", _num(100 * m["thrust_mismatch_fraction"] if m["thrust_mismatch_fraction"] is not None else None, 1, "%"), "difference at peak"),
        ("Clock offset", _num(summary["sync"]["clock_offset_s"], 1, "s"), f"± {_num(summary['sync']['offset_sigma_s'], 1)} s ({summary['sync']['source']})"),
    ]
    card_html = "".join(f'<div class="card"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div>'
                        f'<div class="s">{html.escape(s)}</div></div>' for k, v, s in cards)
    flags_html = "".join(f"<li>{html.escape(f)}</li>" for f in summary["flags"]) or "<li>Nothing flagged.</li>"
    findings_html = findings_section(summary["correlation"]["findings"])
    cells_png = _cells_figure(summary, per_photo)
    strain_html = "".join(f'<figure><img src="data:image/png;base64,{png}"><figcaption>{html.escape(t)}</figcaption></figure>'
                          for t, png in _strain_figures(results_dir, summary))
    synthetic = ('<div class="synthetic">SYNTHETIC DATA — generated test with known answers, not a real experiment.</div>'
                 if summary.get("synthetic") else "")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{html.escape(summary['project'])} — StrainLink report</title>
<style>
{STYLE}
</style></head><body><main>
<h1>{html.escape(summary['project'])}</h1>
<div class="meta">StrainLink {html.escape(summary['version'])} · analysed {html.escape(summary['finished'])} · {summary['frames']} photos · {summary['seconds']:.0f} s</div>
{synthetic}
<h2>Key results</h2><div class="cards">{card_html}</div>
<h2>Needs your review</h2><ul class="flags">{flags_html}</ul>
<h2>Findings (correlation engine)</h2>
<p class="meta">Graded by the strength of the evidence. Every statement is computed from the data; open “Evidence and method” for the numbers.</p>
{findings_html}
<h2>Force and pressures</h2>
<img src="data:image/png;base64,{_force_figure(summary)}">
{f'<img src="data:image/png;base64,{cells_png}">' if cells_png else ''}
<h2>Primary slip surface (band-1) over the test</h2>
<img src="data:image/png;base64,{_band_figure(summary)}">
<h2>Shear strain at key moments</h2><div class="grid">{strain_html}</div>
<h2>Methods</h2>
<p class="meta">Displacements by digital image correlation (IC-GN, ZNSSD, incremental Lagrangian tracking); maximum shear strain γmax = ε1 − ε2 (Hencky);
shear bands by multi-scale ridge detection with non-maximum suppression, junction-graph tracing and sub-grid centrelines;
band shape by BIC over straight, log-spiral and composite models with a 5 mm practical-straightness rule; lift-off angle = band direction over its first 50 mm;
clocks matched through the wall push; base friction removed using the calibration run; Kp = 2Pp/(γH²w).
Accuracy of each step is documented in StrainLink's validation records.</p>
<div class="foot">Generated by StrainLink {html.escape(__version__)}. Flags mark results that need an engineer's check; unflagged results are within the validated tolerances.</div>
</main></body></html>"""
    out = results_dir / "report.html"
    out.write_text(page)
    return out
