#!/usr/bin/env python3
"""Plot land-mean SIMBA GPP factors from `gpp_diag.json`.

    python plot_gpp_diagnostics.py
    python plot_gpp_diagnostics.py --input gpp_diag.json --outdir gpp_diag_plots

Writes light-factor panels, combined light/water/min rates, a water budget
(prescribed P and plant CO2 vs leftover in vegppw), and a preset-knob
comparison. venus_surface / mars_surface are attached as crash-shaped stubs
when missing so P / plant CO2 / β / column knobs still plot; GPP rate
charts skip those points.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atmos_presets import PRESETS
from gpp_terms import (
    LIGHT_FACTORS,
    RATE_FIELDS,
    SURFACE_STUB_ATMOS,
    WATER_BUDGET_TERMS,
    attach_surface_preset_records,
    attribution_vs_reference,
    backfill_preset_fields,
    water_budget_vs_reference,
)

_STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper.mplstyle")
if os.path.exists(_STYLE):
    plt.style.use(_STYLE)
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.titleweight"] = "normal"

FACTOR_PANELS = [
    ("gpp", "GPP (land mean)"),
    ("gppl", r"$GPP_{light}$"),
    ("gppw", r"$GPP_{water}$"),
    ("beta", r"$\beta(CO_2)$"),
    ("fT", r"$f(T_{sfc})$"),
    ("fPAR", "fPAR / cover"),
    ("SWdn", r"$SW\downarrow$ [W m$^{-2}$]"),
    ("frac_light_limited", "fraction light-limited"),
]

RATE_STYLES = (
    ("gppl", r"$GPP_{light}$", dict(ls="--", marker="o", color="steelblue")),
    ("gppw", r"$GPP_{water}$", dict(ls=":", marker="s", color="darkorange")),
    ("gpp", r"$GPP=\min$", dict(ls="-", marker="D", color="black", lw=1.8)),
)

WATER_BUDGET_LABELS = {
    "p_total_bar": r"$P$",
    "co2veg_ppmv": r"plant $CO_2$",
    "gppw_climate": "climate leftover",
}
WATER_BUDGET_COLORS = {
    "p_total_bar": "slategray",
    "co2veg_ppmv": "goldenrod",
    "gppw_climate": "seagreen",
}

PRESET_KNOB_PANELS = [
    ("p_total_bar", r"$P$ [bar]", True),
    ("pCO2_bar", r"$p\mathrm{CO}_2$ [bar]", True),
    ("co2_ppmv", r"air $\mathrm{CO}_2$ [ppmv]", True),
    ("co2veg_ppmv", r"plant $\mathrm{CO}_2$ [ppmv]", False),
    ("beta", r"$\beta(\mathrm{CO}_2)$", False),
    ("timestep_p_factor", r"timestep $P$ factor", False),
    ("modeltop", "modeltop [mbar]", True),
    ("mmw", r"mmw [g mol$^{-1}$]", False),
]

# Earth and the two surface-P presets: same marker family, Earth is green X.
PRESET_COMPARE_STYLES = {
    "n2": dict(color="green", marker="x", linestyle="none",
               markersize=9, markeredgewidth=1.8, zorder=5),
    "venus_surface": dict(color="crimson", marker="x", linestyle="none",
                          markersize=9, markeredgewidth=1.8, zorder=4),
    "mars_surface": dict(color="royalblue", marker="x", linestyle="none",
                         markersize=9, markeredgewidth=1.8, zorder=4),
}


def load_bundle(path):
    with open(path, "r") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return {"axis": None, "records": data}
    return data


def infer_axis(records, hinted=None):
    if hinted:
        return hinted
    masses = {rec["mass_ratio"] for rec in records}
    aus = {rec["au"] for rec in records}
    mstars = {rec["mstar"] for rec in records}
    if len(masses) > 1:
        return "mass"
    if len(aus) > 1:
        return "au"
    if len(mstars) > 1:
        return "mstar"
    return "mass"


def axis_key(axis):
    return {"mass": "mass_ratio", "au": "au", "mstar": "mstar"}[axis]


def axis_label(axis):
    return {
        "mass": r"Planet mass [$M_\oplus$]",
        "au": "Semi-major axis [AU]",
        "mstar": r"Stellar mass [$M_\odot$]",
    }[axis]


def grouped(records):
    by_atmos = {}
    for rec in records:
        by_atmos.setdefault(rec["atmos"], []).append(rec)
    for atmos, rows in by_atmos.items():
        rows.sort(key=lambda rec: (rec["mass_ratio"], rec["mstar"], rec["au"]))
    return by_atmos


def atmos_order(by_atmos):
    preferred = ("n2", "co2", "mars", "evolved", "venus_surface", "mars_surface")
    names = [name for name in preferred if name in by_atmos]
    names += [name for name in by_atmos if name not in names]
    return names


def atmos_label(atmos):
    return PRESETS.get(atmos, {}).get("label", atmos)


def knob_label(atmos):
    if atmos == "n2":
        return "Earth"
    return atmos_label(atmos)


def atmos_with_gpp(by_atmos):
    """Atmospheres that have at least one live GPP point. Stubs stay off rate charts."""
    names = []
    for atmos in atmos_order(by_atmos):
        rows = by_atmos[atmos]
        if any((not rec.get("crashed")) and rec.get("gpp") is not None for rec in rows):
            names.append(atmos)
    return names


def pick_reference(records):
    """Prefer 1 Me / 1 Msun / 1 AU / n2; else 1 Me on n2; else first live n2; else first live."""
    def live(rec):
        return (not rec.get("crashed")) and rec.get("gpp") is not None

    for rec in records:
        if live(rec) and rec["atmos"] == "n2" and rec["mass_ratio"] == 1.0 and rec["mstar"] == 1.0 and rec["au"] == 1.0:
            return rec
    for rec in records:
        if live(rec) and rec["atmos"] == "n2" and rec["mass_ratio"] == 1.0:
            return rec
    for rec in records:
        if live(rec) and rec["atmos"] == "n2":
            return rec
    for rec in records:
        if live(rec):
            return rec
    return None


def xy(rows, axis, field, include_crashed=False):
    key = axis_key(axis)
    xs, ys = [], []
    for rec in rows:
        if rec.get("crashed") and not include_crashed:
            continue
        value = rec.get(field)
        if value is None:
            continue
        xs.append(rec[key])
        ys.append(value)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def xy_live_crash(rows, axis, field):
    """Live points and crashed/preset-only points as separate series."""
    key = axis_key(axis)
    live_x, live_y, crash_x, crash_y = [], [], [], []
    for rec in rows:
        value = rec.get(field)
        if value is None:
            continue
        if rec.get("crashed"):
            crash_x.append(rec[key])
            crash_y.append(value)
        else:
            live_x.append(rec[key])
            live_y.append(value)
    return (
        np.array(live_x, dtype=float), np.array(live_y, dtype=float),
        np.array(crash_x, dtype=float), np.array(crash_y, dtype=float),
    )


def plot_factors(by_atmos, axis, outdir):
    atmos_names = atmos_with_gpp(by_atmos)
    if not atmos_names:
        return None
    nrows = len(FACTOR_PANELS)
    ncols = max(len(atmos_names), 1)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(max(4.0, 3.6 * ncols), max(2.2 * nrows, 8.0)),
        sharex=True, squeeze=False,
    )
    for col, atmos in enumerate(atmos_names):
        rows = by_atmos[atmos]
        label = atmos_label(atmos)
        for row, (field, title) in enumerate(FACTOR_PANELS):
            ax = axes[row, col]
            xs, ys = xy(rows, axis, field)
            if xs.size:
                ax.plot(xs, ys, marker="o")
            ax.set_title(f"{label}: {title}" if row == 0 else title)
            if row == nrows - 1:
                ax.set_xlabel(axis_label(axis))
            if field in ("gpp", "gppl", "gppw") and ys.size and np.nanmax(ys) > 0:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.suptitle("SIMBA GPP terms (land mean)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(outdir, "gpp_terms_by_atmos.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_overlay(by_atmos, axis, outdir):
    """Each factor on one panel, one line per atmosphere."""
    atmos_names = atmos_with_gpp(by_atmos)
    if not atmos_names:
        return None
    fig, axes = plt.subplots(
        2, 4, figsize=(11.0, 5.4), sharex=True, squeeze=False
    )
    axes = axes.ravel()
    for ax, (field, title) in zip(axes, FACTOR_PANELS):
        for atmos in atmos_names:
            xs, ys = xy(by_atmos[atmos], axis, field)
            if not xs.size:
                continue
            ax.plot(xs, ys, marker="o", label=atmos_label(atmos))
        ax.set_title(title)
        ax.set_xlabel(axis_label(axis))
        if field in ("gpp", "gppl", "gppw"):
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.suptitle("GPP terms overlaid by atmosphere", y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    path = os.path.join(outdir, "gpp_terms_overlay.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_gpp_ratio(by_atmos, axis, outdir):
    """GPP / n2 GPP at the same mass (or AU / M*). n2 is a flat 1."""
    if "n2" not in atmos_with_gpp(by_atmos):
        return None
    key = axis_key(axis)
    n2_map = {}
    for rec in by_atmos["n2"]:
        if rec.get("crashed") or rec.get("gpp") in (None, 0.0):
            continue
        n2_map[rec[key]] = rec["gpp"]
    if not n2_map:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for atmos in atmos_with_gpp(by_atmos):
        xs, ys = [], []
        for rec in by_atmos[atmos]:
            if rec.get("crashed") or rec.get("gpp") is None:
                continue
            ref = n2_map.get(rec[key])
            if not ref:
                continue
            xs.append(rec[key])
            ys.append(rec["gpp"] / ref)
        if xs:
            ax.plot(xs, ys, marker="o", label=atmos_label(atmos))
    ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel(axis_label(axis))
    ax.set_ylabel("GPP / Earthlike (n2) at same x")
    ax.set_title("Land-mean GPP relative to Earthlike mix")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "gpp_ratio_vs_earth.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_attribution(records, reference, axis, outdir):
    if reference is None:
        return None
    live = [rec for rec in records if not rec.get("crashed") and rec.get("gpp") is not None]
    if not live:
        return None
    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(live) + 2), 3.8))
    x = np.arange(len(live))
    bottoms_pos = np.zeros(len(live))
    bottoms_neg = np.zeros(len(live))
    colors = ["steelblue", "darkorange", "seagreen", "firebrick"]
    for color, name in zip(colors, LIGHT_FACTORS):
        values = []
        for rec in live:
            values.append(attribution_vs_reference(rec, reference)["dln"][name] or 0.0)
        values = np.array(values)
        pos = np.where(values >= 0.0, values, 0.0)
        neg = np.where(values < 0.0, values, 0.0)
        ax.bar(x, pos, bottom=bottoms_pos, color=color, label=name)
        ax.bar(x, neg, bottom=bottoms_neg, color=color)
        bottoms_pos += pos
        bottoms_neg += neg
    labels = [
        "{a}/{p} M={m:g}".format(
            a=rec["atmos"], p=rec.get("physics") or "earth", m=rec[axis_key(axis)]
        )
        for rec in live
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_ylabel(r"$\Delta \ln$ vs reference")
    ax.set_title(
        "Light-factor attribution vs "
        f"{reference['atmos']} M={reference['mass_ratio']:g}, "
        f"M*={reference['mstar']:g}, AU={reference['au']:g}"
    )
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "gpp_dln_attribution.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_rates_by_atmos(by_atmos, axis, outdir):
    """GPP_light, GPP_water, and min on one panel per atmosphere."""
    atmos_names = atmos_with_gpp(by_atmos)
    if not atmos_names:
        return None
    ncols = max(len(atmos_names), 1)
    fig, axes = plt.subplots(
        1, ncols,
        figsize=(max(4.2, 3.8 * ncols), 3.6),
        sharey=False, squeeze=False,
    )
    for col, atmos in enumerate(atmos_names):
        ax = axes[0, col]
        rows = by_atmos[atmos]
        for field, title, style in RATE_STYLES:
            xs, ys = xy(rows, axis, field)
            if xs.size:
                ax.plot(xs, ys, label=title, **style)
        ax.set_title(atmos_label(atmos))
        ax.set_xlabel(axis_label(axis))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if col == 0:
            ax.set_ylabel(r"GPP [kg C m$^{-2}$ s$^{-1}$]")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(r"Light-limited, water-limited, and $GPP=\min$ (land mean)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, "gpp_rates_by_atmos.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_rate_regime_bars(by_atmos, outdir, masses=(1.0, 1.5)):
    """Grouped bars of light / water / min GPP by atmosphere at selected masses."""
    present = sorted({
        rec["mass_ratio"]
        for rows in by_atmos.values()
        for rec in rows
        if (not rec.get("crashed")) and rec.get("gpp") is not None
    })
    masses = [m for m in masses if m in present] or present
    if not masses:
        return None

    def rate_at(atmos, mass, field):
        for rec in by_atmos.get(atmos, []):
            if rec.get("crashed") or rec.get(field) is None:
                continue
            if rec.get("mass_ratio") == mass:
                return rec[field]
        return None

    atmos_names = atmos_with_gpp(by_atmos)
    if not atmos_names:
        return None
    nmass = len(masses)
    fig, axes = plt.subplots(
        1, nmass, figsize=(max(5.0, 3.2 * nmass), 3.8), sharey=True, squeeze=False,
    )
    x = np.arange(len(atmos_names), dtype=float)
    width = 0.8 / len(RATE_FIELDS)
    for col, mass in enumerate(masses):
        ax = axes[0, col]
        positive = []
        for i, (field, title, style) in enumerate(RATE_STYLES):
            ys = [rate_at(atmos, mass, field) for atmos in atmos_names]
            ys = [float("nan") if v is None else v for v in ys]
            positive.extend(v for v in ys if v == v and v > 0)
            offset = (i - 0.5 * (len(RATE_FIELDS) - 1)) * width
            ax.bar(
                x + offset, ys, width, label=title,
                color=style["color"], edgecolor="0.2", linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([atmos_label(a) for a in atmos_names], rotation=30, ha="right")
        ax.set_title(rf"${mass:g}\,M_\oplus$")
        if positive and max(positive) / min(positive) > 50:
            ax.set_yscale("log")
        else:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if col == 0:
            ax.set_ylabel(r"GPP [kg C m$^{-2}$ s$^{-1}$]")
            ax.legend(frameon=False, fontsize=8)
        ax.margins(y=0.14)
    fig.suptitle(r"Light, water, and $GPP=\min$ by atmosphere")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, "gpp_rate_regime_bars.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _water_budget_rows(records, axis):
    """Rows with P, including crashed / preset-only (leftover stays None)."""
    preferred = {name: i for i, name in enumerate(
        ("n2", "co2", "mars", "evolved", "venus_surface", "mars_surface")
    )}
    key = axis_key(axis)
    rows = [rec for rec in records if rec.get("p_total_bar") is not None]
    rows.sort(key=lambda rec: (
        preferred.get(rec.get("atmos"), 99),
        rec.get(key) if rec.get(key) is not None else 0.0,
        rec.get("atmos") or "",
    ))
    return rows


def plot_water_attribution(records, reference, axis, outdir):
    """Stacked Δln(GPP_water) = Δln P + Δln plant CO2 + climate leftover."""
    if reference is None:
        return None
    rows = _water_budget_rows(records, axis)
    if not rows:
        return None
    budgets = [water_budget_vs_reference(rec, reference) for rec in rows]
    if all(b["dln"][name] is None for b in budgets for name in WATER_BUDGET_TERMS):
        return None
    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(rows) + 2), 3.8))
    x = np.arange(len(rows))
    bottoms_pos = np.zeros(len(rows))
    bottoms_neg = np.zeros(len(rows))
    for name in WATER_BUDGET_TERMS:
        values = np.array([
            0.0 if b["dln"][name] is None else b["dln"][name] for b in budgets
        ])
        pos = np.where(values >= 0.0, values, 0.0)
        neg = np.where(values < 0.0, values, 0.0)
        ax.bar(x, pos, bottom=bottoms_pos, color=WATER_BUDGET_COLORS[name],
               label=WATER_BUDGET_LABELS[name])
        ax.bar(x, neg, bottom=bottoms_neg, color=WATER_BUDGET_COLORS[name])
        bottoms_pos += pos
        bottoms_neg += neg
    labels = []
    for rec in rows:
        tag = "{a}/{p} M={m:g}".format(
            a=rec["atmos"], p=rec.get("physics") or "earth", m=rec[axis_key(axis)]
        )
        if rec.get("crashed"):
            tag += "*"
        labels.append(tag)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_ylabel(r"$\Delta \ln GPP_{water}$ vs reference")
    ax.set_title(
        "Water budget vs "
        f"{reference['atmos']} M={reference['mass_ratio']:g}, "
        f"M*={reference['mstar']:g}, AU={reference['au']:g}"
        "\n(P and plant CO2 from config; leftover is ET/VPD/cover;"
        " * crashed / preset-only, leftover unknown)"
    )
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "gpp_water_dln_attribution.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_preset_knobs(by_atmos, axis, outdir):
    """Prescribed mix/column knobs vs Earth (green X) and surface-P presets."""
    atmos_names = atmos_order(by_atmos)
    if not atmos_names:
        return None
    fig, axes = plt.subplots(
        2, 4, figsize=(12.0, 5.6), sharex=True, squeeze=False
    )
    axes = axes.ravel()
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color") or ["C0"]
    others = [name for name in atmos_names if name not in PRESET_COMPARE_STYLES]
    compare = [name for name in ("n2", "venus_surface", "mars_surface") if name in by_atmos]
    for ax, (field, title, log_y) in zip(axes, PRESET_KNOB_PANELS):
        for i, atmos in enumerate(others):
            color = colors[i % len(colors)]
            live_x, live_y, crash_x, crash_y = xy_live_crash(
                by_atmos[atmos], axis, field
            )
            label = atmos_label(atmos)
            if live_x.size:
                ax.plot(live_x, live_y, marker="o", color=color, label=label)
            if crash_x.size:
                crash_label = None if live_x.size else label
                ax.plot(
                    crash_x, crash_y, ls="none", marker="x", ms=8,
                    color=color, label=crash_label,
                )
        for atmos in compare:
            xs, ys = xy(by_atmos[atmos], axis, field, include_crashed=True)
            if not xs.size:
                continue
            ax.plot(xs, ys, label=knob_label(atmos), **PRESET_COMPARE_STYLES[atmos])
        ax.set_title(title)
        ax.set_xlabel(axis_label(axis))
        if log_y:
            ax.set_yscale("log")
    handles, labels = [], []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        for handle, name in zip(h, lab):
            if name not in labels:
                handles.append(handle)
                labels.append(name)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6),
                   bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.suptitle(
        "Prescribed GPP knobs vs Earth\n"
        "Earth = green X; Venus 92 bar / Mars 6.36 mbar = X; "
        "1-bar modeltop is Model default",
        y=1.08,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    path = os.path.join(outdir, "gpp_preset_knobs.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_gpp_regime_bars(by_atmos, outdir, masses=(1.0, 1.5)):
    """Grouped bars of land-mean GPP by atmosphere at selected planet masses.

    Heights are GPP / Earthlike GPP at 1 M⊕ when that point exists.
    """
    present = sorted({
        rec["mass_ratio"]
        for rows in by_atmos.values()
        for rec in rows
        if (not rec.get("crashed")) and rec.get("gpp") is not None
    })
    masses = [m for m in masses if m in present] or present
    if not masses:
        return None

    def gpp_at(atmos, mass):
        for rec in by_atmos.get(atmos, []):
            if rec.get("crashed") or rec.get("gpp") is None:
                continue
            if rec.get("mass_ratio") == mass:
                return rec["gpp"]
        return None

    earth_1 = gpp_at("n2", 1.0)
    atmos_names = atmos_with_gpp(by_atmos)
    if not atmos_names:
        return None
    x = np.arange(len(masses), dtype=float)
    width = 0.8 / max(len(atmos_names), 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, atmos in enumerate(atmos_names):
        ys = []
        for mass in masses:
            value = gpp_at(atmos, mass)
            if value is None:
                ys.append(float("nan"))
            elif earth_1:
                ys.append(value / earth_1)
            else:
                ys.append(value)
        offset = (i - 0.5 * (len(atmos_names) - 1)) * width
        bars = ax.bar(x + offset, ys, width, label=atmos_label(atmos))
        ax.bar_label(bars, fmt="%.2f" if earth_1 else "%.1e", padding=2, fontsize=8)
    if earth_1:
        ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
        ax.set_ylabel("GPP / Earthlike 1 $M_\\oplus$")
    else:
        ax.set_ylabel("Land-mean GPP")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m:g}" for m in masses])
    ax.set_xlabel(r"Planet mass [$M_\oplus$]")
    ax.set_title("GPP by atmosphere")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.margins(y=0.14)
    fig.tight_layout()
    path = os.path.join(outdir, "gpp_regime_bars.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="gpp_diag.json")
    parser.add_argument("--outdir", default="gpp_diag_plots")
    parser.add_argument("--axis", choices=("mass", "au", "mstar"), default=None)
    parser.add_argument(
        "--only-bars",
        action="store_true",
        help="write only gpp_regime_bars.png",
    )
    parser.add_argument(
        "--no-surface-stubs",
        action="store_true",
        help="do not attach venus_surface / mars_surface preset-only records",
    )
    return parser.parse_args(argv)


def prepare_records(records, attach_stubs=True):
    prepared = [backfill_preset_fields(dict(rec)) for rec in records]
    # Always put Earth on the knobs comparison; surface-P stubs are optional.
    prepared = attach_surface_preset_records(prepared, atmos=("n2",))
    if attach_stubs:
        prepared = attach_surface_preset_records(prepared, atmos=SURFACE_STUB_ATMOS)
    return prepared


def main(argv=None):
    args = parse_args(argv)
    bundle = load_bundle(args.input)
    records = bundle["records"]
    if not records:
        raise SystemExit(f"no records in {args.input}")
    records = prepare_records(records, attach_stubs=not args.no_surface_stubs)
    axis = infer_axis(records, bundle.get("axis") or args.axis)
    os.makedirs(args.outdir, exist_ok=True)
    by_atmos = grouped(records)
    if args.only_bars:
        written = [plot_gpp_regime_bars(by_atmos, args.outdir)]
    else:
        ref = pick_reference(records)
        written = [
            plot_factors(by_atmos, axis, args.outdir),
            plot_overlay(by_atmos, axis, args.outdir),
            plot_rates_by_atmos(by_atmos, axis, args.outdir),
            plot_rate_regime_bars(by_atmos, args.outdir),
            plot_gpp_ratio(by_atmos, axis, args.outdir),
            plot_gpp_regime_bars(by_atmos, args.outdir),
            plot_attribution(records, ref, axis, args.outdir),
            plot_water_attribution(records, ref, axis, args.outdir),
            plot_preset_knobs(by_atmos, axis, args.outdir),
        ]
    for path in written:
        if path:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
