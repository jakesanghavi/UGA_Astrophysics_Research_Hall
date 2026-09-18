#!/usr/bin/env python3
"""Plot land-mean SIMBA GPP factors from `gpp_diag.json`.

    python plot_gpp_diagnostics.py
    python plot_gpp_diagnostics.py --input gpp_diag.json --outdir gpp_diag_plots
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
from gpp_terms import LIGHT_FACTORS, attribution_vs_reference

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


def xy(rows, axis, field):
    key = axis_key(axis)
    xs, ys = [], []
    for rec in rows:
        if rec.get("crashed"):
            continue
        value = rec.get(field)
        if value is None:
            continue
        xs.append(rec[key])
        ys.append(value)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def plot_factors(by_atmos, axis, outdir):
    atmos_names = [name for name in ("n2", "evolved", "co2") if name in by_atmos]
    atmos_names += [name for name in by_atmos if name not in atmos_names]
    nrows = len(FACTOR_PANELS)
    ncols = max(len(atmos_names), 1)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(max(4.0, 3.6 * ncols), max(2.2 * nrows, 8.0)),
        sharex=True, squeeze=False,
    )
    for col, atmos in enumerate(atmos_names):
        rows = by_atmos[atmos]
        label = PRESETS.get(atmos, {}).get("label", atmos)
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
    fig, axes = plt.subplots(
        2, 4, figsize=(11.0, 5.4), sharex=True, squeeze=False
    )
    axes = axes.ravel()
    for ax, (field, title) in zip(axes, FACTOR_PANELS):
        for atmos, rows in by_atmos.items():
            xs, ys = xy(rows, axis, field)
            if not xs.size:
                continue
            ax.plot(xs, ys, marker="o", label=atmos)
        ax.set_title(title)
        ax.set_xlabel(axis_label(axis))
        if field in ("gpp", "gppl", "gppw"):
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    fig.suptitle("GPP terms overlaid by atmosphere knob")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, "gpp_terms_overlay.png")
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
        "{a} M={m:g}".format(a=rec["atmos"], m=rec[axis_key(axis)])
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="gpp_diag.json")
    parser.add_argument("--outdir", default="gpp_diag_plots")
    parser.add_argument("--axis", choices=("mass", "au", "mstar"), default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    bundle = load_bundle(args.input)
    records = bundle["records"]
    if not records:
        raise SystemExit(f"no records in {args.input}")
    axis = infer_axis(records, bundle.get("axis") or args.axis)
    os.makedirs(args.outdir, exist_ok=True)
    by_atmos = grouped(records)
    written = [
        plot_factors(by_atmos, axis, args.outdir),
        plot_overlay(by_atmos, axis, args.outdir),
        plot_attribution(records, pick_reference(records), axis, args.outdir),
    ]
    for path in written:
        if path:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
