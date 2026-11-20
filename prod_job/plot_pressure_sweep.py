#!/usr/bin/env python3
"""GPP and surface temperature vs surface pressure at fixed planet mass.

Reads a `run_gpp_diagnostics.py --axis pressure` bundle and writes

    <stem>_gpp.png    per-area land GPP relative to the 1 bar point
    <stem>_total.png  whole-planet GPP [Pg C/yr]
    <stem>_temp.png   global and land mean surface temperature

With --mass-sweep, the GPP plot also shows the mass-scaled-pressure sweep
(`16cpus_test_*_massonly_n2_pmass_co2mix.json`) at its own pressures, so the
pressure-only response can be compared with the one where gravity and radius
change too.

    python plot_pressure_sweep.py pressure_sweep.json --mass-sweep .
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atmos_presets import iter_sweep_json, total_pressure_bar
from gpp_terms import GPP_METRIC_INDEX

SURFACE = "#fcfcfb"
GRID = "#e6e5e0"
REF_LINE = "#b5b4ae"
SERIES = ("#2a78d6", "#eb6834")
MASS_SWEEP_FILE_TAG = "_massonly_n2_pmass_co2mix"


def load_pressure_records(path):
    with open(path) as handle:
        bundle = json.load(handle)
    records = [rec for rec in bundle["records"]
               if rec.get("p_target_bar") is not None and not rec.get("crashed")]
    return sorted(records, key=lambda rec: rec["p_target_bar"])


def reference_record(records, p_ref=1.0):
    return min(records, key=lambda rec: abs(rec["p_target_bar"] - p_ref))


def mass_sweep_points(directory):
    """(pressure_bar, per-area GPP, mass) for the co2mix mass sweep, if present."""
    import model_helpers as mh

    points = []
    for rec in iter_sweep_json(directory):
        if rec["file_tag"] != MASS_SWEEP_FILE_TAG:
            continue
        with open(rec["path"]) as handle:
            entry = json.load(handle)["1.0"]["1.0"]
        index = GPP_METRIC_INDEX["per_area"]
        if len(entry) <= index or entry[index] is None:
            continue
        params, _ = mh.base_atmosphere(rec["mass"], "n2", None, "mass", "mixing_ratio")
        points.append((total_pressure_bar(params), entry[index], rec["mass"]))
    return sorted(points)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Surface pressure [bar]")


def pressure_ticks(ax, pressures):
    ax.set_xticks(pressures)
    ax.set_xticklabels([f"{p:g}" for p in pressures])
    ax.set_xlim(0, max(pressures) * 1.05)


def plot_gpp(records, mass_points, outfile):
    ref = reference_record(records)
    pressures = [rec["p_target_bar"] for rec in records]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    style_axes(ax)
    ax.axhline(1.0, color=REF_LINE, lw=1, zorder=0)
    ax.plot(pressures, [rec["area_gpp"] / ref["area_gpp"] for rec in records],
            color=SERIES[0], lw=2, marker="o", ms=6, markeredgecolor=SURFACE,
            markeredgewidth=1.2, zorder=3, label=r"1 $M_\oplus$, pressure varied")
    if mass_points:
        mass_ref = min(mass_points, key=lambda point: abs(point[2] - 1.0))[1]
        ax.plot([p for p, _, _ in mass_points], [g / mass_ref for _, g, _ in mass_points],
                color=SERIES[1], lw=2, marker="s", ms=6, markeredgecolor=SURFACE,
                markeredgewidth=1.2, zorder=3,
                label="Mass varied, pressure scaled with mass (co2mix)")
        for p, g, mass in mass_points:
            ax.annotate(f"{mass:g}", (p, g / mass_ref), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=7.5, color="#52514e")
    pressure_ticks(ax, pressures)
    ax.set_ylabel(r"GPP per m$^2$ of land (1 bar, 1 $M_\oplus$ = 1)")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {outfile}")


def plot_total(records, outfile):
    pressures = [rec["p_target_bar"] for rec in records]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    style_axes(ax)
    ax.plot(pressures, [rec["planet_gpp_pgc_yr"] for rec in records],
            color=SERIES[0], lw=2, marker="o", ms=6, markeredgecolor=SURFACE,
            markeredgewidth=1.2, zorder=3, label=r"1 $M_\oplus$, pressure varied")
    pressure_ticks(ax, pressures)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r"Whole-planet GPP [Pg C yr$^{-1}$]")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {outfile}")


def plot_temperature(records, outfile):
    pressures = [rec["p_target_bar"] for rec in records]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    style_axes(ax)
    for key, label, color, marker in (
        ("ts_global_C", "Global mean", SERIES[0], "o"),
        ("ts_land_area_C", "Land mean", SERIES[1], "s"),
    ):
        ax.plot(pressures, [rec[key] for rec in records], color=color, lw=2,
                marker=marker, ms=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
                zorder=3, label=label)
    pressure_ticks(ax, pressures)
    ax.set_ylabel(r"Surface temperature [$^\circ$C]")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {outfile}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", help="run_gpp_diagnostics.py --axis pressure JSON")
    parser.add_argument("--mass-sweep", default=None,
                        help="directory with the co2mix mass-sweep JSON to overlay")
    parser.add_argument("--outdir", default=None, help="PNG directory (default: bundle's)")
    args = parser.parse_args(argv)

    records = load_pressure_records(args.bundle)
    if not records:
        print(f"No finished pressure-axis records in {args.bundle}", file=sys.stderr)
        return 1
    stem = os.path.splitext(os.path.basename(args.bundle))[0]
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.bundle))
    mass_points = mass_sweep_points(args.mass_sweep) if args.mass_sweep else []
    plot_gpp(records, mass_points, os.path.join(outdir, f"{stem}_gpp.png"))
    plot_total(records, os.path.join(outdir, f"{stem}_total.png"))
    plot_temperature(records, os.path.join(outdir, f"{stem}_temp.png"))
    for rec in records:
        print(f"  P={rec['p_target_bar']:g} bar  GPP/m2={rec['area_gpp']:.3e}  "
              f"GPP={rec['planet_gpp_pgc_yr']:.1f} Pg C/yr  "
              f"T_global={rec['ts_global_C']:.1f} C  T_land={rec['ts_land_area_C']:.1f} C")
    return 0


if __name__ == "__main__":
    sys.exit(main())
