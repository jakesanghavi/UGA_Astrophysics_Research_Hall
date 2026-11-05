#!/usr/bin/env python3
"""Bar-grid of Earth-normalized GPP from production sweep JSON.

Discovers `16cpus_test_*.json` in a directory, groups them by
(run_mode, atmos, physics, resolution, scenario), and writes

    gpp_<mode>_<atmos>_<physics>[_<scenario>][_<res>][_scale]_<metric>.png

Metrics (--metric, default per_area and planet_total):
    cells         unweighted sum over land cells (no suffix; ignores planet size)
    per_area      GPP per m^2 of land, cells weighted by cos(latitude)  (_perarea)
    planet_total  per_area times the planet's land area                 (_total)

    python plot_veg_by_params.py
    python plot_veg_by_params.py --atmos n2 --mode mass_only
    python plot_veg_by_params.py --metric cells
    python plot_veg_by_params.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gpp_terms import GPP_METRIC_INDEX
from atmos_presets import (
    EARTH_REFERENCE_JSON,
    iter_sweep_json,
    resolve_atmos_type,
    resolve_physics_mode,
    scenario_tag,
    sweep_json_name,
)

VALUE_INDEX = 1
DEFAULT_METRICS = ("per_area", "planet_total")
METRIC_SUFFIX = {"cells": "", "per_area": "_perarea", "planet_total": "_total"}
METRIC_YLABEL = {
    "cells": "GPP / Earth",
    "per_area": "GPP per m$^2$ land / Earth",
    "planet_total": "Planet GPP / Earth",
}
METRIC_TITLE = {
    "cells": "Gross Primary Production, land-cell sum (Earth = 1)",
    "per_area": "Gross Primary Production per unit land area (Earth = 1)",
    "planet_total": "Gross Primary Production, whole planet (Earth = 1)",
}

_STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper.mplstyle")
if os.path.exists(_STYLE):
    plt.style.use(_STYLE)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=".", help="directory with sweep JSON")
    parser.add_argument("--outdir", default=None, help="PNG directory (default: --dir)")
    parser.add_argument("--atmos", default=None, help="n2/co2/mars/... ; omit to plot every mix found")
    parser.add_argument("--physics", default=None, help="earth/mars/other")
    parser.add_argument(
        "--mode",
        choices=("normal", "mass_only"),
        default=None,
        help="omit to plot every run mode found",
    )
    parser.add_argument("--resolution", default=None, help="T21/T42; omit for every resolution found")
    parser.add_argument(
        "--earth-ref",
        default=None,
        help=f"Earth baseline JSON (default: <dir>/{EARTH_REFERENCE_JSON})",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        choices=tuple(GPP_METRIC_INDEX),
        default=list(DEFAULT_METRICS),
        help="per_area / planet_total / cells (default: per_area planet_total)",
    )
    parser.add_argument(
        "--value-index",
        type=int,
        default=None,
        help="raw JSON entry index, overrides --metric (0 avg, 1 cell sum, "
             "4 per-area, 5 planet total)",
    )
    parser.add_argument("--list", action="store_true", help="print discovered files and exit")
    return parser.parse_args(argv)


def regime_key(rec):
    scenario = scenario_tag(rec.get("pressure_scaling", "fixed"),
                            rec.get("co2_scaling", "mixing_ratio"))
    return (rec["run_mode"], rec["atmos"], rec["physics"], rec["resolution"], scenario)


def regime_label(run_mode, atmos, physics, resolution="T21", scenario=""):
    mode = "mass_only" if run_mode == "mass_only" else "normal"
    bits = [mode, atmos, physics]
    if resolution and resolution != "T21":
        bits.append(resolution)
    if scenario:
        bits.append(scenario.lstrip("_").replace("_", " "))
    return " ".join(bits)


def plot_basename(run_mode, atmos, physics, resolution="T21", scale=None, scenario="",
                  metric="cells"):
    mode = "massonly" if run_mode == "mass_only" else "normal"
    name = f"gpp_{mode}_{atmos}_{physics}{scenario}"
    if resolution and resolution != "T21":
        name += f"_{resolution}"
    if scale:
        name += f"_{scale}"
    return name + METRIC_SUFFIX.get(metric, "") + ".png"


def discover(directory):
    groups = {}
    for rec in iter_sweep_json(directory):
        groups.setdefault(regime_key(rec), []).append(rec)
    for records in groups.values():
        records.sort(key=lambda rec: rec["mass"])
    return groups


def filter_groups(groups, args):
    atmos = resolve_atmos_type(args.atmos) if args.atmos else None
    physics = resolve_physics_mode(args.physics) if args.physics else None
    selected = {}
    for key, records in groups.items():
        run_mode, mix, phys, resolution, _scenario = key
        if args.mode and run_mode != args.mode:
            continue
        if atmos and mix != atmos:
            continue
        if physics and phys != physics:
            continue
        if args.resolution and resolution != args.resolution:
            continue
        selected[key] = records
    return selected


def load_regime(records):
    masses = []
    data = {}
    for rec in records:
        with open(rec["path"]) as handle:
            data[rec["mass"]] = json.load(handle)
        masses.append(rec["mass"])
    return masses, data


def first_value(payload, value_index):
    for by_au in payload.values():
        for entry in by_au.values():
            if len(entry) > value_index and entry[value_index]:
                return entry[value_index]
    return None


def load_earth_baseline(path, value_index):
    """Baseline from the Earth reference JSON, or None if it lacks this entry."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as handle:
        return first_value(json.load(handle), value_index)


def fallback_baseline(groups, key, value_index):
    """1 M_earth point of the fixed-pressure sweep matching `key`, else of `key` itself.

    Used when earth_reference.json predates the area metrics.
    """
    run_mode, atmos, physics, resolution, _scenario = key
    for candidate in ((run_mode, atmos, physics, resolution, ""), key):
        for rec in groups.get(candidate, []):
            if rec["mass"] != 1.0:
                continue
            with open(rec["path"]) as handle:
                base = first_value(json.load(handle), value_index)
            if base:
                return base, rec["name"]
    return None, None


def normalized_gpp(data, mass, mstar, au, norm, value_index):
    """Crashed (null) points plot as 0; entries without this metric are NaN (no bar)."""
    try:
        entry = data[mass][mstar][au]
    except (KeyError, TypeError):
        return 0.0
    if entry[0] is None:
        return 0.0
    if len(entry) <= value_index or entry[value_index] is None:
        return float("nan")
    return entry[value_index] / norm


def star_rows(data):
    found = set()
    for by_au in data.values():
        found.update(by_au.keys())
    return sorted(found, key=float)


def au_columns(data, rows):
    per_star = {}
    for mstar in rows:
        aus = []
        for by_au in data.values():
            if mstar in by_au:
                aus = sorted(by_au[mstar].keys(), key=float)
                break
        per_star[mstar] = aus
    return per_star


def plot_grid(data, masses, rows, norm, value_index, yscale="linear", sharey=False,
              title="", outfile="", annotate=False, metric="cells"):
    rows = rows or star_rows(data)
    per_star = au_columns(data, rows)
    ncols = max((len(aus) for aus in per_star.values()), default=1)
    nrows = max(len(rows), 1)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(max(4.0, 4.5 * ncols), max(3.0, 2.8 * nrows)),
        sharex=True, sharey=sharey, squeeze=False,
    )

    x = np.arange(len(masses))
    all_values = []
    global_max = -np.inf

    for i, mstar in enumerate(rows):
        au_keys = per_star[mstar]
        for j in range(ncols):
            ax = axes[i, j]
            if j >= len(au_keys):
                ax.axis("off")
                continue

            au = au_keys[j]
            values = np.array([
                normalized_gpp(data, mass, mstar, au, norm, value_index)
                for mass in masses
            ])
            values = np.clip(values, 1e-20, None)
            all_values.extend(values[np.isfinite(values)])
            global_max = max(global_max, np.nanmax(values, initial=-np.inf))

            bars = ax.bar(x, values)
            ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
            ax.set_title(f"M*={mstar}, AU={round(float(au), 2)}")
            if annotate:
                ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=5)
                ax.margins(y=0.12)

            if i == nrows - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(masses, rotation=90)
                ax.set_xlabel(r"Planet mass [$M_\oplus$]")
            if j == 0:
                ax.set_ylabel(METRIC_YLABEL.get(metric, "GPP / Earth"))
            if yscale == "log":
                ax.set_yscale("log")

    if sharey:
        all_values = np.array(all_values)
        threshold = 1e-15
        valid = all_values[all_values > threshold]
        ymin = valid.min() if valid.size else threshold
        ymax = global_max if global_max > threshold else threshold
        for row in axes:
            for ax in row:
                ax.set_ylim(ymin, ymax)

    heading = METRIC_TITLE.get(metric, "Gross Primary Production (Earth = 1)")
    fig.suptitle(f"{heading}\n{title}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(outfile)
    plt.close(fig)
    print(f"Wrote {outfile}")
    return outfile


def plot_regime(masses, data, run_mode, atmos, physics, resolution, norm,
                value_index, outdir, scenario="", metric="cells"):
    rows = star_rows(data)
    title = regime_label(run_mode, atmos, physics, resolution, scenario)
    written = []
    if run_mode == "mass_only":
        path = os.path.join(outdir, plot_basename(run_mode, atmos, physics, resolution,
                                                  scenario=scenario, metric=metric))
        written.append(plot_grid(
            data, masses, rows, norm, value_index,
            yscale="linear", title=title, outfile=path, annotate=True, metric=metric,
        ))
        return written

    for scale, yscale, sharey in (
        ("shared_linear", "linear", True),
        ("shared_log", "log", True),
        ("indep_linear", "linear", False),
    ):
        path = os.path.join(
            outdir, plot_basename(run_mode, atmos, physics, resolution, scale, scenario,
                                  metric),
        )
        written.append(plot_grid(
            data, masses, rows, norm, value_index,
            yscale=yscale, sharey=sharey, title=title, outfile=path, metric=metric,
        ))
    return written


def print_inventory(groups, file=None):
    stream = sys.stdout if file is None else file
    if not groups:
        print("No sweep JSON files found.", file=stream)
        return
    for key in sorted(groups):
        records = groups[key]
        print(regime_label(*key), file=stream)
        for rec in records:
            print(f"  {rec['mass']:g}  {rec['name']}", file=stream)


def expected_hint(args):
    if not (args.atmos and args.mode):
        return None
    try:
        return sweep_json_name(
            1.0,
            run_mode=args.mode,
            atmos_type=args.atmos,
            physics_mode=args.physics or "earth",
            resolution=args.resolution or "T21",
        )
    except ValueError:
        return None


def main(argv=None):
    args = parse_args(argv)
    directory = os.path.abspath(args.dir)
    outdir = os.path.abspath(args.outdir or directory)
    os.makedirs(outdir, exist_ok=True)

    found = discover(directory)
    groups = filter_groups(found, args)
    if args.list:
        print_inventory(groups)
        if not groups and found:
            print("Other regimes on disk:")
            print_inventory(found)
        return 0
    if not groups:
        print(f"No matching sweep JSON in {directory}", file=sys.stderr)
        hint = expected_hint(args)
        if hint:
            print(f"Example name: {hint}", file=sys.stderr)
        if found:
            print("Found on disk:", file=sys.stderr)
            print_inventory(found, file=sys.stderr)
        else:
            print("Legacy `_massonly2` / `_massonly` files are still read as n2 + earth.",
                  file=sys.stderr)
        return 1

    earth_ref = args.earth_ref or os.path.join(directory, EARTH_REFERENCE_JSON)
    if args.value_index is not None:
        plans = [("cells", args.value_index)]
    else:
        plans = [(metric, GPP_METRIC_INDEX[metric]) for metric in args.metric]

    announced = set()
    for metric, value_index in plans:
        ref_norm = load_earth_baseline(earth_ref, value_index)
        for key, records in sorted(groups.items()):
            run_mode, atmos, physics, resolution, scenario = key
            norm = ref_norm
            if norm is None:
                norm, source = fallback_baseline(found, key, value_index)
                if norm is None:
                    print(f"Warning: no {metric} baseline for {regime_label(*key)}; "
                          "plotting raw values. Run backfill_area_gpp.py?")
                    norm = 1.0
                elif (metric, source) not in announced:
                    announced.add((metric, source))
                    print(f"{metric}: {os.path.basename(earth_ref)} has no value; "
                          f"Earth = {source}")
            masses, data = load_regime(records)
            missing = [m for m in masses
                       if any(np.isnan(normalized_gpp(data, m, ms, au, 1.0, value_index))
                              for ms, by_au in data[m].items() for au in by_au)]
            if missing:
                print(f"Warning: {regime_label(*key)} has no {metric} value for "
                      f"masses {missing}; run backfill_area_gpp.py")
            plot_regime(
                masses, data, run_mode, atmos, physics, resolution,
                norm, value_index, outdir, scenario=scenario, metric=metric,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
