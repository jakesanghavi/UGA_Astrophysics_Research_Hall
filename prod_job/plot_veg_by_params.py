#!/usr/bin/env python3
"""Bar-grid of Earth-normalized GPP from production sweep JSON.

Discovers `16cpus_test_*.json` in a directory, groups them by
(run_mode, atmos, physics, resolution), and writes

    gpp_<mode>_<atmos>_<physics>[_<res>][_scale].png

    python plot_veg_by_params.py
    python plot_veg_by_params.py --atmos n2 --mode mass_only
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

from atmos_presets import (
    EARTH_REFERENCE_JSON,
    iter_sweep_json,
    resolve_atmos_type,
    resolve_physics_mode,
    sweep_json_name,
)

VALUE_INDEX = 1

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
        "--value-index",
        type=int,
        default=VALUE_INDEX,
        help="JSON entry index: 0 avg GPP, 1 tot GPP (default 1)",
    )
    parser.add_argument("--list", action="store_true", help="print discovered files and exit")
    return parser.parse_args(argv)


def regime_key(rec):
    return (rec["run_mode"], rec["atmos"], rec["physics"], rec["resolution"])


def regime_label(run_mode, atmos, physics, resolution="T21"):
    mode = "mass_only" if run_mode == "mass_only" else "normal"
    bits = [mode, atmos, physics]
    if resolution and resolution != "T21":
        bits.append(resolution)
    return " ".join(bits)


def plot_basename(run_mode, atmos, physics, resolution="T21", scale=None):
    mode = "massonly" if run_mode == "mass_only" else "normal"
    name = f"gpp_{mode}_{atmos}_{physics}"
    if resolution and resolution != "T21":
        name += f"_{resolution}"
    if scale:
        name += f"_{scale}"
    return name + ".png"


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
        run_mode, mix, phys, resolution = key
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


def load_earth_baseline(path, value_index):
    if not path or not os.path.exists(path):
        print(f"Warning: {path or EARTH_REFERENCE_JSON} not found; GPP will not be "
              "normalized to Earth (using 1.0).")
        return 1.0
    with open(path) as handle:
        ref = json.load(handle)
    for by_au in ref.values():
        for entry in by_au.values():
            base = entry[value_index]
            if base:
                return base
    print(f"Warning: no usable baseline in {path}; using 1.0.")
    return 1.0


def normalized_gpp(data, mass, mstar, au, norm, value_index):
    try:
        val = data[mass][mstar][au][value_index]
    except (KeyError, IndexError, TypeError):
        return 0.0
    if val is None:
        return 0.0
    return val / norm


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
              title="", outfile="", annotate=False):
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
            all_values.extend(values)
            global_max = max(global_max, values.max())

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
                ax.set_ylabel("GPP / Earth")
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

    fig.suptitle(f"Gross Primary Production (Earth = 1)\n{title}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(outfile)
    plt.close(fig)
    print(f"Wrote {outfile}")
    return outfile


def plot_regime(masses, data, run_mode, atmos, physics, resolution, norm,
                value_index, outdir):
    rows = star_rows(data)
    title = regime_label(run_mode, atmos, physics, resolution)
    written = []
    if run_mode == "mass_only":
        path = os.path.join(outdir, plot_basename(run_mode, atmos, physics, resolution))
        written.append(plot_grid(
            data, masses, rows, norm, value_index,
            yscale="linear", title=title, outfile=path, annotate=True,
        ))
        return written

    for scale, yscale, sharey in (
        ("shared_linear", "linear", True),
        ("shared_log", "log", True),
        ("indep_linear", "linear", False),
    ):
        path = os.path.join(
            outdir, plot_basename(run_mode, atmos, physics, resolution, scale),
        )
        written.append(plot_grid(
            data, masses, rows, norm, value_index,
            yscale=yscale, sharey=sharey, title=title, outfile=path,
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
    norm = load_earth_baseline(earth_ref, args.value_index)

    for (run_mode, atmos, physics, resolution), records in sorted(groups.items()):
        masses, data = load_regime(records)
        plot_regime(
            masses, data, run_mode, atmos, physics, resolution,
            norm, args.value_index, outdir,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
