#!/usr/bin/env python3
"""Small-grid SIMBA GPP-term diagnostic.

Not a production sweep. Default is 3 atmosphere knobs × 3 planet masses at
1 Msun / 1 AU, one model year. Flip the knobs rather than swapping
`model_helpers_n2.py` / `model_helpers_co2.py`:

    evolved  Earth-like mix + H/He from evolve_atmosphere
    n2       Earth-like N2/O2, no H/He overlay
    co2      Venus-like CO2, no H/He overlay

Usage (from prod_job/, with the project venv):

    python run_gpp_diagnostics.py --dry-run
    python run_gpp_diagnostics.py
    python run_gpp_diagnostics.py --atmos n2 --masses 0.5,1 --years 1
    python run_gpp_diagnostics.py --axis au --atmos n2,co2 --masses 1 --years 1
    python plot_gpp_diagnostics.py

Then plot. This script does not launch the full MSTARS × HZ × MASS_RATIOS grid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from atmos_presets import ATMOS_TYPES
from veg_utils import calc_hz_percentiles

DEFAULT_MASSES = [0.25, 1.0, 2.0]
DEFAULT_MSTAR = 1.0
DEFAULT_AU = 1.0
DEFAULT_YEARS = 1
DEFAULT_OUTPUT = "gpp_diag.json"


def parse_float_list(text):
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_atmos_list(text):
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in ATMOS_TYPES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown atmos type(s) {unknown}; expected one of {list(ATMOS_TYPES)}"
        )
    return names


def build_tasks(args):
    """Return (atmos, mass, mstar, au) tuples for the requested tiny grid."""
    atmos_types = args.atmos
    masses = args.masses
    if args.axis == "mass":
        points = [(args.mstar, args.au)]
        masses = args.masses
    elif args.axis == "au":
        if args.hz:
            aus = list(calc_hz_percentiles(args.mstar))
        else:
            aus = args.aus
        points = [(args.mstar, au) for au in aus]
        if len(args.masses) != 1:
            print(f"note: --axis au uses one planet mass; taking {args.masses[0]:g}")
        masses = [args.masses[0]]
    elif args.axis == "mstar":
        points = [(mstar, args.au) for mstar in args.mstars]
        if len(args.masses) != 1:
            print(f"note: --axis mstar uses one planet mass; taking {args.masses[0]:g}")
        masses = [args.masses[0]]
    else:
        raise ValueError(f"unknown axis {args.axis}")

    tasks = []
    for atmos in atmos_types:
        for mass in masses:
            for mstar, au in points:
                tasks.append((atmos, mass, mstar, au))
    return tasks


def record_key(atmos, mass, mstar, au):
    return f"{atmos}|{mass}|{mstar}|{au}"


def load_existing(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    if isinstance(data, list):
        return data
    raise ValueError(f"{path} is not a diagnostic JSON list/object")


def index_records(records):
    return {
        record_key(rec["atmos"], rec["mass_ratio"], rec["mstar"], rec["au"]): rec
        for rec in records
        if rec.get("atmos") is not None
    }


def save_bundle(path, args, records):
    bundle = {
        "axis": args.axis,
        "years": args.years,
        "atmos": args.atmos,
        "records": records,
    }
    with open(path, "w") as handle:
        json.dump(bundle, handle, indent=2)
        handle.write("\n")


def workdir_suffix(atmos, mass, mstar, au):
    def safe(value):
        return str(value).replace(".", "p").replace("-", "m")
    return f"_diag_{atmos}_{safe(mass)}_{safe(mstar)}_{safe(au)}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atmos",
        type=parse_atmos_list,
        default=list(ATMOS_TYPES),
        help="comma-separated knobs: evolved,n2,co2 (default: all three)",
    )
    parser.add_argument(
        "--masses",
        type=parse_float_list,
        default=DEFAULT_MASSES,
        help="comma-separated planet masses in Mearth (default: 0.25,1,2)",
    )
    parser.add_argument(
        "--mstar",
        type=float,
        default=DEFAULT_MSTAR,
        help="stellar mass for --axis mass/au (default: 1.0)",
    )
    parser.add_argument(
        "--mstars",
        type=parse_float_list,
        default=[0.8, 1.0, 1.2],
        help="stellar masses for --axis mstar",
    )
    parser.add_argument(
        "--au",
        type=float,
        default=DEFAULT_AU,
        help="semi-major axis for --axis mass/mstar (default: 1.0)",
    )
    parser.add_argument(
        "--aus",
        type=parse_float_list,
        default=[0.8, 1.0, 1.2],
        help="AU values for --axis au when --hz is not set",
    )
    parser.add_argument(
        "--hz",
        action="store_true",
        help="with --axis au, use the three HZ percentiles of --mstar",
    )
    parser.add_argument(
        "--axis",
        choices=("mass", "au", "mstar"),
        default="mass",
        help="which coordinate to scan (default: mass at 1 Msun / 1 AU)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=DEFAULT_YEARS,
        help="ExoPlaSim years per point (default: 1, not the production N_YEARS)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="JSON bundle to write (default: gpp_diag.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the grid and exit without launching ExoPlaSim",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute points even if they already exist and did not crash",
    )
    return parser.parse_args(argv)


def print_plan(tasks, args):
    print(f"GPP diagnostic: {len(tasks)} ExoPlaSim run(s), {args.years} year(s) each")
    print(f"  axis={args.axis}  atmos={args.atmos}")
    print("  (atmos, mass, M*, AU)")
    for atmos, mass, mstar, au in tasks:
        print(f"    {atmos:8s}  {mass:g}  {mstar:g}  {au:g}")
    print(f"  output: {args.output}")
    print("This is a diagnostic grid, not the production sweep.")


def run_tasks(tasks, args):
    # Imported here so --dry-run does not need ExoPlaSim installed.
    import model_helpers as mh

    mh.N_YEARS = args.years
    records = [] if args.force else load_existing(args.output)
    by_key = index_records(records)
    pending = []
    for atmos, mass, mstar, au in tasks:
        key = record_key(atmos, mass, mstar, au)
        existing = by_key.get(key)
        if (
            not args.force
            and existing is not None
            and not existing.get("crashed")
            and existing.get("gpp") is not None
        ):
            continue
        pending.append((atmos, mass, mstar, au, workdir_suffix(atmos, mass, mstar, au)))

    if not pending:
        print("Nothing to run (all points already present).")
        save_bundle(args.output, args, list(by_key.values()))
        return

    def store(record):
        key = record_key(record["atmos"], record["mass_ratio"], record["mstar"], record["au"])
        by_key[key] = record
        save_bundle(args.output, args, list(by_key.values()))
        crashed = record.get("crashed")
        gpp = record.get("gpp")
        frac = record.get("frac_light_limited")
        print(
            f"  {record['atmos']:8s} M={record['mass_ratio']:g} "
            f"M*={record['mstar']:g} AU={record['au']:g}  "
            f"{'CRASH' if crashed else f'GPP={gpp}  light_frac={frac}'}"
        )

    workers = mh.WORKERS
    first = pending[0]
    rest = pending[1:]
    print(f"Running {len(pending)} point(s) with N_YEARS={mh.N_YEARS}, WORKERS={workers}")
    store(mh.diag_grid_point(first[1], first[2], first[3], mh.RESOLUTION, first[4], first[0]))

    if not rest:
        return

    if workers <= 1:
        for atmos, mass, mstar, au, suffix in rest:
            store(mh.diag_grid_point(mass, mstar, au, mh.RESOLUTION, suffix, atmos))
        return

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                mh.diag_grid_point, mass, mstar, au, mh.RESOLUTION, suffix, atmos
            ): (atmos, mass, mstar, au)
            for atmos, mass, mstar, au, suffix in rest
        }
        for fut in as_completed(future_map):
            atmos, mass, mstar, au = future_map[fut]
            try:
                store(fut.result())
            except Exception as exc:
                print(f"Error! ({atmos}, {mass}, {mstar}, {au}): {exc}")


def main(argv=None):
    args = parse_args(argv)
    tasks = build_tasks(args)
    print_plan(tasks, args)
    if args.dry_run:
        return 0
    run_tasks(tasks, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
