#!/usr/bin/env python3
"""Add per-area and planet-total GPP to sweep JSON written before they existed.

Old entries are [avg, tot, startemp, flux]. This reads each grid point's
final-year NetCDF from its ExoPlaSim workdir and appends
[area_avg, planet_total] (see gpp_terms.land_gpp_summary). An entry is only
updated when the NetCDF's unweighted land mean matches the JSON's `avg`, so a
workdir overwritten by a different run is skipped, not mixed in.

    python backfill_area_gpp.py            # every sweep JSON in this directory
    python backfill_area_gpp.py --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from atmos_presets import (
    EARTH_REFERENCE_JSON,
    WORKDIR_PREFIX,
    grid_workdir_suffix,
    iter_sweep_json,
)
from gpp_terms import land_gpp_summary
from model_helpers import planet_bulk

MATCH_RTOL = 1e-4


def final_netcdf(workdir):
    paths = sorted(glob.glob(os.path.join(workdir, "MOST.*.nc")))
    return paths[-1] if paths else None


def summary_from_netcdf(path, mass):
    import netCDF4

    with netCDF4.Dataset(path) as data:
        gpp = data["veggpp"][:]
        lsm = data["lsm"][:]
        lat = data["lat"][:]
    return land_gpp_summary(gpp, lsm, lat, planet_bulk(mass)["radius"])


def backfill_file(path, workdir_for, mass, dry_run=False):
    """Returns (updated, skipped) messages for one JSON file."""
    with open(path) as handle:
        payload = json.load(handle)
    updated, skipped = [], []
    for ms, by_au in payload.items():
        for au, entry in by_au.items():
            label = f"{os.path.basename(path)} ({ms}, {au})"
            if len(entry) >= 6 or entry[0] is None:
                continue
            workdir = workdir_for(ms, au)
            nc_path = final_netcdf(workdir)
            if nc_path is None:
                skipped.append(f"{label}: no NetCDF in {workdir}")
                continue
            avg, _tot, area_avg, total = summary_from_netcdf(nc_path, mass)
            if abs(avg - entry[0]) > MATCH_RTOL * abs(entry[0]):
                skipped.append(f"{label}: {nc_path} is from a different run "
                               f"(avg {avg:.4g} vs {entry[0]:.4g})")
                continue
            by_au[au] = entry[:4] + [area_avg, total]
            updated.append(label)
    if updated and not dry_run:
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=4)
    return updated, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=".", help="directory with sweep JSON and workdirs")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args(argv)
    directory = os.path.abspath(args.dir)

    jobs = []
    for rec in iter_sweep_json(directory):
        def workdir_for(ms, au, rec=rec):
            return os.path.join(directory, WORKDIR_PREFIX + grid_workdir_suffix(
                rec["mass_token"], ms, au, rec["atmos"], rec["physics"], rec["resolution"],
                rec.get("pressure_scaling", "fixed"), rec.get("co2_scaling", "mixing_ratio"),
            ))
        jobs.append((rec["path"], workdir_for, rec["mass"]))

    ref_path = os.path.join(directory, EARTH_REFERENCE_JSON)
    if os.path.exists(ref_path):
        def ref_workdir(ms, au):
            return os.path.join(directory, WORKDIR_PREFIX + grid_workdir_suffix(
                1.0, ms, au, "n2", "earth", tag="ref_"))
        jobs.append((ref_path, ref_workdir, 1.0))

    n_updated = 0
    for path, workdir_for, mass in jobs:
        updated, skipped = backfill_file(path, workdir_for, mass, args.dry_run)
        n_updated += len(updated)
        for msg in updated:
            print(f"{'would update' if args.dry_run else 'updated'} {msg}")
        for msg in skipped:
            print(f"skipped {msg}")
    print(f"{n_updated} entries {'to update' if args.dry_run else 'updated'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
