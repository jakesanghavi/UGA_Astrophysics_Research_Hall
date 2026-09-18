#!/usr/bin/env python3
"""Unit tests for SIMBA GPP-term reconstruction and the tiny diagnostic grid.

Run from prod_job/:

    python test_gpp_terms.py
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest

import numpy as np

import atmos_presets as ap
import gpp_terms as gt
import plot_gpp_diagnostics as plotter
import run_gpp_diagnostics as runner


class TestBeta(unittest.TestCase):
    def test_docs_vs_fortran_at_low_co2(self):
        # Docs form would clamp at 1; Fortran drops below 1.
        beta = gt.beta_co2(330.0)
        self.assertAlmostEqual(beta, 1.0 + 0.3 * math.log(330.0 / 360.0), places=12)
        self.assertLess(beta, 1.0)

    def test_reference_is_one(self):
        self.assertAlmostEqual(gt.beta_co2(360.0), 1.0, places=12)

    def test_zero_co2(self):
        self.assertEqual(gt.beta_co2(0.0), 0.0)

    def test_shuts_off_near_13_ppm(self):
        # 1 + 0.3 ln(C/360) = 0 => C/360 = exp(-1/0.3)
        cutoff = 360.0 * math.exp(-1.0 / 0.3)
        self.assertEqual(gt.beta_co2(cutoff * 0.5), 0.0)
        self.assertGreater(gt.beta_co2(cutoff * 1.5), 0.0)

    def test_venuslike_is_a_large_lever(self):
        ppmv = ap.co2_ppmv_from_params(ap.VENUSLIKE_GASES)
        beta = gt.beta_co2(ppmv)
        self.assertGreater(ppmv, 9.0e5)
        self.assertGreater(beta, 3.0)


class TestLimitationFunctions(unittest.TestCase):
    def test_temperature_switch(self):
        self.assertEqual(float(gt.f_temperature(272.15)), 0.0)
        self.assertAlmostEqual(float(gt.f_temperature(273.15 + 2.5)), 0.5, places=12)
        self.assertEqual(float(gt.f_temperature(273.15 + 10.0)), 1.0)

    def test_fpar_beer(self):
        self.assertAlmostEqual(float(gt.f_par(0.0)), 0.0, places=12)
        self.assertAlmostEqual(float(gt.f_par(math.log(2.0) / 0.5)), 0.5, places=12)

    def test_reconstruction_identity(self):
        ts = np.array([[273.15 + 10.0, 273.15 - 1.0]])
        lai = np.array([[2.0, 0.0]])
        rss = np.array([[200.0, 200.0]])
        ssru = np.array([[50.0, 50.0]])
        beta = 1.1
        rec = gt.gpp_light(beta, gt.f_temperature(ts), gt.f_par(lai), gt.sw_down(rss, ssru))
        expected_live = 3.4e-10 * 1.1 * 1.0 * (1.0 - math.exp(-0.5 * 2.0)) * 250.0
        self.assertAlmostEqual(float(rec[0, 0]), expected_live, places=16)
        self.assertEqual(float(rec[0, 1]), 0.0)  # frozen


class TestSummarize(unittest.TestCase):
    def _toy_fields(self):
        land = np.array([[1.0, 1.0], [0.0, 1.0]])
        ts = np.array([[283.15, 263.15], [283.15, 283.15]])
        lai = np.array([[1.0, 1.0], [1.0, 0.0]])
        rss = np.array([[100.0, 100.0], [100.0, 100.0]])
        ssru = np.array([[20.0, 20.0], [20.0, 20.0]])
        fT = gt.f_temperature(ts)
        fPAR = gt.f_par(lai)
        sw = gt.sw_down(rss, ssru)
        beta = gt.beta_co2(ap.co2_ppmv_from_params(ap.EARTHLIKE_GASES))
        gppl = gt.gpp_light(beta, fT, fPAR, sw)
        gppw = np.maximum(gppl * 2.0, 1.0e-20)  # light-limited wherever cover/T allow
        gpp = np.minimum(gppl, gppw)
        return dict(
            gpp=gpp, gppl=gppl, gppw=gppw, lai=lai, ts=ts, rss=rss, ssru=ssru,
            land_mask=gt.land_mask_from_lsm(land),
            params=dict(ap.EARTHLIKE_GASES),
            mass_ratio=1.0, mstar=1.0, au=1.0, atmos_type="n2",
        )

    def test_land_mask_ignores_ocean(self):
        rec = gt.summarize_from_fields(**self._toy_fields())
        self.assertFalse(rec["crashed"])
        self.assertAlmostEqual(rec["gppl_recon_relerr"] or 0.0, 0.0, places=12)
        # Frozen land cell is in the mask; fT land-mean is 2/3.
        self.assertAlmostEqual(rec["fT"], 2.0 / 3.0, places=12)
        self.assertGreater(rec["frac_light_limited"], 0.9)
        self.assertEqual(rec["atmos"], "n2")

    def test_water_limitation_fraction(self):
        fields = self._toy_fields()
        fields["gppw"] = fields["gppl"] * 0.1
        rec = gt.summarize_from_fields(**fields)
        self.assertLess(rec["frac_light_limited"], 0.1)

    def test_crashed_record(self):
        rec = gt.base_record(
            mass_ratio=0.25, mstar=1.0, au=1.0, atmos_type="co2", crashed=True
        )
        self.assertTrue(rec["crashed"])
        self.assertIsNone(rec["gpp"])
        self.assertIsNone(rec["beta"])


class TestAttribution(unittest.TestCase):
    def test_sw_dominates_when_only_sw_changes(self):
        ref = {"beta": 1.0, "fT": 1.0, "fPAR": 0.5, "SWdn": 100.0, "gpp": 1.0, "crashed": False}
        rec = dict(ref)
        rec["SWdn"] = 200.0
        rec["gpp"] = 2.0
        attr = gt.attribution_vs_reference(rec, ref)
        self.assertEqual(attr["dominant_light_term"], "SWdn")
        self.assertAlmostEqual(attr["dln"]["SWdn"], math.log(2.0), places=12)
        self.assertEqual(attr["dln"]["beta"], 0.0)

    def test_beta_dominates_n2_vs_co2_if_climate_fixed(self):
        n2 = {
            "beta": gt.beta_co2(ap.co2_ppmv_from_params(ap.EARTHLIKE_GASES)),
            "fT": 1.0, "fPAR": 0.5, "SWdn": 200.0, "gpp": 1.0, "crashed": False,
        }
        co2 = dict(n2)
        co2["beta"] = gt.beta_co2(ap.co2_ppmv_from_params(ap.VENUSLIKE_GASES))
        attr = gt.attribution_vs_reference(co2, n2)
        self.assertEqual(attr["dominant_light_term"], "beta")
        self.assertGreater(attr["dln"]["beta"], math.log(3.0))


class TestPresets(unittest.TestCase):
    def test_knobs_match_helper_files(self):
        self.assertTrue(ap.PRESETS["evolved"]["apply_hhe"])
        self.assertFalse(ap.PRESETS["n2"]["apply_hhe"])
        self.assertFalse(ap.PRESETS["co2"]["apply_hhe"])
        params = dict(ap.VENUSLIKE_GASES)
        params["pN2"] = 99.0
        ap.apply_atmos_preset(params, "n2")
        self.assertAlmostEqual(params["pN2"], ap.EARTHLIKE_GASES["pN2"])
        self.assertAlmostEqual(params["pCO2"], 330.0e-6)

    def test_evolved_hhe_dilutes_co2_ppmv(self):
        params = dict(ap.EARTHLIKE_GASES)
        params["pH2"] = 1.0
        params["pHe"] = 0.5
        diluted = ap.co2_ppmv_from_params(params)
        earth = ap.co2_ppmv_from_params(ap.EARTHLIKE_GASES)
        self.assertLess(diluted, earth)


class TestDiagnosticCli(unittest.TestCase):
    def test_default_grid_is_tiny(self):
        args = runner.parse_args([])
        tasks = runner.build_tasks(args)
        # 3 atmos × 3 masses × 1 (M*, AU)
        self.assertEqual(len(tasks), 9)
        self.assertEqual(args.years, 1)
        self.assertEqual({t[0] for t in tasks}, {"evolved", "n2", "co2"})

    def test_dry_run_does_not_import_exoplasim(self):
        args = runner.parse_args(["--dry-run", "--atmos", "n2", "--masses", "1"])
        tasks = runner.build_tasks(args)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(runner.main(["--dry-run", "--atmos", "n2", "--masses", "1"]), 0)

    def test_axis_au_with_fixed_mass(self):
        args = runner.parse_args(["--axis", "au", "--atmos", "n2", "--masses", "1,2,3"])
        tasks = runner.build_tasks(args)
        # masses collapsed to 1.0 when more than one mass given on au axis
        self.assertTrue(all(t[1] == 1.0 for t in tasks))
        self.assertGreaterEqual(len(tasks), 3)


class TestPlotter(unittest.TestCase):
    def test_plotter_writes_pngs_from_schema(self):
        records = []
        for atmos, beta_scale in (("n2", 1.0), ("co2", 3.2), ("evolved", 0.9)):
            for mass, sw in ((0.25, 180.0), (1.0, 150.0), (2.0, 120.0)):
                fT = 1.0
                fPAR = 0.4 + 0.1 * mass
                gppl = 3.4e-10 * beta_scale * fT * fPAR * sw
                records.append({
                    "atmos": atmos,
                    "mass_ratio": mass,
                    "mstar": 1.0,
                    "au": 1.0,
                    "crashed": False,
                    "gpp": gppl,
                    "gppl": gppl,
                    "gppw": gppl * 2.0,
                    "beta": beta_scale,
                    "fT": fT,
                    "fPAR": fPAR,
                    "SWdn": sw,
                    "frac_light_limited": 1.0,
                    "lai": 1.2,
                    "ts_C": 12.0,
                })
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "gpp_diag.json")
            outdir = os.path.join(tmp, "plots")
            with open(json_path, "w") as handle:
                json.dump({"axis": "mass", "records": records}, handle)
            plotter.main(["--input", json_path, "--outdir", outdir])
            for name in (
                "gpp_terms_by_atmos.png",
                "gpp_terms_overlay.png",
                "gpp_dln_attribution.png",
            ):
                path = os.path.join(outdir, name)
                self.assertTrue(os.path.isfile(path), name)
                self.assertGreater(os.path.getsize(path), 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
