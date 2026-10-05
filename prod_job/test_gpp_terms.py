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

    def test_venuslike_plant_co2_is_clamped(self):
        ppmv = ap.co2_ppmv_from_params(ap.VENUSLIKE_GASES)
        self.assertGreater(ppmv, 9.0e5)
        plant = gt.plant_co2_ppmv(ppmv)
        self.assertAlmostEqual(plant, gt.CO2VEG_MAX)
        beta = gt.beta_co2(ppmv)
        expected = 1.0 + 0.3 * math.log(gt.CO2VEG_MAX / 360.0)
        self.assertAlmostEqual(beta, expected, places=12)
        self.assertLess(beta, 1.4)
        self.assertGreater(beta, 1.2)

    def test_earth_co2_is_below_the_clamp(self):
        ppmv = ap.co2_ppmv_from_params(ap.EARTHLIKE_GASES)
        self.assertLess(ppmv, gt.CO2VEG_MAX)
        self.assertAlmostEqual(gt.plant_co2_ppmv(ppmv), ppmv)


class TestLimitationFunctions(unittest.TestCase):
    def test_temperature_switch(self):
        self.assertEqual(float(gt.f_temperature(272.15)), 0.0)
        self.assertAlmostEqual(float(gt.f_temperature(273.15 + 2.5)), 0.5, places=12)
        self.assertEqual(float(gt.f_temperature(273.15 + 10.0)), 1.0)
        self.assertEqual(float(gt.f_temperature(273.15 + 35.0)), 1.0)
        self.assertAlmostEqual(float(gt.f_temperature(273.15 + 40.0)), 0.5, places=12)
        self.assertEqual(float(gt.f_temperature(273.15 + 45.0)), 0.0)
        self.assertEqual(float(gt.f_temperature(273.15 + 50.0)), 0.0)

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
        self.assertAlmostEqual(rec["co2veg_ppmv"], rec["co2_ppmv"])

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

    def test_venus_air_co2_clamped_in_summary(self):
        fields = self._toy_fields()
        fields["params"] = dict(ap.VENUSLIKE_GASES)
        fields["atmos_type"] = "co2"
        rec = gt.summarize_from_fields(**fields)
        self.assertGreater(rec["co2_ppmv"], 9.0e5)
        self.assertAlmostEqual(rec["co2veg_ppmv"], gt.CO2VEG_MAX)
        self.assertLess(rec["beta"], 1.4)


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
        self.assertGreater(attr["dln"]["beta"], 0.0)
        self.assertLess(attr["dln"]["beta"], math.log(2.0))


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

    def test_aliases_and_mars_preset(self):
        self.assertEqual(ap.resolve_atmos_type("earth"), "n2")
        self.assertEqual(ap.resolve_atmos_type("venus"), "co2")
        self.assertEqual(ap.resolve_atmos_type("mars"), "mars")
        params = {}
        ap.apply_atmos_preset(params, "mars")
        self.assertAlmostEqual(params["pCO2"], ap.MARSLIKE_GASES["pCO2"])
        self.assertGreater(params["pCO2"], params["pN2"])

    def test_overrides_tweak_co2_mix(self):
        params = {}
        ap.apply_atmos_preset(params, "venus", overrides={"pCO2": 0.90, "pN2": 0.10})
        self.assertAlmostEqual(params["pCO2"], 0.90)
        self.assertAlmostEqual(params["pN2"], 0.10)

    def test_surface_presets_are_not_the_defaults(self):
        self.assertAlmostEqual(ap.total_pressure_bar(ap.VENUSLIKE_GASES), 1.0, delta=0.02)
        self.assertAlmostEqual(ap.total_pressure_bar(ap.MARSLIKE_GASES), 1.0, delta=0.02)
        self.assertAlmostEqual(
            ap.total_pressure_bar(ap.PRESETS["venus_surface"]["gases"]),
            ap.VENUS_SURFACE_BAR, places=12,
        )
        self.assertAlmostEqual(
            ap.total_pressure_bar(ap.PRESETS["mars_surface"]["gases"]),
            ap.MARS_SURFACE_BAR, places=12,
        )
        params = {}
        ap.apply_atmos_preset(params, "co2")
        self.assertAlmostEqual(
            ap.total_pressure_bar(params), ap.total_pressure_bar(ap.VENUSLIKE_GASES)
        )
        params = {}
        ap.apply_atmos_preset(params, "mars")
        self.assertAlmostEqual(
            ap.total_pressure_bar(params), ap.total_pressure_bar(ap.MARSLIKE_GASES)
        )

    def test_surface_presets_keep_mole_fractions(self):
        v1 = ap.mole_fractions(ap.VENUSLIKE_GASES)
        vs = ap.mole_fractions(ap.PRESETS["venus_surface"]["gases"])
        for gas in v1:
            self.assertAlmostEqual(v1[gas], vs[gas], places=12)
        m1 = ap.mole_fractions(ap.MARSLIKE_GASES)
        ms = ap.mole_fractions(ap.PRESETS["mars_surface"]["gases"])
        for gas in m1:
            self.assertAlmostEqual(m1[gas], ms[gas], places=12)

    def test_surface_aliases_are_opt_in(self):
        self.assertEqual(ap.resolve_atmos_type("venus"), "co2")
        self.assertEqual(ap.resolve_atmos_type("venus_surface"), "venus_surface")
        self.assertEqual(ap.resolve_atmos_type("mars_surface"), "mars_surface")

    def test_evolved_hhe_dilutes_co2_ppmv(self):
        params = dict(ap.EARTHLIKE_GASES)
        params["pH2"] = 1.0
        params["pHe"] = 0.5
        diluted = ap.co2_ppmv_from_params(params)
        earth = ap.co2_ppmv_from_params(ap.EARTHLIKE_GASES)
        self.assertLess(diluted, earth)


class TestThermoAndPhysics(unittest.TestCase):
    def test_pure_co2_matches_plasim_mars_kappa(self):
        thermo = ap.thermo_from_params({"pCO2": 1.0})
        self.assertAlmostEqual(thermo["mmw"], 44.01, places=6)
        self.assertAlmostEqual(thermo["gascon"], ap.R_UNIV / 44.01, places=6)
        self.assertAlmostEqual(thermo["akap"], 0.2273, places=4)

    def test_earthlike_akap_is_near_diatomic(self):
        thermo = ap.thermo_from_params(ap.EARTHLIKE_GASES)
        self.assertAlmostEqual(thermo["akap"], 0.286, places=2)
        self.assertAlmostEqual(thermo["gascon"], 287.0, delta=8.0)

    def test_venus_and_mars_are_both_co2_but_differ(self):
        venus = ap.thermo_from_params(ap.VENUSLIKE_GASES)
        mars = ap.thermo_from_params(ap.MARSLIKE_GASES)
        self.assertAlmostEqual(venus["akap"], 0.2273, places=2)
        self.assertAlmostEqual(mars["akap"], 0.2273, places=2)
        self.assertNotAlmostEqual(venus["mmw"], mars["mmw"], places=2)

    def test_earth_physics_does_not_write_gascon_or_akap(self):
        params = dict(ap.VENUSLIKE_GASES)
        kwargs = ap.apply_physics_mode(params, "earth")
        self.assertFalse(kwargs["mars"])
        self.assertNotIn("gascon", params)
        self.assertNotIn("AKAP@planet_namelist", params.get("otherargs", {}))

    def test_other_physics_writes_gascon_and_akap(self):
        params = dict(ap.VENUSLIKE_GASES)
        kwargs = ap.apply_physics_mode(params, "other")
        thermo = ap.thermo_from_params(ap.VENUSLIKE_GASES)
        self.assertFalse(kwargs["mars"])
        self.assertAlmostEqual(params["gascon"], thermo["gascon"])
        self.assertEqual(params["otherargs"]["AKAP@planet_namelist"], str(thermo["akap"]))

    def test_mars_physics_sets_model_flag(self):
        params = dict(ap.MARSLIKE_GASES)
        kwargs = ap.apply_physics_mode(params, "mars")
        self.assertTrue(kwargs["mars"])
        self.assertNotIn("gascon", params)

    def test_default_file_tags_match_historical_names(self):
        self.assertEqual(ap.regime_file_tag("normal", "n2", "earth"), "_normal_n2")
        self.assertEqual(ap.regime_file_tag("mass_only", "earth", "earth"), "_massonly2")
        self.assertEqual(ap.regime_file_tag("mass_only", "venus", "other"), "_massonly_co2_other")
        self.assertEqual(ap.regime_file_tag("mass_only", "mars", "other"), "_massonly_mars_other")
        self.assertEqual(
            ap.regime_file_tag("mass_only", "venus_surface", "other"),
            "_massonly_venus_surface_other",
        )
        self.assertEqual(
            ap.regime_file_tag("normal", "mars_surface", "other"),
            "_normal_mars_surface_other",
        )


class TestDiagnosticCli(unittest.TestCase):
    def test_default_grid_is_tiny(self):
        args = runner.parse_args([])
        tasks = runner.build_tasks(args)
        # 3 atmos × 3 masses × 1 (M*, AU)
        self.assertEqual(len(tasks), 9)
        self.assertEqual(args.years, 1)
        self.assertEqual({t[0] for t in tasks}, {"evolved", "n2", "co2"})

    def test_aliases_and_mars_are_accepted(self):
        args = runner.parse_args(["--dry-run", "--atmos", "earth,venus,mars", "--masses", "1"])
        self.assertEqual(args.atmos, ["n2", "co2", "mars"])

    def test_surface_presets_are_accepted(self):
        args = runner.parse_args([
            "--dry-run", "--atmos", "venus_surface,mars_surface", "--masses", "1",
        ])
        self.assertEqual(args.atmos, ["venus_surface", "mars_surface"])

    def test_physics_mapping_matches_demo(self):
        args = runner.parse_args([
            "--dry-run", "--atmos", "n2,co2,mars", "--masses", "1,1.5",
            "--physics", "n2=earth,co2=other,mars=other",
        ])
        tasks = runner.build_tasks(args)
        self.assertEqual(len(tasks), 6)
        by_atmos = {t[0]: t[4] for t in tasks}
        self.assertEqual(by_atmos["n2"], "earth")
        self.assertEqual(by_atmos["co2"], "other")
        self.assertEqual(by_atmos["mars"], "other")

    def test_default_physics_is_earth(self):
        args = runner.parse_args(["--dry-run", "--atmos", "co2", "--masses", "1"])
        tasks = runner.build_tasks(args)
        self.assertEqual(tasks[0][4], "earth")

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


class TestSimbaPatch(unittest.TestCase):
    def test_patch_declares_namelist_vars(self):
        path = os.path.join(os.path.dirname(__file__), "plasim_patches", "simba.f90")
        self.assertTrue(os.path.isfile(path), path)
        text = open(path).read()
        for token in ("co2veg", "co2veg_max", "t_hot", "t_kill", "zco2p"):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
