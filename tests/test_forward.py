"""The forward-evaluation module must reproduce the pooled results reported by Lv, Zheng and Gu (2026, forward benchmark; manuscript in preparation)
from the year-effect tables archived with that paper (tests/data, copied from the authors' analysis results), and the
environment-level scores of the benchmark from per-cell predictions (examples/forward_MU_SOY.csv)."""
import os

import numpy as np
import pandas as pd
import pytest

import gpverdict as gv

D = os.path.join(os.path.dirname(__file__), "data")
EX = os.path.join(os.path.dirname(__file__), "..", "examples", "forward_MU_SOY.csv")
rd = lambda f: pd.read_csv(os.path.join(D, f))
EXP = rd("expected_pooled_31.csv").set_index("contrast")


def check(label, ye, res, level=0.95):
    """Pool with the module and compare with the reported row (estimate, both intervals, both verdicts) to 1e-9."""
    r = gv.pool_year_effects(ye[["dataset", "d", "se"]], resolution=res, level=level).iloc[0]
    e = EXP.loc[label]
    for a, b in (("est", "est"), ("lo", "dl_lo"), ("hi", "dl_hi"), ("tau2", "tau2"), ("hk_lo", "hk_lo"), ("hk_hi", "hk_hi")):
        if np.isfinite(e[b]):
            assert abs(r[a] - e[b]) < 1e-9, (label, a, r[a], e[b])
    assert r["verdict"] == e["dl_verdict"], label
    if e["target_years"] > 1:
        assert r["verdict_hk"] == e["hk_verdict"], label


BENCH = {"reml": "Two-stage GBLUP", "rf": "Random forest", "gbm": "Gradient boosting", "mlp": "Multilayer perceptron",
         "gxe_gbm": "Covariate LightGBM", "rn_ridge": "Covariate reaction-norm ridge", "dl_g": "Within-environment-loss network",
         "R_STK_FW": "Stacking on the forward history", "Oracle": "Best method in hindsight"}


def test_benchmark_contrasts():                       # Fig. 4b; Table 1, row 2
    y = rd("benchmark_year_effects.csv")
    for item, nm in BENCH.items():
        lab = f"Benchmark: {nm} − cell-level GBLUP"
        check(lab, y[y["item"] == item], EXP.loc[lab, "resolution"])


def test_sparse_testing():                            # Fig. 2; Table 1, row 1
    y = rd("sparse_gxe_year_effects.csv")
    for f, lab in ((0.25, "Sparse 25 %: M×E − main-effect GBLUP"), (0.5, "Sparse 50 %: M×E − main-effect GBLUP")):
        check(lab, y[y["fraction"] == f], EXP.loc[lab, "resolution"])


def test_value_of_phenotypes():                       # Fig. 2a; Table 1, row 1
    y = rd("sparse_value_year_effects.csv"); exp = rd("expected_sparse_value.csv")
    for r in exp.itertuples():
        g = y[(y["fraction"] == r.fraction) & (y["contrast"] == r.contrast)]
        out = gv.pool_year_effects(g[["dataset", "d", "se"]], resolution=r.resolution).iloc[0]
        assert abs(out["est"] - r.est) < 1e-9 and abs(out["lo"] - r.lo) < 1e-9 and abs(out["hi"] - r.hi) < 1e-9
        assert out["verdict"] == r.verdict


def test_clac_and_ablations():                        # Fig. 3a, b
    check("CLAC − cell-level GBLUP", rd("clac_year_effects.csv"), EXP.loc["CLAC − cell-level GBLUP", "resolution"],
          EXP.loc["CLAC − cell-level GBLUP", "level"])
    y = rd("clac_ablation_year_effects.csv")
    for k, nm in (("loss_D4_linear_kernel", "linear kernel"), ("loss_D5_single_main_effect", "single main effect"),
                  ("loss_D1_equal_weights", "equal weights"), ("loss_D3_no_cleaning", "no phenotype cleaning"),
                  ("loss_D2_no_spatial", "no spatial adjustment")):
        lab = f"CLAC ablation, loss: {nm}"
        check(lab, y[y["contrast"] == k], EXP.loc[lab, "resolution"], EXP.loc[lab, "level"])


def test_kernel_transfer():                           # Fig. 3c
    y = rd("kernel_year_effects.csv"); exp = rd("expected_kernel.csv").set_index("range")
    check("Kernel transfer: all six datasets", y, exp.loc["all48", "resolution"])
    for d in ("G2F", "NUST", "URSN", "ESWYT", "GEM_IA", "MU_SOY"):
        check(f"Kernel transfer: {d}", y[y["dataset"] == d], exp.loc[d, "resolution"])


def test_t_quantile_matches_scipy():
    scipy = pytest.importorskip("scipy.stats")
    err = max(abs(gv.t_ppf(q, df) - scipy.t.ppf(q, df)) for df in (1, 2, 3, 4, 5, 10, 30, 47)
              for q in (0.6, 0.975, 0.9875, 0.99, 0.991667, 0.995))
    assert err < 1e-9


def test_per_cell_entry_reproduces_benchmark_scores():
    """From per-cell predictions of MU_SOY 2020: environment-level differences equal the benchmark's to 1e-12 for nine of
    the eleven methods. The example file holds the benchmark's archived predictions, which are stored in single precision;
    for the two kNN methods, whose predictions contain near-ties, that rounding changes a few ranks and the differences
    agree to 5e-4 (the benchmark scored the double-precision originals). The bootstrap SE depends on the random stream
    and is only checked for size."""
    v = gv.forward_verdict(EX, reference="cell_reml", min_genotypes=25, B=2000, seed=0)
    E = v["all"]["environments"]
    exp = rd("expected_MU_SOY_per_env.csv")
    m = E.merge(exp, left_on=["year", "environment", "method"], right_on=["target", "env", "item"])
    assert len(m) == len(exp) and len(m) == len(E)
    knn = m["method"].str.startswith("knn")
    assert np.abs(m.loc[~knn, "d"] - m.loc[~knn, "d_spearman"]).max() < 1e-12
    assert np.abs(m.loc[knn, "d"] - m.loc[knn, "d_spearman"]).max() < 5e-4
    assert (m["n_x"] == m["n_y"]).all()
    ye = v["all"]["year_effects"]
    ref = rd("benchmark_year_effects.csv").query("dataset == 'MU_SOY'").set_index("item")
    for r in ye.itertuples():
        assert abs(r.d - ref.loc[r.method, "d"]) < (5e-4 if r.method.startswith("knn") else 1e-12)
        assert 0.5 < r.se / ref.loc[r.method, "se"] < 2.0           # same quantity, different bootstrap stream
    assert {"new", "old"} <= set(v)                                     # first_year marks new and old lines
    md = gv.forward_markdown(v)
    assert "Forward evaluation against `cell_reml`" in md and ("resolved" in md or "tied" in md)
