"""GPverdict must reproduce the published numbers on the spring wheat example (Lv and Gu 2026)."""
import os
import numpy as np
import gpverdict as gv

EX = os.path.join(os.path.dirname(__file__), "..", "examples", "spring_wheat_URSN.csv")


def test_reversal_and_outcome():
    df = gv.load(EX)
    ET = gv.env_table(df, 0.10, 12)
    S = gv.summarise(df, ET)
    rev, pairs = gv.reversal(S, "rmse", "pearson")
    assert pairs == 136 and abs(rev - 0.25) < 1e-12            # Table S4, base panel
    o = gv.outcome_test(df, 0.10, 12, B=400, seed=0)
    rec = {k: round(100 * v, 1) for k, v in o["recovery"].items()}
    assert rec == {"spearman": 58.7, "pearson": 78.5, "rmse": 37.9}   # Table S7, base panel


def test_tiers_follow_the_criterion():
    inv = gv.invariance_check(gv.load(EX), 0.10, 12).set_index("key").tier
    assert set(inv[inv == "I"].index) == {"spearman", "sel_diff", "hit_rate", "ndcg"}
    assert set(inv[inv == "II"].index) == {"pearson"}


def test_cells_law():
    assert abs(gv.k_gauss(0.10) - 2.9657) < 1e-9
    assert abs(gv.cells_needed(0.05) - (2.9657 / 0.05) ** 2) < 1e-6
    assert abs(gv.resolvable_gap(10_000) - 0.029657) < 1e-9
