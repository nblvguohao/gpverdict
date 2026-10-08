"""GPverdict: decision-aligned evaluation of genomic prediction methods across environments.

    from gpverdict import verdict, render_html
    v = verdict("predictions.csv", frac=0.10)
    open("report.html", "w").write(render_html(v))

    # forward evaluation over target years (version 1.2.0)
    from gpverdict import forward_verdict, forward_html
    f = forward_verdict("forward_predictions.csv", reference="GBLUP")

References: G. Lv, R. Zheng, L. Gu, A decision-based criterion and a resolution threshold for ranking genomic prediction
models within trials (2026); G. Lv, R. Zheng, L. Gu, Genomic prediction for selection in a new year: a forward benchmark
across 48 target years of maize, wheat and soybean trials (2026, manuscript).
"""
from .core import (load, from_wide, env_table, summarise, reversal, invariance_check, rank_intervals,
                   outcome_test, resolvable_gap, cells_needed, k_gauss, verdict, METRICS)
from .report import render_markdown, render_html, to_json
from .forward import (load_forward, env_differences, year_effects, pool, pool_year_effects, forward_verdict,
                      forward_markdown, forward_html, forward_json, floor_se, dersimonian_laird, hartung_knapp, t_ppf)

__version__ = "1.2.0"
