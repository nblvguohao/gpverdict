"""GPverdict: decision-aligned evaluation of genomic prediction methods across environments.

    from gpverdict import verdict, render_html
    v = verdict("predictions.csv", frac=0.10)
    open("report.html", "w").write(render_html(v))

Reference: G. Lv, L. Gu, A decision-based criterion and an open tool for choosing genomic
prediction methods across environments (2026).
"""
from .core import (load, env_table, summarise, reversal, invariance_check, rank_intervals,
                   outcome_test, resolvable_gap, cells_needed, k_gauss, verdict, METRICS)
from .report import render_markdown, render_html, to_json

__version__ = "1.0.0"
