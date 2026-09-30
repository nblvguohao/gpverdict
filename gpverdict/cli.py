"""Command line: gpverdict predictions.csv [--frac 0.1] [--out report.html] [--json out.json]"""
import argparse
import sys

from .core import verdict
from .report import render_html, render_markdown, to_json


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gpverdict", description="Decision-aligned evaluation of genomic prediction methods. "
                                 "Input: CSV with columns environment, genotype, observed, predicted, method.")
    ap.add_argument("csv")
    ap.add_argument("--frac", type=float, default=0.10, help="selected fraction within each environment (0.05-0.20; default 0.10)")
    ap.add_argument("--min-genotypes", type=int, default=10, help="skip environments with fewer genotypes (default 10)")
    ap.add_argument("--bootstrap", type=int, default=400, help="environment resamples for rank intervals (default 400)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write an HTML report here (default: print Markdown)")
    ap.add_argument("--json", help="also write all results as JSON")
    ap.add_argument("--no-outcome", action="store_true", help="skip the out-of-sample outcome test")
    a = ap.parse_args(argv)
    v = verdict(a.csv, a.frac, a.min_genotypes, a.bootstrap, a.seed, run_outcome=not a.no_outcome)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(render_html(v))
    else:
        sys.stdout.write(render_markdown(v) + "\n")
    if a.json:
        open(a.json, "w", encoding="utf-8").write(to_json(v))


if __name__ == "__main__":
    main()
