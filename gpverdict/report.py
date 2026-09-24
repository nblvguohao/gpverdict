"""Plain-language report of a GPverdict analysis (Markdown, HTML or JSON)."""
import html
import json
import math

import numpy as np

from .core import METRICS, cells_needed


def _f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def _cells(n):
    if not math.isfinite(n):
        return "no finite trial"
    if n >= 1e9:
        return f"{n/1e9:.1f} billion"
    return f"{n/1e6:.1f} million" if n >= 1e6 else f"{n:,.0f}"


def headline(v):
    r = v["resolution"]; lead = v["leader"]; tied = r["tied_with_leader"]
    lines = [f"**Choose `{lead}`** for within-environment selection of the top {v['settings']['frac']:.0%}: "
             f"it ranks first by the mean within-environment rank correlation, the metric this decision admits."]
    if len(tied) > 1:
        others = ", ".join(f"`{m}`" for m in tied if m != lead)
        lines.append(f"At this trial size ({v['data']['cells']:,} genotype–environment cells per method) it cannot be told apart "
                     f"from {others}: they lie within {_f(r['gap'])} of it, the smallest gap such a trial resolves "
                     f"(Gaussian lower bound; calibrated panels needed {_f(r['panel_range'][0])}–{_f(r['panel_range'][1])}). "
                     f"Choose among them on cost or simplicity.")
    else:
        lines.append(f"Its lead of {_f(r['leader_margin'])} exceeds {_f(r['gap'])}, the smallest gap this trial resolves "
                     f"(Gaussian lower bound; calibrated panels needed {_f(r['panel_range'][0])}–{_f(r['panel_range'][1])}).")
    if math.isfinite(r["leader_margin"]) and r["leader_margin"] < 1e-4:
        lines.append("The first two methods are practically identical on this metric (gap below 0.0001).")
    elif math.isfinite(r["leader_margin"]):
        lines.append(f"Resolving the gap between the first two methods ({_f(r['leader_margin'], 4)}) would take about "
                     f"{_cells(r['cells_to_resolve_margin'])} cells.")
    if v["rmse_pick"] != lead:
        lines.append(f"Ranking by RMSE would pick `{v['rmse_pick']}` instead; RMSE and Pearson r order "
                     f"{_f(100*v['reversal_rmse_pearson'], 0)} % of method pairs in opposite directions.")
    o = v.get("outcome")
    if o:
        rec = o["recovery"]
        lines.append(f"Out of sample ({o['design']}, {o['splits']} splits, outcome reliability {_f(o['reliability'], 2)}), the method each rule "
                     f"picks recovers {_f(100*rec['spearman'], 0)} % (rank correlation), {_f(100*rec['pearson'], 0)} % (Pearson r) and "
                     f"{_f(100*rec['rmse'], 0)} % (RMSE) of the selection gain attainable over an average method.")
        if not math.isfinite(o["reliability"]):
            lines.append("With fewer than six methods the outcome's reliability cannot be estimated; read these recoveries as descriptive, not as a comparison.")
        elif o["reliability"] < 0.5:
            lines.append("The outcome's split-half reliability is below 0.5, so these recoveries should not be compared.")
    else:
        lines.append(f"The out-of-sample test was skipped: it needs at least three methods and either eight environments or "
                     f"{2 * v['settings']['min_genotypes']} genotypes in an environment.")
    return lines


def planning_table(v):
    f = v["settings"]["frac"]
    return [(g, cells_needed(g, f)) for g in (0.10, 0.05, 0.03, 0.02, 0.01)]


def render_markdown(v):
    S = v["summary"].copy(); R = v["ranks"]; RI = v["rank_intervals"]
    out = ["# GPverdict report", "",
           f"{v['data']['methods']} methods, {v['data']['environments']} environments, {v['data']['genotypes']:,} genotypes; "
           f"selection of the top {v['settings']['frac']:.0%} within each environment.", "", "## Verdict", ""]
    out += [f"- {l}" for l in headline(v)]
    out += ["", "## Methods ranked by the admissible metric", "",
            "| Rank | Method | Rank correlation | Pearson r | RMSE | Realised selection differential | Rank by RMSE |"
            + (" 95 % rank interval |" if RI is not None else ""),
            "|---:|---|---:|---:|---:|---:|---:|" + ("---:|" if RI is not None else "")]
    for m in S.sort_values("spearman", ascending=False).index:
        row = f"| {R.loc[m, 'spearman']} | `{m}` | {_f(S.loc[m, 'spearman'])} | {_f(S.loc[m, 'pearson'])} | {_f(S.loc[m, 'rmse'])} | {_f(S.loc[m, 'sel_diff'])} | {R.loc[m, 'rmse']} |"
        if RI is not None:
            row += f" [{RI.loc[m, 'rank_lo']}, {RI.loc[m, 'rank_hi']}] |"
        out.append(row)
    inv = v.get("invariance")
    if inv is not None:
        out += ["", "## Which metrics can rank methods for this decision", "",
                "Checked on your own predictions: median relative change of each metric when the predictions are transformed "
                "within every environment in ways that cannot change which genotypes are selected, and when their order is reversed.", "",
                "| Metric | Affine maps | Increasing maps | Order reversed | Tier |", "|---|---:|---:|---:|:---:|"]
        for _, r in inv.iterrows():
            out.append(f"| {r.metric} | {r.affine_drift:.1e} | {r.monotone_drift:.1e} | {r.reversal_change:.2f} | {r.tier} |")
        out += ["", "Tier I metrics represent the decision; Tier II (Pearson r) is unmoved by calibration only; "
                "Tier III metrics can move without any change in the selected material and should not rank methods for it."]
    out += ["", "## Planning a trial", "", "| Accuracy gain to resolve | Genotype–environment cells needed (Gaussian lower bound) |", "|---:|---:|"]
    out += [f"| {g:.2f} | {_cells(n)} |" for g, n in planning_table(v)]
    out += ["", "Calibrated panels needed 1.5–4.2 times these values; treat them as minimums.", "",
            "---", "GPverdict " + __import__("gpverdict").__version__ + " · Lv & Gu (2026) · computations run locally."]
    return "\n".join(out)


def render_html(v):
    md = render_markdown(v)
    body = []; in_table = False
    for line in md.split("\n"):
        esc = html.escape(line)
        esc = __import__("re").sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc)
        esc = __import__("re").sub(r"`(.+?)`", r"<code>\1</code>", esc)
        if line.startswith("|"):
            cells = [c.strip() for c in esc.strip("|").split("|")]
            if set(line.replace("|", "").strip()) <= set("-: "):
                continue
            if not in_table:
                body.append("<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr></thead><tbody>"); in_table = True
            else:
                body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            body.append("</tbody></table>"); in_table = False
        if line.startswith("# "): body.append(f"<h1>{esc[2:]}</h1>")
        elif line.startswith("## "): body.append(f"<h2>{esc[3:]}</h2>")
        elif line.startswith("- "): body.append(f"<p class='v'>{esc[2:]}</p>")
        elif line == "---": body.append("<hr>")
        elif line.strip(): body.append(f"<p>{esc}</p>")
    if in_table: body.append("</tbody></table>")
    css = ("body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1d2430;line-height:1.5}"
           "table{border-collapse:collapse;width:100%;margin:8px 0 16px;font-size:14px}th,td{border-bottom:1px solid #dde2ea;padding:5px 8px;text-align:right}"
           "th:nth-child(2),td:nth-child(2),th:first-child,td:first-child{text-align:left}p.v{background:#eef5ee;border-left:4px solid #3c8c4a;padding:8px 12px;margin:6px 0}"
           "code{background:#f1f3f6;padding:1px 4px;border-radius:3px}h2{margin-top:28px;border-bottom:2px solid #e8ebf0;padding-bottom:4px}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>GPverdict report</title><style>{css}</style></head><body>{''.join(body)}</body></html>"


def to_json(v):
    def conv(x):
        if hasattr(x, "to_dict"):
            return x.reset_index().to_dict(orient="records")
        if isinstance(x, (np.floating, float)):
            return None if not math.isfinite(float(x)) else float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, dict):
            return {k: conv(val) for k, val in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(i) for i in x]
        return x
    return json.dumps(conv(v), indent=1)
