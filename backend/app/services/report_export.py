"""Exportable report generation.

Produces a self-contained HTML report (opens in any browser; "Print -> Save
as PDF" gives a PDF without adding a PDF-rendering dependency we can't test
in this environment) and a JSON export of the same analysis payload.

We deliberately avoid a PDF-generation library here: none was available to
verify in the offline sandbox this MVP was built in, and a broken PDF export
is worse than an honestly-scoped HTML one. Swapping in reportlab/weasyprint
later is a contained change limited to this file.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any

SEVERITY_COLORS = {
    "critical": "#dc2626",
    "high": "#f97316",
    "medium": "#f59e0b",
    "low": "#94a3b8",
}


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _render_dqi_table(dqi: dict) -> str:
    rows = []
    for c in dqi["components"]:
        score = f"{c['score']:.3f}" if c["available"] else "unavailable"
        rows.append(
            f"<tr><td>{_esc(c['name'].capitalize())}</td><td>{score}</td>"
            f"<td>{c['effective_weight']:.2f}</td><td>{c['contribution']:.4f}</td></tr>"
        )
    return "\n".join(rows)


def _render_issues(issues: list[dict]) -> str:
    if not issues:
        return "<p>No issues detected.</p>"
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_issues = sorted(issues, key=lambda i: order.get(i["severity"], 9))
    rows = []
    for issue in sorted_issues:
        color = SEVERITY_COLORS.get(issue["severity"], "#94a3b8")
        cols = ", ".join(issue["columns"]) if issue.get("columns") else "-"
        rows.append(
            f"<tr><td><span class='badge' style='background:{color}'>{_esc(issue['severity'])}</span></td>"
            f"<td>{_esc(issue['issue_type'].replace('_', ' '))}</td>"
            f"<td>{_esc(cols)}</td><td>{_esc(issue['explanation'])}</td></tr>"
        )
    return "\n".join(rows)


def _render_recommendations(recommendations: list[dict]) -> str:
    if not recommendations:
        return "<p>No recommendations -- no issues were detected to act on.</p>"
    items = []
    for rec in recommendations:
        color = SEVERITY_COLORS.get(rec["severity"], "#94a3b8")
        items.append(
            f"<li><span class='badge' style='background:{color}'>{_esc(rec['severity'])}</span> {_esc(rec['message'])}</li>"
        )
    return "<ul class='rec-list'>" + "\n".join(items) + "</ul>"


def _render_baseline(baseline: dict) -> str:
    if not baseline.get("available"):
        return f"<p>Baseline model performance unavailable: {_esc(baseline.get('reason_unavailable', ''))}</p>"

    def render_feature_set(fs: dict, title: str) -> str:
        if not fs:
            return ""
        blocks = [f"<h4>{_esc(title)} ({fs['n_features']} features)</h4>"]
        for r in fs["results"]:
            blocks.append(f"<p><strong>{_esc(r['model_name'])}</strong></p><ul>")
            for metric, stats in r["metrics"].items():
                if isinstance(stats, dict):
                    blocks.append(f"<li>{_esc(metric)}: {stats['mean']:.4f} &plusmn; {stats['std']:.4f}</li>")
                else:
                    blocks.append(f"<li>{_esc(metric)}: error - {_esc(stats)}</li>")
            blocks.append("</ul>")
        return "\n".join(blocks)

    parts = [
        f"<p>Task: {_esc(baseline['task_type'])} &middot; Target: {_esc(baseline['target_column'])} "
        f"&middot; {_esc(baseline['cv_folds'])}-fold cross-validation &middot; "
        f"Readiness: <strong>{_esc(baseline['readiness'])}</strong></p>",
        render_feature_set(baseline.get("full_features"), "All features"),
    ]
    if baseline.get("cleaned_features"):
        excluded = ", ".join(baseline.get("excluded_columns", []))
        parts.append(
            f"<p style='color:#b45309'>Excluding columns flagged as potential leakage/redundancy: {_esc(excluded)}</p>"
        )
        parts.append(render_feature_set(baseline["cleaned_features"], "Excluding flagged features"))
    return "\n".join(parts)


def _render_informativeness(informativeness: dict) -> str:
    if not informativeness.get("available"):
        return f"<p>Informativeness diagnostic unavailable: {_esc(informativeness.get('reason_unavailable', ''))}</p>"
    rows = []
    for f in informativeness.get("top_features", [])[:5]:
        rows.append(f"<li>{_esc(f['column'])}: {f['normalized_mi'] * 100:.1f}%</li>")
    return (
        f"<p>{_esc(informativeness['explanation'])}</p>"
        f"<ul>{''.join(rows)}</ul>"
        f"<p class='note'>{_esc(informativeness['note'])}</p>"
    )


def build_html_report(analysis: dict, dataset_meta: dict) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    dqi = analysis["dqi"]

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AutoDQ Report — {_esc(dataset_meta.get('filename', 'dataset'))}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; color: #1e293b; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ color: #1c3fae; margin-bottom: 4px; }}
  h2 {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-top: 40px; }}
  h4 {{ margin-bottom: 4px; }}
  .meta {{ color: #64748b; font-size: 14px; margin-bottom: 24px; }}
  .dqi-score {{ font-size: 48px; font-weight: bold; color: #1c3fae; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
  th {{ background: #f8fafc; text-transform: uppercase; font-size: 11px; color: #64748b; }}
  .badge {{ color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; text-transform: uppercase; }}
  .rec-list li {{ margin-bottom: 8px; }}
  .note {{ font-size: 12px; color: #94a3b8; margin-top: 8px; }}
  @media print {{ body {{ margin: 0; }} }}
</style>
</head>
<body>
  <h1>AutoDQ Report</h1>
  <p class="meta">
    Dataset: {_esc(dataset_meta.get('filename', 'unknown'))} &middot;
    {_esc(dataset_meta.get('n_rows', '?'))} rows &times; {_esc(dataset_meta.get('n_columns', '?'))} columns &middot;
    Generated {generated_at}
  </p>

  <h2>Dataset Quality Index</h2>
  <div class="dqi-score">{dqi['dqi_0_100']} / 100</div>
  <p>Interpretation: <strong>{_esc(dqi['interpretation'])}</strong> &middot; Domain preset: {_esc(dqi['domain_preset'])}</p>
  <table>
    <tr><th>Component</th><th>Score</th><th>Weight</th><th>Contribution</th></tr>
    {_render_dqi_table(dqi)}
  </table>
  <p class="note">{_esc(dqi['note'])}</p>

  <h2>Detected Issues ({len(analysis['issues'])})</h2>
  <table>
    <tr><th>Severity</th><th>Type</th><th>Column(s)</th><th>Explanation</th></tr>
    {_render_issues(analysis['issues'])}
  </table>

  <h2>Recommendations</h2>
  {_render_recommendations(analysis['recommendations'])}

  <h2>Baseline Model Performance</h2>
  {_render_baseline(analysis['baseline'])}

  <h2>Feature-Target Informativeness <span style="font-size:12px;color:#7c3aed;font-weight:normal;">(experimental, separate from DQI)</span></h2>
  {_render_informativeness(analysis['informativeness'])}

  <p class="note">
    Generated by AutoDQ. Weights and interpretation bands are configurable heuristics, not
    universal standards. Leakage findings are reported as "potential," not confirmed.
  </p>
</body>
</html>"""
