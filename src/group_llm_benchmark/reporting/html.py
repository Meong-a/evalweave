from __future__ import annotations

import html
import json
from typing import Any


STYLE = """
body{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1180px;margin:32px auto;padding:0 24px;color:#182230}
h1,h2{color:#173b63}table{border-collapse:collapse;width:100%;margin:12px 0 28px}
th,td{border:1px solid #d8e0e8;padding:8px;text-align:left;vertical-align:top}th{background:#eef4f8}
.good{color:#08783e}.bad{color:#b42318}.muted{color:#5c6b7a}code{background:#f4f6f8;padding:2px 4px}
"""


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class='muted'>No data.</p>"
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    head = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_run_html(metrics: dict[str, Any], config: dict[str, Any]) -> str:
    score_rows = [{"metric": key, "value": f"{value:.6f}"} for key, value in metrics.get("scores", {}).items()]
    capability_rows = []
    for capability, item in metrics.get("per_capability", {}).items():
        capability_rows.append({"capability": capability, **item})
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Benchmark report</title><style>{STYLE}</style></head>
<body><h1>{html.escape(str(config.get('experiment_name', 'Benchmark run')))}</h1>
<p>Run ID: <code>{html.escape(str(metrics.get('run_id', '')))}</code></p>
<h2>Summary</h2>{_table([{'samples': metrics.get('count'), 'errors': metrics.get('error_count'), 'invalid': metrics.get('invalid_count'), 'latency_ms': round(float(metrics.get('total_latency_ms', 0)), 2)}])}
<h2>Metrics</h2>{_table(score_rows)}
<h2>Capability breakdown</h2>{_table(capability_rows)}
<h2>Resolved configuration</h2><pre>{html.escape(json.dumps(config, ensure_ascii=False, indent=2))}</pre>
</body></html>"""


def render_comparison_html(result: dict[str, Any]) -> str:
    rows = []
    for capability, values in result.get("capability_matrix", {}).items():
        rows.append({"capability": capability, **values})
    transitions = [{"transition": key, "count": value} for key, value in result.get("transitions", {}).items()]
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Benchmark comparison</title><style>{STYLE}</style></head>
<body><h1>Experiment comparison</h1>
<p><code>{html.escape(str(result.get('baseline_run')))}</code> compared with <code>{html.escape(str(result.get('candidate_run')))}</code></p>
<h2>Compatibility checks</h2>{_table([result.get('compatibility', {})])}
<h2>Overall result</h2>{_table([result.get('overall', {})])}
<h2>Capability change matrix</h2>{_table(rows)}
<h2>Per-sample transitions</h2>{_table(transitions)}
<h2>Statistical analysis</h2><pre>{html.escape(json.dumps(result.get('statistics', {}), ensure_ascii=False, indent=2))}</pre>
</body></html>"""


def render_leaderboard_html(result: dict[str, Any]) -> str:
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Benchmark leaderboard</title><style>{STYLE}</style></head>
<body><h1>{html.escape(str(result.get('name', 'Benchmark leaderboard')))}</h1>
<h2>Model ranking</h2>{_table(result.get('ranking', []))}
<h2>Capability matrix</h2>{_table(result.get('capability_matrix', []))}
</body></html>"""


def render_suite_matrix_html(result: dict[str, Any]) -> str:
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Benchmark suite matrix</title><style>{STYLE}</style></head>
<body><h1>{html.escape(str(result.get('name', 'Benchmark suite matrix')))}</h1>
<p class='muted'>不同任务的指标不做统一排名；每行保留自己的任务和主指标。</p>
<h2>Run summary</h2>{_table(result.get('runs', []))}
<h2>Capability matrix</h2>{_table(result.get('capability_matrix', []))}
</body></html>"""
