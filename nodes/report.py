"""
Node 6: Report Node
Generates HTML report using Jinja2 template and JSON export.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from models.schemas import GraphState, ParsedJD, RankedResult


# Resolve template directory relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
OUTPUT_DIR = PROJECT_ROOT / "output"


def generate_report_node(state: GraphState) -> dict:
    """
    Generate HTML and JSON reports from ranked results.

    Input state keys: ranked_result, parsed_jd, overrides
    Output state keys: report_html, report_json
    """
    ranked_result = RankedResult(**state["ranked_result"])
    parsed_jd = ParsedJD(**state["parsed_jd"])
    overrides = state.get("overrides", [])

    # Prepare template data
    template_data = {
        "jd": parsed_jd.model_dump(),
        "ranked_candidates": [c.model_dump() for c in ranked_result.ranked_candidates],
        "tier_groups": ranked_result.tier_groups,
        "bias_audit": [f.model_dump() for f in ranked_result.bias_audit],
        "overrides": overrides,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_candidates": len(ranked_result.ranked_candidates),
        "strong_count": len(ranked_result.tier_groups.get("Strong", [])),
        "borderline_count": len(ranked_result.tier_groups.get("Borderline", [])),
        "weak_count": len(ranked_result.tier_groups.get("Weak", [])),
    }

    # Render HTML report
    report_html = _render_html(template_data)

    # Generate JSON report
    report_json = json.dumps(template_data, indent=2, default=str)

    # Save to output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "report.html").write_text(report_html, encoding="utf-8")
    (OUTPUT_DIR / "report.json").write_text(report_json, encoding="utf-8")

    print(f"[Report] Generated HTML report: {OUTPUT_DIR / 'report.html'}")
    print(f"[Report] Generated JSON report: {OUTPUT_DIR / 'report.json'}")

    return {
        "report_html": report_html,
        "report_json": report_json,
    }


def _render_html(data: dict) -> str:
    """Render the HTML report using Jinja2 template."""
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
    )

    try:
        template = env.get_template("report.html")
        return template.render(**data)
    except Exception as e:
        print(f"[Report] Template rendering error: {e}")
        # Fallback: generate a basic HTML report
        return _fallback_html(data)


def _fallback_html(data: dict) -> str:
    """Generate a basic HTML report if the Jinja2 template is missing or broken."""
    candidates_html = ""
    for i, c in enumerate(data.get("ranked_candidates", []), 1):
        tier_color = {"Strong": "#22c55e", "Borderline": "#f59e0b", "Weak": "#ef4444"}.get(c["tier"], "#6b7280")
        scores_html = ""
        for ds in c.get("dimension_scores", []):
            scores_html += f"<tr><td>{ds['dimension']}</td><td>{ds['score']}/10</td><td>{ds['weight']}</td><td>{ds['justification']}</td></tr>"

        gaps = ", ".join(c.get("skill_gaps", [])) or "None"
        flags = "; ".join(c.get("bias_flags", [])) or "None"

        candidates_html += f"""
        <div style="border:1px solid #e5e7eb; border-radius:8px; padding:16px; margin:12px 0;">
            <h3>{i}. {c['candidate_name']}
                <span style="background:{tier_color}; color:white; padding:2px 10px; border-radius:12px; font-size:0.8em;">{c['tier']}</span>
                <span style="font-size:0.9em; color:#6b7280;"> — {c['recommendation']}</span>
            </h3>
            <p><strong>Weighted Total:</strong> {c['weighted_total']}/10 | <strong>JD Similarity:</strong> {c['similarity_score']}</p>
            <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; width:100%; margin:8px 0;">
                <tr style="background:#f3f4f6;"><th>Dimension</th><th>Score</th><th>Weight</th><th>Justification</th></tr>
                {scores_html}
            </table>
            <p><strong>Skill Gaps:</strong> {gaps}</p>
            <p><strong>Bias Flags:</strong> {flags}</p>
        </div>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>HR Shortlist Report — {data.get('jd', {}).get('title', 'N/A')}</title>
<style>body{{font-family:Inter,sans-serif;max-width:900px;margin:auto;padding:20px;}}table{{font-size:0.9em;}}</style></head>
<body>
<h1>HR Shortlist Report</h1>
<p><strong>Position:</strong> {data.get('jd', {}).get('title', 'N/A')} | <strong>Generated:</strong> {data.get('generated_at', 'N/A')}</p>
<p><strong>Total:</strong> {data.get('total_candidates', 0)} | 
   <span style="color:#22c55e;">Strong: {data.get('strong_count', 0)}</span> | 
   <span style="color:#f59e0b;">Borderline: {data.get('borderline_count', 0)}</span> | 
   <span style="color:#ef4444;">Weak: {data.get('weak_count', 0)}</span></p>
<hr>
{candidates_html}
</body></html>"""
