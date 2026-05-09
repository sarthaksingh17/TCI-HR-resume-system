"""
HR Resume & LinkedIn Shortlisting Agent — Streamlit UI.

Features:
- Password-protected access via APP_PASSWORD env var
- Upload JD (PDF/DOCX/TXT) + resumes (PDF/DOCX) + LinkedIn JSON
- Run the full 7-node LangGraph pipeline with spinner feedback
- View ranked results with rubric breakdowns
- HITL override form with audit trail
- Download HTML and JSON reports
"""

import json
import os

import streamlit as st
from dotenv import load_dotenv

from main import run_pipeline
from models.schemas import RankedResult, ScoringDimension
from nodes.hitl import apply_hitl_override
from nodes.rank import rank_candidates_node
from nodes.report import generate_report_node
from utils.doc_reader import read_document_from_bytes

load_dotenv()

# ── Page Config ──────────────────────────────────────────
st.set_page_config(page_title="HR Shortlisting Agent", page_icon="📋", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .stExpander { border: 1px solid #e5e7eb; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Auth Gate ────────────────────────────────────────────
def _check_auth():
    """Password gate — only active if APP_PASSWORD is set in .env."""
    app_password = os.getenv("APP_PASSWORD", "")
    if not app_password:
        return  # No password set → open access (dev mode)

    if st.session_state.get("authenticated"):
        return

    st.markdown("### 🔐 Login Required")
    password = st.text_input("Enter password:", type="password", key="pw_input")
    if st.button("Login", key="login_btn"):
        if password == app_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


def main():
    _check_auth()

    st.markdown('<div class="main-header">📋 HR Resume Shortlisting Agent</div>', unsafe_allow_html=True)
    st.caption("AI-powered candidate evaluation with LangGraph • Groq • sentence-transformers")
    st.divider()

    # ── Sidebar: File Uploads ────────────────────────────
    with st.sidebar:
        st.header("📁 Upload Files")
        st.subheader("1. Job Description")
        jd_file = st.file_uploader("Upload JD (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"], key="jd_upload")

        st.subheader("2. Resumes")
        resume_files = st.file_uploader("Upload resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True, key="resume_upload")

        st.subheader("3. LinkedIn Profiles (Optional)")
        linkedin_files = st.file_uploader("Upload LinkedIn JSON exports", type=["json"], accept_multiple_files=True, key="linkedin_upload")

        st.divider()
        run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True, disabled=not (jd_file and resume_files))

    if run_btn:
        _run_pipeline(jd_file, resume_files, linkedin_files)
    elif "pipeline_result" in st.session_state:
        _display_results()
    else:
        _show_landing()


def _show_landing():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1:** Upload a Job Description in the sidebar")
    with col2:
        st.info("**Step 2:** Upload candidate resumes (PDF/DOCX)")
    with col3:
        st.info("**Step 3:** Click 'Run Pipeline' to evaluate")

    st.markdown("### How It Works")
    st.markdown("""
    | Node | Step | Description |
    |------|------|-------------|
    | 1 | **JD Parser** | Extracts structured requirements from the job description |
    | 2A | **Resume Parser** | Parses PDF/DOCX resumes into structured profiles |
    | 2B | **LinkedIn Parser** | Parses exported LinkedIn JSON profiles |
    | 3 | **Profile Merge** | Merges resume + LinkedIn data (resume wins on conflicts) |
    | 4 | **Scoring** | LLM scores 5 dimensions (0-10) with rubric descriptors |
    | 5 | **Ranking** | Ranks by weighted total + runs bias audit |
    | 6 | **Report** | Generates HTML and JSON reports |
    | 7 | **HITL Override** | Human reviewer can adjust scores with audit trail |
    """)


def _run_pipeline(jd_file, resume_files, linkedin_files):
    with st.spinner("📄 Reading Job Description..."):
        try:
            jd_text = read_document_from_bytes(jd_file.read(), jd_file.name)
        except Exception as e:
            st.error(f"Error reading JD: {e}")
            return

    with st.spinner(f"📑 Reading {len(resume_files)} resume(s)..."):
        resume_texts = {}
        for rf in resume_files:
            try:
                resume_texts[rf.name] = read_document_from_bytes(rf.read(), rf.name)
            except Exception as e:
                st.warning(f"Could not read {rf.name}: {e}")
        if not resume_texts:
            st.error("No resumes could be parsed.")
            return

    linkedin_data = {}
    if linkedin_files:
        with st.spinner("🔗 Reading LinkedIn profiles..."):
            for lf in linkedin_files:
                try:
                    content = lf.read().decode("utf-8")
                    json.loads(content)
                    linkedin_data[lf.name] = content
                except Exception as e:
                    st.warning(f"Could not read {lf.name}: {e}")

    with st.spinner("🤖 Running AI pipeline — parsing, scoring, ranking... (this may take a minute)"):
        try:
            result = run_pipeline(jd_text, resume_texts, linkedin_data)
            st.session_state["pipeline_result"] = result
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    st.success("✅ Pipeline completed successfully!")

    # Warn about any resumes that failed to parse
    parse_failures = result.get("parse_failures", [])
    if parse_failures:
        for fname in parse_failures:
            st.warning(f"⚠️ Resume **{fname}** could not be parsed by the AI and was skipped. Try re-uploading it.")

    _display_results()


def _display_results():
    result = st.session_state.get("pipeline_result")
    if not result:
        return

    ranked = RankedResult(**result.get("ranked_result", {}))

    st.header("📊 Results Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Candidates", len(ranked.ranked_candidates))
    c2.metric("✅ Strong (Hire)", len(ranked.tier_groups.get("Strong", [])))
    c3.metric("⚠️ Borderline", len(ranked.tier_groups.get("Borderline", [])))
    c4.metric("❌ Weak (No Hire)", len(ranked.tier_groups.get("Weak", [])))
    st.divider()

    st.header("🏆 Ranked Candidates")
    for i, candidate in enumerate(ranked.ranked_candidates):
        tier_emoji = {"Strong": "🟢", "Borderline": "🟡", "Weak": "🔴"}.get(candidate.tier, "⚪")
        pct = candidate.weighted_total * 10
        with st.expander(
            f"{tier_emoji} #{i+1} — {candidate.candidate_name} | "
            f"{candidate.weighted_total}/10 ({pct:.0f}%) | {candidate.tier} → {candidate.recommendation}"
        ):
            cols = st.columns(5)
            for j, ds in enumerate(candidate.dimension_scores):
                with cols[j]:
                    st.metric(ds.dimension, f"{ds.score}/10", help=ds.justification)

            st.caption(f"**JD Similarity:** {candidate.similarity_score}")
            st.markdown("**Score Justifications:**")
            for ds in candidate.dimension_scores:
                st.markdown(f"- **{ds.dimension}** ({ds.score}/10): {ds.justification}")

            if candidate.skill_gaps:
                st.warning(f"**Skill Gaps:** {', '.join(candidate.skill_gaps)}")
            if candidate.bias_flags:
                for flag in candidate.bias_flags:
                    st.error(f"**Bias Flag:** {flag}")

    if ranked.bias_audit:
        st.divider()
        st.header("🔍 Bias Audit")
        for flag in ranked.bias_audit:
            st.warning(f"**[{flag.severity.upper()}] {flag.flag_type}**\n\n{flag.description}\n\n*Candidates:* {', '.join(flag.candidate_names)}")

    st.divider()
    st.header("✏️ Human-in-the-Loop Override")
    _hitl_form(ranked)

    st.divider()
    st.header("📥 Download Reports")
    dl1, dl2 = st.columns(2)
    with dl1:
        html_report = result.get("report_html", "")
        if html_report:
            st.download_button("📄 Download HTML Report", data=html_report, file_name="shortlist_report.html", mime="text/html", use_container_width=True)
    with dl2:
        json_report = result.get("report_json", "")
        if json_report:
            st.download_button("📋 Download JSON Report", data=json_report, file_name="shortlist_report.json", mime="application/json", use_container_width=True)

    overrides = result.get("overrides", [])
    if overrides:
        st.divider()
        st.header("📝 Override Audit Trail")
        for o in overrides:
            st.info(f"**{o['candidate_name']}** — {o['dimension']}: {o['original_score']} → {o['new_score']}\n\n*Reason:* {o['reason']} | *By:* {o['reviewer']} | *At:* {o['timestamp']}")


def _hitl_form(ranked):
    if not ranked.ranked_candidates:
        st.info("No candidates to override.")
        return

    candidate_names = [c.candidate_name for c in ranked.ranked_candidates]
    dimensions = [d.value for d in ScoringDimension]

    with st.form("hitl_form"):
        col1, col2 = st.columns(2)
        with col1:
            selected_candidate = st.selectbox("Select Candidate", candidate_names)
            selected_dimension = st.selectbox("Select Dimension", dimensions)
        with col2:
            new_score = st.slider("New Score", 0.0, 10.0, 5.0, 0.5)
            reason = st.text_input("Reason for Override", placeholder="e.g. Verified via interview")
        reviewer = st.text_input("Reviewer Name", value="HR Reviewer")
        submitted = st.form_submit_button("Apply Override", type="primary")

    if submitted and reason:
        with st.spinner("Applying override and re-ranking..."):
            result = st.session_state["pipeline_result"]
            updates = apply_hitl_override(state=result, candidate_name=selected_candidate, dimension=selected_dimension, new_score=new_score, reason=reason, reviewer=reviewer)
            if updates:
                result.update(updates)
                rank_updates = rank_candidates_node(result)
                result.update(rank_updates)
                report_updates = generate_report_node(result)
                result.update(report_updates)
                st.session_state["pipeline_result"] = result
                st.success(f"Override applied! {selected_candidate} / {selected_dimension}: → {new_score}")
                st.rerun()
    elif submitted and not reason:
        st.warning("Please provide a reason for the override.")


if __name__ == "__main__":
    main()
