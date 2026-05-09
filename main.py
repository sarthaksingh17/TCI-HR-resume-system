"""
HR Resume & LinkedIn Shortlisting Agent — LangGraph Pipeline Assembly.

7-Node Pipeline:
  1. JD Parser → 2A. Resume Parser → 2B. LinkedIn Parser
  → 3. Profile Merge → 4. Scoring → 5. Rank → 6. Report
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from models.schemas import GraphState
from nodes.jd_parser import parse_jd_node
from nodes.linkedin_parser import parse_linkedin_node
from nodes.profile_merge import merge_profiles_node
from nodes.rank import rank_candidates_node
from nodes.report import generate_report_node
from nodes.resume_parser import parse_resumes_node
from nodes.scoring import score_candidates_node


def build_graph():
    """Build and compile the LangGraph pipeline."""
    workflow = StateGraph(GraphState)

    workflow.add_node("jd_parser", parse_jd_node)
    workflow.add_node("resume_parser", parse_resumes_node)
    workflow.add_node("linkedin_parser", parse_linkedin_node)
    workflow.add_node("profile_merge", merge_profiles_node)
    workflow.add_node("scoring", score_candidates_node)
    workflow.add_node("rank", rank_candidates_node)
    workflow.add_node("report", generate_report_node)

    workflow.set_entry_point("jd_parser")
    workflow.add_edge("jd_parser", "resume_parser")
    workflow.add_edge("resume_parser", "linkedin_parser")
    workflow.add_edge("linkedin_parser", "profile_merge")
    workflow.add_edge("profile_merge", "scoring")
    workflow.add_edge("scoring", "rank")
    workflow.add_edge("rank", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


def run_pipeline(jd_text, resume_texts, linkedin_data=None):
    """Run the full shortlisting pipeline."""
    graph = build_graph()

    initial_state = {
        "jd_text": jd_text,
        "resume_texts": resume_texts,
        "linkedin_data": linkedin_data or {},
    }

    print("=" * 60)
    print("HR SHORTLISTING PIPELINE — Starting")
    print(f"Resumes: {len(resume_texts)} | LinkedIn: {len(linkedin_data or {})}")
    print("=" * 60)

    result = graph.invoke(initial_state)

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return result


if __name__ == "__main__":
    import glob
    from utils.doc_reader import read_document

    # Read JD from data/jd/ (only document files)
    jd_files = glob.glob("data/jd/*.pdf") + glob.glob("data/jd/*.docx") + glob.glob("data/jd/*.txt")
    if not jd_files:
        print("ERROR: No JD file found in data/jd/ — drop a PDF/DOCX/TXT there.")
        exit(1)
    jd_text = read_document(jd_files[0])
    print(f"Loaded JD: {jd_files[0]}")

    # Read resumes from data/resumes/ (only document files)
    resume_files = glob.glob("data/resumes/*.pdf") + glob.glob("data/resumes/*.docx") + glob.glob("data/resumes/*.txt")
    if not resume_files:
        print("ERROR: No resumes found in data/resumes/ — drop PDF/DOCX/TXT files there.")
        exit(1)
    resume_texts = {}
    for rf in resume_files:
        resume_texts[rf] = read_document(rf)
        print(f"Loaded resume: {rf}")

    # Read LinkedIn JSONs from data/linkedin/ (optional)
    linkedin_data = {}
    for lf in glob.glob("data/linkedin/*.json"):
        with open(lf, "r", encoding="utf-8") as f:
            linkedin_data[lf] = f.read()
        print(f"Loaded LinkedIn: {lf}")

    run_pipeline(jd_text=jd_text, resume_texts=resume_texts, linkedin_data=linkedin_data)

