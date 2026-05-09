"""
Node 1: JD Parser Node
Parses a Job Description file and generates a structured ParsedJD + JD embedding.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from models.schemas import GraphState, ParsedJD
from utils.embeddings import get_embedding
from utils.llm_client import get_llm, invoke_with_retry
from utils.sanitiser import sanitise_text


JD_PARSE_PROMPT = """You are an expert HR analyst. Extract structured information from the following Job Description.

## Job Description Text:
{jd_text}

## Instructions:
Extract the following fields accurately:
- title: The exact job title
- required_skills: List of explicitly required skills, technologies, and competencies
- preferred_skills: List of preferred/nice-to-have skills (if mentioned)
- min_experience_years: Minimum years of experience required (use 0 if not specified)
- education_requirements: List of education requirements (degrees, fields of study)
- responsibilities: List of key job responsibilities

Be thorough and extract ALL mentioned skills and requirements. Do not invent information not present in the JD."""


def parse_jd_node(state: GraphState) -> dict:
    """
    Parse the Job Description text into a structured format and generate its embedding.

    Input state keys: jd_text
    Output state keys: parsed_jd, jd_embedding
    """
    jd_text = state["jd_text"]
    sanitised_jd = sanitise_text(jd_text, source="jd")

    # Use LLM to extract structured JD
    llm = get_llm()
    structured_llm = llm.with_structured_output(ParsedJD)
    parsed_jd: ParsedJD = invoke_with_retry(
        structured_llm,
        JD_PARSE_PROMPT.format(jd_text=sanitised_jd),
    )
    parsed_jd.raw_text = jd_text

    # Generate JD embedding from key fields
    embedding_text = (
        f"{parsed_jd.title}. "
        f"Required skills: {', '.join(parsed_jd.required_skills)}. "
        f"Preferred skills: {', '.join(parsed_jd.preferred_skills)}. "
        f"Responsibilities: {'. '.join(parsed_jd.responsibilities)}. "
        f"Education: {', '.join(parsed_jd.education_requirements)}."
    )
    jd_embedding = get_embedding(embedding_text)

    print(f"[JD Parser] Parsed JD: {parsed_jd.title}")
    print(f"[JD Parser] Required skills: {len(parsed_jd.required_skills)}")
    print(f"[JD Parser] Preferred skills: {len(parsed_jd.preferred_skills)}")

    return {
        "parsed_jd": parsed_jd.model_dump(),
        "jd_embedding": jd_embedding,
    }
