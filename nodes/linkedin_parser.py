"""
Node 2B: LinkedIn Parser Node
Parses exported LinkedIn JSON files into structured CandidateProfile objects.
Uses async parallel processing via asyncio.gather().
"""

from __future__ import annotations

import asyncio
import json

from models.schemas import CandidateProfile, GraphState
from utils.llm_client import ainvoke_with_retry, get_llm, run_async_safe
from utils.sanitiser import mask_pii, sanitise_text


LINKEDIN_PARSE_PROMPT = """You are an expert profile parser. Extract structured candidate information from the following LinkedIn profile data (exported as JSON).

## LinkedIn Profile Data:
{linkedin_json}

## Instructions:
Extract the following fields:
- name: Candidate's full name
- email: Email address (null if not found)
- phone: Phone number (null if not found)
- skills: List of ALL skills mentioned in the profile
- experience: List of work experiences, each with company, title, duration_months (estimate if needed), description, and domain
- education: List of education entries with institution, degree, field, year, and gpa (if mentioned)
- projects: List of projects with name, description, technologies used, and url (if any)
- certifications: List of certifications, courses, or licenses
- summary: The professional headline or summary from the profile

Be thorough and extract all available information from the LinkedIn data."""


async def _parse_single_linkedin(structured_llm, filename: str, json_string: str) -> dict | None:
    """Parse a single LinkedIn JSON file asynchronously."""
    try:
        json_obj = json.loads(json_string) if isinstance(json_string, str) else json_string
        formatted = json.dumps(json_obj, indent=2)
    except json.JSONDecodeError as e:
        print(f"[LinkedIn Parser] Invalid JSON in {filename}: {e}")
        return None

    sanitised = sanitise_text(formatted, source=f"linkedin:{filename}")
    try:
        profile: CandidateProfile = await ainvoke_with_retry(
            structured_llm,
            LINKEDIN_PARSE_PROMPT.format(linkedin_json=sanitised),
        )
        profile.source = "linkedin"
        profile.file_name = filename
        print(f"[LinkedIn Parser] Parsed: {mask_pii(profile.name)} from {filename}")
        return profile.model_dump()
    except Exception as e:
        print(f"[LinkedIn Parser] Error parsing {filename}: {e}")
        return None


async def _parse_all_linkedin(structured_llm, items: list[tuple[str, str]]) -> list[dict]:
    """Parse all LinkedIn profiles in parallel using asyncio.gather()."""
    sem = asyncio.Semaphore(3)

    async def _guarded(filename, json_str):
        async with sem:
            return await _parse_single_linkedin(structured_llm, filename, json_str)

    tasks = [_guarded(fn, js) for fn, js in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    profiles = []
    for r in results:
        if isinstance(r, Exception):
            print(f"[LinkedIn Parser] Async error: {r}")
        elif r is not None:
            profiles.append(r)
    return profiles


def parse_linkedin_node(state: GraphState) -> dict:
    """
    Parse all LinkedIn JSON files into structured CandidateProfile objects (async parallel).

    Input state keys: linkedin_data
    Output state keys: linkedin_profiles
    """
    linkedin_data: dict = state.get("linkedin_data", {})
    if not linkedin_data:
        print("[LinkedIn Parser] No LinkedIn data provided, skipping.")
        return {"linkedin_profiles": []}

    llm = get_llm()
    structured_llm = llm.with_structured_output(CandidateProfile)

    items = list(linkedin_data.items())
    profiles = run_async_safe(_parse_all_linkedin(structured_llm, items))

    return {"linkedin_profiles": profiles}
