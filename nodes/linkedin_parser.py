"""
Node 2B: LinkedIn Parser Node
Parses exported LinkedIn JSON files into structured CandidateProfile objects.
Uses sequential processing with aggressive retry to ensure all profiles are parsed.
"""

from __future__ import annotations

import json
import time

from models.schemas import CandidateProfile, GraphState
from utils.llm_client import get_llm, invoke_with_retry
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


def _parse_single_linkedin(structured_llm, filename: str, json_string: str) -> dict | None:
    """Parse a single LinkedIn JSON file with aggressive retry — will NOT give up easily."""
    try:
        json_obj = json.loads(json_string) if isinstance(json_string, str) else json_string
        formatted = json.dumps(json_obj, indent=2)
    except json.JSONDecodeError as e:
        print(f"[LinkedIn Parser] Invalid JSON in {filename}: {e}")
        return None

    sanitised = sanitise_text(formatted, source=f"linkedin:{filename}")

    # Aggressive retry: 8 attempts with increasing waits
    max_attempts = 8
    for attempt in range(max_attempts):
        try:
            profile: CandidateProfile = invoke_with_retry(
                structured_llm,
                LINKEDIN_PARSE_PROMPT.format(linkedin_json=sanitised),
                max_retries=5,
                wait_seconds=5,
            )
            profile.source = "linkedin"
            profile.file_name = filename
            print(f"[LinkedIn Parser] Parsed: {mask_pii(profile.name)} from {filename}")
            return profile.model_dump()
        except Exception as e:
            wait = 10 * (attempt + 1)  # 10s, 20s, 30s, ...
            print(f"[LinkedIn Parser] Attempt {attempt+1}/{max_attempts} failed for {filename}: {e}")
            if attempt < max_attempts - 1:
                print(f"[LinkedIn Parser] Waiting {wait}s before retrying...")
                time.sleep(wait)

    print(f"[LinkedIn Parser] FAILED all {max_attempts} attempts for {filename}")
    return None


def parse_linkedin_node(state: GraphState) -> dict:
    """
    Parse all LinkedIn JSON files into structured CandidateProfile objects.
    Waits for Groq rate limits to reset before starting, then processes each profile.

    Input state keys: linkedin_data
    Output state keys: linkedin_profiles
    """
    linkedin_data: dict = state.get("linkedin_data", {})
    if not linkedin_data:
        print("[LinkedIn Parser] No LinkedIn data provided, skipping.")
        return {"linkedin_profiles": []}

    # Cooldown: wait for Groq rate limit to reset after resume parsing
    print("[LinkedIn Parser] Waiting 15s for Groq rate limit cooldown...")
    time.sleep(15)

    llm = get_llm()
    structured_llm = llm.with_structured_output(CandidateProfile)

    items = list(linkedin_data.items())
    profiles = []

    for idx, (filename, json_str) in enumerate(items):
        print(f"[LinkedIn Parser] Parsing profile {idx + 1}/{len(items)}: {filename}")
        result = _parse_single_linkedin(structured_llm, filename, json_str)

        if result is not None:
            profiles.append(result)

        # Delay between profiles to avoid rate limits
        if idx < len(items) - 1:
            print("[LinkedIn Parser] Waiting 5s between profiles...")
            time.sleep(5)

    print(f"[LinkedIn Parser] Successfully parsed {len(profiles)}/{len(items)} profile(s)")
    return {"linkedin_profiles": profiles}
