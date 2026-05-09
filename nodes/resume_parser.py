"""
Node 2A: Resume Parser Node
Parses resume files (PDF/DOCX) into structured CandidateProfile objects.
Uses async parallel processing via asyncio.gather().
"""

from __future__ import annotations

import asyncio

from models.schemas import CandidateProfile, GraphState
from utils.llm_client import ainvoke_with_retry, get_llm, run_async_safe
from utils.sanitiser import mask_pii, sanitise_text


RESUME_PARSE_PROMPT = """You are an expert resume parser. Extract structured information from the following resume text.

## Resume Text:
{resume_text}

## Instructions:
Extract the following fields:
- name: Candidate's full name
- email: Email address (null if not found)
- phone: Phone number (null if not found)
- skills: List of ALL technical and soft skills mentioned
- experience: List of work experiences, each with company, title, duration_months (estimate from dates), description, and domain
- education: List of education entries with institution, degree, field, year, and gpa (if mentioned)
- projects: List of projects with name, description, technologies used, and url (if any)
- certifications: List of certifications or courses
- summary: A brief professional summary based on the resume content

Be thorough. Extract every skill, experience, and project mentioned. For duration_months, calculate from start and end dates if provided."""


async def _parse_single_resume(structured_llm, filename: str, text: str) -> dict | None:
    """Parse a single resume asynchronously."""
    if not text.strip():
        print(f"[Resume Parser] Skipping empty file: {filename}")
        return None

    sanitised = sanitise_text(text, source=f"resume:{filename}")
    try:
        profile: CandidateProfile = await ainvoke_with_retry(
            structured_llm,
            RESUME_PARSE_PROMPT.format(resume_text=sanitised),
        )
        profile.source = "resume"
        profile.file_name = filename
        print(f"[Resume Parser] Parsed: {mask_pii(profile.name)} from {filename}")
        return profile.model_dump()
    except Exception as e:
        print(f"[Resume Parser] Error parsing {filename}: {e}")
        return None


async def _parse_all_resumes(structured_llm, items: list[tuple[str, str]]) -> list[dict]:
    """Parse all resumes in parallel using asyncio.gather(), with sequential retry for failures."""
    sem = asyncio.Semaphore(2)  # limit to 2 concurrent LLM calls to avoid Groq rate limits

    async def _guarded(filename, text):
        async with sem:
            return await _parse_single_resume(structured_llm, filename, text)

    tasks = [_guarded(fn, txt) for fn, txt in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    profiles = []
    failed_items = []  # track failures for sequential retry
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"[Resume Parser] Async error for {items[i][0]}: {r}")
            failed_items.append(items[i])
        elif r is not None:
            profiles.append(r)
        else:
            print(f"[Resume Parser] Returned None for {items[i][0]}")
            failed_items.append(items[i])

    # Sequential retry for any resumes that failed in parallel
    for filename, text in failed_items:
        print(f"[Resume Parser] Retrying {filename} sequentially...")
        await asyncio.sleep(3)  # wait before retry to respect rate limits
        try:
            result = await _parse_single_resume(structured_llm, filename, text)
            if result is not None:
                profiles.append(result)
                print(f"[Resume Parser] Retry succeeded for {filename}")
            else:
                print(f"[Resume Parser] Retry returned None for {filename}")
        except Exception as e:
            print(f"[Resume Parser] Retry also failed for {filename}: {e}")

    return profiles


def parse_resumes_node(state: GraphState) -> dict:
    """
    Parse all resume files into structured CandidateProfile objects (async parallel).

    Input state keys: resume_texts
    Output state keys: resume_profiles, parse_failures
    """
    resume_texts: dict = state.get("resume_texts", {})
    if not resume_texts:
        return {"resume_profiles": [], "parse_failures": []}

    llm = get_llm()
    structured_llm = llm.with_structured_output(CandidateProfile)

    items = list(resume_texts.items())
    profiles = run_async_safe(_parse_all_resumes(structured_llm, items))

    # Identify which resumes failed to parse
    parsed_filenames = {p.get("file_name") for p in profiles if p.get("file_name")}
    parse_failures = [fn for fn, _ in items if fn not in parsed_filenames]

    if parse_failures:
        print(f"[Resume Parser] WARNING: {len(parse_failures)} resume(s) could not be parsed: {parse_failures}")

    print(f"[Resume Parser] Successfully parsed {len(profiles)}/{len(items)} resume(s)")
    return {"resume_profiles": profiles, "parse_failures": parse_failures}
