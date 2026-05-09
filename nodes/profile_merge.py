"""
Node 3: Profile Merge Node
Merges resume and LinkedIn profiles for the same candidate.
Resume data wins on conflicts.
"""

from __future__ import annotations

from models.schemas import CandidateProfile, GraphState
from utils.sanitiser import mask_pii


def _normalize_name(name: str) -> str:
    """Normalize a name for matching (lowercase, strip whitespace)."""
    return " ".join(name.lower().strip().split())


def _merge_two_profiles(resume_profile: dict, linkedin_profile: dict) -> dict:
    """
    Merge a resume profile with a LinkedIn profile.
    Resume data takes priority on conflicts.
    LinkedIn data fills in gaps.
    """
    resume = CandidateProfile(**resume_profile)
    linkedin = CandidateProfile(**linkedin_profile)
    merged = resume.model_copy()

    # Fill in missing contact info from LinkedIn
    if not merged.email and linkedin.email:
        merged.email = linkedin.email
    if not merged.phone and linkedin.phone:
        merged.phone = linkedin.phone

    # Merge skills (union, deduplicated)
    all_skills = list(set(
        [s.lower().strip() for s in merged.skills] +
        [s.lower().strip() for s in linkedin.skills]
    ))
    merged.skills = sorted(all_skills)

    # Add LinkedIn projects not in resume
    resume_project_names = {p.get("name", "").lower() for p in resume_profile.get("projects", [])}
    for proj in linkedin.projects:
        if proj.name.lower() not in resume_project_names:
            merged.projects.append(proj)

    # Add LinkedIn certifications not in resume
    resume_certs = {c.lower() for c in merged.certifications}
    for cert in linkedin.certifications:
        if cert.lower() not in resume_certs:
            merged.certifications.append(cert)

    # Use LinkedIn summary if resume summary is empty
    if not merged.summary.strip() and linkedin.summary.strip():
        merged.summary = linkedin.summary

    merged.source = "merged"
    return merged.model_dump()


def merge_profiles_node(state: GraphState) -> dict:
    """
    Merge resume and LinkedIn profiles. Match by normalized name.
    Candidates with only one source pass through unchanged.

    Input state keys: resume_profiles, linkedin_profiles
    Output state keys: candidate_profiles
    """
    resume_profiles: list[dict] = state.get("resume_profiles", [])
    linkedin_profiles: list[dict] = state.get("linkedin_profiles", [])

    # Index LinkedIn profiles by normalized name
    linkedin_by_name: dict[str, dict] = {}
    for lp in linkedin_profiles:
        norm_name = _normalize_name(lp.get("name", ""))
        if norm_name:
            linkedin_by_name[norm_name] = lp

    merged_profiles = []
    matched_linkedin_names = set()

    # Process each resume profile
    for rp in resume_profiles:
        norm_name = _normalize_name(rp.get("name", ""))
        linkedin_match = linkedin_by_name.get(norm_name)

        if linkedin_match:
            # Merge resume + LinkedIn
            merged = _merge_two_profiles(rp, linkedin_match)
            matched_linkedin_names.add(norm_name)
            print(f"[Profile Merge] Merged resume + LinkedIn for: {mask_pii(rp.get('name', ''))}")
        else:
            # Resume only — pass through
            merged = rp.copy()
            print(f"[Profile Merge] Resume only for: {mask_pii(rp.get('name', ''))}")

        merged_profiles.append(merged)

    # Add any LinkedIn-only profiles
    for lp in linkedin_profiles:
        norm_name = _normalize_name(lp.get("name", ""))
        if norm_name not in matched_linkedin_names:
            merged_profiles.append(lp)
            print(f"[Profile Merge] LinkedIn only for: {mask_pii(lp.get('name', ''))}")

    print(f"[Profile Merge] Total candidates: {len(merged_profiles)}")
    return {"candidate_profiles": merged_profiles}
