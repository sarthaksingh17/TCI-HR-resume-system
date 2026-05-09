"""
Node 4: Scoring Node
Scores each candidate against the JD using LLM-based rubric evaluation.
Uses 5 dimensions with assignment-specified weights and 0-10 scale.
"""

from __future__ import annotations

from models.schemas import (
    CandidateProfile,
    DimensionScore,
    GraphState,
    LLMScoringOutput,
    ParsedJD,
    Recommendation,
    RubricScore,
    ScoringDimension,
    Tier,
)
from utils.embeddings import cosine_similarity, get_embedding
from utils.llm_client import get_llm, invoke_with_retry
from utils.sanitiser import mask_pii, sanitise_text


# Weights from the assignment (must sum to 1.0)
DIMENSION_WEIGHTS = {
    ScoringDimension.SKILLS_MATCH: 0.30,
    ScoringDimension.EXPERIENCE_RELEVANCE: 0.25,
    ScoringDimension.EDUCATION_CERTS: 0.15,
    ScoringDimension.PROJECT_PORTFOLIO: 0.20,
    ScoringDimension.COMMUNICATION_QUALITY: 0.10,
}


SCORING_PROMPT = """You are an expert HR evaluator. Score the following candidate against the Job Description using the rubric below.

## Job Description:
Title: {jd_title}
Required Skills: {required_skills}
Preferred Skills: {preferred_skills}
Min Experience: {min_experience} years
Education Requirements: {education_reqs}
Responsibilities: {responsibilities}

## Candidate Profile:
Name: {candidate_name}
Skills: {candidate_skills}
Experience: {candidate_experience}
Education: {candidate_education}
Projects: {candidate_projects}
Certifications: {candidate_certs}
Summary: {candidate_summary}

## Scoring Rubric (score each dimension 0-10):

### 1. Skills Match (Weight: 30%)
- 0: Less than 30% of required skills matched
- 5: 50-70% of required skills matched, some preferred skills
- 10: Over 85% of required skills matched plus preferred skills

### 2. Experience Relevance (Weight: 25%)
- 0: No relevant experience or less than minimum years
- 5: Meets minimum experience, some domain relevance
- 10: Exceeds experience requirements with highly relevant roles and domain expertise

### 3. Education & Certifications (Weight: 15%)
- 0: Does not meet education requirements
- 5: Meets basic education requirements
- 10: Exceeds education requirements with relevant certifications and advanced degrees

### 4. Project / Portfolio (Weight: 20%)
- 0: No relevant projects or portfolio
- 5: Some projects demonstrating relevant skills
- 10: Strong portfolio with projects directly relevant to the role using required technologies

### 5. Communication Quality (Weight: 10%)
- 0: Poorly written resume, unclear descriptions
- 5: Adequate writing, reasonably clear descriptions
- 10: Exceptionally well-written, clear and concise descriptions showing strong communication

## Instructions:
- Score each dimension from 0 to 10 based on the rubric descriptors above.
- Provide a one-line justification for each score.
- Identify any required skills the candidate is missing (skill_gaps).
- Be fair and consistent. Score based only on evidence in the profile."""


def _format_experience(experience: list[dict]) -> str:
    """Format work experience list into readable text."""
    if not experience:
        return "None listed"
    parts = []
    for exp in experience:
        duration = f"{exp.get('duration_months', 'N/A')} months" if exp.get('duration_months') else "N/A duration"
        parts.append(
            f"- {exp.get('title', 'N/A')} at {exp.get('company', 'N/A')} ({duration}): "
            f"{exp.get('description', 'N/A')}"
        )
    return "\n".join(parts)


def _format_education(education: list[dict]) -> str:
    """Format education list into readable text."""
    if not education:
        return "None listed"
    parts = []
    for edu in education:
        gpa = f", GPA: {edu['gpa']}" if edu.get('gpa') else ""
        parts.append(
            f"- {edu.get('degree', 'N/A')} in {edu.get('field', 'N/A')} "
            f"from {edu.get('institution', 'N/A')} ({edu.get('year', 'N/A')}{gpa})"
        )
    return "\n".join(parts)


def _format_projects(projects: list[dict]) -> str:
    """Format projects list into readable text."""
    if not projects:
        return "None listed"
    parts = []
    for proj in projects:
        tech = ", ".join(proj.get("technologies", []))
        parts.append(
            f"- {proj.get('name', 'N/A')}: {proj.get('description', 'N/A')} "
            f"[Technologies: {tech or 'N/A'}]"
        )
    return "\n".join(parts)


def _compute_tier_and_recommendation(weighted_total: float) -> tuple[str, str]:
    """Map weighted total to tier and hire recommendation.
    Converts 0-10 score to 0-100 percentage for threshold comparison:
    Strong >= 75%, Borderline 50-74%, Weak < 50% (per assignment rubric).
    """
    percentage = weighted_total * 10  # Convert 0-10 scale to 0-100
    if percentage >= 75:
        return Tier.STRONG.value, Recommendation.HIRE.value
    elif percentage >= 50:
        return Tier.BORDERLINE.value, Recommendation.MAYBE.value
    else:
        return Tier.WEAK.value, Recommendation.NO_HIRE.value


def score_candidates_node(state: GraphState) -> dict:
    """
    Score each candidate against the parsed JD using LLM rubric evaluation.

    Input state keys: parsed_jd, jd_embedding, candidate_profiles
    Output state keys: rubric_scores, candidate_embeddings
    """
    parsed_jd = ParsedJD(**state["parsed_jd"])
    jd_embedding = state["jd_embedding"]
    candidate_profiles: list[dict] = state.get("candidate_profiles", [])

    llm = get_llm()
    structured_llm = llm.with_structured_output(LLMScoringOutput)

    rubric_scores = []
    candidate_embeddings = {}

    for cp_dict in candidate_profiles:
        cp = CandidateProfile(**cp_dict)
        print(f"[Scoring] Scoring candidate: {mask_pii(cp.name)}")

        # Generate candidate embedding
        candidate_text = (
            f"{cp.name}. Skills: {', '.join(cp.skills)}. "
            f"Summary: {cp.summary}. "
            f"Certifications: {', '.join(cp.certifications)}."
        )
        candidate_emb = get_embedding(candidate_text)
        candidate_embeddings[cp.name] = candidate_emb

        # Compute cosine similarity with JD
        sim_score = cosine_similarity(jd_embedding, candidate_emb)

        # Build the scoring prompt
        prompt = SCORING_PROMPT.format(
            jd_title=parsed_jd.title,
            required_skills=", ".join(parsed_jd.required_skills),
            preferred_skills=", ".join(parsed_jd.preferred_skills),
            min_experience=parsed_jd.min_experience_years,
            education_reqs=", ".join(parsed_jd.education_requirements),
            responsibilities="; ".join(parsed_jd.responsibilities),
            candidate_name=cp.name,
            candidate_skills=", ".join(cp.skills),
            candidate_experience=_format_experience([e.model_dump() if hasattr(e, 'model_dump') else e for e in cp.experience]),
            candidate_education=_format_education([e.model_dump() if hasattr(e, 'model_dump') else e for e in cp.education]),
            candidate_projects=_format_projects([p.model_dump() if hasattr(p, 'model_dump') else p for p in cp.projects]),
            candidate_certs=", ".join(cp.certifications) if cp.certifications else "None",
            candidate_summary=cp.summary or "Not provided",
        )

        try:
            llm_output: LLMScoringOutput = invoke_with_retry(structured_llm, prompt)

            # Build dimension scores
            dimension_scores = [
                DimensionScore(
                    dimension=ScoringDimension.SKILLS_MATCH.value,
                    score=llm_output.skills_match_score,
                    weight=DIMENSION_WEIGHTS[ScoringDimension.SKILLS_MATCH],
                    justification=llm_output.skills_match_justification,
                ),
                DimensionScore(
                    dimension=ScoringDimension.EXPERIENCE_RELEVANCE.value,
                    score=llm_output.experience_score,
                    weight=DIMENSION_WEIGHTS[ScoringDimension.EXPERIENCE_RELEVANCE],
                    justification=llm_output.experience_justification,
                ),
                DimensionScore(
                    dimension=ScoringDimension.EDUCATION_CERTS.value,
                    score=llm_output.education_score,
                    weight=DIMENSION_WEIGHTS[ScoringDimension.EDUCATION_CERTS],
                    justification=llm_output.education_justification,
                ),
                DimensionScore(
                    dimension=ScoringDimension.PROJECT_PORTFOLIO.value,
                    score=llm_output.project_score,
                    weight=DIMENSION_WEIGHTS[ScoringDimension.PROJECT_PORTFOLIO],
                    justification=llm_output.project_justification,
                ),
                DimensionScore(
                    dimension=ScoringDimension.COMMUNICATION_QUALITY.value,
                    score=llm_output.communication_score,
                    weight=DIMENSION_WEIGHTS[ScoringDimension.COMMUNICATION_QUALITY],
                    justification=llm_output.communication_justification,
                ),
            ]

            # Compute weighted total
            weighted_total = sum(ds.score * ds.weight for ds in dimension_scores)
            tier, recommendation = _compute_tier_and_recommendation(weighted_total)

            rubric_score = RubricScore(
                candidate_name=cp.name,
                dimension_scores=dimension_scores,
                weighted_total=round(weighted_total, 2),
                similarity_score=round(sim_score, 4),
                recommendation=recommendation,
                tier=tier,
                skill_gaps=llm_output.skill_gaps,
            )
            rubric_scores.append(rubric_score.model_dump())
            print(f"[Scoring] {mask_pii(cp.name)}: {weighted_total:.2f}/10 -> {tier} ({recommendation})")

        except Exception as e:
            print(f"[Scoring] Error scoring {mask_pii(cp.name)}: {e}")

    return {
        "rubric_scores": rubric_scores,
        "candidate_embeddings": candidate_embeddings,
    }
