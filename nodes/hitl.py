"""
Node 7: Human-in-the-Loop (HITL) Override Node
Accepts score overrides from reviewers, recalculates totals, and logs the audit trail.
"""

from __future__ import annotations

from datetime import datetime

from models.schemas import (
    DimensionScore,
    GraphState,
    HITLOverride,
    Recommendation,
    RubricScore,
    Tier,
)
from utils.sanitiser import mask_pii


def _compute_tier_and_recommendation(weighted_total: float) -> tuple[str, str]:
    """Map weighted total to tier and hire recommendation.
    Converts 0-10 score to 0-100 percentage for threshold comparison.
    """
    percentage = weighted_total * 10
    if percentage >= 75:
        return Tier.STRONG.value, Recommendation.HIRE.value
    elif percentage >= 50:
        return Tier.BORDERLINE.value, Recommendation.MAYBE.value
    else:
        return Tier.WEAK.value, Recommendation.NO_HIRE.value


def apply_hitl_override(
    state: GraphState,
    candidate_name: str,
    dimension: str,
    new_score: float,
    reason: str,
    reviewer: str = "HR Reviewer",
) -> dict:
    """
    Apply a human override to a candidate's dimension score.

    This function:
    1. Finds the candidate in rubric_scores
    2. Logs the original → new score change
    3. Recalculates weighted total, tier, and recommendation
    4. Returns updated state keys

    Args:
        state: Current graph state.
        candidate_name: Name of the candidate to override.
        dimension: Which scoring dimension to change.
        new_score: New score (0-10).
        reason: Reason for the override.
        reviewer: Who made the override.

    Returns:
        Dict with updated rubric_scores and overrides.
    """
    rubric_scores: list[dict] = list(state.get("rubric_scores", []))
    overrides: list[dict] = list(state.get("overrides", []))

    # Find the candidate
    target_idx = None
    for i, score_dict in enumerate(rubric_scores):
        if score_dict["candidate_name"] == candidate_name:
            target_idx = i
            break

    if target_idx is None:
        print(f"[HITL] Candidate '{candidate_name}' not found. Override skipped.")
        return {}

    score = RubricScore(**rubric_scores[target_idx])

    # Find the dimension to override
    original_score = None
    for ds in score.dimension_scores:
        if ds.dimension == dimension:
            original_score = ds.score
            ds.score = new_score
            ds.justification = f"[HITL Override] {ds.justification} -> Overridden to {new_score}: {reason}"
            break

    if original_score is None:
        print(f"[HITL] Dimension '{dimension}' not found for {candidate_name}. Override skipped.")
        return {}

    # Log the override
    override_record = HITLOverride(
        candidate_name=candidate_name,
        dimension=dimension,
        original_score=original_score,
        new_score=new_score,
        reason=reason,
        timestamp=datetime.now().isoformat(),
        reviewer=reviewer,
    )
    overrides.append(override_record.model_dump())

    # Recalculate weighted total
    new_weighted_total = sum(ds.score * ds.weight for ds in score.dimension_scores)
    score.weighted_total = round(new_weighted_total, 2)

    # Recalculate tier and recommendation
    score.tier, score.recommendation = _compute_tier_and_recommendation(new_weighted_total)

    # Update the score in the list
    rubric_scores[target_idx] = score.model_dump()

    print(
        f"[HITL] Override applied: {mask_pii(candidate_name)} / {dimension}: "
        f"{original_score} -> {new_score} (reason: {reason})"
    )
    print(f"[HITL] New weighted total: {score.weighted_total}/10 -> {score.tier} ({score.recommendation})")

    return {
        "rubric_scores": rubric_scores,
        "overrides": overrides,
    }
