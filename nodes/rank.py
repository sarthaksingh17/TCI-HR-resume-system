"""
Node 5: Rank Node
Sorts candidates by weighted total score, groups into tiers, and runs bias audit.
"""

from __future__ import annotations

from models.schemas import BiasFlag, GraphState, RankedResult, RubricScore, Tier
from utils.bias_audit import run_bias_audit
from utils.sanitiser import mask_pii


def rank_candidates_node(state: GraphState) -> dict:
    """
    Rank all scored candidates and run bias audit.

    Input state keys: rubric_scores, candidate_embeddings
    Output state keys: ranked_result
    """
    raw_scores: list[dict] = state.get("rubric_scores", [])
    candidate_embeddings: dict = state.get("candidate_embeddings", {})

    if not raw_scores:
        print("[Rank] No scores to rank.")
        return {
            "ranked_result": RankedResult(
                ranked_candidates=[],
                tier_groups={},
                bias_audit=[],
            ).model_dump()
        }

    # Deserialize into RubricScore objects
    scores = [RubricScore(**s) for s in raw_scores]

    # Sort descending by weighted total
    scores.sort(key=lambda s: s.weighted_total, reverse=True)

    # Group into tiers
    tier_groups: dict[str, list[str]] = {
        Tier.STRONG.value: [],
        Tier.BORDERLINE.value: [],
        Tier.WEAK.value: [],
    }
    for score in scores:
        tier_groups[score.tier].append(score.candidate_name)

    print(f"[Rank] Ranking {len(scores)} candidates:")
    for i, s in enumerate(scores, 1):
        print(f"  {i}. {mask_pii(s.candidate_name)}: {s.weighted_total:.2f}/10 ({s.tier})")

    # Run bias audit
    bias_flags: list[BiasFlag] = run_bias_audit(scores, candidate_embeddings)
    if bias_flags:
        print(f"[Rank] Bias audit found {len(bias_flags)} flag(s):")
        for flag in bias_flags:
            print(f"  - [{flag.severity.upper()}] {flag.flag_type}: {flag.description}")
    else:
        print("[Rank] Bias audit: No flags raised.")

    # Attach bias flags to individual candidate scores
    serialized_scores = []
    for score in scores:
        score_dict = score.model_dump()
        # Collect flags that mention this candidate
        candidate_flags = [
            f.description for f in bias_flags
            if score.candidate_name in f.candidate_names
        ]
        score_dict["bias_flags"] = candidate_flags
        serialized_scores.append(score_dict)

    ranked_result = RankedResult(
        ranked_candidates=[RubricScore(**s) for s in serialized_scores],
        tier_groups=tier_groups,
        bias_audit=bias_flags,
    )

    return {"ranked_result": ranked_result.model_dump()}
