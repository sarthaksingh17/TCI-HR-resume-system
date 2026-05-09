"""
Bias audit utility — checks for scoring anomalies and potential biases.

Checks performed:
1. Similar candidates (by embedding) with very different scores → flag
2. Keyword stuffing: high skill count but no project evidence → flag
3. Scoring variance too high across the batch → flag
"""

from __future__ import annotations

import numpy as np

from models.schemas import BiasFlag, RubricScore
from utils.embeddings import cosine_similarity


def run_bias_audit(
    scores: list[RubricScore],
    candidate_embeddings: dict[str, list[float]] | None = None,
) -> list[BiasFlag]:
    """
    Run bias audit checks on a batch of scored candidates.

    Args:
        scores: List of RubricScore objects for all candidates.
        candidate_embeddings: Optional dict mapping candidate names to their embeddings.

    Returns:
        List of BiasFlag objects describing any detected issues.
    """
    flags: list[BiasFlag] = []

    flags.extend(_check_similar_candidates_different_scores(scores, candidate_embeddings))
    flags.extend(_check_keyword_stuffing(scores))
    flags.extend(_check_scoring_variance(scores))

    return flags


def _check_similar_candidates_different_scores(
    scores: list[RubricScore],
    candidate_embeddings: dict[str, list[float]] | None,
) -> list[BiasFlag]:
    """Flag pairs of candidates with similar profiles but very different scores."""
    flags = []
    if not candidate_embeddings or len(scores) < 2:
        return flags

    names = [s.candidate_name for s in scores]
    score_map = {s.candidate_name: s.weighted_total for s in scores}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            emb_a = candidate_embeddings.get(name_a)
            emb_b = candidate_embeddings.get(name_b)

            if emb_a is None or emb_b is None:
                continue

            sim = cosine_similarity(emb_a, emb_b)
            score_diff = abs(score_map[name_a] - score_map[name_b])

            # High similarity (>0.85) but large score difference (>2.0 on 0-10 scale)
            if sim > 0.85 and score_diff > 2.0:
                flags.append(BiasFlag(
                    flag_type="Similar Profile Score Discrepancy",
                    description=(
                        f"{name_a} and {name_b} have similar profiles "
                        f"(similarity: {sim:.2f}) but a score difference of {score_diff:.1f}. "
                        f"Review for potential inconsistency."
                    ),
                    candidate_names=[name_a, name_b],
                    severity="high" if score_diff > 3.0 else "medium",
                ))

    return flags


def _check_keyword_stuffing(scores: list[RubricScore]) -> list[BiasFlag]:
    """Flag candidates with high skills-match but low project scores (keyword stuffing)."""
    flags = []
    for score in scores:
        skills_score = None
        project_score = None
        for ds in score.dimension_scores:
            if ds.dimension == "Skills Match":
                skills_score = ds.score
            elif ds.dimension == "Project / Portfolio":
                project_score = ds.score

        if skills_score is not None and project_score is not None:
            # High skills (>=7) but very low projects (<=3) → suspicious
            if skills_score >= 7.0 and project_score <= 3.0:
                flags.append(BiasFlag(
                    flag_type="Potential Keyword Stuffing",
                    description=(
                        f"{score.candidate_name} has high skills match ({skills_score:.1f}/10) "
                        f"but low project evidence ({project_score:.1f}/10). "
                        f"Skills may be listed without practical demonstration."
                    ),
                    candidate_names=[score.candidate_name],
                    severity="medium",
                ))

    return flags


def _check_scoring_variance(scores: list[RubricScore]) -> list[BiasFlag]:
    """Flag if scoring variance across the batch is unusually high."""
    flags = []
    if len(scores) < 3:
        return flags

    totals = [s.weighted_total for s in scores]
    variance = float(np.var(totals))
    std_dev = float(np.std(totals))

    # High variance threshold (std dev > 3.0 on a 0-10 scale is very spread out)
    if std_dev > 3.0:
        flags.append(BiasFlag(
            flag_type="High Scoring Variance",
            description=(
                f"Scoring variance across {len(scores)} candidates is unusually high "
                f"(std dev: {std_dev:.2f}). This may indicate inconsistent evaluation criteria."
            ),
            candidate_names=[s.candidate_name for s in scores],
            severity="medium",
        ))

    return flags
