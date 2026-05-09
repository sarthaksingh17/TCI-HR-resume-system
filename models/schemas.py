"""
Pydantic models and LangGraph state definition for the HR Resume Shortlisting Agent.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, TypedDict

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class ScoringDimension(str, Enum):
    SKILLS_MATCH = "Skills Match"
    EXPERIENCE_RELEVANCE = "Experience Relevance"
    EDUCATION_CERTS = "Education & Certs"
    PROJECT_PORTFOLIO = "Project / Portfolio"
    COMMUNICATION_QUALITY = "Communication Quality"


class Tier(str, Enum):
    STRONG = "Strong"
    BORDERLINE = "Borderline"
    WEAK = "Weak"


class Recommendation(str, Enum):
    HIRE = "Hire"
    MAYBE = "Maybe"
    NO_HIRE = "No Hire"


# ─────────────────────────────────────────────
# JD Models
# ─────────────────────────────────────────────

class ParsedJD(BaseModel):
    """Structured representation of a Job Description."""
    title: str = Field(description="Job title")
    required_skills: list[str] = Field(description="List of required skills/technologies")
    preferred_skills: list[str] = Field(default_factory=list, description="List of preferred/nice-to-have skills")
    min_experience_years: int = Field(default=0, description="Minimum years of experience required")
    education_requirements: list[str] = Field(default_factory=list, description="Education requirements")
    responsibilities: list[str] = Field(default_factory=list, description="Key responsibilities")
    raw_text: str = Field(default="", description="Original JD text")


# ─────────────────────────────────────────────
# Candidate Profile Models
# ─────────────────────────────────────────────

class WorkExperience(BaseModel):
    """A single work experience entry."""
    company: str = Field(description="Company name")
    title: str = Field(description="Job title")
    duration_months: Optional[int] = Field(default=None, description="Duration in months")
    description: str = Field(default="", description="Role description")
    domain: Optional[str] = Field(default=None, description="Industry/domain")


class Education(BaseModel):
    """A single education entry."""
    institution: str = Field(description="Institution name")
    degree: str = Field(description="Degree obtained")
    field: str = Field(default="", description="Field of study")
    year: Optional[int] = Field(default=None, description="Graduation year")
    gpa: Optional[float] = Field(default=None, description="GPA if available")


class Project(BaseModel):
    """A single project entry."""
    name: str = Field(description="Project name")
    description: str = Field(default="", description="Project description")
    technologies: list[str] = Field(default_factory=list, description="Technologies used")
    url: Optional[str] = Field(default=None, description="Project URL if available")


class CandidateProfile(BaseModel):
    """Structured candidate profile extracted from resume or LinkedIn."""
    name: str = Field(description="Candidate full name")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    skills: list[str] = Field(default_factory=list, description="List of skills")
    experience: list[WorkExperience] = Field(default_factory=list, description="Work experience entries")
    education: list[Education] = Field(default_factory=list, description="Education entries")
    projects: list[Project] = Field(default_factory=list, description="Project entries")
    certifications: list[str] = Field(default_factory=list, description="Certifications")
    summary: str = Field(default="", description="Professional summary")
    source: str = Field(default="resume", description="Source: resume, linkedin, or merged")
    file_name: Optional[str] = Field(default=None, description="Original file name")


# ─────────────────────────────────────────────
# Scoring Models
# ─────────────────────────────────────────────

class DimensionScore(BaseModel):
    """Score for a single rubric dimension."""
    dimension: str = Field(description="Dimension name")
    score: float = Field(ge=0, le=10, description="Score from 0 to 10")
    weight: float = Field(description="Weight as a decimal (e.g., 0.30)")
    justification: str = Field(description="One-line justification for the score")


class RubricScore(BaseModel):
    """Complete rubric scoring for a candidate."""
    candidate_name: str = Field(description="Candidate name")
    dimension_scores: list[DimensionScore] = Field(description="Scores for each dimension")
    weighted_total: float = Field(description="Weighted total score (0-10 scale)")
    similarity_score: float = Field(default=0.0, description="Cosine similarity with JD embedding")
    recommendation: str = Field(description="Hire / Maybe / No Hire")
    tier: str = Field(description="Strong / Borderline / Weak")
    skill_gaps: list[str] = Field(default_factory=list, description="Skills the candidate is missing")
    bias_flags: list[str] = Field(default_factory=list, description="Any bias flags for this candidate")


# ─────────────────────────────────────────────
# LLM Structured Output for Scoring
# ─────────────────────────────────────────────

class LLMScoringOutput(BaseModel):
    """Structured output from the LLM for candidate scoring."""
    skills_match_score: float = Field(ge=0, le=10, description="Skills Match score (0-10)")
    skills_match_justification: str = Field(description="One-line justification for Skills Match score")
    experience_score: float = Field(ge=0, le=10, description="Experience Relevance score (0-10)")
    experience_justification: str = Field(description="One-line justification for Experience score")
    education_score: float = Field(ge=0, le=10, description="Education & Certs score (0-10)")
    education_justification: str = Field(description="One-line justification for Education score")
    project_score: float = Field(ge=0, le=10, description="Project/Portfolio score (0-10)")
    project_justification: str = Field(description="One-line justification for Project score")
    communication_score: float = Field(ge=0, le=10, description="Communication Quality score (0-10)")
    communication_justification: str = Field(description="One-line justification for Communication score")
    skill_gaps: list[str] = Field(default_factory=list, description="Required skills the candidate is missing")


# ─────────────────────────────────────────────
# Bias Audit Models
# ─────────────────────────────────────────────

class BiasFlag(BaseModel):
    """A single bias audit flag."""
    flag_type: str = Field(description="Type of bias flag")
    description: str = Field(description="Description of the issue")
    candidate_names: list[str] = Field(default_factory=list, description="Candidates involved")
    severity: str = Field(default="medium", description="low / medium / high")


# ─────────────────────────────────────────────
# Ranking Models
# ─────────────────────────────────────────────

class RankedResult(BaseModel):
    """Complete ranking result with bias audit."""
    ranked_candidates: list[RubricScore] = Field(description="Candidates sorted by score descending")
    tier_groups: dict[str, list[str]] = Field(description="Tier -> list of candidate names")
    bias_audit: list[BiasFlag] = Field(default_factory=list, description="Bias audit flags")


# ─────────────────────────────────────────────
# HITL Override Models
# ─────────────────────────────────────────────

class HITLOverride(BaseModel):
    """Record of a human-in-the-loop score override."""
    candidate_name: str = Field(description="Candidate whose score was overridden")
    dimension: str = Field(description="Dimension that was changed")
    original_score: float = Field(description="Original score before override")
    new_score: float = Field(ge=0, le=10, description="New score after override")
    reason: str = Field(description="Reason for the override")
    timestamp: str = Field(description="ISO timestamp of the override")
    reviewer: str = Field(default="HR Reviewer", description="Who made the override")


# ─────────────────────────────────────────────
# LangGraph State
# ─────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    """State object that flows through the LangGraph pipeline."""
    # Inputs
    jd_text: str
    resume_texts: dict  # {filename: extracted_text}
    linkedin_data: dict  # {filename: json_string}

    # After JD parsing
    parsed_jd: dict  # serialized ParsedJD
    jd_embedding: list

    # After candidate parsing
    resume_profiles: list  # list of serialized CandidateProfile
    linkedin_profiles: list
    candidate_profiles: list  # merged/final profiles
    parse_failures: list  # filenames that failed to parse

    # After scoring
    rubric_scores: list  # list of serialized RubricScore
    candidate_embeddings: dict  # {candidate_name: embedding_vector}

    # After ranking
    ranked_result: dict  # serialized RankedResult

    # Output
    report_html: str
    report_json: str

    # HITL
    overrides: list  # list of serialized HITLOverride
