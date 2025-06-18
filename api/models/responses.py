"""
API response models for the Career Roadmap AI application.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class SkillAssessment(BaseModel):
    """Skill assessment details."""
    skill_name: str
    current_level: int = Field(..., ge=0, le=10)
    required_level: int = Field(..., ge=0, le=10)
    gap_size: int
    transferable: bool
    priority: str  # "high", "medium", "low"
    related_skills: List[str] = []


class LearningResource(BaseModel):
    """Learning resource details."""
    resource_id: str
    title: str
    type: str  # "course", "book", "project", "certification", "article"
    provider: str
    url: Optional[str] = None
    duration_hours: int
    difficulty_level: str
    cost: float = 0.0
    rating: Optional[float] = None
    skills_covered: List[str]
    prerequisites: List[str] = []


class Milestone(BaseModel):
    """Roadmap milestone."""
    milestone_id: str
    title: str
    description: str
    target_date: datetime
    skills_to_achieve: List[str]
    resources: List[LearningResource]
    projects: List[str]
    estimated_hours: int
    checkpoint_criteria: List[str]


class RoadmapMetrics(BaseModel):
    """Roadmap metrics and estimates."""
    total_hours_required: int
    estimated_completion_months: float
    difficulty_rating: float = Field(..., ge=1, le=10)
    confidence_score: float = Field(..., ge=0, le=1)
    total_cost_estimate: float
    job_market_demand: str  # "high", "medium", "low"
    success_probability: float = Field(..., ge=0, le=1)


class CareerTransitionRoadmap(BaseModel):
    """Complete career transition roadmap response."""
    roadmap_id: str
    created_at: datetime
    current_role: str
    target_role: str
    
    # Skills analysis
    transferable_skills: List[SkillAssessment]
    skills_to_acquire: List[SkillAssessment]
    skill_gap_summary: str
    
    # Learning path
    milestones: List[Milestone]
    recommended_resources: List[LearningResource]
    recommended_projects: List[Dict[str, Any]]
    certifications: List[Dict[str, Any]]
    
    # Metrics and estimates
    metrics: RoadmapMetrics
    
    # Personalized insights
    personalized_advice: str
    potential_challenges: List[str]
    success_factors: List[str]
    alternative_paths: List[Dict[str, Any]]


class ProgressReport(BaseModel):
    """Progress tracking report."""
    roadmap_id: str
    overall_progress_percentage: float
    completed_milestones: int
    total_milestones: int
    skills_acquired: List[str]
    current_phase: str
    estimated_completion_date: datetime
    recommendations: List[str]
    achievement_badges: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SuccessResponse(BaseModel):
    """Standard success response."""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
