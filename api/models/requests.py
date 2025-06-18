"""
API request models for the Career Roadmap AI application.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class ExperienceLevel(str, Enum):
    """Experience level enumeration."""
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    EXPERT = "expert"


class CareerTransitionRequest(BaseModel):
    """Request model for career transition analysis."""
    current_role: str = Field(..., description="Current job title/role")
    target_role: str = Field(..., description="Target job title/role")
    current_skills: List[str] = Field(default=[], description="List of current skills")
    experience_years: int = Field(..., ge=0, le=50, description="Years of experience")
    experience_level: ExperienceLevel = Field(..., description="Current experience level")
    available_hours_per_week: int = Field(..., ge=1, le=100, description="Hours available for learning per week")
    preferred_learning_style: Optional[str] = Field(None, description="Preferred learning style")
    budget_constraint: Optional[float] = Field(None, ge=0, description="Monthly budget for learning")
    target_timeline_months: Optional[int] = Field(None, ge=1, le=60, description="Target timeline in months")
    industry_preference: Optional[str] = Field(None, description="Preferred industry")
    location: Optional[str] = Field(None, description="Location for job market analysis")
    
    @validator('current_skills')
    def validate_skills(cls, v):
        return [skill.strip() for skill in v if skill.strip()]


class SkillsAnalysisRequest(BaseModel):
    """Request model for skills gap analysis."""
    current_role: str
    target_role: str
    current_skills: List[str]
    include_soft_skills: bool = True
    include_technical_skills: bool = True


class ProgressUpdateRequest(BaseModel):
    """Request model for progress updates."""
    roadmap_id: str
    completed_items: List[str]
    current_skill_levels: Dict[str, int]  # skill_name: proficiency_level (1-10)
    feedback: Optional[str] = None
