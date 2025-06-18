"""
Roadmap generation endpoints.
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import Optional, Dict, Any
import uuid
from datetime import datetime
from api.models.requests import CareerTransitionRequest, ProgressUpdateRequest
from api.models.responses import (
    CareerTransitionRoadmap, 
    ProgressReport, 
    ErrorResponse,
    SuccessResponse
)
from core.chains.orchestrator import CareerRoadmapOrchestrator
from services.cache_service import get_cache
from services.database_service import get_db
from app.dependencies import get_current_user_optional
from utils.logger import get_logger
from app.config import settings

router = APIRouter()
logger = get_logger(__name__)

# Initialize orchestrator
orchestrator = CareerRoadmapOrchestrator()


@router.post("/generate", response_model=CareerTransitionRoadmap)
async def generate_roadmap(
    request: CareerTransitionRequest,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = Depends(get_current_user_optional),
    cache = Depends(get_cache)
) -> CareerTransitionRoadmap:
    """
    Generate a personalized career transition roadmap.
    
    This endpoint analyzes the career transition request and generates a comprehensive
    roadmap including:
    - Skills gap analysis
    - Learning milestones
    - Resource recommendations
    - Time and complexity estimates
    """
    try:

        logger.info(f"START Generating roadmap for ")

        # Generate cache key
        cache_key = f"roadmap:{request.current_role}:{request.target_role}:{user_id}"
        
        # Check cache
        if settings.enable_caching:
            cached_roadmap = await cache.get(cache_key)
            if cached_roadmap:
                logger.info(f"Returning cached roadmap for user {user_id}")
                return CareerTransitionRoadmap(**cached_roadmap)
        
        # Generate new roadmap
        logger.info(f"Generating roadmap for {request.current_role} -> {request.target_role}")
        roadmap = await orchestrator.generate_roadmap(request)
        
        # Cache the result
        if settings.enable_caching:
            await cache.set(cache_key, roadmap.dict(), expire=settings.cache_ttl)
        
        # Save to database in background
        background_tasks.add_task(save_roadmap_to_db, roadmap, user_id)
        
        # Track analytics in background
        if settings.enable_analytics:
            background_tasks.add_task(track_roadmap_generation, roadmap, user_id)
        
        return roadmap
        
    except Exception as e:
        logger.error(f"Error generating roadmap: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate roadmap: {str(e)}"
        )


@router.get("/roadmap/{roadmap_id}", response_model=CareerTransitionRoadmap)
async def get_roadmap(
    roadmap_id: str,
    user_id: Optional[str] = Depends(get_current_user_optional),
    db = Depends(get_db)
) -> CareerTransitionRoadmap:
    """Get a specific roadmap by ID."""
    try:
        # Fetch from database
        roadmap = await db.get_roadmap(roadmap_id, user_id)
        
        if not roadmap:
            raise HTTPException(
                status_code=404,
                detail=f"Roadmap {roadmap_id} not found"
            )
        
        return roadmap
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching roadmap: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch roadmap: {str(e)}"
        )


# Helper functions
async def save_roadmap_to_db(roadmap: CareerTransitionRoadmap, user_id: str):
    """Save roadmap to database (background task)."""
    try:
        # Implementation would save to actual database
        logger.info(f"Saved roadmap {roadmap.roadmap_id} for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to save roadmap: {str(e)}")


async def track_roadmap_generation(roadmap: CareerTransitionRoadmap, user_id: str):
    """Track analytics for roadmap generation (background task)."""
    try:
        # Implementation would send to analytics service
        logger.info(f"Tracked roadmap generation for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to track analytics: {str(e)}")


def calculate_progress(
    roadmap: CareerTransitionRoadmap, 
    update: ProgressUpdateRequest
) -> ProgressReport:
    """Calculate progress based on completed items."""
    total_milestones = len(roadmap.milestones)
    completed_milestones = len([
        m for m in roadmap.milestones 
        if m.milestone_id in update.completed_items
    ])
    
    progress_percentage = (completed_milestones / total_milestones * 100) if total_milestones > 0 else 0
    
    return ProgressReport(
        roadmap_id=roadmap.roadmap_id,
        overall_progress_percentage=progress_percentage,
        completed_milestones=completed_milestones,
        total_milestones=total_milestones,
        skills_acquired=list(update.current_skill_levels.keys()),
        current_phase=f"Milestone {completed_milestones + 1}" if completed_milestones < total_milestones else "Completed",
        estimated_completion_date=datetime.utcnow(),  # Would be calculated based on pace
        recommendations=[
            "Keep up the great work!",
            "Consider joining study groups for peer learning"
        ],
        achievement_badges=[]
    )
