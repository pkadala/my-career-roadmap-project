"""
API v1 router configuration.
"""
from fastapi import APIRouter
from api.v1.endpoints import roadmap, career_analysis, skills, progress

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(roadmap.router, prefix="/roadmap", tags=["roadmap"])
api_router.include_router(career_analysis.router, prefix="/career", tags=["career"])
api_router.include_router(skills.router, prefix="/skills", tags=["skills"])
api_router.include_router(progress.router, prefix="/progress", tags=["progress"])
