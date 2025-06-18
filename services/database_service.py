"""
Database service for roadmap persistence.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, DateTime, JSON, Integer, Float, Boolean, select, func
from app.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()

# Database Models
class RoadmapModel(Base):
    """Database model for roadmaps."""
    __tablename__ = "roadmaps"
    
    roadmap_id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    current_role = Column(String)
    target_role = Column(String)
    data = Column(JSON)  # Full roadmap data
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProgressModel(Base):
    """Database model for progress tracking."""
    __tablename__ = "progress"
    
    id = Column(Integer, primary_key=True)
    roadmap_id = Column(String, index=True)
    user_id = Column(String, index=True)
    progress_percentage = Column(Float)
    completed_items = Column(JSON)
    skill_levels = Column(JSON)
    updated_at = Column(DateTime, default=datetime.utcnow)


class DatabaseService:
    """Database service for roadmap persistence."""
    
    def __init__(self):
        self.engine = None
        self.async_session = None
    
    async def init(self):
        """Initialize database connection."""
        try:
            self.engine = create_async_engine(
                settings.database_url,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                echo=settings.debug
            )
            
            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    async def save_roadmap(self, roadmap_data: Dict[str, Any], user_id: str) -> bool:
        """Save roadmap to database."""
        try:
            async with self.async_session() as session:
                roadmap = RoadmapModel(
                    roadmap_id=roadmap_data["roadmap_id"],
                    user_id=user_id,
                    current_role=roadmap_data["current_role"],
                    target_role=roadmap_data["target_role"],
                    data=roadmap_data
                )
                session.add(roadmap)
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save roadmap: {str(e)}")
            return False
    
    async def get_roadmap(self, roadmap_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get roadmap from database."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(RoadmapModel).where(
                        RoadmapModel.roadmap_id == roadmap_id,
                        RoadmapModel.user_id == user_id
                    )
                )
                roadmap = result.scalar_one_or_none()
                return roadmap.data if roadmap else None
        except Exception as e:
            logger.error(f"Failed to get roadmap: {str(e)}")
            return None
    
    async def list_roadmaps(
        self, 
        user_id: str, 
        skip: int = 0, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """List roadmaps for a user."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(RoadmapModel)
                    .where(RoadmapModel.user_id == user_id)
                    .order_by(RoadmapModel.created_at.desc())
                    .offset(skip)
                    .limit(limit)
                )
                roadmaps = result.scalars().all()
                return [r.data for r in roadmaps]
        except Exception as e:
            logger.error(f"Failed to list roadmaps: {str(e)}")
            return []
    
    async def count_roadmaps(self, user_id: str) -> int:
        """Count roadmaps for a user."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(func.count())
                    .select_from(RoadmapModel)
                    .where(RoadmapModel.user_id == user_id)
                )
                return result.scalar() or 0
        except Exception as e:
            logger.error(f"Failed to count roadmaps: {str(e)}")
            return 0
    
    async def delete_roadmap(self, roadmap_id: str, user_id: str) -> bool:
        """Delete a roadmap."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(RoadmapModel).where(
                        RoadmapModel.roadmap_id == roadmap_id,
                        RoadmapModel.user_id == user_id
                    )
                )
                roadmap = result.scalar_one_or_none()
                if roadmap:
                    await session.delete(roadmap)
                    await session.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to delete roadmap: {str(e)}")
            return False
    
    async def save_progress(
        self, 
        roadmap_id: str, 
        user_id: str, 
        progress_data: Dict[str, Any]
    ) -> bool:
        """Save progress update."""
        try:
            async with self.async_session() as session:
                progress = ProgressModel(
                    roadmap_id=roadmap_id,
                    user_id=user_id,
                    progress_percentage=progress_data.get("overall_progress_percentage", 0),
                    completed_items=progress_data.get("completed_items", []),
                    skill_levels=progress_data.get("skill_levels", {})
                )
                session.add(progress)
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save progress: {str(e)}")
            return False


# Singleton instance
db_service = DatabaseService()

async def init_db():
    """Initialize database service."""
    await db_service.init()

async def get_db() -> DatabaseService:
    """Get database service instance."""
    return db_service
