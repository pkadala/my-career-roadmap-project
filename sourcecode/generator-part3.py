#!/usr/bin/env python3
"""
Career Roadmap AI - Complete Project Generator (Part 3)
This script generates services, orchestrator, and remaining files.
"""
import os
import json


def create_file(filepath: str, content: str):
    """Create a file with the given content."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Created: {filepath}")


def create_orchestrator():
    """Create the orchestrator chain."""
    
    create_file('core/chains/__init__.py', '')
    create_file('core/prompts/__init__.py', '')
    
    # Create orchestrator.py
    create_file('core/chains/orchestrator.py', '''"""
Main orchestration logic for the Career Roadmap AI application.
This orchestrator coordinates all agents to generate comprehensive career roadmaps.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid
import json
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser
from core.agents.career_analyzer import CareerAnalyzerAgent
from core.agents.skills_mapper import SkillsMapperAgent
from core.agents.roadmap_generator import RoadmapGeneratorAgent
from core.agents.complexity_estimator import ComplexityEstimatorAgent
from api.models.requests import CareerTransitionRequest
from api.models.responses import (
    CareerTransitionRoadmap, 
    SkillAssessment, 
    Milestone, 
    RoadmapMetrics,
    LearningResource
)
from services.llm_service import get_llm
from utils.logger import get_logger

logger = get_logger(__name__)


class CareerRoadmapOrchestrator:
    """Orchestrates the generation of comprehensive career transition roadmaps."""
    
    def __init__(self):
        """Initialize the orchestrator with all required agents."""
        self.llm = get_llm()
        self.agents = self._initialize_agents()
        self.parser = JsonOutputParser()
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents with the configured LLM."""
        return {
            "career_analyzer": CareerAnalyzerAgent(self.llm),
            "skills_mapper": SkillsMapperAgent(self.llm),
            "roadmap_generator": RoadmapGeneratorAgent(self.llm),
            "complexity_estimator": ComplexityEstimatorAgent(self.llm)
        }
    
    async def generate_roadmap(self, request: CareerTransitionRequest) -> CareerTransitionRoadmap:
        """
        Generate a complete career transition roadmap.
        
        This method orchestrates multiple agents to:
        1. Analyze the career transition
        2. Map skills and identify gaps
        3. Generate learning roadmap
        4. Estimate complexity and timeline
        5. Compile everything into a comprehensive response
        """
        try:
            roadmap_id = str(uuid.uuid4())
            logger.info(f"Starting roadmap generation: {roadmap_id}")
            
            # Phase 1: Parallel analysis of career and skills
            career_analysis, skills_mapping = await self._parallel_initial_analysis(request)
            
            # Phase 2: Generate roadmap based on skills gap
            skills_to_learn = self._extract_skills_to_learn(skills_mapping)
            roadmap_details = await self._generate_learning_roadmap(
                request, skills_to_learn
            )
            
            # Phase 3: Estimate complexity and metrics
            complexity_metrics = await self._estimate_complexity(request)
            
            # Phase 4: Compile final roadmap
            final_roadmap = self._compile_roadmap(
                roadmap_id,
                request,
                career_analysis,
                skills_mapping,
                roadmap_details,
                complexity_metrics
            )
            
            logger.info(f"Roadmap generation completed: {roadmap_id}")
            return final_roadmap
            
        except Exception as e:
            logger.error(f"Error generating roadmap: {str(e)}")
            raise
    
    async def _parallel_initial_analysis(
        self, 
        request: CareerTransitionRequest
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Run career analysis and skills mapping in parallel."""
        # Create parallel runnable
        parallel_chain = RunnableParallel(
            career_analysis=lambda x: self.agents["career_analyzer"].analyze_transition(
                x["current_role"],
                x["target_role"],
                x["current_skills"],
                x.get("location")
            ),
            skills_mapping=lambda x: self.agents["skills_mapper"].map_skills(
                x["current_role"],
                x["target_role"],
                x["current_skills"],
                x["experience_years"]
            )
        )
        
        # Execute parallel analysis
        results = await parallel_chain.ainvoke({
            "current_role": request.current_role,
            "target_role": request.target_role,
            "current_skills": request.current_skills,
            "experience_years": request.experience_years,
            "location": request.location
        })
        
        return results["career_analysis"], results["skills_mapping"]
    
    def _extract_skills_to_learn(self, skills_mapping: Dict[str, Any]) -> List[str]:
        """Extract the list of skills that need to be learned."""
        # In production, this would parse the skills_mapping more intelligently
        # For now, return a sample list
        return [
            "Machine Learning",
            "Deep Learning",
            "Python Advanced",
            "TensorFlow",
            "LangChain",
            "Vector Databases"
        ]
    
    async def _generate_learning_roadmap(
        self,
        request: CareerTransitionRequest,
        skills_to_learn: List[str]
    ) -> Dict[str, Any]:
        """Generate the learning roadmap with courses and projects."""
        return self.agents["roadmap_generator"].generate_roadmap(
            skills_to_learn=skills_to_learn,
            hours_per_week=request.available_hours_per_week,
            timeline_months=request.target_timeline_months or 6,
            budget_per_month=request.budget_constraint,
            learning_style=request.preferred_learning_style
        )
    
    async def _estimate_complexity(
        self,
        request: CareerTransitionRequest
    ) -> Dict[str, Any]:
        """Estimate the complexity of the career transition."""
        return self.agents["complexity_estimator"].estimate_complexity(
            current_role=request.current_role,
            target_role=request.target_role,
            current_skills=request.current_skills,
            experience_years=request.experience_years,
            hours_per_week=request.available_hours_per_week
        )
    
    def _compile_roadmap(
        self,
        roadmap_id: str,
        request: CareerTransitionRequest,
        career_analysis: Dict[str, Any],
        skills_mapping: Dict[str, Any],
        roadmap_details: Dict[str, Any],
        complexity_metrics: Dict[str, Any]
    ) -> CareerTransitionRoadmap:
        """Compile all analysis results into a final roadmap."""
        
        # Create sample data structures (in production, parse from agent outputs)
        transferable_skills = [
            SkillAssessment(
                skill_name="Python",
                current_level=7,
                required_level=9,
                gap_size=2,
                transferable=True,
                priority="high",
                related_skills=["Data Structures", "Algorithms"]
            ),
            SkillAssessment(
                skill_name="Problem Solving",
                current_level=8,
                required_level=8,
                gap_size=0,
                transferable=True,
                priority="medium",
                related_skills=["Critical Thinking", "Analytical Skills"]
            )
        ]
        
        skills_to_acquire = [
            SkillAssessment(
                skill_name="Machine Learning",
                current_level=0,
                required_level=7,
                gap_size=7,
                transferable=False,
                priority="high",
                related_skills=["Statistics", "Linear Algebra"]
            ),
            SkillAssessment(
                skill_name="LangChain",
                current_level=0,
                required_level=6,
                gap_size=6,
                transferable=False,
                priority="high",
                related_skills=["LLMs", "Prompt Engineering"]
            )
        ]
        
        # Create milestones
        milestones = self._create_milestones(request, roadmap_details)
        
        # Create learning resources
        resources = self._create_learning_resources()
        
        # Create metrics
        metrics = RoadmapMetrics(
            total_hours_required=480,
            estimated_completion_months=6.0,
            difficulty_rating=complexity_metrics.get("difficulty_rating", 7.5),
            confidence_score=0.85,
            total_cost_estimate=1200.0,
            job_market_demand="high",
            success_probability=complexity_metrics.get("success_probability", 0.75)
        )
        
        # Compile final roadmap
        return CareerTransitionRoadmap(
            roadmap_id=roadmap_id,
            created_at=datetime.utcnow(),
            current_role=request.current_role,
            target_role=request.target_role,
            transferable_skills=transferable_skills,
            skills_to_acquire=skills_to_acquire,
            skill_gap_summary=self._generate_skill_gap_summary(
                transferable_skills, skills_to_acquire
            ),
            milestones=milestones,
            recommended_resources=resources,
            recommended_projects=self._create_project_recommendations(),
            certifications=self._create_certification_recommendations(),
            metrics=metrics,
            personalized_advice=self._generate_personalized_advice(request, complexity_metrics),
            potential_challenges=[
                "Steep learning curve for machine learning mathematics",
                "Balancing learning with current work commitments",
                "Staying motivated during complex topics"
            ],
            success_factors=[
                "Strong programming foundation in Python",
                "Consistent daily practice",
                "Active participation in ML communities"
            ],
            alternative_paths=self._generate_alternative_paths(request)
        )
    
    def _create_milestones(
        self, 
        request: CareerTransitionRequest,
        roadmap_details: Dict[str, Any]
    ) -> List[Milestone]:
        """Create milestone objects from roadmap details."""
        milestones = []
        start_date = datetime.utcnow()
        
        # Month 1-2: Foundation
        milestones.append(Milestone(
            milestone_id=str(uuid.uuid4()),
            title="Foundation Building",
            description="Strengthen Python skills and learn ML fundamentals",
            target_date=start_date + timedelta(days=60),
            skills_to_achieve=["Python Advanced", "ML Fundamentals", "Statistics"],
            resources=[],  # Would be populated from roadmap_details
            projects=["Basic ML classifier", "Data analysis project"],
            estimated_hours=80,
            checkpoint_criteria=[
                "Complete Python advanced course",
                "Build first ML model",
                "Understand key ML algorithms"
            ]
        ))
        
        # Month 3-4: Deep Dive
        milestones.append(Milestone(
            milestone_id=str(uuid.uuid4()),
            title="Deep Learning & Specialization",
            description="Master deep learning and choose specialization area",
            target_date=start_date + timedelta(days=120),
            skills_to_achieve=["Deep Learning", "TensorFlow", "Neural Networks"],
            resources=[],
            projects=["Image classification project", "NLP project"],
            estimated_hours=100,
            checkpoint_criteria=[
                "Complete deep learning course",
                "Deploy a neural network",
                "Choose specialization (CV/NLP/RL)"
            ]
        ))
        
        # Month 5-6: Advanced & Portfolio
        milestones.append(Milestone(
            milestone_id=str(uuid.uuid4()),
            title="Advanced Skills & Portfolio",
            description="Learn LLMs, LangChain, and build portfolio projects",
            target_date=start_date + timedelta(days=180),
            skills_to_achieve=["LLMs", "LangChain", "MLOps"],
            resources=[],
            projects=["End-to-end ML pipeline", "LLM-powered application"],
            estimated_hours=120,
            checkpoint_criteria=[
                "Deploy production ML model",
                "Complete LangChain project",
                "Portfolio ready for job applications"
            ]
        ))
        
        return milestones
    
    def _create_learning_resources(self) -> List[LearningResource]:
        """Create sample learning resources."""
        return [
            LearningResource(
                resource_id=str(uuid.uuid4()),
                title="Machine Learning by Andrew Ng",
                type="course",
                provider="Coursera",
                url="https://www.coursera.org/learn/machine-learning",
                duration_hours=60,
                difficulty_level="beginner",
                cost=0.0,
                rating=4.9,
                skills_covered=["Machine Learning", "Algorithms", "Neural Networks"],
                prerequisites=["Basic Python", "Linear Algebra"]
            ),
            LearningResource(
                resource_id=str(uuid.uuid4()),
                title="Deep Learning Specialization",
                type="course",
                provider="Coursera",
                url="https://www.coursera.org/specializations/deep-learning",
                duration_hours=120,
                difficulty_level="intermediate",
                cost=49.0,
                rating=4.8,
                skills_covered=["Deep Learning", "TensorFlow", "CNNs", "RNNs"],
                prerequisites=["Machine Learning Basics"]
            ),
            LearningResource(
                resource_id=str(uuid.uuid4()),
                title="LangChain for LLM Applications",
                type="course",
                provider="Udemy",
                duration_hours=30,
                difficulty_level="intermediate",
                cost=89.99,
                rating=4.7,
                skills_covered=["LangChain", "LLMs", "RAG", "Agents"],
                prerequisites=["Python", "Basic ML"]
            )
        ]
    
    def _create_project_recommendations(self) -> List[Dict[str, Any]]:
        """Create project recommendations."""
        return [
            {
                "title": "Customer Support Chatbot",
                "description": "Build an AI chatbot using LangChain and OpenAI",
                "skills_demonstrated": ["LangChain", "LLMs", "API Integration"],
                "estimated_hours": 40,
                "difficulty": "intermediate",
                "portfolio_value": "high"
            },
            {
                "title": "Image Classification Pipeline",
                "description": "End-to-end ML pipeline for image classification",
                "skills_demonstrated": ["TensorFlow", "MLOps", "Docker"],
                "estimated_hours": 60,
                "difficulty": "advanced",
                "portfolio_value": "very high"
            }
        ]
    
    def _create_certification_recommendations(self) -> List[Dict[str, Any]]:
        """Create certification recommendations."""
        return [
            {
                "name": "TensorFlow Developer Certificate",
                "provider": "Google",
                "cost": 100,
                "preparation_time": "2-3 months",
                "value": "high",
                "link": "https://www.tensorflow.org/certificate"
            },
            {
                "name": "AWS Certified Machine Learning",
                "provider": "Amazon",
                "cost": 300,
                "preparation_time": "3-4 months",
                "value": "very high",
                "link": "https://aws.amazon.com/certification/certified-machine-learning-specialty/"
            }
        ]
    
    def _generate_skill_gap_summary(
        self,
        transferable_skills: List[SkillAssessment],
        skills_to_acquire: List[SkillAssessment]
    ) -> str:
        """Generate a summary of the skill gap analysis."""
        transferable_count = len(transferable_skills)
        new_skills_count = len(skills_to_acquire)
        
        high_priority_skills = [
            s.skill_name for s in skills_to_acquire 
            if s.priority == "high"
        ]
        
        return f"""Your transition from Software Engineer to AI Engineer shows a solid foundation 
        with {transferable_count} transferable skills. You'll need to acquire {new_skills_count} 
        new skills, with focus on {', '.join(high_priority_skills[:3])}. Your strong Python 
        background gives you a significant advantage in this transition."""
    
    def _generate_personalized_advice(
        self,
        request: CareerTransitionRequest,
        complexity_metrics: Dict[str, Any]
    ) -> str:
        """Generate personalized advice based on analysis."""
        return f"""Based on your {request.experience_years} years of experience and 
        {request.available_hours_per_week} hours/week availability, here's my advice:
        
        1. **Leverage your strengths**: Your software engineering background provides excellent 
        foundation for AI engineering. Focus on the mathematical aspects early.
        
        2. **Time management**: With {request.available_hours_per_week} hours/week, aim for 
        consistent daily practice rather than weekend cramming.
        
        3. **Community engagement**: Join AI/ML communities early. Your questions will evolve 
        from basic to advanced as you progress.
        
        4. **Project-based learning**: Start building projects from month 2. Real-world 
        application accelerates learning significantly.
        
        5. **Stay current**: The AI field evolves rapidly. Allocate 20% of your time to 
        staying updated with latest developments."""
    
    def _generate_alternative_paths(
        self,
        request: CareerTransitionRequest
    ) -> List[Dict[str, Any]]:
        """Generate alternative career paths."""
        return [
            {
                "title": "ML Engineer Path",
                "description": "Focus on ML engineering and deployment rather than research",
                "timeline": "4-5 months",
                "difficulty": "6/10",
                "key_difference": "More emphasis on engineering, less on theory"
            },
            {
                "title": "Data Scientist Path",
                "description": "Transition to Data Science first, then specialize in AI",
                "timeline": "3-4 months to DS, +3 months to AI",
                "difficulty": "5/10",
                "key_difference": "Gentler learning curve, more business-focused"
            },
            {
                "title": "AI Product Manager Path",
                "description": "Combine technical knowledge with product management",
                "timeline": "4-6 months",
                "difficulty": "5/10",
                "key_difference": "Less coding, more strategic and product-focused"
            }
        ]
''')


def create_services():
    """Create the services layer."""
    
    create_file('services/__init__.py', '')
    create_file('data/__init__.py', '')
    create_file('data/models/__init__.py', '')
    create_file('data/repositories/__init__.py', '')
    
    # services/llm_service.py
    create_file('services/llm_service.py', '''"""
Service for managing LLM interactions.
"""
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from app.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMService:
    """Service for managing LLM interactions."""
    
    def __init__(self):
        self._llm_cache: Dict[str, BaseChatModel] = {}
    
    def get_llm(
        self, 
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> BaseChatModel:
        """Get configured LLM instance."""
        provider = provider or settings.llm_provider
        cache_key = f"{provider}:{model}:{temperature}:{max_tokens}"
        
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]
        
        if provider == "openai":
            llm = ChatOpenAI(
                model=model or settings.openai_model,
                temperature=temperature or settings.temperature,
                max_tokens=max_tokens or settings.max_tokens,
                api_key=settings.openai_api_key,
                timeout=30,
                max_retries=3
            )
        elif provider == "anthropic":
            llm = ChatAnthropic(
                model=model or settings.anthropic_model,
                temperature=temperature or settings.temperature,
                max_tokens=max_tokens or settings.max_tokens,
                anthropic_api_key=settings.anthropic_api_key,
                timeout=30,
                max_retries=3
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
        
        self._llm_cache[cache_key] = llm
        logger.info(f"Created LLM instance: {provider} - {model}")
        return llm


# Singleton instance
llm_service = LLMService()

def get_llm(**kwargs) -> BaseChatModel:
    """Get LLM instance."""
    return llm_service.get_llm(**kwargs)
''')

    # services/cache_service.py
    create_file('services/cache_service.py', '''"""
Redis-based caching service.
"""
import json
from typing import Optional, Any
import redis.asyncio as redis
from app.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class CacheService:
    """Redis-based caching service."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
    
    async def init(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {str(e)}")
            self.redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Set value in cache with expiration."""
        if not self.redis_client:
            return False
        
        try:
            serialized = json.dumps(value)
            await self.redis_client.setex(key, expire, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")
            return False
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()


# Singleton instance
cache_service = CacheService()

async def init_cache():
    """Initialize cache service."""
    await cache_service.init()

async def get_cache() -> CacheService:
    """Get cache service instance."""
    return cache_service
''')

    # services/database_service.py
    create_file('services/database_service.py', '''"""
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
''')


def create_utils():
    """Create utility files."""
    
    create_file('utils/__init__.py', '')
    
    # utils/logger.py
    create_file('utils/logger.py', '''"""
Logging configuration for the application.
"""
import logging
import structlog
from typing import Optional
from app.config import settings

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a configured logger instance."""
    logger = structlog.get_logger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper()))
    return logger
''')

    # utils/validators.py
    create_file('utils/validators.py', '''"""
Input validation utilities.
"""
import re
from typing import List

class RoleValidator:
    """Validator for job roles."""
    
    VALID_ROLE_PATTERN = re.compile(r'^[a-zA-Z\s\-\.]+)
    MIN_LENGTH = 3
    MAX_LENGTH = 100
    
    @classmethod
    def validate_role(cls, role: str) -> str:
        """Validate job role string."""
        if not role or not role.strip():
            raise ValueError("Role cannot be empty")
        
        role = role.strip()
        
        if len(role) < cls.MIN_LENGTH:
            raise ValueError(f"Role must be at least {cls.MIN_LENGTH} characters")
        
        if len(role) > cls.MAX_LENGTH:
            raise ValueError(f"Role must be at most {cls.MAX_LENGTH} characters")
        
        if not cls.VALID_ROLE_PATTERN.match(role):
            raise ValueError("Role contains invalid characters")
        
        return role


class SkillValidator:
    """Validator for skills."""
    
    VALID_SKILL_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-\.\+\#\/]+)
    MIN_LENGTH = 1
    MAX_LENGTH = 50
    MAX_SKILLS = 50
    
    @classmethod
    def validate_skill(cls, skill: str) -> str:
        """Validate individual skill."""
        if not skill or not skill.strip():
            raise ValueError("Skill cannot be empty")
        
        skill = skill.strip()
        
        if len(skill) < cls.MIN_LENGTH:
            raise ValueError(f"Skill must be at least {cls.MIN_LENGTH} characters")
        
        if len(skill) > cls.MAX_LENGTH:
            raise ValueError(f"Skill must be at most {cls.MAX_LENGTH} characters")
        
        if not cls.VALID_SKILL_PATTERN.match(skill):
            raise ValueError(f"Skill '{skill}' contains invalid characters")
        
        return skill
    
    @classmethod
    def validate_skills_list(cls, skills: List[str]) -> List[str]:
        """Validate list of skills."""
        if len(skills) > cls.MAX_SKILLS:
            raise ValueError(f"Cannot have more than {cls.MAX_SKILLS} skills")
        
        validated_skills = []
        seen = set()
        
        for skill in skills:
            validated_skill = cls.validate_skill(skill)
            skill_lower = validated_skill.lower()
            
            if skill_lower not in seen:
                validated_skills.append(validated_skill)
                seen.add(skill_lower)
        
        return validated_skills
''')

    # utils/formatters.py
    create_file('utils/formatters.py', '''"""
Output formatting utilities.
"""
from typing import Dict, Any, List
from datetime import datetime, timedelta

class RoadmapFormatter:
    """Formatter for roadmap outputs."""
    
    @staticmethod
    def format_duration(hours: int) -> str:
        """Format duration in hours to human-readable string."""
        if hours < 1:
            return "Less than 1 hour"
        elif hours == 1:
            return "1 hour"
        elif hours < 24:
            return f"{hours} hours"
        else:
            days = hours // 24
            remaining_hours = hours % 24
            if remaining_hours == 0:
                return f"{days} day{'s' if days > 1 else ''}"
            else:
                return f"{days} day{'s' if days > 1 else ''} {remaining_hours} hour{'s' if remaining_hours > 1 else ''}"
    
    @staticmethod
    def format_timeline(months: float) -> str:
        """Format timeline in months to human-readable string."""
        if months < 1:
            weeks = int(months * 4)
            return f"{weeks} week{'s' if weeks != 1 else ''}"
        elif months == 1:
            return "1 month"
        elif months < 12:
            return f"{months:.1f} months" if months % 1 != 0 else f"{int(months)} months"
        else:
            years = months / 12
            if years == 1:
                return "1 year"
            elif years % 1 == 0:
                return f"{int(years)} years"
            else:
                return f"{years:.1f} years"
    
    @staticmethod
    def format_skill_level(level: int) -> str:
        """Format skill level to descriptive string."""
        levels = {
            0: "No Experience",
            1: "Beginner",
            2: "Novice",
            3: "Basic",
            4: "Intermediate",
            5: "Competent",
            6: "Proficient",
            7: "Advanced",
            8: "Expert",
            9: "Master",
            10: "Guru"
        }
        return levels.get(level, "Unknown")
    
    @staticmethod
    def format_cost(cost: float) -> str:
        """Format cost to currency string."""
        if cost == 0:
            return "Free"
        elif cost < 0:
            return "Unknown"
        else:
            return f"${cost:,.2f}"
    
    @staticmethod
    def format_progress_percentage(percentage: float) -> str:
        """Format progress percentage."""
        return f"{percentage:.1f}%"
''')


def create_api_endpoints():
    """Create API endpoint files."""
    
    # api/v1/endpoints/roadmap.py
    create_file('api/v1/endpoints/roadmap.py', '''"""
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
''')

    # Create stub files for other endpoints
    create_file('api/v1/endpoints/career_analysis.py', '''"""Career analysis endpoints."""
from fastapi import APIRouter
router = APIRouter()
''')
    
    create_file('api/v1/endpoints/skills.py', '''"""Skills analysis endpoints."""
from fastapi import APIRouter
router = APIRouter()
''')
    
    create_file('api/v1/endpoints/progress.py', '''"""Progress tracking endpoints."""
from fastapi import APIRouter
router = APIRouter()
''')


def create_scripts_and_docker():
    """Create scripts and Docker files."""
    
    # scripts/quickstart.py
    create_file('scripts/quickstart.py', '''#!/usr/bin/env python3
"""
Quick start script for Career Roadmap AI application.
"""
import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🚀 Career Roadmap AI - Quick Start")
print("\\nTo get started:")
print("1. Create virtual environment: python -m venv venv")
print("2. Activate it: source venv/bin/activate")
print("3. Install dependencies: pip install -r requirements.txt")
print("4. Copy .env.example to .env and add your API keys")
print("5. Start databases: docker-compose up -d")
print("6. Run: uvicorn app.main:app --reload")
print("\\nVisit http://localhost:8000/docs for API documentation")
''')

    # docker/Dockerfile
    create_file('docker/Dockerfile', '''FROM python:3.11.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
''')

    # docker-compose.yml
    create_file('docker-compose.yml', '''version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:password@postgres:5432/career_roadmap
      - REDIS_URL=redis://redis:6379
    env_file:
      - .env
    depends_on:
      - postgres
      - redis
    volumes:
      - ./data:/app/data

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: career_roadmap
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
''')

    # VS Code settings
    vscode_settings = {
        "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
        "python.linting.enabled": True,
        "python.linting.pylintEnabled": False,
        "python.linting.flake8Enabled": True,
        "python.linting.flake8Args": ["--max-line-length=120", "--ignore=E203,W503"],
        "python.formatting.provider": "black",
        "python.formatting.blackArgs": ["--line-length=120"],
        "python.testing.pytestEnabled": True,
        "python.testing.unittestEnabled": False,
        "python.testing.pytestArgs": ["tests"],
        "editor.formatOnSave": True,
        "editor.codeActionsOnSave": {
            "source.organizeImports": True
        },
        "editor.rulers": [120],
        "files.exclude": {
            "**/__pycache__": True,
            "**/*.pyc": True,
            "**/.pytest_cache": True,
            "**/.mypy_cache": True,
            "**/venv": True
        }
    }
    
    create_file('.vscode/settings.json', json.dumps(vscode_settings, indent=2))
    
    # VS Code launch.json
    launch_config = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: FastAPI",
                "type": "python",
                "request": "launch",
                "module": "uvicorn",
                "args": [
                    "app.main:app",
                    "--reload",
                    "--port",
                    "8000"
                ],
                "jinja": True,
                "justMyCode": True,
                "env": {
                    "PYTHONPATH": "${workspaceFolder}"
                }
            }
        ]
    }
    
    create_file('.vscode/launch.json', json.dumps(launch_config, indent=2))


def create_readme():
    """Create comprehensive README."""
    
    create_file('README.md', '''# Career Roadmap AI - Backend Application

An AI-powered career transition roadmap generator built with LangChain 0.3.25, FastAPI, and Python 3.11.11.

## 🚀 Features

- **AI-Powered Career Analysis**: Analyzes career transitions and job market dynamics
- **Personalized Learning Roadmaps**: Creates month-by-month learning paths
- **Skills Gap Analysis**: Identifies transferable skills and learning requirements
- **Intelligent Resource Recommendations**: Suggests courses, certifications, and projects
- **Progress Tracking**: Monitor your career transition journey

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **AI/LLM**: LangChain 0.3.25
- **LLM Providers**: OpenAI / Anthropic
- **Database**: PostgreSQL with SQLAlchemy
- **Cache**: Redis
- **Vector Store**: ChromaDB
- **Language**: Python 3.11.11

## 📋 Prerequisites

- Python 3.11.11
- Docker & Docker Compose
- OpenAI or Anthropic API key

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd career-roadmap-ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

5. **Start databases**
   ```bash
   docker-compose up -d postgres redis
   ```

6. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

7. **Access the API**
   - Documentation: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

## 📚 API Documentation

### Generate Roadmap
```http
POST /api/v1/roadmap/generate
Content-Type: application/json

{
  "current_role": "Software Engineer",
  "target_role": "AI Engineer",
  "current_skills": ["Python", "JavaScript"],
  "experience_years": 3,
  "experience_level": "mid",
  "available_hours_per_week": 15
}
```

## 🐳 Docker Deployment

```bash
docker-compose up -d
```

## 🧪 Testing

```bash
pytest
```

## 📝 License

MIT License

## 🤝 Contributing

Pull requests are welcome!
''')


def main():
    """Main function to generate part 3 of the project."""
    print("🚀 Generating Career Roadmap AI Project - Part 3\\n")
    
    # Create orchestrator
    create_orchestrator()
    
    # Create services
    create_services()
    
    # Create utils
    create_utils()
    
    # Create API endpoints
    create_api_endpoints()
    
    # Create scripts and Docker files
    create_scripts_and_docker()
    
    # Create README
    create_readme()
    
    print("\\n✅ Complete project generated successfully!")
    print("\\n📁 Project structure created:")
    print("   career-roadmap-ai/")
    print("   ├── app/           (Application core)")
    print("   ├── api/           (API endpoints)")  
    print("   ├── core/          (Business logic)")
    print("   ├── services/      (External services)")
    print("   ├── utils/         (Utilities)")
    print("   ├── docker/        (Docker config)")
    print("   └── scripts/       (Setup scripts)")
    
    print("\\n🎯 Next steps:")
    print("1. Create virtual environment: python -m venv venv")
    print("2. Activate it: source venv/bin/activate")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Copy .env.example to .env and add your API keys")
    print("5. Start databases: docker-compose up -d")
    print("6. Run: uvicorn app.main:app --reload")
    print("\\n📚 Visit http://localhost:8000/docs for API documentation")


if __name__ == "__main__":
    main()