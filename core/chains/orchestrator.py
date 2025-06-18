"""
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
            roadmap_details = self._generate_learning_roadmap(
                request, skills_to_learn
            )
            
            # Phase 3: Estimate complexity and metrics
            complexity_metrics = self._estimate_complexity(request)
            
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
        # Use synchronous calls instead of RunnableParallel to avoid agent_scratchpad issues
        career_analysis = self.agents["career_analyzer"].analyze_transition(
            request.current_role,
            request.target_role,
            request.current_skills,
            request.location
        )
        
        skills_mapping = self.agents["skills_mapper"].map_skills(
            request.current_skills,
            request.target_role,
            request.experience_level
        )
        
        return career_analysis, skills_mapping
    
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
    
    def _generate_learning_roadmap(
        self,
        request: CareerTransitionRequest,
        skills_to_learn: List[str]
    ) -> Dict[str, Any]:
        """Generate the learning roadmap with courses and projects."""
        return self.agents["roadmap_generator"].generate_roadmap(
            current_role=request.current_role,
            target_role=request.target_role,
            current_skills=request.current_skills,
            experience_years=request.experience_years,
            available_hours=request.available_hours_per_week,
            preferred_learning_style=request.preferred_learning_style
        )
    
    def _estimate_complexity(
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
