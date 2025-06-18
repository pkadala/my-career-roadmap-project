"""
Skills database tool for querying role requirements.
"""
from typing import Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
import json


class SkillsDatabaseInput(BaseModel):
    """Input schema for skills database tool."""
    role: str = Field(description="Job role to analyze")
    skill_type: Optional[str] = Field(default=None, description="Type of skills: technical, soft, or both")


class SkillsDatabaseTool(BaseTool):
    """Tool for querying skills database and relationships."""
    
    name: str = "skills_database"
    description: str = "Query comprehensive skills database for role requirements and skill relationships"
    args_schema: type = SkillsDatabaseInput
    
    def _run(
        self,
        role: str,
        skill_type: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Query skills database."""
        # Simulated skills database
        # In production, this would query a vector database or skills ontology
        
        skills_data = {
            "software_engineer": {
                "technical": {
                    "core": ["Programming", "Data Structures", "Algorithms", "Git", "Testing"],
                    "frameworks": ["React", "Node.js", "Django", "Spring Boot"],
                    "tools": ["Docker", "CI/CD", "Cloud Platforms"],
                    "databases": ["SQL", "NoSQL", "Redis"]
                },
                "soft": ["Problem Solving", "Communication", "Teamwork", "Time Management"]
            },
            "ai_engineer": {
                "technical": {
                    "core": ["Python", "Machine Learning", "Deep Learning", "Mathematics", "Statistics"],
                    "frameworks": ["TensorFlow", "PyTorch", "Scikit-learn", "Transformers"],
                    "tools": ["Jupyter", "MLflow", "Weights & Biases", "Docker"],
                    "specialized": ["NLP", "Computer Vision", "LLMs", "Fine-tuning"]
                },
                "soft": ["Analytical Thinking", "Research Skills", "Communication", "Collaboration"]
            },
            "data_scientist": {
                "technical": {
                    "core": ["Python/R", "Statistics", "Machine Learning", "Data Visualization"],
                    "tools": ["Pandas", "NumPy", "Matplotlib", "Tableau", "SQL"],
                    "advanced": ["A/B Testing", "Experimentation", "Causal Inference"]
                },
                "soft": ["Business Acumen", "Storytelling", "Critical Thinking"]
            }
        }
        
        # Normalize role name
        role_key = role.lower().replace(" ", "_")
        
        # Get skills for the role
        if role_key in skills_data:
            role_skills = skills_data[role_key]
        else:
            # Return generic skills if role not found
            role_skills = {
                "technical": {
                    "core": ["Programming", "Problem Solving", "Technical Documentation"],
                    "tools": ["Version Control", "IDEs", "Debugging Tools"]
                },
                "soft": ["Communication", "Teamwork", "Adaptability"]
            }
        
        # Filter by skill type if specified
        if skill_type:
            if skill_type.lower() in role_skills:
                return json.dumps({role: {skill_type: role_skills[skill_type]}}, indent=2)
        
        return json.dumps({role: role_skills}, indent=2)
    
    async def _arun(
        self,
        role: str,
        skill_type: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async version."""
        return self._run(role, skill_type, run_manager)
