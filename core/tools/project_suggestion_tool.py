"""
Project suggestion tool for portfolio building.
"""
from typing import List, Optional
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import json


class ProjectSuggestionTool(BaseTool):
    """Tool for suggesting hands-on projects."""
    
    name: str = "project_suggester"
    description: str = "Suggest hands-on projects to build portfolio and demonstrate skills"
    
    def _run(
        self,
        target_role: str,
        skills_to_practice: List[str],
        difficulty_level: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Suggest relevant projects."""
        projects = {
            "ai_engineer": {
                "beginner": [
                    {
                        "title": "Image Classification Web App",
                        "description": "Build a web app that classifies images using pre-trained models",
                        "skills_demonstrated": ["Python", "TensorFlow", "Flask", "Basic ML"],
                        "estimated_hours": 20,
                        "portfolio_value": "medium"
                    }
                ],
                "intermediate": [
                    {
                        "title": "Custom Chatbot with RAG",
                        "description": "Create a domain-specific chatbot using LangChain and vector databases",
                        "skills_demonstrated": ["LLMs", "LangChain", "Vector DBs", "API Integration"],
                        "estimated_hours": 40,
                        "portfolio_value": "high"
                    }
                ],
                "advanced": [
                    {
                        "title": "End-to-End ML Pipeline",
                        "description": "Build a complete ML pipeline with monitoring and deployment",
                        "skills_demonstrated": ["MLOps", "Docker", "CI/CD", "Model Monitoring"],
                        "estimated_hours": 80,
                        "portfolio_value": "very high"
                    }
                ]
            }
        }
        
        role_key = target_role.lower().replace(" ", "_")
        role_projects = projects.get(role_key, {})
        level_projects = role_projects.get(difficulty_level, [])
        
        return json.dumps({
            "suggested_projects": level_projects,
            "total_projects": len(level_projects),
            "advice": "Choose projects that align with your interests and target role"
        }, indent=2)
    
    async def _arun(
        self,
        target_role: str,
        skills_to_practice: List[str],
        difficulty_level: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async version."""
        return self._run(target_role, skills_to_practice, difficulty_level, run_manager)


def create_career_tools() -> List[BaseTool]:
    """Create and return all career transition tools."""
    return [
        JobMarketResearchTool(),
        SkillsDatabaseTool(),
        CourseFindingTool(),
        CertificationFinderTool(),
        ProjectSuggestionTool()
    ]
