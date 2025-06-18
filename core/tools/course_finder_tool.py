"""
Course finding tool for learning resources.
"""
from typing import List, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
import json


class CourseFindInput(BaseModel):
    """Input schema for course finder tool."""
    skills: List[str] = Field(description="Skills to find courses for")
    level: str = Field(description="Skill level: beginner, intermediate, advanced")
    max_duration_hours: Optional[int] = Field(default=None, description="Maximum course duration")
    free_only: bool = Field(default=False, description="Only free courses")


class CourseFindingTool(BaseTool):
    """Tool for finding relevant courses and learning resources."""
    
    name: str = "course_finder"
    description: str = "Find relevant courses and learning resources for specific skills"
    args_schema: type = CourseFindInput
    
    def _run(
        self,
        skills: List[str],
        level: str,
        max_duration_hours: Optional[int] = None,
        free_only: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Find courses for given skills."""
        # Simulated course database
        # In production, integrate with Coursera, Udemy, edX APIs
        
        courses = []
        
        # Sample course templates
        course_templates = {
            "python": {
                "beginner": [
                    {
                        "title": "Python for Everybody",
                        "provider": "Coursera",
                        "duration_hours": 40,
                        "cost": 0,
                        "rating": 4.8,
                        "url": "https://coursera.org/python-everybody"
                    },
                    {
                        "title": "Complete Python Bootcamp",
                        "provider": "Udemy",
                        "duration_hours": 60,
                        "cost": 89.99,
                        "rating": 4.7,
                        "url": "https://udemy.com/python-bootcamp"
                    }
                ],
                "intermediate": [
                    {
                        "title": "Advanced Python Programming",
                        "provider": "Pluralsight",
                        "duration_hours": 30,
                        "cost": 29.99,
                        "rating": 4.6
                    }
                ]
            },
            "machine learning": {
                "beginner": [
                    {
                        "title": "Machine Learning by Andrew Ng",
                        "provider": "Coursera",
                        "duration_hours": 60,
                        "cost": 0,
                        "rating": 4.9,
                        "url": "https://coursera.org/ml-andrew-ng"
                    }
                ],
                "intermediate": [
                    {
                        "title": "Deep Learning Specialization",
                        "provider": "Coursera",
                        "duration_hours": 120,
                        "cost": 49.99,
                        "rating": 4.8
                    }
                ]
            }
        }
        
        # Generate courses for each skill
        for skill in skills:
            skill_lower = skill.lower()
            
            if skill_lower in course_templates and level in course_templates[skill_lower]:
                skill_courses = course_templates[skill_lower][level]
            else:
                # Generate generic course
                skill_courses = [
                    {
                        "title": f"{skill} Fundamentals",
                        "provider": "Online Platform",
                        "duration_hours": 20,
                        "cost": 0 if free_only else 49.99,
                        "rating": 4.5
                    }
                ]
            
            # Filter by duration and cost
            for course in skill_courses:
                if max_duration_hours and course["duration_hours"] > max_duration_hours:
                    continue
                if free_only and course["cost"] > 0:
                    continue
                
                course["skills_covered"] = [skill]
                course["level"] = level
                courses.append(course)
        
        return json.dumps({
            "courses": courses,
            "total_found": len(courses),
            "search_criteria": {
                "skills": skills,
                "level": level,
                "free_only": free_only,
                "max_duration": max_duration_hours
            }
        }, indent=2)
    
    async def _arun(
        self,
        skills: List[str],
        level: str,
        max_duration_hours: Optional[int] = None,
        free_only: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async version."""
        return self._run(skills, level, max_duration_hours, free_only, run_manager)
