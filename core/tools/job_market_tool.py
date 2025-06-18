"""
Job market research tool for career transitions.
"""
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
import json


class JobMarketInput(BaseModel):
    """Input schema for job market research tool."""
    role: str = Field(description="Job role to research")
    location: Optional[str] = Field(default=None, description="Location for job market data")
    include_salary: bool = Field(default=True, description="Include salary information")


class JobMarketResearchTool(BaseTool):
    """Tool for researching job market data and trends."""
    
    name: str = "job_market_research"
    description: str = "Research job market data including demand, salary ranges, and required skills for a specific role"
    args_schema: type = JobMarketInput
    
    def _run(
        self,
        role: str,
        location: Optional[str] = None,
        include_salary: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute job market research."""
        # Simulate job market data retrieval
        # In production, this would connect to real APIs (LinkedIn, Indeed, etc.)
        
        market_data = {
            "role": role,
            "location": location or "United States",
            "demand_level": "high",
            "growth_rate": "15% annually",
            "average_salary": {
                "entry_level": "$65,000 - $85,000",
                "mid_level": "$95,000 - $130,000",
                "senior_level": "$140,000 - $200,000"
            },
            "top_companies": ["Google", "Microsoft", "Amazon", "Meta", "Apple"],
            "required_skills": {
                "must_have": ["Python", "Machine Learning", "Deep Learning", "TensorFlow/PyTorch"],
                "nice_to_have": ["MLOps", "Cloud Platforms", "Docker", "Kubernetes"],
                "emerging": ["LLMs", "Prompt Engineering", "Vector Databases"]
            },
            "job_postings_count": 15420,
            "remote_percentage": 65,
            "market_insights": [
                "High demand for AI/ML engineers with LLM experience",
                "Companies prioritizing production ML skills",
                "Growing emphasis on MLOps and deployment"
            ]
        }
        
        if include_salary:
            return json.dumps(market_data, indent=2)
        else:
            market_data.pop("average_salary", None)
            return json.dumps(market_data, indent=2)
    
    async def _arun(
        self,
        role: str,
        location: Optional[str] = None,
        include_salary: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async version of job market research."""
        return self._run(role, location, include_salary, run_manager)
