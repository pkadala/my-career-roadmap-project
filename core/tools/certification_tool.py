"""
Certification finder tool.
"""
from typing import List, Optional
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import json


class CertificationFinderTool(BaseTool):
    """Tool for finding relevant certifications."""
    
    name: str = "certification_finder"
    description: str = "Find industry-recognized certifications for career advancement"
    
    def _run(
        self,
        role: str,
        skills: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Find relevant certifications."""
        certifications = {
            "ai_engineer": [
                {
                    "name": "TensorFlow Developer Certificate",
                    "provider": "Google",
                    "cost": 100,
                    "duration_months": 3,
                    "difficulty": "intermediate",
                    "value": "high"
                },
                {
                    "name": "AWS Certified Machine Learning",
                    "provider": "Amazon",
                    "cost": 300,
                    "duration_months": 6,
                    "difficulty": "advanced",
                    "value": "very high"
                }
            ],
            "software_engineer": [
                {
                    "name": "AWS Certified Developer",
                    "provider": "Amazon",
                    "cost": 300,
                    "duration_months": 4,
                    "difficulty": "intermediate",
                    "value": "high"
                }
            ]
        }
        
        role_key = role.lower().replace(" ", "_")
        certs = certifications.get(role_key, [])
        
        return json.dumps({
            "certifications": certs,
            "role": role,
            "recommendation": "Focus on certifications that align with your target role"
        }, indent=2)
    
    async def _arun(
        self,
        role: str,
        skills: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async version."""
        return self._run(role, skills, run_manager)
