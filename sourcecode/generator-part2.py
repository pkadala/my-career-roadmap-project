#!/usr/bin/env python3
"""
Career Roadmap AI - Complete Project Generator (Part 2)
This script generates the core tools and agents.
"""
import os
from pathlib import Path


def create_file(filepath: str, content: str):
    """Create a file with the given content."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Created: {filepath}")


def create_core_tools():
    """Create the core tools."""
    
    # Create __init__.py files
    create_file('core/__init__.py', '')
    create_file('core/tools/__init__.py', '')
    
    # core/tools/job_market_tool.py
    create_file('core/tools/job_market_tool.py', '''"""
Job market research tool for career transitions.
"""
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
import json


class JobMarketInput(BaseModel):
    """Input schema for job market research tool."""
    role: str = Field(description="Job role to research")
    location: Optional[str] = Field(default=None, description="Location for job market data")
    include_salary: bool = Field(default=True, description="Include salary information")


class JobMarketResearchTool(BaseTool):
    """Tool for researching job market data and trends."""
    
    name = "job_market_research"
    description = "Research job market data including demand, salary ranges, and required skills for a specific role"
    args_schema = JobMarketInput
    
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
''')

    # core/tools/skills_database_tool.py
    create_file('core/tools/skills_database_tool.py', '''"""
Skills database tool for querying role requirements.
"""
from typing import Optional
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
import json


class SkillsDatabaseInput(BaseModel):
    """Input schema for skills database tool."""
    role: str = Field(description="Job role to analyze")
    skill_type: Optional[str] = Field(default=None, description="Type of skills: technical, soft, or both")


class SkillsDatabaseTool(BaseTool):
    """Tool for querying skills database and relationships."""
    
    name = "skills_database"
    description = "Query comprehensive skills database for role requirements and skill relationships"
    args_schema = SkillsDatabaseInput
    
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
''')

    # core/tools/course_finder_tool.py
    create_file('core/tools/course_finder_tool.py', '''"""
Course finding tool for learning resources.
"""
from typing import List, Optional
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
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
    
    name = "course_finder"
    description = "Find relevant courses and learning resources for specific skills"
    args_schema = CourseFindInput
    
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
''')

    # core/tools/certification_tool.py
    create_file('core/tools/certification_tool.py', '''"""
Certification finder tool.
"""
from typing import List
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import json


class CertificationFinderTool(BaseTool):
    """Tool for finding relevant certifications."""
    
    name = "certification_finder"
    description = "Find industry-recognized certifications for career advancement"
    
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
''')

    # core/tools/project_suggestion_tool.py
    create_file('core/tools/project_suggestion_tool.py', '''"""
Project suggestion tool for portfolio building.
"""
from typing import List
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import json


class ProjectSuggestionTool(BaseTool):
    """Tool for suggesting hands-on projects."""
    
    name = "project_suggester"
    description = "Suggest hands-on projects to build portfolio and demonstrate skills"
    
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
''')


def create_core_agents():
    """Create the core agents."""
    
    create_file('core/agents/__init__.py', '')
    
    # core/agents/career_analyzer.py
    create_file('core/agents/career_analyzer.py', '''"""
Career analyzer agent for analyzing career transitions.
"""
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from app.config import settings
from core.tools.job_market_tool import JobMarketResearchTool
from core.tools.skills_database_tool import SkillsDatabaseTool
from utils.logger import get_logger

logger = get_logger(__name__)


class CareerAnalyzerAgent:
    """Agent for analyzing career transitions and market dynamics."""
    
    def __init__(self, llm=None):
        """Initialize the career analyzer agent."""
        self.llm = llm or self._get_default_llm()
        self.tools = [JobMarketResearchTool(), SkillsDatabaseTool()]
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5
        )
        self.agent = self._create_agent()
    
    def _get_default_llm(self):
        """Get default LLM based on configuration."""
        if settings.llm_provider == "openai":
            return ChatOpenAI(
                model=settings.openai_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                api_key=settings.openai_api_key
            )
        else:
            return ChatAnthropic(
                model=settings.anthropic_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                api_key=settings.anthropic_api_key
            )
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor."""
        system_message = """You are a Career Transition Analyst specializing in tech careers.
        Your role is to analyze career transitions, understand job market dynamics, and identify skill requirements.
        
        When analyzing a career transition:
        1. Research the current and target roles thoroughly
        2. Understand the job market demand and trends
        3. Identify the key skills required for each role
        4. Analyze the gap between current and target positions
        5. Consider industry standards and best practices
        
        Always provide data-driven insights based on your research tools.
        Be realistic about transition complexity and timeframes.
        Focus on actionable information that helps users plan their career journey."""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def analyze_transition(self, current_role: str, target_role: str, 
                         current_skills: List[str], location: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a career transition."""
        query = f"""Analyze the career transition from {current_role} to {target_role}.
        Current skills: {', '.join(current_skills)}
        Location: {location or 'United States'}
        
        Please provide:
        1. Job market analysis for both roles
        2. Required skills for the target role
        3. Skill gap analysis
        4. Market demand and salary expectations
        5. Key insights about this transition"""
        
        result = self.agent.invoke({"input": query})
        return {
            "analysis": result["output"],
            "current_role": current_role,
            "target_role": target_role,
            "location": location
        }
''')

    # core/agents/skills_mapper.py
    create_file('core/agents/skills_mapper.py', '''"""
Skills mapper agent for identifying transferable skills.
"""
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from app.config import settings
from core.tools.skills_database_tool import SkillsDatabaseTool
from utils.logger import get_logger

logger = get_logger(__name__)


class SkillsMapperAgent:
    """Agent for mapping skills and identifying transferable competencies."""
    
    def __init__(self, llm=None):
        """Initialize the skills mapper agent."""
        self.llm = llm or self._get_default_llm()
        self.tools = [SkillsDatabaseTool()]
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=3
        )
        self.agent = self._create_agent()
    
    def _get_default_llm(self):
        """Get default LLM based on configuration."""
        if settings.llm_provider == "openai":
            return ChatOpenAI(
                model=settings.openai_model,
                temperature=0.3,  # Lower temperature for more consistent skill mapping
                max_tokens=settings.max_tokens,
                api_key=settings.openai_api_key
            )
        else:
            return ChatAnthropic(
                model=settings.anthropic_model,
                temperature=0.3,
                max_tokens=settings.max_tokens,
                api_key=settings.anthropic_api_key
            )
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor."""
        system_message = """You are a Skills Analysis Expert specializing in identifying transferable skills and competency mapping.
        
        Your responsibilities:
        1. Identify transferable skills between roles
        2. Assess skill proficiency levels (1-10 scale)
        3. Map skills to industry standards
        4. Identify skill relationships and dependencies
        5. Prioritize skills based on market demand
        
        When analyzing skills:
        - Consider both technical and soft skills
        - Identify hidden or implicit skills from experience
        - Recognize skill clusters and related competencies
        - Assess the effort required to bridge skill gaps
        
        Provide structured skill assessments with clear proficiency levels and gap analysis."""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
    
    def map_skills(self, current_role: str, target_role: str, 
                   current_skills: List[str], experience_years: int) -> Dict[str, Any]:
        """Map skills and identify gaps."""
        query = f"""Analyze skills for transition from {current_role} to {target_role}.
        Current skills: {', '.join(current_skills)}
        Years of experience: {experience_years}
        
        Please provide:
        1. Complete list of transferable skills with proficiency levels (1-10)
        2. Skills that need to be acquired with required proficiency levels
        3. Skills that need improvement with current and target levels
        4. Skill priorities based on importance for the target role
        5. Hidden skills that the person likely has based on their role and experience"""
        
        result = self.agent.invoke({"input": query})
        return self._parse_skills_analysis(result["output"])
    
    def _parse_skills_analysis(self, analysis: str) -> Dict[str, Any]:
        """Parse the skills analysis into structured format."""
        # This would include more sophisticated parsing in production
        return {
            "raw_analysis": analysis,
            "transferable_skills": [],
            "skills_to_acquire": [],
            "skills_to_improve": [],
            "skill_priorities": []
        }
''')

    # core/agents/roadmap_generator.py
    create_file('core/agents/roadmap_generator.py', '''"""
Roadmap generator agent for creating personalized learning paths.
"""
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from app.config import settings
from core.tools.course_finder_tool import CourseFindingTool
from core.tools.certification_tool import CertificationFinderTool
from core.tools.project_suggestion_tool import ProjectSuggestionTool
from utils.logger import get_logger

logger = get_logger(__name__)


class RoadmapGeneratorAgent:
    """Agent for generating personalized learning roadmaps."""
    
    def __init__(self, llm=None):
        """Initialize the roadmap generator agent."""
        self.llm = llm or self._get_default_llm()
        self.tools = [
            CourseFindingTool(),
            CertificationFinderTool(),
            ProjectSuggestionTool()
        ]
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=3
        )
        self.agent = self._create_agent()
    
    def _get_default_llm(self):
        """Get default LLM based on configuration."""
        if settings.llm_provider == "openai":
            return ChatOpenAI(
                model=settings.openai_model,
                temperature=0.7,
                max_tokens=settings.max_tokens,
                api_key=settings.openai_api_key
            )
        else:
            return ChatAnthropic(
                model=settings.anthropic_model,
                temperature=0.7,
                max_tokens=settings.max_tokens,
                api_key=settings.anthropic_api_key
            )
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor."""
        system_message = """You are a Learning Path Architect specializing in creating personalized career development roadmaps.
        
        Your role is to:
        1. Design structured learning paths with clear milestones
        2. Recommend specific courses, certifications, and projects
        3. Create realistic timelines based on available hours per week
        4. Balance theoretical learning with practical application
        5. Consider budget constraints and learning preferences
        
        When creating roadmaps:
        - Break down the journey into manageable milestones (1-2 month chunks)
        - Mix different types of learning resources (courses, books, projects)
        - Include hands-on projects to build portfolio
        - Add checkpoints to measure progress
        - Provide alternative paths based on different constraints
        
        Always create actionable, specific roadmaps that users can follow step-by-step."""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=6
        )
    
    def generate_roadmap(self, 
                        skills_to_learn: List[str],
                        hours_per_week: int,
                        timeline_months: int,
                        budget_per_month: Optional[float] = None,
                        learning_style: Optional[str] = None) -> Dict[str, Any]:
        """Generate a personalized learning roadmap."""
        query = f"""Create a detailed learning roadmap with the following requirements:
        Skills to learn: {', '.join(skills_to_learn)}
        Available hours per week: {hours_per_week}
        Target timeline: {timeline_months} months
        Budget per month: ${budget_per_month or 'flexible'}
        Learning style preference: {learning_style or 'mixed'}
        
        Please create:
        1. Monthly milestones with specific goals
        2. Recommended courses for each skill (use course_finder tool)
        3. Relevant certifications to pursue (use certification_finder tool)
        4. Hands-on projects for portfolio (use project_suggester tool)
        5. Time allocation for each activity
        6. Progress checkpoints and success criteria"""
        
        result = self.agent.invoke({"input": query})
        return {
            "roadmap": result["output"],
            "skills": skills_to_learn,
            "timeline": timeline_months,
            "hours_per_week": hours_per_week
        }
''')

    # core/agents/complexity_estimator.py
    create_file('core/agents/complexity_estimator.py', '''"""
Complexity estimator agent for assessing transition difficulty.
"""
from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from app.config import settings
from core.tools.job_market_tool import JobMarketResearchTool
from core.tools.skills_database_tool import SkillsDatabaseTool
from utils.logger import get_logger

logger = get_logger(__name__)


class ComplexityEstimatorAgent:
    """Agent for estimating transition complexity and success probability."""
    
    def __init__(self, llm=None):
        """Initialize the complexity estimator agent."""
        self.llm = llm or self._get_default_llm()
        self.tools = [JobMarketResearchTool(), SkillsDatabaseTool()]
        self.agent = self._create_agent()
    
    def _get_default_llm(self):
        """Get default LLM based on configuration."""
        if settings.llm_provider == "openai":
            return ChatOpenAI(
                model=settings.openai_model,
                temperature=0.3,  # Lower temperature for consistent estimates
                max_tokens=2000,
                api_key=settings.openai_api_key
            )
        else:
            return ChatAnthropic(
                model=settings.anthropic_model,
                temperature=0.3,
                max_tokens=2000,
                api_key=settings.anthropic_api_key
            )
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor."""
        system_message = """You are a Career Transition Complexity Analyst.
        
        Your role is to assess:
        1. Difficulty rating (1-10) for career transitions
        2. Realistic time estimates based on starting point
        3. Success probability based on market conditions
        4. Potential challenges and risk factors
        5. Success factors and accelerators
        
        Consider factors like:
        - Size of skill gap
        - Market demand for target role
        - Competition level
        - Industry barriers
        - Required experience levels
        - Certification requirements
        
        Provide honest, data-driven assessments to help users set realistic expectations."""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
    
    def estimate_complexity(self,
                          current_role: str,
                          target_role: str,
                          current_skills: List[str],
                          experience_years: int,
                          hours_per_week: int) -> Dict[str, Any]:
        """Estimate transition complexity and metrics."""
        query = f"""Analyze the complexity of transitioning from {current_role} to {target_role}.
        Current skills: {', '.join(current_skills)}
        Years of experience: {experience_years}
        Available hours per week for learning: {hours_per_week}
        
        Please provide:
        1. Difficulty rating (1-10) with justification
        2. Realistic time estimate in months
        3. Success probability percentage
        4. Top 3 challenges
        5. Top 3 success factors
        6. Risk mitigation strategies"""
        
        result = self.agent.invoke({"input": query})
        return self._parse_complexity_estimate(result["output"])
    
    def _parse_complexity_estimate(self, estimate: str) -> Dict[str, Any]:
        """Parse complexity estimate into structured format."""
        # Simplified parsing - in production, use more sophisticated NLP
        return {
            "raw_estimate": estimate,
            "difficulty_rating": 7.5,
            "time_estimate_months": 6,
            "success_probability": 0.75,
            "challenges": [],
            "success_factors": []
        }


def create_career_agents(llm=None) -> Dict[str, Any]:
    """Create and return all career transition agents."""
    return {
        "career_analyzer": CareerAnalyzerAgent(llm),
        "skills_mapper": SkillsMapperAgent(llm),
        "roadmap_generator": RoadmapGeneratorAgent(llm),
        "complexity_estimator": ComplexityEstimatorAgent(llm)
    }
''')


def main():
    """Main function to generate part 2 of the project."""
    print("🚀 Generating Career Roadmap AI Project - Part 2\n")
    
    # Create core tools
    create_core_tools()
    
    # Create core agents
    create_core_agents()
    
    print("\n✅ Part 2 completed! Run generate_project_part3.py next.")


if __name__ == "__main__":
    main()
