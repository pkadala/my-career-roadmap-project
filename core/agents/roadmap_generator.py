"""
Roadmap generator agent for creating personalized career roadmaps.
"""
import warnings
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

# Suppress deprecation warning for ConversationBufferWindowMemory
warnings.filterwarnings("ignore", message=".*ConversationBufferWindowMemory.*", category=DeprecationWarning)
from langchain.memory import ConversationBufferWindowMemory

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from app.config import settings
from core.tools.project_suggestion_tool import ProjectSuggestionTool
from core.tools.course_finder_tool import CourseFindingTool
from core.tools.certification_tool import CertificationFinderTool
from utils.logger import get_logger

logger = get_logger(__name__)


class RoadmapGeneratorAgent:
    """Agent for generating personalized career roadmaps."""
    
    def __init__(self, llm=None):
        """Initialize the roadmap generator agent."""
        self.llm = llm or self._get_default_llm()
        self.tools = [ProjectSuggestionTool(), CourseFindingTool(), CertificationFinderTool()]
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
                temperature=0.3
            )
        else:
            return ChatAnthropic(
                model=settings.anthropic_model,
                temperature=0.3
            )
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor."""
        system_message = """You are a Career Roadmap Generator. Your role is to create personalized, step-by-step career transition roadmaps.
        
        When creating roadmaps:
        1. Consider the user's current skills and experience
        2. Break down the transition into manageable phases
        3. Include specific projects, courses, and certifications
        4. Provide realistic timelines and milestones
        5. Consider different learning paths and alternatives
        
        Always create actionable, detailed roadmaps that users can follow."""
        
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ],
            input_variables=["input", "chat_history", "agent_scratchpad"]
        )
        
        agent = create_openai_functions_agent(
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
    
    def generate_roadmap(self, current_role: str, target_role: str, 
                        current_skills: List[str], experience_years: int,
                        available_hours: int, preferred_learning_style: str = "mixed") -> Dict[str, Any]:
        """Generate a personalized career roadmap."""
        query = f"""Generate a detailed career roadmap for transitioning from {current_role} to {target_role}.
        Current skills: {', '.join(current_skills)}
        Years of experience: {experience_years}
        Available hours per week: {available_hours}
        Preferred learning style: {preferred_learning_style}
        
        Please provide:
        1. Phase-by-phase breakdown
        2. Specific projects to build
        3. Recommended courses and certifications
        4. Timeline for each phase
        5. Success metrics and milestones
        6. Alternative paths and contingencies"""
        
        result = self.agent.invoke({"input": query})
        return {
            "roadmap": result["output"],
            "current_role": current_role,
            "target_role": target_role,
            "experience_years": experience_years,
            "available_hours": available_hours,
            "learning_style": preferred_learning_style
        }
