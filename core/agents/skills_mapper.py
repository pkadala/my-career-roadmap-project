"""
Skills mapper agent for mapping and analyzing skill requirements.
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
from core.tools.skills_database_tool import SkillsDatabaseTool
from core.tools.certification_tool import CertificationFinderTool
from core.tools.course_finder_tool import CourseFindingTool
from utils.logger import get_logger

logger = get_logger(__name__)


class SkillsMapperAgent:
    """Agent for mapping skills and identifying learning paths."""
    
    def __init__(self, llm=None):
        """Initialize the skills mapper agent."""
        self.llm = llm or self._get_default_llm()
        self.tools = [SkillsDatabaseTool(), CertificationFinderTool(), CourseFindingTool()]
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
        system_message = """You are a Skills Mapping Specialist. Your role is to:
        1. Analyze skill requirements for target roles
        2. Map current skills to target skills
        3. Identify skill gaps and learning needs
        4. Recommend certifications and courses
        5. Create personalized learning paths
        
        Always provide specific, actionable recommendations based on your tools."""
        
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
    
    def map_skills(self, current_skills: List[str], target_role: str, 
                  experience_level: str = "mid-level") -> Dict[str, Any]:
        """Map current skills to target role requirements."""
        query = f"""Map skills for transitioning to {target_role} role.
        Current skills: {', '.join(current_skills)}
        Experience level: {experience_level}
        
        Please provide:
        1. Required skills for the target role
        2. Skill gap analysis
        3. Recommended certifications
        4. Suggested courses and learning resources
        5. Prioritized learning path"""
        
        result = self.agent.invoke({"input": query})
        return {
            "skill_mapping": result["output"],
            "current_skills": current_skills,
            "target_role": target_role,
            "experience_level": experience_level
        }
