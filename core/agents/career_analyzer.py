"""
Career analyzer agent for analyzing career transitions.
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
                temperature=0.3
            )
        else:
            return ChatAnthropic(
                model=settings.anthropic_model,
                temperature=0.3
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
