"""
Complexity estimator agent for assessing transition difficulty.
"""
import warnings
from typing import List, Dict, Any
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


class ComplexityEstimatorAgent:
    """Agent for estimating transition complexity and success probability."""
    
    def __init__(self, llm=None):
        """Initialize the complexity estimator agent."""
        self.llm = llm or self._get_default_llm()
        self.tools = [JobMarketResearchTool(), SkillsDatabaseTool()]
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
                temperature=0.3
            )
        else:
            return ChatAnthropic(
                model=settings.anthropic_model,
                temperature=0.3
            )
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor."""
        system_message = """You are a Complexity Estimator Agent. Your job is to estimate the complexity, risks, and timeline for career transitions based on user background and roadmap."""
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
    
    def estimate_complexity(self,
                          current_role: str,
                          target_role: str,
                          current_skills: List[str],
                          experience_years: int,
                          hours_per_week: int) -> Dict[str, Any]:
        """Estimate the complexity of the career transition."""
        query = f"""Estimate the complexity of transitioning from {current_role} to {target_role}.\nCurrent skills: {', '.join(current_skills)}\nYears of experience: {experience_years}\nAvailable hours per week: {hours_per_week}\n\nPlease provide:\n1. Difficulty rating (1-10)\n2. Estimated timeline in months\n3. Key risk factors\n4. Success probability\n5. Risk mitigation strategies"""
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
