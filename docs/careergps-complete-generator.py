                        {"name": "3Blue1Brown Essence of Linear Algebra", "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab", "type": "video"}
                    ]
                ),
                SkillSchema(
                    name="Statistics & Probability",
                    category="Mathematics",
                    difficulty=6,
                    resources=[
                        {"name": "Think Stats", "url": "https://greenteapress.com/thinkstats/", "type": "book"},
                        {"name": "StatQuest YouTube", "url": "https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw", "type": "video"}
                    ]
                ),
                SkillSchema(
                    name="Python Programming",
                    category="Programming",
                    difficulty=5,
                    resources=[
                        {"name": "Python for Data Science", "url": "https://www.coursera.org/learn/python-for-data-science", "type": "course"},
                        {"name": "Automate the Boring Stuff", "url": "https://automatetheboringstuff.com/", "type": "book"}
                    ]
                )
            ],
            projects=[
                ProjectSchema(
                    title="Implement Gradient Descent from Scratch",
                    description="Build gradient descent optimizer using NumPy",
                    difficulty=6,
                    estimated_hours=20
                ),
                ProjectSchema(
                    title="Statistical Analysis Project",
                    description="Analyze a real dataset using Python and statistics",
                    difficulty=5,
                    estimated_hours=15
                )
            ]
        )
        phases.append(phase1)
        
        # Phase 2: Machine Learning Fundamentals
        phase2 = PhaseSchema(
            title="Machine Learning Fundamentals",
            description="Master classical ML algorithms and techniques",
            duration_weeks=10,
            order=2,
            status="pending",
            skills=[
                SkillSchema(
                    name="Supervised Learning",
                    category="Machine Learning",
                    difficulty=7,
                    resources=[
                        {"name": "Andrew Ng's ML Course", "url": "https://www.coursera.org/learn/machine-learning", "type": "course"},
                        {"name": "Hands-On ML Book", "url": "https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/", "type": "book"}
                    ]
                ),
                SkillSchema(
                    name="scikit-learn",
                    category="Tools",
                    difficulty=5,
                    resources=[
                        {"name": "scikit-learn Documentation", "url": "https://scikit-learn.org/stable/", "type": "documentation"}
                    ]
                )
            ],
            projects=[
                ProjectSchema(
                    title="End-to-End ML Pipeline",
                    description="Build complete ML pipeline with data preprocessing, model training, and evaluation",
                    difficulty=7,
                    estimated_hours=30
                )
            ]
        )
        phases.append(phase2)
        
        # Phase 3: Deep Learning
        phase3 = PhaseSchema(
            title="Deep Learning & Neural Networks",
            description="Master deep learning concepts and frameworks",
            duration_weeks=12,
            order=3,
            status="pending",
            skills=[
                SkillSchema(
                    name="Neural Networks",
                    category="Deep Learning",
                    difficulty=8,
                    resources=[
                        {"name": "Fast.ai Course", "url": "https://www.fast.ai/", "type": "course"},
                        {"name": "Deep Learning Book", "url": "https://www.deeplearningbook.org/", "type": "book"}
                    ]
                ),
                SkillSchema(
                    name="PyTorch",
                    category="Frameworks",
                    difficulty=7,
                    resources=[
                        {"name": "PyTorch Tutorials", "url": "https://pytorch.org/tutorials/", "type": "documentation"}
                    ]
                )
            ],
            projects=[
                ProjectSchema(
                    title="Image Classification System",
                    description="Build CNN for image classification",
                    difficulty=8,
                    estimated_hours=40
                )
            ]
        )
        phases.append(phase3)
        
        # Calculate total duration
        total_weeks = sum(phase.duration_weeks for phase in phases)
        total_months = int(total_weeks / 4.33)
        
        return RoadmapSchema(
            id=1,  # Will be set by database
            user_id=user_profile['user_id'],
            current_role=user_profile['current_role'],
            target_role=user_profile['target_role'],
            total_duration_months=total_months,
            difficulty_level=7,
            success_probability=0.75,
            phases=phases,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    async def save_roadmap(self, db: AsyncSession, user_id: int, roadmap_schema: RoadmapSchema):
        """Save roadmap to database"""
        # Create roadmap model
        roadmap = Roadmap(
            user_id=user_id,
            current_role=roadmap_schema.current_role,
            target_role=roadmap_schema.target_role,
            total_duration_months=roadmap_schema.total_duration_months,
            difficulty_level=roadmap_schema.difficulty_level,
            success_probability=roadmap_schema.success_probability
        )
        
        db.add(roadmap)
        await db.flush()
        
        # Create phases
        for phase_schema in roadmap_schema.phases:
            phase = Phase(
                roadmap_id=roadmap.id,
                title=phase_schema.title,
                description=phase_schema.description,
                duration_weeks=phase_schema.duration_weeks,
                order=phase_schema.order,
                status=phase_schema.status
            )
            
            db.add(phase)
            await db.flush()
            
            # Create skills
            for skill_schema in phase_schema.skills:
                skill = Skill(
                    phase_id=phase.id,
                    name=skill_schema.name,
                    category=skill_schema.category,
                    difficulty=skill_schema.difficulty,
                    resources=json.dumps(skill_schema.resources)
                )
                db.add(skill)
            
            # Create projects
            for project_schema in phase_schema.projects:
                project = Project(
                    phase_id=phase.id,
                    title=project_schema.title,
                    description=project_schema.description,
                    difficulty=project_schema.difficulty,
                    estimated_hours=project_schema.estimated_hours
                )
                db.add(project)
        
        await db.commit()
        await db.refresh(roadmap)
        
        return roadmap
    
    def _analyze_skills(self, skills_text: str) -> str:
        """Tool: Analyze skills depth and relevance"""
        return f"Skills analyzed: {skills_text}"
    
    def _identify_gaps(self, current_target: str) -> str:
        """Tool: Identify skill gaps"""
        return f"Gaps identified for: {current_target}"
    
    def _calculate_timeline(self, complexity: str) -> str:
        """Tool: Calculate realistic timeline"""
        return f"Timeline calculated: {complexity}"
    
    def _find_resources(self, skills_needed: str) -> str:
        """Tool: Find learning resources"""
        return f"Resources found for: {skills_needed}"
'''

    def _get_job_matching_engine(self):
        return '''from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple
import asyncio
import logging

from app.schemas.job import JobMatchSchema
from app.core.config import settings

logger = logging.getLogger(__name__)

class JobMatchingEngine:
    """AI-powered job matching with skill gap analysis"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.skill_extractor = SkillExtractionAgent()
    
    async def match_jobs(
        self, 
        user_profile: Dict,
        filters: Dict = None
    ) -> List[JobMatchSchema]:
        """Find and rank job matches for user"""
        
        # Get user's skill embedding
        user_skills = user_profile.get('skills', [])
        user_embedding = self._get_skills_embedding(user_skills)
        
        # Fetch jobs (in production, this would query the database)
        jobs = await self._fetch_jobs(filters)
        
        # Calculate matches
        matches = []
        for job in jobs:
            match_data = await self._calculate_match(
                user_embedding,
                user_profile,
                job
            )
            matches.append(match_data)
        
        # Sort by opportunity score
        matches.sort(
            key=lambda x: x.opportunity_score,
            reverse=True
        )
        
        return matches[:50]  # Return top 50 matches
    
    async def _calculate_match(
        self,
        user_embedding: np.ndarray,
        user_profile: Dict,
        job: Dict
    ) -> JobMatchSchema:
        """Calculate detailed match metrics"""
        
        # Extract job requirements
        job_requirements = await self.skill_extractor.extract(job['description'])
        job_embedding = self._get_skills_embedding(job_requirements)
        
        # Calculate similarity
        similarity = cosine_similarity(
            [user_embedding],
            [job_embedding]
        )[0][0]
        
        # Analyze skill gaps
        skill_gaps = self._analyze_skill_gaps(
            user_profile['skills'],
            job_requirements
        )
        
        # Calculate growth potential
        growth_potential = self._calculate_growth_potential(
            skill_gaps,
            user_profile
        )
        
        # Calculate opportunity score
        opportunity_score = self._calculate_opportunity_score(
            similarity,
            growth_potential,
            job
        )
        
        return JobMatchSchema(
            job_id=job['id'],
            company=job['company'],
            title=job['title'],
            location=job.get('location', 'Remote'),
            remote=job.get('remote', False),
            salary_range=self._format_salary_range(job),
            match_percentage=int(similarity * 100),
            skill_gaps=skill_gaps,
            growth_potential=growth_potential,
            opportunity_score=opportunity_score,
            estimated_prep_weeks=self._estimate_prep_time(skill_gaps),
            application_tips=self._generate_application_tips(
                user_profile,
                job,
                skill_gaps
            )
        )
    
    def _get_skills_embedding(self, skills: List[str]) -> np.ndarray:
        """Get embedding for skill set"""
        if not skills:
            return np.zeros(768)  # Model dimension
        
        skills_text = ", ".join(skills)
        return self.model.encode(skills_text)
    
    def _analyze_skill_gaps(
        self,
        user_skills: List[str],
        job_requirements: List[str]
    ) -> List[Dict]:
        """Identify and categorize skill gaps"""
        user_skills_set = set(s.lower() for s in user_skills)
        gaps = []
        
        for req in job_requirements:
            if req.lower() not in user_skills_set:
                gaps.append({
                    "skill": req,
                    "difficulty": self._assess_skill_difficulty(req),
                    "learning_hours": self._estimate_learning_hours(req)
                })
        
        return gaps
    
    def _calculate_growth_potential(
        self,
        skill_gaps: List[Dict],
        user_profile: Dict
    ) -> float:
        """Calculate career growth potential"""
        if not skill_gaps:
            return 0.2  # Low growth if no new skills
        
        avg_difficulty = np.mean([g['difficulty'] for g in skill_gaps])
        skill_count_factor = min(len(skill_gaps) / 5, 1.0)
        
        growth_score = (0.4 * skill_count_factor + 
                       0.3 * (avg_difficulty / 10) +
                       0.3 * self._career_alignment_score(user_profile))
        
        return round(growth_score, 2)
    
    def _calculate_opportunity_score(
        self,
        similarity: float,
        growth_potential: float,
        job: Dict
    ) -> float:
        """Calculate overall opportunity score"""
        # Sweet spot is 70-85% match (room to grow)
        if 0.7 <= similarity <= 0.85:
            match_bonus = 0.2
        else:
            match_bonus = 0
        
        # Factor in company quality, salary, etc.
        company_score = job.get('company_rating', 3.5) / 5
        
        score = (0.4 * similarity + 
                0.3 * growth_potential + 
                0.2 * company_score +
                0.1 * match_bonus)
        
        return round(score, 2)
    
    def _assess_skill_difficulty(self, skill: str) -> int:
        """Assess learning difficulty (1-10)"""
        difficult_skills = ['machine learning', 'deep learning', 'kubernetes', 'system design']
        medium_skills = ['python', 'docker', 'sql', 'git']
        
        skill_lower = skill.lower()
        if any(d in skill_lower for d in difficult_skills):
            return 8
        elif any(m in skill_lower for m in medium_skills):
            return 5
        return 3
    
    def _estimate_learning_hours(self, skill: str) -> int:
        """Estimate hours needed to learn skill"""
        difficulty = self._assess_skill_difficulty(skill)
        base_hours = {
            1: 20, 2: 40, 3: 60, 4: 80, 5: 100,
            6: 150, 7: 200, 8: 300, 9: 400, 10: 500
        }
        return base_hours.get(difficulty, 100)
    
    def _estimate_prep_time(self, skill_gaps: List[Dict]) -> int:
        """Estimate weeks needed for preparation"""
        if not skill_gaps:
            return 0
        
        total_hours = sum(gap['learning_hours'] for gap in skill_gaps)
        # Assuming 15 hours/week study time
        weeks = total_hours / 15
        
        return max(1, int(weeks))
    
    def _generate_application_tips(
        self,
        user_profile: Dict,
        job: Dict,
        skill_gaps: List[Dict]
    ) -> List[str]:
        """Generate personalized application tips"""
        tips = []
        
        # Highlight transferable skills
        tips.append(f"Emphasize your {user_profile['current_role']} experience")
        
        # Address skill gaps
        if skill_gaps:
            learnable = [g['skill'] for g in skill_gaps if g['difficulty'] <= 5]
            if learnable:
                tips.append(f"Show enthusiasm to learn {', '.join(learnable[:2])}")
        
        # Company-specific tips
        if job.get('company_culture'):
            tips.append(f"Align with {job['company']} culture")
        
        return tips
    
    def _format_salary_range(self, job: Dict) -> str:
        """Format salary range for display"""
        if job.get('salary_min') and job.get('salary_max'):
            return f"${job['salary_min']:,} - ${job['salary_max']:,}"
        return "Not specified"
    
    def _career_alignment_score(self, user_profile: Dict) -> float:
        """Calculate how well job aligns with career goals"""
        # Simplified - would be more sophisticated in production
        return 0.8
    
    async def _fetch_jobs(self, filters: Dict) -> List[Dict]:
        """Fetch jobs from database or external APIs"""
        # Mock data for demonstration
        return [
            {
                "id": "job1",
                "title": "ML Engineer",
                "company": "TechCorp",
                "location": "San Francisco, CA",
                "remote": True,
                "description": "Looking for ML Engineer with Python, TensorFlow, and Kubernetes experience...",
                "salary_min": 150000,
                "salary_max": 250000,
                "company_rating": 4.5
            },
            {
                "id": "job2",
                "title": "AI Engineer",
                "company": "AI Startup",
                "location": "Remote",
                "remote": True,
                "description": "Seeking AI Engineer with experience in NLP, Computer Vision, and PyTorch...",
                "salary_min": 130000,
                "salary_max": 200000,
                "company_rating": 4.2
            }
        ]
    
    async def get_job_insights(self, job: Dict, user: any) -> Dict:
        """Get personalized insights for a specific job"""
        return {
            "fit_analysis": "Strong technical fit with growth opportunities",
            "preparation_timeline": "4-6 weeks of focused preparation recommended",
            "interview_topics": ["System Design", "ML Algorithms", "Production ML"],
            "salary_negotiation": {
                "market_rate": "$180,000",
                "your_leverage": "Strong Python background",
                "suggested_range": "$170,000 - $200,000"
            }
        }

class SkillExtractionAgent:
    """Extract skills from job descriptions using AI"""
    
    def __init__(self):
        from langchain.chat_models import ChatOpenAI
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )
    
    async def extract(self, job_description: str) -> List[str]:
        """Extract required skills from job description"""
        prompt = f"""
        Extract all technical skills and requirements from this job description.
        Return only the skills as a comma-separated list.
        
        Job Description:
        {job_description}
        
        Skills:
        """
        
        response = await self.llm.apredict(prompt)
        skills = [s.strip() for s in response.split(',')]
        
        return skills
'''

    def _get_interview_coach(self):
        return '''from typing import Dict, List, Optional
import json
import asyncio
import logging
from datetime import datetime

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from app.schemas.interview import InterviewSessionSchema, InterviewQuestionSchema, InterviewFeedbackSchema
from app.core.config import settings

logger = logging.getLogger(__name__)

class AIInterviewCoach:
    """AI-powered interview coaching system"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Question templates by type
        self.question_templates = {
            "behavioral": [
                "Tell me about a time when you had to learn a new technology quickly",
                "Describe a challenging problem you solved",
                "How do you handle conflicting priorities?",
                "Give an example of when you failed and what you learned",
                "How do you stay updated with technology trends?"
            ],
            "technical": [
                "Explain how you would design a distributed system",
                "What's your approach to optimizing database queries?",
                "How would you implement a recommendation system?",
                "Describe your experience with microservices",
                "How do you ensure code quality in your projects?"
            ],
            "system_design": [
                "Design a URL shortening service like bit.ly",
                "How would you build a real-time chat application?",
                "Design a ride-sharing service",
                "How would you architect a video streaming platform?",
                "Design a distributed cache system"
            ]
        }
    
    async def create_interview_session(
        self,
        user_id: str,
        job_role: str,
        company: str,
        interview_type: str = "behavioral"
    ) -> InterviewSessionSchema:
        """Create a new interview session"""
        
        # Generate questions based on role and company
        questions = await self._generate_questions(
            job_role,
            company,
            interview_type
        )
        
        session = InterviewSessionSchema(
            id=f"session_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            job_role=job_role,
            company=company,
            interview_type=interview_type,
            questions=questions,
            status="active",
            started_at=datetime.utcnow()
        )
        
        return session
    
    async def _generate_questions(
        self,
        job_role: str,
        company: str,
        interview_type: str
    ) -> List[InterviewQuestionSchema]:
        """Generate role-specific interview questions"""
        
        prompt = f"""
        Generate 5 interview questions for:
        Role: {job_role}
        Company: {company}
        Type: {interview_type}
        
        For each question provide:
        1. The question text
        2. What the interviewer is looking for
        3. Key points to cover in a good answer
        
        Format as a list of questions with details.
        """
        
        response = await self.llm.apredict(prompt)
        
        # Parse response and create question schemas
        questions = []
        base_questions = self.question_templates.get(interview_type, self.question_templates["behavioral"])
        
        for i, question_text in enumerate(base_questions[:5]):
            # Customize question for role/company
            customized = await self._customize_question(question_text, job_role, company)
            
            questions.append(InterviewQuestionSchema(
                id=f"q{i+1}",
                text=customized,
                category=interview_type,
                difficulty="medium" if i < 3 else "hard",
                hints=[
                    "Use the STAR method (Situation, Task, Action, Result)",
                    f"Relate your answer to {company}'s values",
                    "Be specific with examples"
                ]
            ))
        
        return questions
    
    async def _customize_question(self, base_question: str, role: str, company: str) -> str:
        """Customize question for specific role and company"""
        prompt = f"""
        Customize this interview question for a {role} position at {company}:
        
        Base question: {base_question}
        
        Make it specific to the role and company while keeping the core intent.
        Return only the customized question.
        """
        
        response = await self.llm.apredict(prompt)
        return response.strip()
    
    async def process_answer(
        self,
        session_id: str,
        audio_data: bytes = None,
        text_answer: str = None
    ) -> InterviewFeedbackSchema:
        """Process user's answer and provide feedback"""
        
        # For demo, we'll use text answer directly
        # In production, would transcribe audio_data
        
        if not text_answer and audio_data:
            text_answer = "Transcribed answer would go here"
        
        # Analyze answer
        analysis = await self._analyze_answer(text_answer)
        
        # Generate feedback
        feedback = InterviewFeedbackSchema(
            question_id="current_question",
            answer_text=text_answer,
            content_score=analysis["content_score"],
            structure_score=analysis["structure_score"],
            relevance_score=analysis["relevance_score"],
            delivery_score=analysis.get("delivery_score", 0.8),
            suggestions=analysis["suggestions"],
            example_answer=analysis.get("example_answer")
        )
        
        return feedback
    
    async def _analyze_answer(self, answer: str) -> Dict:
        """Analyze answer quality using AI"""
        
        analysis_prompt = f"""
        Analyze this interview answer and provide scores and feedback:
        
        Answer: {answer}
        
        Evaluate:
        1. Content Quality (0-1): Does it answer the question thoroughly?
        2. Structure (0-1): Is it well-organized (e.g., STAR format)?
        3. Relevance (0-1): Is it relevant to the role?
        
        Provide:
        - Scores for each criterion
        - 3 specific suggestions for improvement
        - A brief example of how to improve the answer
        
        Format as JSON with keys: content_score, structure_score, relevance_score, suggestions, example_answer
        """
        
        response = await self.llm.apredict(analysis_prompt)
        
        # Parse JSON response
        try:
            analysis = json.loads(response)
        except:
            # Fallback if JSON parsing fails
            analysis = {
                "content_score": 0.7,
                "structure_score": 0.6,
                "relevance_score": 0.8,
                "suggestions": [
                    "Provide more specific examples",
                    "Use the STAR method for better structure",
                    "Quantify your achievements"
                ],
                "example_answer": "In my previous role at XYZ Corp, I faced a similar situation..."
            }
        
        return analysis
    
    async def generate_session_analysis(self, session_id: str) -> Dict:
        """Generate complete session analysis"""
        return {
            "overall_score": 0.75,
            "strengths": [
                "Good technical knowledge",
                "Clear communication",
                "Relevant examples"
            ],
            "areas_for_improvement": [
                "More structured responses",
                "Quantify achievements",
                "Show more enthusiasm"
            ],
            "recommendations": [
                "Practice STAR method for behavioral questions",
                "Prepare 5-7 strong examples from your experience",
                "Research the company culture and values"
            ]
        }
    
    async def get_question_bank(
        self,
        role: Optional[str] = None,
        company: Optional[str] = None,
        interview_type: Optional[str] = None
    ) -> List[Dict]:
        """Get interview questions from the bank"""
        questions = []
        
        # Filter by type if specified
        types = [interview_type] if interview_type else self.question_templates.keys()
        
        for q_type in types:
            for question in self.question_templates.get(q_type, []):
                questions.append({
                    "text": question,
                    "type": q_type,
                    "difficulty": "medium",
                    "tags": [role, company] if role or company else []
                })
        
        return questions
'''

    def _get_learning_agent(self):
        return '''from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime

from langchain.agents import Tool, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from app.core.config import settings

logger = logging.getLogger(__name__)

class AdaptiveLearningAgent:
    """Agentic AI that adapts learning content based on user progress"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.memory = ConversationSummaryMemory(llm=self.llm)
        
        self.tools = [
            Tool(
                name="assess_understanding",
                func=self._assess_understanding,
                description="Assess user's understanding of current topic"
            ),
            Tool(
                name="adjust_difficulty",
                func=self._adjust_difficulty,
                description="Adjust content difficulty based on performance"
            ),
            Tool(
                name="generate_exercise",
                func=self._generate_exercise,
                description="Generate practice exercise at appropriate level"
            ),
            Tool(
                name="provide_hint",
                func=self._provide_hint,
                description="Provide contextual hints when user is stuck"
            )
        ]
    
    async def generate_learning_module(
        self,
        user_id: int,
        skill: any,
        phase: any
    ) -> Dict:
        """Generate adaptive learning module"""
        
        # Get user's current level
        user_level = await self._assess_user_level(user_id, skill.name)
        
        # Generate content based on level
        content = await self._generate_content(
            skill_name=skill.name,
            user_level=user_level,
            phase_context=phase.title
        )
        
        # Create exercises
        exercises = await self._generate_exercises(
            skill_name=skill.name,
            difficulty=user_level
        )
        
        return {
            "module_id": f"{skill.id}_{datetime.utcnow().timestamp()}",
            "skill_name": skill.name,
            "difficulty_level": user_level,
            "estimated_time_minutes": 45,
            "content": content,
            "exercises": exercises,
            "resources": self._get_curated_resources(skill.name)
        }
    
    async def _assess_user_level(self, user_id: int, skill_name: str) -> str:
        """Assess user's current level for a skill"""
        # In production, this would analyze user's history
        # For now, return a default
        return "intermediate"
    
    async def _generate_content(
        self,
        skill_name: str,
        user_level: str,
        phase_context: str
    ) -> Dict:
        """Generate learning content"""
        
        prompt = f"""
        Create a learning module for:
        Skill: {skill_name}
        User Level: {user_level}
        Context: {phase_context}
        
        Include:
        1. Introduction (2-3 paragraphs)
        2. Key concepts (3-5 points)
        3. Practical examples
        4. Common pitfalls to avoid
        
        Make it engaging and appropriate for the user's level.
        """
        
        response = await self.llm.apredict(prompt)
        
        return {
            "introduction": "Introduction to " + skill_name,
            "key_concepts": [
                "Concept 1: Understanding the basics",
                "Concept 2: Practical applications",
                "Concept 3: Advanced techniques"
            ],
            "examples": [
                {"title": "Example 1", "code": "# Code example here"},
                {"title": "Example 2", "code": "# Another example"}
            ],
            "pitfalls": [
                "Common mistake 1 and how to avoid it",
                "Common mistake 2 and how to avoid it"
            ]
        }
    
    async def _generate_exercises(
        self,
        skill_name: str,
        difficulty: str
    ) -> List[Dict]:
        """Generate practice exercises"""
        
        exercises = []
        
        # Generate coding exercise
        coding_prompt = f"""
        Create a coding exercise for {skill_name} at {difficulty} level.
        Include:
        1. Problem statement
        2. Input/Output examples
        3. Hints
        4. Solution approach (don't give full solution)
        """
        
        coding_exercise = await self.llm.apredict(coding_prompt)
        
        exercises.append({
            "type": "coding",
            "title": f"{skill_name} Coding Challenge",
            "description": "Implement a solution for the following problem",
            "content": coding_exercise,
            "estimated_time": 20
        })
        
        # Generate conceptual questions
        exercises.append({
            "type": "quiz",
            "title": f"{skill_name} Concepts Quiz",
            "questions": [
                {
                    "question": f"What is the main purpose of {skill_name}?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct": 0
                },
                {
                    "question": f"Which approach is best for {skill_name}?",
                    "options": ["Approach 1", "Approach 2", "Approach 3", "Approach 4"],
                    "correct": 1
                }
            ]
        })
        
        return exercises
    
    def _get_curated_resources(self, skill_name: str) -> List[Dict]:
        """Get curated learning resources"""
        # In production, this would query a resource database
        return [
            {
                "title": f"Official {skill_name} Documentation",
                "url": "https://docs.example.com",
                "type": "documentation",
                "difficulty": "all"
            },
            {
                "title": f"{skill_name} Video Tutorial",
                "url": "https://youtube.com/example",
                "type": "video",
                "difficulty": "beginner"
            },
            {
                "title": f"Advanced {skill_name} Techniques",
                "url": "https://blog.example.com",
                "type": "article",
                "difficulty": "advanced"
            }
        ]
    
    async def check_skill_mastery(
        self,
        user_id: int,
        skill_id: str
    ) -> float:
        """Check if user has mastered a skill"""
        # In production, analyze user's exercise performance
        # For now, return a mock value
        return 0.75
    
    async def get_next_module(self, user_id: int) -> Dict:
        """Get the next recommended module"""
        return {
            "module_id": "next_module",
            "skill_name": "Next Skill to Learn",
            "reason": "Based on your progress, this is the logical next step"
        }
    
    async def get_skill_resources(
        self,
        skill_name: str,
        user_level: int
    ) -> List[Dict]:
        """Get personalized skill resources"""
        
        prompt = f"""
        Recommend learning resources for:
        Skill: {skill_name}
        Experience Level: {user_level} years
        
        Include:
        - Free resources
        - Paid courses
        - Books
        - Practice platforms
        
        Format as a list with title, description, type, and estimated time.
        """
        
        response = await self.llm.apredict(prompt)
        
        # Parse and structure response
        resources = [
            {
                "title": "Resource 1",
                "description": "Great for beginners",
                "type": "course",
                "url": "https://example.com",
                "estimated_hours": 20,
                "cost": "free"
            },
            {
                "title": "Resource 2",
                "description": "Comprehensive guide",
                "type": "book",
                "url": "https://example.com",
                "estimated_hours": 40,
                "cost": "$30"
            }
        ]
        
        return resources
    
    def _assess_understanding(self, user_response: str) -> str:
        """Tool: Assess user's understanding"""
        return f"Understanding assessed for: {user_response}"
    
    def _adjust_difficulty(self, current_difficulty: str) -> str:
        """Tool: Adjust difficulty level"""
        return f"Difficulty adjusted from: {current_difficulty}"
    
    def _generate_exercise(self, topic: str) -> str:
        """Tool: Generate new exercise"""
        return f"Exercise generated for: {topic}"
    
    def _provide_hint(self, problem_context: str) -> str:
        """Tool: Provide hint for problem"""
        return f"Hint provided for: {problem_context}"
'''

    def _get_analytics_tracker(self):
        return '''from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.learning import LearningSession
from app.models.roadmap import Phase, Skill
from app.models.user import User

class ProgressAnalytics:
    """Track and analyze user progress"""
    
    async def analyze_user_progress(
        self,
        db: AsyncSession,
        user_id: int
    ) -> Dict:
        """Comprehensive progress analysis"""
        
        # Get learning history
        learning_data = await self._fetch_learning_history(db, user_id)
        
        # Calculate metrics
        metrics = {
            "completion_rate": await self._calculate_completion_rate(db, user_id),
            "learning_velocity": self._calculate_velocity(learning_data),
            "skill_mastery": await self._assess_skill_mastery(db, user_id),
            "engagement_score": self._calculate_engagement(learning_data),
            "current_streak": self._calculate_streak(learning_data)
        }
        
        # Predict future progress
        predictions = self._predict_timeline(metrics)
        
        # Identify at-risk areas
        risk_analysis = self._identify_risks(metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            metrics,
            predictions,
            risk_analysis
        )
        
        return {
            "current_metrics": metrics,
            "predictions": predictions,
            "risks": risk_analysis,
            "recommendations": recommendations,
            "milestone_progress": await self._get_milestone_progress(db, user_id)
        }
    
    async def _fetch_learning_history(
        self,
        db: AsyncSession,
        user_id: int
    ) -> List[Dict]:
        """Fetch user's learning history"""
        
        result = await db.execute(
            select(LearningSession)
            .where(LearningSession.user_id == user_id)
            .order_by(LearningSession.started_at.desc())
            .limit(100)
        )
        
        sessions = result.scalars().all()
        
        return [
            {
                "started_at": session.started_at,
                "duration_minutes": session.duration_minutes,
                "score": session.score,
                "completed_exercises": session.completed_exercises
            }
            for session in sessions
        ]
    
    async def _calculate_completion_rate(
        self,
        db: AsyncSession,
        user_id: int
    ) -> float:
        """Calculate overall completion rate"""
        
        # Get total and completed phases
        total_phases = await db.execute(
            select(func.count(Phase.id))
            .join(Roadmap)
            .where(Roadmap.user_id == user_id)
        )
        total = total_phases.scalar()
        
        completed_phases = await db.execute(
            select(func.count(Phase.id))
            .join(Roadmap)
            .where(Roadmap.user_id == user_id)
            .where(Phase.status == "completed")
        )
        completed = completed_phases.scalar()
        
        if total == 0:
            return 0.0
        
        return round(completed / total, 2)
    
    def _calculate_velocity(self, learning_data: List[Dict]) -> Dict:
        """Calculate learning velocity (progress per week)"""
        
        if not learning_data:
            return {"current": 0, "trend": "stable"}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(learning_data)
        df['started_at'] = pd.to_datetime(df['started_at'])
        
        # Calculate weekly progress
        df['week'] = df['started_at'].dt.isocalendar().week
        weekly_hours = df.groupby('week')['duration_minutes'].sum() / 60
        
        # Current week vs previous week
        if len(weekly_hours) >= 2:
            current = weekly_hours.iloc[-1]
            previous = weekly_hours.iloc[-2]
            
            if current > previous * 1.1:
                trend = "increasing"
            elif current < previous * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            current = weekly_hours.iloc[-1] if len(weekly_hours) > 0 else 0
            trend = "new"
        
        return {
            "current_hours_per_week": round(current, 1),
            "trend": trend
        }
    
    async def _assess_skill_mastery(
        self,
        db: AsyncSession,
        user_id: int
    ) -> Dict:
        """Assess mastery level for each skill"""
        
        # Get skills with learning data
        result = await db.execute(
            select(
                Skill.name,
                func.avg(LearningSession.score).label('avg_score'),
                func.count(LearningSession.id).label('practice_count')
            )
            .join(LearningSession, LearningSession.skill_id == Skill.id)
            .where(LearningSession.user_id == user_id)
            .group_by(Skill.name)
        )
        
        skill_data = result.all()
        
        mastery_levels = {}
        for skill_name, avg_score, practice_count in skill_data:
            # Calculate mastery based on score and practice
            if avg_score >= 0.9 and practice_count >= 5:
                level = "mastered"
            elif avg_score >= 0.7 and practice_count >= 3:
                level = "proficient"
            elif avg_score >= 0.5:
                level = "developing"
            else:
                level = "beginner"
            
            mastery_levels[skill_name] = {
                "level": level,
                "score": round(avg_score, 2),
                "practice_sessions": practice_count
            }
        
        return mastery_levels
    
    def _calculate_engagement(self, learning_data: List[Dict]) -> float:
        """Calculate engagement score based on consistency"""
        
        if not learning_data:
            return 0.0
        
        # Check learning frequency
        dates = [session['started_at'].date() for session in learning_data]
        unique_days = len(set(dates))
        
        # Calculate engagement (days active / days in period)
        if dates:
            period_days = (dates[0] - dates[-1]).days + 1
            engagement = min(unique_days / max(period_days, 1), 1.0)
        else:
            engagement = 0.0
        
        return round(engagement, 2)
    
    def _calculate_streak(self, learning_data: List[Dict]) -> int:
        """Calculate current learning streak"""
        
        if not learning_data:
            return 0
        
        dates = sorted(set(session['started_at'].date() for session in learning_data), reverse=True)
        
        streak = 0
        expected_date = datetime.now().date()
        
        for date in dates:
            if date == expected_date:
                streak += 1
                expected_date = date - timedelta(days=1)
            else:
                break
        
        return streak
    
    def _predict_timeline(self, metrics: Dict) -> Dict:
        """Predict completion timeline based on current progress"""
        
        completion_rate = metrics['completion_rate']
        velocity = metrics['learning_velocity']['current_hours_per_week']
        
        if velocity == 0:
            weeks_remaining = float('inf')
        else:
            # Estimate based on typical requirements
            remaining_work = (1 - completion_rate) * 200  # Assume 200 hours total
            weeks_remaining = remaining_work / velocity
        
        return {
            "estimated_weeks_remaining": round(weeks_remaining, 1),
            "projected_completion_date": (
                datetime.now() + timedelta(weeks=weeks_remaining)
            ).strftime("%Y-%m-%d"),
            "confidence": "high" if velocity > 10 else "medium"
        }
    
    def _identify_risks(self, metrics: Dict) -> List[Dict]:
        """Identify potential risks to completion"""
        
        risks = []
        
        # Low velocity risk
        if metrics['learning_velocity']['current_hours_per_week'] < 5:
            risks.append({
                "type": "low_velocity",
                "severity": "high",
                "message": "Current pace is too slow to meet goals",
                "suggestion": "Increase weekly study time to at least 10 hours"
            })
        
        # Engagement risk
        if metrics['engagement_score'] < 0.5:
            risks.append({
                "type": "low_engagement",
                "severity": "medium",
                "message": "Inconsistent learning pattern detected",
                "suggestion": "Set a regular daily schedule for learning"
            })
        
        # Skill gaps
        mastery = metrics['skill_mastery']
        struggling_skills = [
            skill for skill, data in mastery.items()
            if data['level'] == 'beginner' and data['practice_sessions'] > 3
        ]
        
        if struggling_skills:
            risks.append({
                "type": "skill_difficulty",
                "severity": "medium",
                "message": f"Struggling with: {', '.join(struggling_skills)}",
                "suggestion": "Consider additional resources or mentorship"
            })
        
        return risks
    
    def _generate_recommendations(
        self,
        metrics: Dict,
        predictions: Dict,
        risks: List[Dict]
    ) -> List[str]:
        """Generate personalized recommendations"""
        
        recommendations = []
        
        # Based on velocity
        if metrics['learning_velocity']['current_hours_per_week'] < 10:
            recommendations.append(
                "Increase study time to 2 hours daily for faster progress"
            )
        
        # Based on streak
        if metrics['current_streak'] < 7:
            recommendations.append(
                "Build consistency with 30-minute daily sessions"
            )
        
        # Based on mastery
        struggling_skills = [
            skill for skill, data in metrics['skill_mastery'].items()
            if data['level'] in ['beginner', 'developing']
        ]
        
        if struggling_skills:
            recommendations.append(
                f"Focus on mastering {struggling_skills[0]} before moving forward"
            )
        
        # Based on risks
        for risk in risks:
            if risk['severity'] == 'high':
                recommendations.append(risk['suggestion'])
        
        return recommendations[:5]  # Top 5 recommendations
    
    async def _get_milestone_progress(
        self,
        db: AsyncSession,
        user_id: int
    ) -> List[Dict]:
        """Get progress towards major milestones"""
        
        # Get roadmap phases
        result = await db.execute(
            select(Phase)
            .join(Roadmap)
            .where(Roadmap.user_id == user_id)
            .order_by(Phase.order)
        )
        
        phases = result.scalars().all()
        
        milestones = []
        for phase in phases:
            milestone = {
                "title": phase.title,
                "status": phase.status,
                "progress": 1.0 if phase.status == "completed" else 0.5 if phase.status == "active" else 0.0,
                "estimated_completion": self._estimate_phase_completion(phase)
            }
            milestones.append(milestone)
        
        return milestones
    
    def _estimate_phase_completion(self, phase) -> str:
        """Estimate when a phase will be completed"""
        
        if phase.status == "completed":
            return "Completed"
        elif phase.status == "active":
            # Estimate based on duration
            weeks_remaining = phase.duration_weeks // 2  # Assume halfway
            completion_date = datetime.now() + timedelta(weeks=weeks_remaining)
            return completion_date.strftime("%Y-%m-%d")
        else:
            return "Not started"
'''

    # Frontend content methods
    def _get_frontend_layout(self):
        return '''import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from './providers'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'CareerGPS - AI Career Navigation Platform',
  description: 'Navigate your career transition with AI-powered guidance',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <Providers>
          <div className="min-h-screen bg-gray-900">
            {children}
          </div>
        </Providers>
      </body>
    </html>
  )
}
'''

    def _get_frontend_home_page(self):
        return '''
'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/hooks/useAuth'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import Link from 'next/link'

export default function Home() {
  const { user, loading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (user && !loading) {
      router.push('/dashboard')
    }
  }, [user, loading, router])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-20">
        <div className="text-center max-w-4xl mx-auto">
          <h1 className="text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Navigate Your AI Career with CareerGPS
          </h1>
          <p className="text-xl text-gray-300 mb-8">
            Transform from Software Engineer to AI Engineer with personalized roadmaps, 
            smart job matching, and AI-powered interview coaching
          </p>
          <div className="flex gap-4 justify-center">
            <Link href="/register">
              <Button size="lg" className="bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600">
                Start Your Journey
              </Button>
            </Link>
            <Link href="/login">
              <Button size="lg" variant="outline">
                Sign In
              </Button>
            </Link>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="container mx-auto px-4 py-20">
        <h2 className="text-3xl font-bold text-center mb-12">
          Everything You Need to Succeed
        </h2>
        <div className="grid md:grid-cols-3 gap-8">
          <Card className="p-6 bg-gray-800 border-gray-700">
            <div className="text-4xl mb-4">🗺️</div>
            <h3 className="text-xl font-semibold mb-2">AI Career Roadmap</h3>
            <p className="text-gray-400">
              Get a personalized learning path tailored to your background and goals
            </p>
          </Card>
          
          <Card className="p-6 bg-gray-800 border-gray-700">
            <div className="text-4xl mb-4">🎯</div>
            <h3 className="text-xl font-semibold mb-2">Smart Job Matching</h3>
            <p className="text-gray-400">
              Find AI roles that match your skills and help you grow
            </p>
          </Card>
          
          <Card className="p-6 bg-gray-800 border-gray-700">
            <div className="text-4xl mb-4">🤖</div>
            <h3 className="text-xl font-semibold mb-2">AI Interview Coach</h3>
            <p className="text-gray-400">
              Practice with AI that knows what top companies are looking for
            </p>
          </Card>
        </div>
      </div>

      {/* CTA Section */}
      <div className="container mx-auto px-4 py-20">
        <div className="bg-gradient-to-r from-blue-600 to-cyan-600 rounded-2xl p-12 text-center">
          <h2 className="text-3xl font-bold mb-4">
            Ready to Launch Your AI Career?
          </h2>
          <p className="text-xl mb-8 opacity-90">
            Join thousands of engineers successfully transitioning to AI
          </p>
          <Link href="/register">
            <Button size="lg" variant="secondary">
              Get Started Free
            </Button>
          </Link>
        </div>
      </div>
    </div>
  )
}
'''

    def _get_dashboard_page(self):
        return '''
'use client'

import { useEffect, useState } from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { useAuth } from '@/hooks/useAuth'
import { api } from '@/lib/api'
import Link from 'next/link'
import { BarChart, Calendar, Target, TrendingUp } from 'lucide-react'

export default function DashboardPage() {
  const { user } = useAuth()
  const [analytics, setAnalytics] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchAnalytics()
  }, [])

  const fetchAnalytics = async () => {
    try {
      const response = await api.analytics.getOverview()
      setAnalytics(response.data)
    } catch (error) {
      console.error('Failed to fetch analytics:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Welcome Header */}
      <div>
        <h1 className="text-3xl font-bold">Welcome back, {user?.full_name || user?.username}!</h1>
        <p className="text-gray-400 mt-2">Here's your progress overview</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-400">Learning Streak</p>
              <p className="text-2xl font-bold mt-1">{analytics?.current_streak || 0} days</p>
            </div>
            <Calendar className="h-8 w-8 text-blue-500" />
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-400">Overall Progress</p>
              <p className="text-2xl font-bold mt-1">{analytics?.overall_progress || 0}%</p>
            </div>
            <Target className="h-8 w-8 text-green-500" />
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-400">Job Matches</p>
              <p className="text-2xl font-bold mt-1">{analytics?.job_matches || 0}</p>
            </div>
            <TrendingUp className="h-8 w-8 text-purple-500" />
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-400">Hours Learned</p>
              <p className="text-2xl font-bold mt-1">{analytics?.total_hours || 0}h</p>
            </div>
            <BarChart className="h-8 w-8 text-orange-500" />
          </div>
        </Card>
      </div>

      {/* Current Phase */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Current Learning Phase</h2>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="font-medium">{analytics?.current_phase?.title || 'Getting Started'}</span>
              <span className="text-sm text-gray-400">
                {analytics?.current_phase?.progress || 0}% complete
              </span>
            </div>
            <Progress value={analytics?.current_phase?.progress || 0} className="h-2" />
          </div>
          <p className="text-gray-400">
            {analytics?.current_phase?.description || 'Begin your AI journey with foundational concepts'}
          </p>
          <div className="flex gap-4">
            <Link href="/learning">
              <Button>Continue Learning</Button>
            </Link>
            <Link href="/roadmap">
              <Button variant="outline">View Roadmap</Button>
            </Link>
          </div>
        </div>
      </Card>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Recent Job Matches</h2>
          <div className="space-y-3">
            {analytics?.recent_jobs?.map((job: any) => (
              <div key={job.id} className="flex justify-between items-center">
                <div>
                  <p className="font-medium">{job.title}</p>
                  <p className="text-sm text-gray-400">{job.company}</p>
                </div>
                <span className="text-sm text-green-500">{job.match}% match</span>
              </div>
            )) || (
              <p className="text-gray-400">No recent job matches</p>
            )}
          </div>
          <Link href="/jobs">
            <Button variant="outline" className="w-full mt-4">
              View All Jobs
            </Button>
          </Link>
        </Card>

        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Upcoming Tasks</h2>
          <div className="space-y-3">
            {analytics?.upcoming_tasks?.map((task: any, index: number) => (
              <div key={index} className="flex items-center gap-3">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <p>{task.title}</p>
              </div>
            )) || (
              <p className="text-gray-400">No upcoming tasks</p>
            )}
          </div>
          <Link href="/learning">
            <Button variant="outline" className="w-full mt-4">
              View Learning Plan
            </Button>
          </Link>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link href="/interview">
            <Button variant="outline" className="w-full">
              Practice Interview
            </Button>
          </Link>
          <Link href="/jobs">
            <Button variant="outline" className="w-full">
              Browse Jobs
            </Button>
          </Link>
          <Link href="/learning">
            <Button variant="outline" className="w-full">
              Continue Learning
            </Button>
          </Link>
        </div>
      </Card>
    </div>
  )
}
'''

    def _get_roadmap_page(self):
        return '''
'use client'

import { useEffect, useState } from 'react'
import { RoadmapVisualization } from './components/RoadmapVisualization'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { useRoadmap } from './hooks/useRoadmap'
import { Loader2 } from 'lucide-react'

export default function RoadmapPage() {
  const { roadmap, loading, generateRoadmap, updateProgress } = useRoadmap()
  const [generating, setGenerating] = useState(false)

  const handleGenerateRoadmap = async () => {
    setGenerating(true)
    try {
      await generateRoadmap({
        current_role: "Software Engineer",
        target_role: "AI Engineer",
        time_commitment: "part_time",
        background: {
          skills: ["Python", "JavaScript", "SQL"],
          experience_years: 5
        }
      })
    } finally {
      setGenerating(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-12 w-12 animate-spin text-blue-500" />
      </div>
    )
  }

  if (!roadmap) {
    return (
      <div className="max-w-2xl mx-auto">
        <Card className="p-8 text-center">
          <h2 className="text-2xl font-bold mb-4">Create Your AI Career Roadmap</h2>
          <p className="text-gray-400 mb-6">
            Get a personalized learning path based on your background and goals
          </p>
          <Button 
            onClick={handleGenerateRoadmap}
            disabled={generating}
            size="lg"
          >
            {generating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating...
              </>
            ) : (
              'Generate My Roadmap'
            )}
          </Button>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Your AI Career Roadmap</h1>
          <p className="text-gray-400 mt-2">
            From {roadmap.current_role} to {roadmap.target_role}
          </p>
        </div>
        <div className="text-right">
          <p className="text-sm text-gray-400">Estimated Duration</p>
          <p className="text-2xl font-bold">{roadmap.total_duration_months} months</p>
        </div>
      </div>

      <RoadmapVisualization 
        phases={roadmap.phases} 
        onPhaseComplete={updateProgress}
      />
    </div>
  )
}
'''

    def _get_roadmap_visualization(self):
        return '''import React from 'react'
import { motion } from 'framer-motion'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { CheckCircle, Circle, Lock } from 'lucide-react'

interface Phase {
  id: number
  title: string
  description: string
  duration_weeks: number
  status: 'completed' | 'active' | 'pending'
  skills: Skill[]
  projects: Project[]
}

interface Skill {
  id: number
  name: string
  category: string
  difficulty: number
}

interface Project {
  id: number
  title: string
  description: string
  difficulty: number
  estimated_hours: number
}

interface RoadmapVisualizationProps {
  phases: Phase[]
  onPhaseComplete?: (phaseId: number) => void
}

export const RoadmapVisualization: React.FC<RoadmapVisualizationProps> = ({
  phases,
  onPhaseComplete
}) => {
  const getPhaseIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-6 h-6 text-green-500" />
      case 'active':
        return <Circle className="w-6 h-6 text-blue-500 animate-pulse" />
      default:
        return <Lock className="w-6 h-6 text-gray-500" />
    }
  }

  const getPhaseProgress = (phase: Phase) => {
    if (phase.status === 'completed') return 100
    if (phase.status === 'active') return 50
    return 0
  }

  return (
    <div className="relative">
      {/* Timeline Line */}
      <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gray-700"></div>
      
      {/* Phases */}
      <div className="space-y-8">
        {phases.map((phase, index) => (
          <motion.div
            key={phase.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="relative"
          >
            {/* Phase Marker */}
            <div className="absolute left-5 top-8 z-10">
              {getPhaseIcon(phase.status)}
            </div>
            
            {/* Phase Content */}
            <Card className={`ml-16 p-6 ${
              phase.status === 'active' ? 'border-blue-500' : ''
            } ${
              phase.status === 'pending' ? 'opacity-60' : ''
            }`}>
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-xl font-semibold mb-2">{phase.title}</h3>
                  <p className="text-gray-400">{phase.description}</p>
                </div>
                <Badge variant={phase.status === 'active' ? 'default' : 'secondary'}>
                  {phase.duration_weeks} weeks
                </Badge>
              </div>
              
              {/* Progress Bar */}
              {phase.status !== 'pending' && (
                <div className="mb-4">
                  <Progress value={getPhaseProgress(phase)} className="h-2" />
                </div>
              )}
              
              {/* Skills Grid */}
              <div className="mb-4">
                <h4 className="font-medium mb-2">Skills to Learn</h4>
                <div className="flex flex-wrap gap-2">
                  {phase.skills.map((skill) => (
                    <Badge key={skill.id} variant="outline">
                      {skill.name}
                    </Badge>
                  ))}
                </div>
              </div>
              
              {/* Projects */}
              <div>
                <h4 className="font-medium mb-2">Projects</h4>
                <div className="space-y-2">
                  {phase.projects.map((project) => (
                    <div key={project.id} className="text-sm">
                      <span className="text-gray-300">• {project.title}</span>
                      <span className="text-gray-500 ml-2">
                        ({project.estimated_hours}h)
                      </span>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Action Button */}
              {phase.status === 'active' && (
                <div className="mt-4">
                  <Button
                    onClick={() => onPhaseComplete?.(phase.id)}
                    size="sm"
                  >
                    Mark as Complete
                  </Button>
                </div>
              )}
            </Card>
          </motion.div>
        ))}
      </div>
    </div>
  )
}
'''

    def _get_jobs_page(self):
        return '''
'use client'

import { useState, useEffect } from 'react'
import { JobCard } from './components/JobCard'
import { JobFilters } from './components/JobFilters'
import { Card } from '@/components/ui/card'
import { api } from '@/lib/api'
import { Loader2 } from 'lucide-react'

export default function JobsPage() {
  const [jobs, setJobs] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [filters, setFilters] = useState({
    location: '',
    remote_only: false,
    min_match: 60,
    salary_min: null
  })

  useEffect(() => {
    fetchJobs()
  }, [filters])

  const fetchJobs = async () => {
    try {
      setLoading(true)
      const response = await api.jobs.getMatches(filters)
      setJobs(response.data)
    } catch (error) {
      console.error('Failed to fetch jobs:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">AI Job Opportunities</h1>
        <p className="text-gray-400 mt-2">Personalized matches based on your profile</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Filters Sidebar */}
        <div className="lg:col-span-1">
          <JobFilters filters={filters} onFilterChange={setFilters} />
        </div>

        {/* Job Listings */}
        <div className="lg:col-span-3">
          {loading ? (
            <div className="flex items-center justify-center h-96">
              <Loader2 className="h-12 w-12 animate-spin text-blue-500" />
            </div>
          ) : jobs.length > 0 ? (
            <div className="space-y-4">
              {jobs.map((job) => (
                <JobCard key={job.job_id} job={job} />
              ))}
            </div>
          ) : (
            <Card className="p-8 text-center">
              <p className="text-gray-400">No jobs match your criteria</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
'''

    def _get_job_card(self):
        return '''import React from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { MapPin, DollarSign, TrendingUp, Clock } from 'lucide-react'
import Link from 'next/link'

interface JobCardProps {
  job: {
    job_id: string
    company: string
    title: string
    location: string
    remote: boolean
    salary_range: string
    match_percentage: number
    skill_gaps: Array<{ skill: string; difficulty: number }>
    growth_potential: number
    opportunity_score: number
    estimated_prep_weeks: number
    application_tips: string[]
  }
}

export const JobCard: React.FC<JobCardProps> = ({ job }) => {
  const getMatchColor = (percentage: number) => {
    if (percentage >= 80) return 'text-green-500'
    if (percentage >= 60) return 'text-blue-500'
    return 'text-yellow-500'
  }

  return (
    <Card className="p-6 hover:border-blue-500 transition-colors">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-xl font-semibold mb-1">{job.title}</h3>
          <p className="text-gray-400">{job.company}</p>
        </div>
        <div className="text-right">
          <p className={`text-2xl font-bold ${getMatchColor(job.match_percentage)}`}>
            {job.match_percentage}%
          </p>
          <p className="text-sm text-gray-400">match</p>
        </div>
      </div>

      <div className="flex flex-wrap gap-4 mb-4 text-sm text-gray-400">
        <div className="flex items-center gap-1">
          <MapPin className="w-4 h-4" />
          {job.location}
        </div>
        {job.salary_range && (
          <div className="flex items-center gap-1">
            <DollarSign className="w-4 h-4" />
            {job.salary_range}
          </div>
        )}
        <div className="flex items-center gap-1">
          <TrendingUp className="w-4 h-4" />
          Growth: {(job.growth_potential * 100).toFixed(0)}%
        </div>
        <div className="flex items-center gap-1">
          <Clock className="w-4 h-4" />
          Prep: {job.estimated_prep_weeks} weeks
        </div>
      </div>

      {job.skill_gaps.length > 0 && (
        <div className="mb-4">
          <p className="text-sm font-medium mb-2">Skills to Learn:</p>
          <div className="flex flex-wrap gap-2">
            {job.skill_gaps.slice(0, 5).map((gap, index) => (
              <Badge key={index} variant="outline">
                {gap.skill}
              </Badge>
            ))}
            {job.skill_gaps.length > 5 && (
              <Badge variant="outline">+{job.skill_gaps.length - 5} more</Badge>
            )}
          </div>
        </div>
      )}

      <div className="flex justify-between items-center">
        <Badge variant={job.remote ? 'default' : 'secondary'}>
          {job.remote ? 'Remote' : 'On-site'}
        </Badge>
        <Link href={`/jobs/${job.job_id}`}>
          <Button size="sm">View Details</Button>
        </Link>
      </div>
    </Card>
  )
}
'''

    def _get_interview_simulator(self):
        return '''import React, { useState, useRef, useEffect } from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Camera, Mic, MicOff, Play, Square } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { api } from '@/lib/api'

export const InterviewSimulator: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false)
  const [currentQuestion, setCurrentQuestion] = useState<any>(null)
  const [feedback, setFeedback] = useState<any>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  useEffect(() => {
    // Initialize webcam
    navigator.mediaDevices.getUserMedia({ video: true, audio: true })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
      })
      .catch(err => console.error('Error accessing media devices:', err))
  }, [])

  const startInterview = async () => {
    try {
      const response = await api.interview.createSession({
        job_role: "AI Engineer",
        company: "Tech Company",
        interview_type: "behavioral"
      })
      
      setSessionId(response.data.id)
      setCurrentQuestion(response.data.questions[0])
    } catch (error) {
      console.error('Failed to start interview:', error)
    }
  }

  const startRecording = () => {
    const stream = videoRef.current?.srcObject as MediaStream
    if (!stream) return

    const mediaRecorder = new MediaRecorder(stream)
    mediaRecorderRef.current = mediaRecorder
    chunksRef.current = []

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data)
      }
    }

    mediaRecorder.start()
    setIsRecording(true)
  }

  const stopRecording = async () => {
    if (!mediaRecorderRef.current) return

    mediaRecorderRef.current.stop()
    setIsRecording(false)

    // Process answer
    const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' })
    
    // For demo, we'll use a text answer
    const response = await api.interview.submitAnswer(sessionId!, {
      answer_text: "This is my answer to the question...",
      audio_data: null // Would send audioBlob in production
    })

    setFeedback(response.data)
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <Card className="p-6">
        <h2 className="text-2xl font-bold mb-4">AI Interview Practice</h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Video Preview */}
          <div>
            <div className="relative aspect-video bg-gray-800 rounded-lg overflow-hidden">
              <video
                ref={videoRef}
                autoPlay
                muted
                className="w-full h-full object-cover"
              />
              
              {isRecording && (
                <div className="absolute top-4 right-4 flex items-center gap-2 bg-red-500 px-3 py-1 rounded-full">
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                  <span className="text-sm">Recording</span>
                </div>
              )}
            </div>
            
            <div className="flex justify-center gap-4 mt-4">
              {!sessionId ? (
                <Button onClick={startInterview} size="lg">
                  <Play className="mr-2 h-4 w-4" />
                  Start Interview
                </Button>
              ) : !isRecording ? (
                <Button onClick={startRecording} size="lg">
                  <Mic className="mr-2 h-4 w-4" />
                  Record Answer
                </Button>
              ) : (
                <Button onClick={stopRecording} size="lg" variant="destructive">
                  <Square className="mr-2 h-4 w-4" />
                  Stop Recording
                </Button>
              )}
            </div>
          </div>

          {/* Question and Feedback */}
          <div className="space-y-4">
            {currentQuestion && (
              <Card className="p-4">
                <h3 className="font-semibold mb-2">Question</h3>
                <p className="text-lg">{currentQuestion.text}</p>
                
                <div className="mt-4">
                  <p className="text-sm text-gray-400">Tips:</p>
                  <ul className="text-sm text-gray-400 mt-1">
                    {currentQuestion.hints.map((hint: string, index: number) => (
                      <li key={index}>• {hint}</li>
                    ))}
                  </ul>
                </div>
              </Card>
            )}

            <AnimatePresence>
              {feedback && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                >
                  <Card className="p-4 border-blue-500">
                    <h3 className="font-semibold mb-3">AI Feedback</h3>
                    
                    <div className="space-y-3">
                      <div>
                        <p className="text-sm text-gray-400">Content Quality</p>
                        <div className="flex items-center gap-2">
                          <div className="flex-1 bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-blue-500 h-2 rounded-full"
                              style={{ width: `${feedback.content_score * 100}%` }}
                            />
                          </div>
                          <span className="text-sm">{(feedback.content_score * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                      
                      <div>
                        <p className="text-sm text-gray-400">Structure</p>
                        <div className="flex items-center gap-2">
                          <div className="flex-1 bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-green-500 h-2 rounded-full"
                              style={{ width: `${feedback.structure_score * 100}%` }}
                            />
                          </div>
                          <span className="text-sm">{(feedback.structure_score * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                      
                      <div className="mt-4">
                        <p className="text-sm font-medium mb-2">Suggestions:</p>
                        <ul className="text-sm space-y-1">
                          {feedback.suggestions.map((suggestion: string, index: number) => (
                            <li key={index} className="text-gray-300">• {suggestion}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </Card>
    </div>
  )
}
'''

    def _get_api_client(self):
        return '''import axios, { AxiosInstance } from 'axios'

class ApiClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    // Request interceptor for auth
    this.client.interceptors.request.use(async (config) => {
      const token = localStorage.getItem('access_token')
      if (token) {
        config.headers.Authorization = `Bearer ${token}`
      }
      return config
    })

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (error.response?.status === 401) {
          // Token expired, try to refresh
          try {
            const refreshToken = localStorage.getItem('refresh_token')
            const response = await this.client.post('/api/v1/auth/refresh', {
              refresh_token: refreshToken
            })
            
            localStorage.setItem('access_token', response.data.access_token)
            
            // Retry original request
            error.config.headers.Authorization = `Bearer ${response.data.access_token}`
            return this.client.request(error.config)
          } catch (refreshError) {
            // Refresh failed, redirect to login
            localStorage.removeItem('access_token')
            localStorage.removeItem('refresh_token')
            window.location.href = '/login'
          }
        }
        return Promise.reject(error)
      }
    )
  }

  // Auth endpoints
  auth = {
    login: (credentials: { username: string; password: string }) =>
      this.client.post('/api/v1/auth/login', credentials),
    
    register: (data: any) =>
      this.client.post('/api/v1/auth/register', data),
    
    me: () =>
      this.client.get('/api/v1/auth/me'),
    
    refresh: () =>
      this.client.post('/api/v1/auth/refresh'),
  }

  // Roadmap endpoints
  roadmap = {
    generate: (data: any) =>
      this.client.post('/api/v1/roadmap/generate', data),
    
    getCurrent: () =>
      this.client.get('/api/v1/roadmap/current'),
    
    updateProgress: (phaseId: number) =>
      this.client.post(`/api/v1/roadmap/progress/${phaseId}/complete`),
    
    getAnalytics: () =>
      this.client.get('/api/v1/roadmap/progress/analytics'),
  }

  // Jobs endpoints
  jobs = {
    getMatches: (filters?: any) =>
      this.client.get('/api/v1/jobs/matches', { params: filters }),
    
    getDetails: (id: string) =>
      this.client.get(`/api/v1/jobs/${id}`),
    
    trackApplication: (jobId: string, data: any) =>
      this.client.post(`/api/v1/jobs/${jobId}/apply`, data),
    
    getApplications: (status?: string) =>
      this.client.get('/api/v1/jobs/applications/mine', { params: { status } }),
  }

  // Interview endpoints
  interview = {
    createSession: (data: any) =>
      this.client.post('/api/v1/interview/session', data),
    
    submitAnswer: (sessionId: string, data: any) =>
      this.client.post(`/api/v1/interview/session/${sessionId}/answer`, data),
    
    getAnalysis: (sessionId: string) =>
      this.client.get(`/api/v1/interview/session/${sessionId}/analysis`),
    
    getHistory: () =>
      this.client.get('/api/v1/interview/history'),
    
    getQuestionBank: (filters?: any) =>
      this.client.get('/api/v1/interview/questions/bank', { params: filters }),
  }

  // Learning endpoints
  learning = {
    getCurrentModule: () =>
      this.client.get('/api/v1/learning/current-module'),
    
    completeModule: (moduleId: string, data: any) =>
      this.client.post(`/api/v1/learning/module/${moduleId}/complete`, data),
    
    getResources: (skillId: number) =>
      this.client.get(`/api/v1/learning/resources/${skillId}`),
    
    getProgress: () =>
      this.client.get('/api/v1/learning/progress/summary'),
  }

  // Analytics endpoints
  analytics = {
    getOverview: () =>
      this.client.get('/api/v1/analytics/overview'),
    
    getSkillProgress: () =>
      this.client.get('/api/v1/analytics/skills'),
    
    getLearningVelocity: () =>
      this.client.get('/api/v1/analytics/velocity'),
  }
}

export const api = new ApiClient()
'''

    def _get_button_component(self):
        return '''import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive:
          "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        outline:
          "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
'''

    def _get_card_component(self):
        return '''import * as React from "react"
import { cn } from "@/lib/utils"

const Card = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "rounded-lg border border-slate-200 bg-white text-slate-950 shadow-sm dark:border-slate-800 dark:bg-slate-950 dark:text-slate-50",
      className
    )}
    {...props}
  />
))
Card.displayName = "Card"

const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1.5 p-6", className)}
    {...props}
  />
))
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn(
      "text-2xl font-semibold leading-none tracking-tight",
      className
    )}
    {...props}
  />
))
CardTitle.displayName = "CardTitle"

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-slate-500 dark:text-slate-400", className)}
    {...props}
  />
))
CardDescription.displayName = "CardDescription"

const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
))
CardContent.displayName = "CardContent"

const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 pt-0", className)}
    {...props}
  />
))
CardFooter.displayName = "CardFooter"

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent }
'''

    # Additional helper methods
    def _get_frontend_globals_css(self):
        return '''@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;
    --card: 0 0% 100%;
    --card-foreground: 222.2 84% 4.9%;
    --popover: 0 0% 100%;
    --popover-foreground: 222.2 84% 4.9%;
    --primary: 222.2 47.4% 11.2%;
    --primary-foreground: 210 40% 98%;
    --secondary: 210 40% 96.1%;
    --secondary-foreground: 222.2 47.4% 11.2%;
    --muted: 210 40% 96.1%;
    --muted-foreground: 215.4 16.3% 46.9%;
    --accent: 210 40% 96.1%;
    --accent-foreground: 222.2 47.4% 11.2%;
    --destructive: 0 84.2% 60.2%;
    --destructive-foreground: 210 40% 98%;
    --border: 214.3 31.8% 91.4%;
    --input: 214.3 31.8% 91.4%;
    --ring: 222.2 84% 4.9%;
    --radius: 0.5rem;
  }

  .dark {
    --background: 222.2 84% 4.9%;
    --foreground: 210 40% 98%;
    --card: 222.2 84% 4.9%;
    --card-foreground: 210 40% 98%;
    --popover: 222.2 84% 4.9%;
    --popover-foreground: 210 40% 98%;
    --primary: 210 40% 98%;
    --primary-foreground: 222.2 47.4% 11.2%;
    --secondary: 217.2 32.6% 17.5%;
    --secondary-foreground: 210 40% 98%;
    --muted: 217.2 32.6% 17.5%;
    --muted-foreground: 215 20.2% 65.1%;
    --accent: 217.2 32.6% 17.5%;
    --accent-foreground: 210 40% 98%;
    --destructive: 0 62.8% 30.6%;
    --destructive-foreground: 210 40% 98%;
    --border: 217.2 32.6% 17.5%;
    --input: 217.2 32.6% 17.5%;
    --ring: 212.7 26.8% 83.9%;
  }
}

@layer base {
  * {
    @apply border-border;
  }
  body {
    @apply bg-background text-foreground;
  }
}
'''

    def _get_frontend_package_json(self):
        return json.dumps({
            "name": "careergps-frontend",
            "version": "0.1.0",
            "private": True,
            "scripts": {
                "dev": "next dev",
                "build": "next build",
                "start": "next start",
                "lint": "next lint",
                "test": "jest",
                "test:watch": "jest --watch"
            },
            "dependencies": {
                "next": "14.0.3",
                "react": "^18",
                "react-dom": "^18",
                "typescript": "^5",
                "@tanstack/react-query": "^5.8.4",
                "axios": "^1.6.2",
                "framer-motion": "^10.16.5",
                "lucide-react": "^0.292.0",
                "tailwindcss": "^3.3.6",
                "zustand": "^4.4.7",
                "next-auth": "^4.24.5",
                "socket.io-client": "^4.5.4",
                "recharts": "^2.9.3",
                "@radix-ui/react-dialog": "^1.0.5",
                "@radix-ui/react-slot": "^1.0.2",
                "@radix-ui/react-progress": "^1.0.3",
                "class-variance-authority": "^0.7.0",
                "clsx": "^2.0.0",
                "tailwind-merge": "^2.0.0"
            },
            "devDependencies": {
                "@types/node": "^20",
                "@types/react": "^18",
                "@types/react-dom": "^18",
                "autoprefixer": "^10.4.16",
                "postcss": "^8.4.31",
                "eslint": "^8",
                "eslint-config-next": "14.0.3",
                "jest": "^29.7.0",
                "@testing-library/react": "^14.1.2",
                "@testing-library/jest-dom": "^6.1.4"
            }
        }, indent=2)
    
    def _get_backend_requirements(self):
        return '''fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
sqlalchemy==2.0.23
asyncpg==0.29.0
alembic==1.12.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
redis==5.0.1
celery==5.3.4
langchain==0.0.340
openai==1.3.5
sentence-transformers==2.2.2
transformers==4.35.2
torch==2.1.1
numpy==1.26.2
pandas==2.1.3
scikit-learn==1.3.2
pinecone-client==2.2.4
boto3==1.29.7
python-dotenv==1.0.0
httpx==0.25.2
websockets==12.0
'''

    # Root file content methods
    def _get_readme_content(self):
        return '''# CareerGPS - AI-Powered Career Navigation Platform

Navigate your career transition from Software Engineer to AI Engineer with personalized roadmaps, smart job matching, and AI-powered interview coaching.

## 🚀 Features

- **AI Career Roadmap**: Personalized learning paths based on your background
- **Smart Job Matching**: AI-powered job recommendations with skill gap analysis
- **Interview Coach**: Practice with AI that provides real-time feedback
- **Adaptive Learning**: Content that adjusts to your pace and understanding
- **Progress Analytics**: Track your journey with detailed insights

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **AI/ML**: LangChain, OpenAI, HuggingFace
- **Database**: PostgreSQL, Redis
- **Queue**: Celery + RabbitMQ
- **Vector DB**: Pinecone/Qdrant

### Frontend
- **Framework**: Next.js 14 (TypeScript)
- **Styling**: Tailwind CSS
- **State**: Zustand + React Query
- **UI Components**: Radix UI + Custom components
- **Real-time**: WebSocket

## 📋 Prerequisites

- Docker & Docker Compose
- Node.js 18+
- Python 3.10+
- PostgreSQL 15+
- Redis 7+

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/careergps.git
   cd careergps
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start with Docker**
   ```bash
   docker-compose up
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 🔧 Development Setup

### Backend Development
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

## 📁 Project Structure

```
careergps/
├── backend/           # FastAPI backend
│   ├── app/          # Application code
│   ├── alembic/      # Database migrations
│   └── tests/        # Backend tests
├── frontend/         # Next.js frontend
│   ├── app/          # App router pages
│   ├── components/   # React components
│   └── lib/          # Utilities
├── infrastructure/   # Deployment configs
└── docs/            # Documentation
```

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

## 📖 API Documentation

Once the backend is running, visit http://localhost:8000/docs for interactive API documentation.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- LangChain for AI orchestration
- The open-source community

---

Built with ❤️ by the CareerGPS team
'''

    def _get_docker_compose_content(self):
        return '''version: '3.8'

services:
  frontend:
    build: 
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
      - NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
    volumes:
      - ./frontend:/app
      - /app/node_modules
    depends_on:
      - backend

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/careergps
      - REDIS_URL=redis://redis:6379
      - CELERY_BROKER_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - HUME_API_KEY=${HUME_API_KEY}
    volumes:
      - ./backend:/app
    depends_on:
      - db
      - redis

  worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
    command: celery -A app.workers.celery_app worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/careergps
      - REDIS_URL=redis://redis:6379
      - CELERY_BROKER_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=careergps
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
'''

    def _get_vscode_settings(self):
        return json.dumps({
            "editor.formatOnSave": True,
            "editor.codeActionsOnSave": {
                "source.fixAll.eslint": True
            },
            "python.linting.enabled": True,
            "python.linting.pylintEnabled": True,
            "python.formatting.provider": "black",
            "python.defaultInterpreterPath": "${workspaceFolder}/backend/venv/bin/python",
            "[python]": {
                "editor.defaultFormatter": "ms-python.black-formatter"
            },
            "[typescript]": {
                "editor.defaultFormatter": "esbenp.prettier-vscode"
            },
            "[typescriptreact]": {
                "editor.defaultFormatter": "esbenp.prettier-vscode"
            },
            "files.exclude": {
                "**/__pycache__": True,
                "**/*.pyc": True,
                "**/node_modules": True,
                "**/.next": True
            }
        }, indent=2)

    def _get_vscode_launch(self):
        return json.dumps({
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "FastAPI Backend",
                    "type": "python",
                    "request": "launch",
                    "module": "uvicorn",
                    "args": [
                        "app.main:app",
                        "--reload",
                        "--port",
                        "8000"
                    ],
                    "cwd": "${workspaceFolder}/backend",
                    "envFile": "${workspaceFolder}/backend/.env"
                },
                {
                    "name": "Next.js Frontend",
                    "type": "node",
                    "request": "launch",
                    "runtimeExecutable": "npm",
                    "runtimeArgs": ["run", "dev"],
                    "cwd": "${workspaceFolder}/frontend",
                    "envFile": "${workspaceFolder}/frontend/.env"
                }
            ]
        }, indent=2)

    def _get_vscode_extensions(self):
        return json.dumps({
            "recommendations": [
                "ms-python.python",
                "ms-python.vscode-pylance",
                "ms-python.black-formatter",
                "dbaeumer.vscode-eslint",
                "esbenp.prettier-vscode",
                "bradlc.vscode-tailwindcss",
                "prisma.prisma",
                "ms-azuretools.vscode-docker",
                "github.copilot"
            ]
        }, indent=2)

    # Continue with remaining helper methods...
    def _get_setup_script(self):
        return '''#!/bin/bash

echo "🚀 Setting up CareerGPS development environment..."

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js is required but not installed. Aborting." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting." >&2; exit 1; }

# Create environment files
echo "📝 Creating environment files..."
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

echo "⚠️  Please update the .env files with your API keys!"

# Install backend dependencies
echo "🐍 Setting up backend..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run database migrations
echo "🗄️  Setting up database..."
alembic upgrade head

cd ..

# Install frontend dependencies
echo "⚛️  Setting up frontend..."
cd frontend
npm install

cd ..

# Pull Docker images
echo "🐳 Pulling Docker images..."
docker-compose pull

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/postgres
mkdir -p data/redis
mkdir -p data/uploads
mkdir -p logs

echo "✅ Setup complete! Run 'docker-compose up' to start the development environment."
'''

    def create_zip_file(self):
        """Create a zip file of the entire project"""
        import shutil
        
        # Create all directories
        self.create_directory_structure()
        
        # Create all files
        print("📝 Creating root files...")
        self.create_root_files()
        
        print("🐍 Creating backend files...")
        self.create_backend_files()
        
        print("⚛️ Creating frontend files...")
        self.create_frontend_files()
        
        print("📜 Creating scripts...")
        self.create_scripts()
        
        print("📚 Creating documentation...")
        self.create_documentation()
        
        # Create zip file
        print("📦 Creating zip file...")
        shutil.make_archive('careergps-complete', 'zip', self.base_dir)
        
        # Clean up
        print("🧹 Cleaning up...")
        shutil.rmtree(self.base_dir)
        
        print("✅ Complete! Project saved as: careergps-complete.zip")
        
        return "careergps-complete.zip"

# Main execution
if __name__ == "__main__":
    generator = CareerGPSGenerator()
    zip_file = generator.create_zip_file()
    print(f"\n🎉 CareerGPS project successfully generated!")
    print(f"📦 Your complete project is ready in: {zip_file}")
    print("\n🚀 To get started:")
    print("1. Unzip the file: unzip careergps-complete.zip")
    print("2. Navigate to directory: cd careergps")
    print("3. Follow the setup instructions in README.md")
#!/usr/bin/env python3
"""
CareerGPS Complete Project Generator
Creates the full-stack application with UI and backend code
"""

import os
import json
import zipfile
from pathlib import Path
import textwrap

class CareerGPSGenerator:
    def __init__(self):
        self.base_dir = "careergps"
        
    def create_directory_structure(self):
        """Create the complete directory structure"""
        directories = [
            # Backend directories
            f"{self.base_dir}/backend/app/api/v1/endpoints",
            f"{self.base_dir}/backend/app/models",
            f"{self.base_dir}/backend/app/schemas",
            f"{self.base_dir}/backend/app/modules/roadmap",
            f"{self.base_dir}/backend/app/modules/jobs",
            f"{self.base_dir}/backend/app/modules/interview",
            f"{self.base_dir}/backend/app/modules/learning",
            f"{self.base_dir}/backend/app/modules/analytics",
            f"{self.base_dir}/backend/app/core",
            f"{self.base_dir}/backend/app/workers",
            f"{self.base_dir}/backend/app/utils",
            f"{self.base_dir}/backend/alembic/versions",
            f"{self.base_dir}/backend/tests",
            
            # Frontend directories
            f"{self.base_dir}/frontend/app/(auth)/login",
            f"{self.base_dir}/frontend/app/(auth)/register",
            f"{self.base_dir}/frontend/app/dashboard",
            f"{self.base_dir}/frontend/app/roadmap/components",
            f"{self.base_dir}/frontend/app/roadmap/hooks",
            f"{self.base_dir}/frontend/app/jobs/[id]",
            f"{self.base_dir}/frontend/app/jobs/components",
            f"{self.base_dir}/frontend/app/interview/components",
            f"{self.base_dir}/frontend/app/learning/components",
            f"{self.base_dir}/frontend/components/ui",
            f"{self.base_dir}/frontend/components/layout",
            f"{self.base_dir}/frontend/components/shared",
            f"{self.base_dir}/frontend/lib",
            f"{self.base_dir}/frontend/hooks",
            f"{self.base_dir}/frontend/public/images",
            f"{self.base_dir}/frontend/public/fonts",
            f"{self.base_dir}/frontend/styles",
            
            # Infrastructure
            f"{self.base_dir}/infrastructure/kubernetes/deployments",
            f"{self.base_dir}/infrastructure/kubernetes/services",
            f"{self.base_dir}/infrastructure/terraform",
            f"{self.base_dir}/infrastructure/scripts",
            
            # Other directories
            f"{self.base_dir}/docs",
            f"{self.base_dir}/scripts",
            f"{self.base_dir}/.vscode",
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def create_root_files(self):
        """Create root configuration files"""
        files = {
            f"{self.base_dir}/README.md": self._get_readme_content(),
            f"{self.base_dir}/docker-compose.yml": self._get_docker_compose_content(),
            f"{self.base_dir}/.env.example": self._get_env_example_content(),
            f"{self.base_dir}/.gitignore": self._get_gitignore_content(),
            f"{self.base_dir}/LICENSE": self._get_license_content(),
            f"{self.base_dir}/Makefile": self._get_makefile_content(),
            f"{self.base_dir}/.vscode/settings.json": self._get_vscode_settings(),
            f"{self.base_dir}/.vscode/launch.json": self._get_vscode_launch(),
            f"{self.base_dir}/.vscode/extensions.json": self._get_vscode_extensions(),
        }
        
        for file_path, content in files.items():
            with open(file_path, 'w') as f:
                f.write(content)
    
    def create_backend_files(self):
        """Create all backend files with complete implementation"""
        backend_files = {
            # Main application files
            f"{self.base_dir}/backend/app/main.py": self._get_backend_main(),
            f"{self.base_dir}/backend/app/config.py": self._get_backend_config(),
            f"{self.base_dir}/backend/app/database.py": self._get_backend_database(),
            f"{self.base_dir}/backend/app/dependencies.py": self._get_backend_dependencies(),
            
            # Models
            f"{self.base_dir}/backend/app/models/__init__.py": self._get_models_init(),
            f"{self.base_dir}/backend/app/models/user.py": self._get_user_model(),
            f"{self.base_dir}/backend/app/models/roadmap.py": self._get_roadmap_model(),
            f"{self.base_dir}/backend/app/models/job.py": self._get_job_model(),
            f"{self.base_dir}/backend/app/models/learning.py": self._get_learning_model(),
            
            # Schemas
            f"{self.base_dir}/backend/app/schemas/__init__.py": self._get_schemas_init(),
            f"{self.base_dir}/backend/app/schemas/user.py": self._get_user_schema(),
            f"{self.base_dir}/backend/app/schemas/roadmap.py": self._get_roadmap_schema(),
            f"{self.base_dir}/backend/app/schemas/job.py": self._get_job_schema(),
            f"{self.base_dir}/backend/app/schemas/interview.py": self._get_interview_schema(),
            
            # API Endpoints
            f"{self.base_dir}/backend/app/api/v1/router.py": self._get_api_router(),
            f"{self.base_dir}/backend/app/api/v1/endpoints/auth.py": self._get_auth_endpoint(),
            f"{self.base_dir}/backend/app/api/v1/endpoints/roadmap.py": self._get_roadmap_endpoint(),
            f"{self.base_dir}/backend/app/api/v1/endpoints/jobs.py": self._get_jobs_endpoint(),
            f"{self.base_dir}/backend/app/api/v1/endpoints/interview.py": self._get_interview_endpoint(),
            f"{self.base_dir}/backend/app/api/v1/endpoints/learning.py": self._get_learning_endpoint(),
            
            # Core modules
            f"{self.base_dir}/backend/app/core/__init__.py": "",
            f"{self.base_dir}/backend/app/core/auth.py": self._get_auth_core(),
            f"{self.base_dir}/backend/app/core/security.py": self._get_security_core(),
            f"{self.base_dir}/backend/app/core/websocket.py": self._get_websocket_core(),
            
            # AI/ML Modules
            f"{self.base_dir}/backend/app/modules/roadmap/agent.py": self._get_roadmap_agent(),
            f"{self.base_dir}/backend/app/modules/roadmap/generator.py": self._get_roadmap_generator(),
            f"{self.base_dir}/backend/app/modules/jobs/matching_engine.py": self._get_job_matching_engine(),
            f"{self.base_dir}/backend/app/modules/jobs/scraper.py": self._get_job_scraper(),
            f"{self.base_dir}/backend/app/modules/interview/coach.py": self._get_interview_coach(),
            f"{self.base_dir}/backend/app/modules/learning/adaptive_agent.py": self._get_learning_agent(),
            f"{self.base_dir}/backend/app/modules/analytics/tracker.py": self._get_analytics_tracker(),
            
            # Workers
            f"{self.base_dir}/backend/app/workers/celery_app.py": self._get_celery_app(),
            f"{self.base_dir}/backend/app/workers/tasks.py": self._get_celery_tasks(),
            
            # Utils
            f"{self.base_dir}/backend/app/utils/__init__.py": "",
            f"{self.base_dir}/backend/app/utils/embeddings.py": self._get_embeddings_util(),
            
            # Config files
            f"{self.base_dir}/backend/requirements.txt": self._get_backend_requirements(),
            f"{self.base_dir}/backend/requirements-dev.txt": self._get_backend_requirements_dev(),
            f"{self.base_dir}/backend/Dockerfile": self._get_backend_dockerfile(),
            f"{self.base_dir}/backend/.env.example": self._get_backend_env_example(),
            f"{self.base_dir}/backend/alembic.ini": self._get_alembic_ini(),
            f"{self.base_dir}/backend/pytest.ini": self._get_pytest_ini(),
            
            # Tests
            f"{self.base_dir}/backend/tests/conftest.py": self._get_test_conftest(),
            f"{self.base_dir}/backend/tests/test_roadmap.py": self._get_test_roadmap(),
        }
        
        # Create __init__.py files
        init_files = [
            f"{self.base_dir}/backend/app/__init__.py",
            f"{self.base_dir}/backend/app/api/__init__.py",
            f"{self.base_dir}/backend/app/api/v1/__init__.py",
            f"{self.base_dir}/backend/app/api/v1/endpoints/__init__.py",
            f"{self.base_dir}/backend/app/modules/__init__.py",
            f"{self.base_dir}/backend/app/modules/roadmap/__init__.py",
            f"{self.base_dir}/backend/app/modules/jobs/__init__.py",
            f"{self.base_dir}/backend/app/modules/interview/__init__.py",
            f"{self.base_dir}/backend/app/modules/learning/__init__.py",
            f"{self.base_dir}/backend/app/modules/analytics/__init__.py",
            f"{self.base_dir}/backend/app/workers/__init__.py",
            f"{self.base_dir}/backend/tests/__init__.py",
        ]
        
        for init_file in init_files:
            Path(init_file).touch()
        
        for file_path, content in backend_files.items():
            with open(file_path, 'w') as f:
                f.write(content)
    
    def create_frontend_files(self):
        """Create all frontend files with complete implementation"""
        frontend_files = {
            # Root app files
            f"{self.base_dir}/frontend/app/layout.tsx": self._get_frontend_layout(),
            f"{self.base_dir}/frontend/app/page.tsx": self._get_frontend_home_page(),
            f"{self.base_dir}/frontend/app/globals.css": self._get_frontend_globals_css(),
            f"{self.base_dir}/frontend/app/providers.tsx": self._get_frontend_providers(),
            
            # Auth pages
            f"{self.base_dir}/frontend/app/(auth)/layout.tsx": self._get_auth_layout(),
            f"{self.base_dir}/frontend/app/(auth)/login/page.tsx": self._get_login_page(),
            f"{self.base_dir}/frontend/app/(auth)/register/page.tsx": self._get_register_page(),
            
            # Dashboard
            f"{self.base_dir}/frontend/app/dashboard/page.tsx": self._get_dashboard_page(),
            f"{self.base_dir}/frontend/app/dashboard/layout.tsx": self._get_dashboard_layout(),
            
            # Roadmap
            f"{self.base_dir}/frontend/app/roadmap/page.tsx": self._get_roadmap_page(),
            f"{self.base_dir}/frontend/app/roadmap/components/RoadmapVisualization.tsx": self._get_roadmap_visualization(),
            f"{self.base_dir}/frontend/app/roadmap/components/PhaseCard.tsx": self._get_phase_card(),
            f"{self.base_dir}/frontend/app/roadmap/hooks/useRoadmap.ts": self._get_use_roadmap_hook(),
            
            # Jobs
            f"{self.base_dir}/frontend/app/jobs/page.tsx": self._get_jobs_page(),
            f"{self.base_dir}/frontend/app/jobs/[id]/page.tsx": self._get_job_detail_page(),
            f"{self.base_dir}/frontend/app/jobs/components/JobCard.tsx": self._get_job_card(),
            f"{self.base_dir}/frontend/app/jobs/components/JobFilters.tsx": self._get_job_filters(),
            
            # Interview
            f"{self.base_dir}/frontend/app/interview/page.tsx": self._get_interview_page(),
            f"{self.base_dir}/frontend/app/interview/components/InterviewSimulator.tsx": self._get_interview_simulator(),
            
            # Learning
            f"{self.base_dir}/frontend/app/learning/page.tsx": self._get_learning_page(),
            f"{self.base_dir}/frontend/app/learning/components/LearningModule.tsx": self._get_learning_module(),
            
            # Components
            f"{self.base_dir}/frontend/components/ui/button.tsx": self._get_button_component(),
            f"{self.base_dir}/frontend/components/ui/card.tsx": self._get_card_component(),
            f"{self.base_dir}/frontend/components/ui/input.tsx": self._get_input_component(),
            f"{self.base_dir}/frontend/components/ui/dialog.tsx": self._get_dialog_component(),
            f"{self.base_dir}/frontend/components/layout/Header.tsx": self._get_header_component(),
            f"{self.base_dir}/frontend/components/layout/Sidebar.tsx": self._get_sidebar_component(),
            f"{self.base_dir}/frontend/components/shared/LoadingSpinner.tsx": self._get_loading_spinner(),
            
            # Lib files
            f"{self.base_dir}/frontend/lib/api.ts": self._get_api_client(),
            f"{self.base_dir}/frontend/lib/auth.ts": self._get_auth_lib(),
            f"{self.base_dir}/frontend/lib/utils.ts": self._get_utils_lib(),
            f"{self.base_dir}/frontend/lib/constants.ts": self._get_constants(),
            
            # Hooks
            f"{self.base_dir}/frontend/hooks/useAuth.ts": self._get_use_auth_hook(),
            f"{self.base_dir}/frontend/hooks/useWebSocket.ts": self._get_use_websocket_hook(),
            f"{self.base_dir}/frontend/hooks/useApi.ts": self._get_use_api_hook(),
            
            # Config files
            f"{self.base_dir}/frontend/package.json": self._get_frontend_package_json(),
            f"{self.base_dir}/frontend/tsconfig.json": self._get_tsconfig(),
            f"{self.base_dir}/frontend/next.config.js": self._get_next_config(),
            f"{self.base_dir}/frontend/tailwind.config.js": self._get_tailwind_config(),
            f"{self.base_dir}/frontend/postcss.config.js": self._get_postcss_config(),
            f"{self.base_dir}/frontend/.env.example": self._get_frontend_env_example(),
            f"{self.base_dir}/frontend/Dockerfile": self._get_frontend_dockerfile(),
        }
        
        for file_path, content in frontend_files.items():
            with open(file_path, 'w') as f:
                f.write(content)
    
    def create_scripts(self):
        """Create utility scripts"""
        scripts = {
            f"{self.base_dir}/scripts/setup.sh": self._get_setup_script(),
            f"{self.base_dir}/scripts/test.sh": self._get_test_script(),
            f"{self.base_dir}/scripts/deploy.sh": self._get_deploy_script(),
        }
        
        for script_path, content in scripts.items():
            with open(script_path, 'w') as f:
                f.write(content)
            os.chmod(script_path, 0o755)
    
    def create_documentation(self):
        """Create documentation files"""
        docs = {
            f"{self.base_dir}/docs/API.md": self._get_api_documentation(),
            f"{self.base_dir}/docs/ARCHITECTURE.md": self._get_architecture_documentation(),
            f"{self.base_dir}/docs/DEPLOYMENT.md": self._get_deployment_documentation(),
            f"{self.base_dir}/docs/CONTRIBUTING.md": self._get_contributing_documentation(),
        }
        
        for doc_path, content in docs.items():
            with open(doc_path, 'w') as f:
                f.write(content)
    
    # Content generation methods (Backend)
    def _get_backend_main(self):
        return '''from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api.v1.router import api_router
from app.core.config import settings
from app.database import engine, Base
from app.core.websocket import websocket_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting up CareerGPS API...")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Initialize AI models
    from app.modules.roadmap.agent import CareerRoadmapAgent
    from app.modules.jobs.matching_engine import JobMatchingEngine
    
    app.state.roadmap_agent = CareerRoadmapAgent()
    app.state.matching_engine = JobMatchingEngine()
    
    yield
    
    logger.info("Shutting down...")

app = FastAPI(
    title="CareerGPS API",
    description="AI-powered career transition platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix="/api/v1")
app.include_router(websocket_router, prefix="/ws")

@app.get("/")
async def root():
    return {
        "message": "Welcome to CareerGPS API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
'''

    def _get_backend_config(self):
        return '''from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "CareerGPS"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str
    
    # Redis
    REDIS_URL: str
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000"
    ]
    
    # AI/ML
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = "us-east1-gcp"
    HUME_API_KEY: str = ""
    
    # AWS
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = ""
    
    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
'''

    def _get_backend_database(self):
        return '''from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
'''

    def _get_backend_dependencies(self):
        return '''from typing import AsyncGenerator
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database import get_db
from app.models.user import User
from app.schemas.user import TokenData

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

async def get_current_user(
    db: AsyncSession = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id)
    except JWTError:
        raise credentials_exception
    
    user = await db.get(User, token_data.user_id)
    if user is None:
        raise credentials_exception
    
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    if not current_user.is_active:
        raise HTTPException(
            status_code=400, 
            detail="Inactive user"
        )
    return current_user
'''

    def _get_user_model(self):
        return '''from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

from app.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Profile information
    current_role = Column(String)
    target_role = Column(String)
    experience_years = Column(Integer)
    skills = Column(String)  # JSON string
    
    # Relationships
    roadmaps = relationship("Roadmap", back_populates="user")
    job_applications = relationship("JobApplication", back_populates="user")
    learning_sessions = relationship("LearningSession", back_populates="user")
'''

    def _get_roadmap_model(self):
        return '''from sqlalchemy import Column, Integer, String, ForeignKey, Float, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime

from app.database import Base

class Roadmap(Base):
    __tablename__ = "roadmaps"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    current_role = Column(String, nullable=False)
    target_role = Column(String, nullable=False)
    total_duration_months = Column(Integer)
    difficulty_level = Column(Integer)
    success_probability = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="roadmaps")
    phases = relationship("Phase", back_populates="roadmap", cascade="all, delete-orphan")

class Phase(Base):
    __tablename__ = "phases"
    
    id = Column(Integer, primary_key=True, index=True)
    roadmap_id = Column(Integer, ForeignKey("roadmaps.id"), nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text)
    duration_weeks = Column(Integer)
    order = Column(Integer)
    status = Column(String, default="pending")  # pending, active, completed
    
    # Relationships
    roadmap = relationship("Roadmap", back_populates="phases")
    skills = relationship("Skill", back_populates="phase", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="phase", cascade="all, delete-orphan")

class Skill(Base):
    __tablename__ = "skills"
    
    id = Column(Integer, primary_key=True, index=True)
    phase_id = Column(Integer, ForeignKey("phases.id"), nullable=False)
    name = Column(String, nullable=False)
    category = Column(String)
    difficulty = Column(Integer)
    resources = Column(Text)  # JSON string
    
    # Relationships
    phase = relationship("Phase", back_populates="skills")

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    phase_id = Column(Integer, ForeignKey("phases.id"), nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text)
    difficulty = Column(Integer)
    estimated_hours = Column(Integer)
    
    # Relationships
    phase = relationship("Phase", back_populates="projects")
'''

    def _get_job_model(self):
        return '''from sqlalchemy import Column, Integer, String, ForeignKey, Float, DateTime, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime

from app.database import Base

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String, unique=True, index=True)
    title = Column(String, nullable=False)
    company = Column(String, nullable=False)
    location = Column(String)
    remote = Column(Boolean, default=False)
    salary_min = Column(Integer)
    salary_max = Column(Integer)
    description = Column(Text)
    requirements = Column(Text)  # JSON string
    posted_date = Column(DateTime)
    url = Column(String)
    
    # AI-generated fields
    skill_requirements = Column(Text)  # JSON string
    seniority_level = Column(String)
    
    # Relationships
    applications = relationship("JobApplication", back_populates="job")

class JobApplication(Base):
    __tablename__ = "job_applications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    status = Column(String, default="applied")  # applied, interviewing, rejected, accepted
    match_score = Column(Float)
    applied_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="job_applications")
    job = relationship("Job", back_populates="applications")
'''

    def _get_learning_model(self):
        return '''from sqlalchemy import Column, Integer, String, ForeignKey, Float, DateTime, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime

from app.database import Base

class LearningSession(Base):
    __tablename__ = "learning_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    phase_id = Column(Integer, ForeignKey("phases.id"))
    skill_id = Column(Integer, ForeignKey("skills.id"))
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    duration_minutes = Column(Integer)
    completed_exercises = Column(Integer, default=0)
    score = Column(Float)
    
    # Relationships
    user = relationship("User", back_populates="learning_sessions")

class InterviewSession(Base):
    __tablename__ = "interview_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    job_role = Column(String)
    company = Column(String)
    interview_type = Column(String)  # behavioral, technical, system_design
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    overall_score = Column(Float)
    feedback = Column(Text)  # JSON string
    
    # Relationships
    questions = relationship("InterviewQuestion", back_populates="session")

class InterviewQuestion(Base):
    __tablename__ = "interview_questions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("interview_sessions.id"), nullable=False)
    question_text = Column(Text)
    answer_text = Column(Text)
    score = Column(Float)
    feedback = Column(Text)
    
    # Relationships
    session = relationship("InterviewSession", back_populates="questions")
'''

    def _get_user_schema(self):
        return '''from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    current_role: Optional[str] = None
    target_role: Optional[str] = None
    experience_years: Optional[int] = None
    skills: Optional[List[str]] = None

class UserCreate(UserBase):
    password: str

class UserUpdate(UserBase):
    password: Optional[str] = None

class UserInDB(UserBase):
    id: int
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class User(UserInDB):
    pass

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None
'''

    def _get_roadmap_schema(self):
        return '''from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class SkillSchema(BaseModel):
    id: Optional[int] = None
    name: str
    category: Optional[str] = None
    difficulty: Optional[int] = None
    resources: Optional[List[dict]] = None
    
    class Config:
        from_attributes = True

class ProjectSchema(BaseModel):
    id: Optional[int] = None
    title: str
    description: Optional[str] = None
    difficulty: Optional[int] = None
    estimated_hours: Optional[int] = None
    
    class Config:
        from_attributes = True

class PhaseSchema(BaseModel):
    id: Optional[int] = None
    title: str
    description: Optional[str] = None
    duration_weeks: int
    order: int
    status: str = "pending"
    skills: List[SkillSchema] = []
    projects: List[ProjectSchema] = []
    
    class Config:
        from_attributes = True

class RoadmapCreate(BaseModel):
    current_role: str
    target_role: str
    time_commitment: str  # full_time, part_time, weekends
    background: dict

class RoadmapSchema(BaseModel):
    id: int
    user_id: int
    current_role: str
    target_role: str
    total_duration_months: int
    difficulty_level: int
    success_probability: float
    phases: List[PhaseSchema]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
'''

    def _get_job_schema(self):
        return '''from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class JobMatchSchema(BaseModel):
    job_id: str
    company: str
    title: str
    location: str
    remote: bool
    salary_range: Optional[str] = None
    match_percentage: int
    skill_gaps: List[dict]
    growth_potential: float
    opportunity_score: float
    estimated_prep_weeks: int
    application_tips: List[str]
    
    class Config:
        from_attributes = True

class JobApplicationCreate(BaseModel):
    job_id: str
    resume_version: Optional[str] = None
    cover_letter: Optional[str] = None
    notes: Optional[str] = None

class JobApplicationSchema(BaseModel):
    id: int
    user_id: int
    job_id: int
    status: str
    match_score: float
    applied_at: datetime
    notes: Optional[str] = None
    
    class Config:
        from_attributes = True

class JobSearchFilters(BaseModel):
    location: Optional[str] = None
    remote_only: bool = False
    min_match: int = 60
    salary_min: Optional[int] = None
    company_size: Optional[str] = None
    industry: Optional[List[str]] = None
'''

    def _get_interview_schema(self):
        return '''from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class InterviewSessionCreate(BaseModel):
    job_role: str
    company: str
    interview_type: str = "behavioral"  # behavioral, technical, system_design

class InterviewQuestionSchema(BaseModel):
    id: str
    text: str
    category: str
    difficulty: str
    hints: List[str]
    
class InterviewFeedbackSchema(BaseModel):
    question_id: str
    answer_text: str
    content_score: float
    structure_score: float
    relevance_score: float
    delivery_score: Optional[float] = None
    suggestions: List[str]
    example_answer: Optional[str] = None
    
class InterviewSessionSchema(BaseModel):
    id: str
    user_id: str
    job_role: str
    company: str
    interview_type: str
    questions: List[InterviewQuestionSchema]
    status: str = "active"
    started_at: datetime
    
    class Config:
        from_attributes = True

class InterviewAnalysisSchema(BaseModel):
    session_id: str
    overall_score: float
    strengths: List[str]
    areas_for_improvement: List[str]
    detailed_feedback: Dict[str, InterviewFeedbackSchema]
    recommendations: List[str]
'''

    def _get_api_router(self):
        return '''from fastapi import APIRouter

from app.api.v1.endpoints import auth, roadmap, jobs, interview, learning

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(roadmap.router, prefix="/roadmap", tags=["roadmap"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_router.include_router(interview.router, prefix="/interview", tags=["interview"])
api_router.include_router(learning.router, prefix="/learning", tags=["learning"])
'''

    def _get_auth_endpoint(self):
        return '''from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta

from app.database import get_db
from app.core import auth
from app.models.user import User
from app.schemas.user import User as UserSchema, UserCreate, Token

router = APIRouter()

@router.post("/register", response_model=UserSchema)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""
    # Check if user exists
    existing_user = await auth.get_user_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Create new user
    user = await auth.create_user(db, user_data)
    return user

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Login and receive access token"""
    user = await auth.authenticate_user(
        db, 
        form_data.username, 
        form_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth.create_access_token(
        data={"sub": str(user.id)}
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.post("/refresh", response_model=Token)
async def refresh_token(
    current_user: User = Depends(auth.get_current_user)
):
    """Refresh access token"""
    access_token = auth.create_access_token(
        data={"sub": str(current_user.id)}
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.get("/me", response_model=UserSchema)
async def get_current_user(
    current_user: User = Depends(auth.get_current_user)
):
    """Get current user information"""
    return current_user
'''

    def _get_roadmap_endpoint(self):
        return '''from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.database import get_db
from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.roadmap import RoadmapCreate, RoadmapSchema
from app.modules.roadmap.agent import CareerRoadmapAgent

router = APIRouter()

@router.post("/generate", response_model=RoadmapSchema)
async def generate_roadmap(
    roadmap_data: RoadmapCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate personalized AI roadmap"""
    # Initialize agent
    agent = CareerRoadmapAgent()
    
    # Prepare user profile
    user_profile = {
        "user_id": current_user.id,
        "current_role": roadmap_data.current_role,
        "target_role": roadmap_data.target_role,
        "time_commitment": roadmap_data.time_commitment,
        "skills": current_user.skills.split(",") if current_user.skills else [],
        "experience_years": current_user.experience_years,
        **roadmap_data.background
    }
    
    # Generate roadmap
    roadmap = await agent.generate_roadmap(user_profile)
    
    # Save to database
    roadmap_model = await agent.save_roadmap(db, current_user.id, roadmap)
    
    return roadmap_model

@router.get("/current", response_model=RoadmapSchema)
async def get_current_roadmap(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's current roadmap"""
    # Get most recent roadmap
    roadmap = await db.execute(
        select(Roadmap)
        .where(Roadmap.user_id == current_user.id)
        .order_by(Roadmap.created_at.desc())
    )
    roadmap = roadmap.scalar_one_or_none()
    
    if not roadmap:
        raise HTTPException(
            status_code=404,
            detail="No roadmap found. Please generate one first."
        )
    
    return roadmap

@router.post("/progress/{phase_id}/complete")
async def complete_phase(
    phase_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Mark a phase as completed"""
    # Get phase
    phase = await db.get(Phase, phase_id)
    
    if not phase:
        raise HTTPException(status_code=404, detail="Phase not found")
    
    # Verify ownership
    roadmap = await db.get(Roadmap, phase.roadmap_id)
    if roadmap.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Update phase status
    phase.status = "completed"
    await db.commit()
    
    # Check if next phase should be activated
    next_phase = await db.execute(
        select(Phase)
        .where(Phase.roadmap_id == roadmap.id)
        .where(Phase.order == phase.order + 1)
    )
    next_phase = next_phase.scalar_one_or_none()
    
    if next_phase:
        next_phase.status = "active"
        await db.commit()
    
    return {"message": "Phase completed successfully"}

@router.get("/progress/analytics")
async def get_progress_analytics(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed progress analytics"""
    from app.modules.analytics.tracker import ProgressAnalytics
    
    analytics = ProgressAnalytics()
    progress_data = await analytics.analyze_user_progress(
        db, 
        current_user.id
    )
    
    return progress_data
'''

    def _get_jobs_endpoint(self):
        return '''from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from app.database import get_db
from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.job import JobMatchSchema, JobApplicationCreate, JobSearchFilters
from app.modules.jobs.matching_engine import JobMatchingEngine

router = APIRouter()

@router.get("/matches", response_model=List[JobMatchSchema])
async def get_job_matches(
    location: Optional[str] = Query(None),
    remote_only: bool = Query(False),
    min_match: int = Query(60),
    salary_min: Optional[int] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get AI-matched job opportunities"""
    # Initialize matching engine
    engine = JobMatchingEngine()
    
    # Prepare user profile
    user_profile = {
        "user_id": current_user.id,
        "skills": current_user.skills.split(",") if current_user.skills else [],
        "current_role": current_user.current_role,
        "target_role": current_user.target_role,
        "experience_years": current_user.experience_years
    }
    
    # Prepare filters
    filters = JobSearchFilters(
        location=location,
        remote_only=remote_only,
        min_match=min_match,
        salary_min=salary_min
    )
    
    # Get matches
    matches = await engine.match_jobs(user_profile, filters.dict())
    
    return matches

@router.get("/{job_id}")
async def get_job_details(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed job information with personalized insights"""
    # Get job from database
    job = await db.execute(
        select(Job).where(Job.external_id == job_id)
    )
    job = job.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get personalized insights
    engine = JobMatchingEngine()
    insights = await engine.get_job_insights(
        job,
        current_user
    )
    
    return {
        "job": job,
        "insights": insights
    }

@router.post("/{job_id}/apply")
async def track_job_application(
    job_id: str,
    application_data: JobApplicationCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Track a job application"""
    # Get job
    job = await db.execute(
        select(Job).where(Job.external_id == job_id)
    )
    job = job.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check if already applied
    existing = await db.execute(
        select(JobApplication)
        .where(JobApplication.user_id == current_user.id)
        .where(JobApplication.job_id == job.id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=400,
            detail="Already applied to this job"
        )
    
    # Create application record
    application = JobApplication(
        user_id=current_user.id,
        job_id=job.id,
        notes=application_data.notes
    )
    
    db.add(application)
    await db.commit()
    
    # Schedule follow-up reminders
    from app.workers.tasks import schedule_application_followup
    schedule_application_followup.delay(application.id)
    
    return {
        "message": "Application tracked successfully",
        "application_id": application.id
    }

@router.get("/applications/mine")
async def get_my_applications(
    status: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's job applications"""
    query = select(JobApplication).where(
        JobApplication.user_id == current_user.id
    )
    
    if status:
        query = query.where(JobApplication.status == status)
    
    result = await db.execute(query.order_by(JobApplication.applied_at.desc()))
    applications = result.scalars().all()
    
    return applications
'''

    def _get_interview_endpoint(self):
        return '''from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.database import get_db
from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.interview import (
    InterviewSessionCreate,
    InterviewSessionSchema,
    InterviewFeedbackSchema,
    InterviewAnalysisSchema
)
from app.modules.interview.coach import AIInterviewCoach

router = APIRouter()

# In-memory session storage (use Redis in production)
interview_sessions = {}

@router.post("/session", response_model=InterviewSessionSchema)
async def create_interview_session(
    session_data: InterviewSessionCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new AI interview session"""
    coach = AIInterviewCoach()
    
    session = await coach.create_interview_session(
        user_id=str(current_user.id),
        job_role=session_data.job_role,
        company=session_data.company,
        interview_type=session_data.interview_type
    )
    
    # Store session
    interview_sessions[session.id] = session
    
    return session

@router.post("/session/{session_id}/answer", response_model=InterviewFeedbackSchema)
async def submit_answer(
    session_id: str,
    answer_text: str,
    audio_data: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Submit answer and get AI feedback"""
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = interview_sessions[session_id]
    
    # Verify ownership
    if session.user_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    coach = AIInterviewCoach()
    
    # Process answer
    feedback = await coach.process_answer(
        session_id=session_id,
        text_answer=answer_text,
        audio_data=audio_data.encode() if audio_data else None
    )
    
    return feedback

@router.get("/session/{session_id}/analysis", response_model=InterviewAnalysisSchema)
async def get_session_analysis(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get complete session analysis"""
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = interview_sessions[session_id]
    
    # Verify ownership
    if session.user_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    coach = AIInterviewCoach()
    
    # Generate analysis
    analysis = await coach.generate_session_analysis(session_id)
    
    # Save to database
    db_session = InterviewSession(
        user_id=current_user.id,
        job_role=session.job_role,
        company=session.company,
        interview_type=session.interview_type,
        overall_score=analysis.overall_score,
        feedback=json.dumps(analysis.dict())
    )
    
    db.add(db_session)
    await db.commit()
    
    return analysis

@router.get("/history")
async def get_interview_history(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's interview history"""
    sessions = await db.execute(
        select(InterviewSession)
        .where(InterviewSession.user_id == current_user.id)
        .order_by(InterviewSession.started_at.desc())
    )
    
    return sessions.scalars().all()

@router.get("/questions/bank")
async def get_question_bank(
    role: Optional[str] = None,
    company: Optional[str] = None,
    interview_type: Optional[str] = None
):
    """Get interview questions from the bank"""
    coach = AIInterviewCoach()
    
    questions = await coach.get_question_bank(
        role=role,
        company=company,
        interview_type=interview_type
    )
    
    return questions
'''

    def _get_learning_endpoint(self):
        return '''from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.database import get_db
from app.dependencies import get_current_active_user
from app.models.user import User
from app.modules.learning.adaptive_agent import AdaptiveLearningAgent

router = APIRouter()

@router.get("/current-module")
async def get_current_module(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current learning module based on roadmap progress"""
    # Get active phase
    active_phase = await db.execute(
        select(Phase)
        .join(Roadmap)
        .where(Roadmap.user_id == current_user.id)
        .where(Phase.status == "active")
    )
    active_phase = active_phase.scalar_one_or_none()
    
    if not active_phase:
        raise HTTPException(
            status_code=404,
            detail="No active learning phase found"
        )
    
    # Get current skill to focus on
    current_skill = await db.execute(
        select(Skill)
        .where(Skill.phase_id == active_phase.id)
        .order_by(Skill.id)
    )
    current_skill = current_skill.first()
    
    if not current_skill:
        raise HTTPException(
            status_code=404,
            detail="No skills found in current phase"
        )
    
    # Generate adaptive content
    agent = AdaptiveLearningAgent()
    module = await agent.generate_learning_module(
        user_id=current_user.id,
        skill=current_skill[0],
        phase=active_phase
    )
    
    return module

@router.post("/module/{module_id}/complete")
async def complete_module(
    module_id: str,
    score: float,
    time_spent_minutes: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Mark a learning module as completed"""
    # Record learning session
    session = LearningSession(
        user_id=current_user.id,
        duration_minutes=time_spent_minutes,
        score=score,
        completed_exercises=1
    )
    
    db.add(session)
    await db.commit()
    
    # Check if skill is mastered
    agent = AdaptiveLearningAgent()
    mastery = await agent.check_skill_mastery(
        user_id=current_user.id,
        skill_id=module_id
    )
    
    return {
        "message": "Module completed",
        "mastery_level": mastery,
        "next_module": await agent.get_next_module(current_user.id)
    }

@router.get("/resources/{skill_id}")
async def get_learning_resources(
    skill_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get curated learning resources for a skill"""
    skill = await db.get(Skill, skill_id)
    
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    
    # Get resources
    agent = AdaptiveLearningAgent()
    resources = await agent.get_skill_resources(
        skill_name=skill.name,
        user_level=current_user.experience_years
    )
    
    return resources

@router.get("/progress/summary")
async def get_learning_progress(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get learning progress summary"""
    # Get all learning sessions
    sessions = await db.execute(
        select(LearningSession)
        .where(LearningSession.user_id == current_user.id)
        .order_by(LearningSession.started_at.desc())
    )
    sessions = sessions.scalars().all()
    
    # Calculate statistics
    total_hours = sum(s.duration_minutes for s in sessions) / 60
    avg_score = sum(s.score for s in sessions if s.score) / len(sessions) if sessions else 0
    
    # Get skill progress
    skill_progress = await db.execute(
        select(
            Skill.name,
            func.count(LearningSession.id).label("sessions"),
            func.avg(LearningSession.score).label("avg_score")
        )
        .join(LearningSession, LearningSession.skill_id == Skill.id)
        .where(LearningSession.user_id == current_user.id)
        .group_by(Skill.name)
    )
    
    return {
        "total_learning_hours": round(total_hours, 1),
        "average_score": round(avg_score, 2),
        "total_sessions": len(sessions),
        "skill_progress": skill_progress.all(),
        "learning_streak": calculate_learning_streak(sessions)
    }

def calculate_learning_streak(sessions):
    """Calculate current learning streak in days"""
    if not sessions:
        return 0
    
    streak = 1
    last_date = sessions[0].started_at.date()
    
    for session in sessions[1:]:
        session_date = session.started_at.date()
        diff = (last_date - session_date).days
        
        if diff == 1:
            streak += 1
            last_date = session_date
        elif diff > 1:
            break
    
    return streak
'''

    def _get_roadmap_agent(self):
        return '''from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from typing import Dict, List
import asyncio
import logging
import json

from app.core.config import settings
from app.schemas.roadmap import RoadmapSchema, PhaseSchema, SkillSchema, ProjectSchema

logger = logging.getLogger(__name__)

class CareerRoadmapAgent:
    """AI Agent for generating personalized career roadmaps"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Define tools for the agent
        self.tools = [
            Tool(
                name="analyze_skills",
                func=self._analyze_skills,
                description="Analyze user's current skills and experience"
            ),
            Tool(
                name="identify_gaps",
                func=self._identify_gaps,
                description="Identify skill gaps for target role"
            ),
            Tool(
                name="calculate_timeline",
                func=self._calculate_timeline,
                description="Calculate realistic timeline for transition"
            ),
            Tool(
                name="find_resources",
                func=self._find_resources,
                description="Find learning resources and courses"
            )
        ]
    
    async def generate_roadmap(self, user_profile: Dict) -> RoadmapSchema:
        """Generate a personalized career roadmap"""
        try:
            # Create prompt for roadmap generation
            prompt = f"""
            Create a detailed career roadmap for transitioning from {user_profile['current_role']} to {user_profile['target_role']}.
            
            User Profile:
            - Current Role: {user_profile['current_role']}
            - Target Role: {user_profile['target_role']}
            - Experience: {user_profile.get('experience_years', 0)} years
            - Current Skills: {', '.join(user_profile.get('skills', []))}
            - Time Commitment: {user_profile.get('time_commitment', 'part_time')}
            
            Generate a roadmap with:
            1. 4-6 learning phases
            2. Specific skills for each phase
            3. Practical projects to build portfolio
            4. Realistic timeline based on time commitment
            5. Resources and learning materials
            
            Format the response as a structured roadmap with phases, skills, and projects.
            """
            
            # Generate roadmap using LLM
            response = await self.llm.apredict(prompt)
            
            # Parse and structure the response
            roadmap = self._parse_roadmap_response(response, user_profile)
            
            return roadmap
            
        except Exception as e:
            logger.error(f"Error generating roadmap: {str(e)}")
            raise
    
    def _parse_roadmap_response(self, response: str, user_profile: Dict) -> RoadmapSchema:
        """Parse LLM response into structured roadmap"""
        # This is a simplified parser - in production, use more sophisticated parsing
        
        phases = []
        
        # Phase 1: Foundations
        phase1 = PhaseSchema(
            title="Mathematical & Programming Foundations",
            description="Build strong foundation in mathematics and Python programming",
            duration_weeks=8,
            order=1,
            status="active",
            skills=[
                SkillSchema(
                    name="Linear Algebra",
                    category="Mathematics",
                    difficulty=7,
                    resources=[
                        {"name": "Khan Academy Linear Algebra", "url": "https://www.khanacademy.org/math/linear-algebra", "type": "video"},
                        {"name": "3Blue1Brown Essence of Linear Algebra", "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3M