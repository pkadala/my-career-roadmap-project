"""
Input validation utilities.
"""
import re
from typing import List

class RoleValidator:
    """Validator for job roles."""
    
    VALID_ROLE_PATTERN = re.compile(r'^[a-zA-Z\s\-\.]+)
    MIN_LENGTH = 3
    MAX_LENGTH = 100
    
    @classmethod
    def validate_role(cls, role: str) -> str:
        """Validate job role string."""
        if not role or not role.strip():
            raise ValueError("Role cannot be empty")
        
        role = role.strip()
        
        if len(role) < cls.MIN_LENGTH:
            raise ValueError(f"Role must be at least {cls.MIN_LENGTH} characters")
        
        if len(role) > cls.MAX_LENGTH:
            raise ValueError(f"Role must be at most {cls.MAX_LENGTH} characters")
        
        if not cls.VALID_ROLE_PATTERN.match(role):
            raise ValueError("Role contains invalid characters")
        
        return role


class SkillValidator:
    """Validator for skills."""
    
    VALID_SKILL_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-\.\+\#\/]+)
    MIN_LENGTH = 1
    MAX_LENGTH = 50
    MAX_SKILLS = 50
    
    @classmethod
    def validate_skill(cls, skill: str) -> str:
        """Validate individual skill."""
        if not skill or not skill.strip():
            raise ValueError("Skill cannot be empty")
        
        skill = skill.strip()
        
        if len(skill) < cls.MIN_LENGTH:
            raise ValueError(f"Skill must be at least {cls.MIN_LENGTH} characters")
        
        if len(skill) > cls.MAX_LENGTH:
            raise ValueError(f"Skill must be at most {cls.MAX_LENGTH} characters")
        
        if not cls.VALID_SKILL_PATTERN.match(skill):
            raise ValueError(f"Skill '{skill}' contains invalid characters")
        
        return skill
    
    @classmethod
    def validate_skills_list(cls, skills: List[str]) -> List[str]:
        """Validate list of skills."""
        if len(skills) > cls.MAX_SKILLS:
            raise ValueError(f"Cannot have more than {cls.MAX_SKILLS} skills")
        
        validated_skills = []
        seen = set()
        
        for skill in skills:
            validated_skill = cls.validate_skill(skill)
            skill_lower = validated_skill.lower()
            
            if skill_lower not in seen:
                validated_skills.append(validated_skill)
                seen.add(skill_lower)
        
        return validated_skills
