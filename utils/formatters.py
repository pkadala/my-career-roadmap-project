"""
Output formatting utilities.
"""
from typing import Dict, Any, List
from datetime import datetime, timedelta

class RoadmapFormatter:
    """Formatter for roadmap outputs."""
    
    @staticmethod
    def format_duration(hours: int) -> str:
        """Format duration in hours to human-readable string."""
        if hours < 1:
            return "Less than 1 hour"
        elif hours == 1:
            return "1 hour"
        elif hours < 24:
            return f"{hours} hours"
        else:
            days = hours // 24
            remaining_hours = hours % 24
            if remaining_hours == 0:
                return f"{days} day{'s' if days > 1 else ''}"
            else:
                return f"{days} day{'s' if days > 1 else ''} {remaining_hours} hour{'s' if remaining_hours > 1 else ''}"
    
    @staticmethod
    def format_timeline(months: float) -> str:
        """Format timeline in months to human-readable string."""
        if months < 1:
            weeks = int(months * 4)
            return f"{weeks} week{'s' if weeks != 1 else ''}"
        elif months == 1:
            return "1 month"
        elif months < 12:
            return f"{months:.1f} months" if months % 1 != 0 else f"{int(months)} months"
        else:
            years = months / 12
            if years == 1:
                return "1 year"
            elif years % 1 == 0:
                return f"{int(years)} years"
            else:
                return f"{years:.1f} years"
    
    @staticmethod
    def format_skill_level(level: int) -> str:
        """Format skill level to descriptive string."""
        levels = {
            0: "No Experience",
            1: "Beginner",
            2: "Novice",
            3: "Basic",
            4: "Intermediate",
            5: "Competent",
            6: "Proficient",
            7: "Advanced",
            8: "Expert",
            9: "Master",
            10: "Guru"
        }
        return levels.get(level, "Unknown")
    
    @staticmethod
    def format_cost(cost: float) -> str:
        """Format cost to currency string."""
        if cost == 0:
            return "Free"
        elif cost < 0:
            return "Unknown"
        else:
            return f"${cost:,.2f}"
    
    @staticmethod
    def format_progress_percentage(percentage: float) -> str:
        """Format progress percentage."""
        return f"{percentage:.1f}%"
