#!/usr/bin/env python3
"""
Quick start script for Career Roadmap AI application.
"""
import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🚀 Career Roadmap AI - Quick Start")
print("\nTo get started:")
print("1. Create virtual environment: python -m venv venv")
print("2. Activate it: source venv/bin/activate")
print("3. Install dependencies: pip install -r requirements.txt")
print("4. Copy .env.example to .env and add your API keys")
print("5. Start databases: docker-compose up -d")
print("6. Run: uvicorn app.main:app --reload")
print("\nVisit http://localhost:8000/docs for API documentation")
