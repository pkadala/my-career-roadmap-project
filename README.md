# Career Roadmap AI - Backend Application

An AI-powered career transition roadmap generator built with LangChain 0.3.25, FastAPI, and Python 3.11.11.

## 🚀 Features

- **AI-Powered Career Analysis**: Analyzes career transitions and job market dynamics
- **Personalized Learning Roadmaps**: Creates month-by-month learning paths
- **Skills Gap Analysis**: Identifies transferable skills and learning requirements
- **Intelligent Resource Recommendations**: Suggests courses, certifications, and projects
- **Progress Tracking**: Monitor your career transition journey

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **AI/LLM**: LangChain 0.3.25
- **LLM Providers**: OpenAI / Anthropic
- **Database**: PostgreSQL with SQLAlchemy
- **Cache**: Redis
- **Vector Store**: ChromaDB
- **Language**: Python 3.11.11

## 📋 Prerequisites

- Python 3.11.11
- Docker & Docker Compose
- OpenAI or Anthropic API key

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd career-roadmap-ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

5. **Start databases**
   ```bash
   docker-compose up -d postgres redis
   ```

6. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

7. **Access the API**
   - Documentation: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

## 📚 API Documentation

### Generate Roadmap
```http
POST /api/v1/roadmap/generate
Content-Type: application/json

{
  "current_role": "Software Engineer",
  "target_role": "AI Engineer",
  "current_skills": ["Python", "JavaScript"],
  "experience_years": 3,
  "experience_level": "mid",
  "available_hours_per_week": 15
}
```

## 🐳 Docker Deployment

```bash
docker-compose up -d
```

## 🧪 Testing

```bash
pytest
```

## 📝 License

MIT License

## 🤝 Contributing

Pull requests are welcome!
