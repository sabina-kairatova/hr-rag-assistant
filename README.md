# HR RAG Assistant

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangChain 1.3+](https://img.shields.io/badge/langchain-1.0+-green.svg)](https://github.com/langchain-ai/langchain)
[![LangGraph 1.2+](https://img.shields.io/badge/langgraph-1.0+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-f90050?style=flat&logo=qdrant&logoColor=white)](https://qdrant.tech/)

**HR-RAG-ASSISTANT** is a production-focused Retrieval-Augmented Generation (RAG) system designed to assist company employees with the HR-related questions. The system combines Qdrant's powerful vector database with advanced search capabilities and presents the following features:

✅ Combines semantic and keywords search (hybrid retrieval)  
✅ Gives source-aware answers with retrieval-backed context  
✅ Implements observability with LangSmith tracing for optimized performance  
✅ Provides FastAPI application with health checks  

## 🏛️ High-Level Architecture

<div align="center">

```mermaid
graph TB
    A[User Query]--> B[Hybrid Search Engine]
    B --> C[Vector Search]
    B --> D[Keyword Search]
    C --> E[Qdrant Vector DB]
    D --> E
    E --> F[Query + Context]
    F --> G[GPT-4o API]
    G --> H[Source-aware Response]

    I[PDF upload] --> J[Chunking]
    J --> K[Embedding Service]
    K --> L[OpenAI API]
    L --> E

    style E fill:#ff6b6b
    style G fill:#4ecdc4
    style H fill:#45b7d1
```

</div>

## 🚀 Quick Start

### 📋 Prerequisites

- **Python 3.12+**
- **UV Package Manager** ([Install Guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Docker Desktop**
- **API Keys:** OpenAI, LangSmith, Qdrant

**1. Clone repository**
```bash
git clone https://github.com/sabina-kairatova/hr-rag-assistant.git
cd hr-rag-assistant
```

**2. Configure environment**
```bash
cp .env.example .env
```
Edit `.env` with your configuration.

**3. Install dependencies**
```bash
uv sync
```

**4. Start all services**
```bash
docker compose up --build -d
```

**4. Start all services**
```bash
curl -f http://localhost:8000/health
```

### **🏗️ Project Structure**

```
hr-rag-assistant/
├── app/                        # Main application code
│   ├── main.py                 # API endpoints (chat, health, metrics)
│   ├── agents.py               # LangGraph agent
│   ├── models.py               # Pydantic validation schemas
│   ├── config.py               # Environment configuration
│   ├── cache.py                # In-memory response cache with TTL 
│   ├── monitoring.py           # Log records for log aggregation
│   ├── security.py             # Security pipeline for input and output processing
│   ├── tools.py                # RAG retrieval tools
│   ├── prompts.py              # Prompts for LLM 
│   ├── embeddings.py           # OpenAI embeddings configuration
│   ├── vectore_store.py        # Qdrant vector store configuration
│   ├── hybrid_search.py        # Hybrid retrieval
│   └── build_knowledge_base.py # Build and load HR knowledge base
├── tests/                      # Test suite
├── data/                       # RAG knowledge base
├── Dockerfile                  # Project container
└── docker-compose.yml          # Docker service orchestration
```

### **📡 API Endpoints Reference**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/metrics` | GET | Metrics for monitoring dashboard|
| `/cache/stats` | GET | Cache performance statistics |
| `/chat` | POST | Main chat endpoint |

**API Documentation:** Visit http://localhost:8000/docs for interactive API explorer
