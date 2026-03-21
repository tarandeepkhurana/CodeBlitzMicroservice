# CodeBlitz Question Variant Generator Microservice

A FastAPI microservice that generates unique coding question variants for the CodeBlitz 1v1 competitive coding platform.

---

## Overview

When two users are matched for a coding battle, this service:

1. Selects a base LeetCode-style question based on player ratings
2. Generates a harder variant using GPT-5.2 (via OpenAI)
3. Validates the variant's solution against test cases using PISTON code execution
4. Auto-fixes failing solutions with retry logic
5. Caches validated variants for reuse
6. Returns the variant with verified test cases

---

## Features

- **Smart Question Selection** - Selects questions based on average player rating and avoids questions either player has seen before
- **LLM-Powered Variant Generation** - Uses GPT-5.2 to create harder variants of base questions
- **Solution Verification** - Validates generated solutions via PISTON code execution engine
- **Auto-Fix Retry Logic** - Automatically fixes failing solutions based on test failures
- **Multi-Language Support** - Python, Java, and C++
- **Rate Limiting** - Per-user and global rate limits for generation endpoints
- **Metrics & Monitoring** - Built-in metrics for success rates, latency, and LLM costs
- **Health Checks** - Kubernetes-ready health/readiness probes

---

## Tech Stack

- **Framework**: FastAPI
- **Language**: Python 3.11+
- **Database**: Supabase (PostgreSQL)
- **LLM**: OpenAI GPT-5.2
- **Code Execution**: PISTON API
- **Package Manager**: uv

---

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager
- **Docker Desktop** - Required for running PISTON
- **Supabase Project** - Database with base questions and variants tables
- **OpenAI API Key** - For variant generation

### Install uv

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Mac/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## PISTON Setup (Code Execution Engine)

PISTON must be running before starting this microservice.

### 1. Clone and Start PISTON

```bash
git clone https://github.com/engineer-man/piston.git
cd piston
docker-compose up -d
```

This starts PISTON API on `http://localhost:2000`.

### 2. Install Language Runtimes

PISTON starts with no languages. Install Python, Java, and C++:

```bash
# Install Python 3.10
curl -X POST http://localhost:2000/api/v2/packages \
  -H "Content-Type: application/json" \
  -d '{"language":"python","version":"3.10.0"}'

# Install Java 15
curl -X POST http://localhost:2000/api/v2/packages \
  -H "Content-Type: application/json" \
  -d '{"language":"java","version":"15.0.2"}'

# Install GCC 10.2 (provides C++)
curl -X POST http://localhost:2000/api/v2/packages \
  -H "Content-Type: application/json" \
  -d '{"language":"gcc","version":"10.2.0"}'
```

Each install takes 30-120 seconds. Wait for 200 response.

### 3. Verify Installation

```bash
curl http://localhost:2000/api/v2/runtimes
```

Should show python, java, c++ in the response.

---

## Setup

### 1. Clone the Repository

```bash
git clone <repo-url>
cd CodeBlitzMicroservice
```

### 2. Create Virtual Environment

```bash
uv venv
```

### 3. Activate Virtual Environment

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Mac/Linux:**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
uv sync
```

### 5. Setup Environment Variables

```bash
cp .env.example .env
```

Then edit `.env` and fill in your actual values:
- `SUPABASE_URL` and `SUPABASE_KEY` - from your Supabase project
- `OPENAI_API_KEY` - your OpenAI API key

### 6. Run the Server

**Development (with hot reload):**

```bash
uv run uvicorn app.main:app --reload --port 8001
```

**Production:**

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Server starts at: http://localhost:8001

**Interactive API Docs:** http://localhost:8001/docs

---

## Project Structure

```
CodeBlitzMicroservice/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point, middleware, lifespan
│   ├── config.py               # Pydantic settings & environment config
│   ├── api/
│   │   └── routes/
│   │       ├── health.py       # Health check endpoints
│   │       ├── question.py     # Question & code execution endpoints
│   │       └── metrics.py      # Monitoring endpoints
│   ├── clients/
│   │   ├── openai_client.py    # OpenAI/Azure AI Foundry client
│   │   ├── piston_client.py    # PISTON code execution client
│   │   └── supabase_client.py  # Supabase database client
│   ├── core/
│   │   ├── error_handling.py   # Global error handling middleware
│   │   ├── metrics.py          # Metrics collection & tracking
│   │   └── rate_limiter.py     # Rate limiting logic
│   ├── schemas/
│   │   └── api.py              # Pydantic request/response models
│   └── services/
│       └── variant_service.py  # Variant generation & verification logic
├── pyproject.toml              # Dependencies & project metadata
├── .env                        # Environment variables (not committed)
└── README.md
```

---

## API Endpoint

### `POST /api/v1/questions/for-match`

Get a question variant for a 1v1 match.

**Request Body:**

```json
{
  "match_id": "uuid-string",
  "user1_id": "uuid-string",
  "user2_id": "uuid-string",
  "user1_rating": 1200,
  "user2_rating": 1350,
  "preferred_categories": [],
  "exclude_categories": []
}
```

**Response:**

```json
{
  "success": true,
  "question": {
    "variant_id": "uuid-string",
    "title": "Two Sum Variant",
    "problem_statement": "Given an array of integers...",
    "input_format": { ... },
    "output_format": { ... },
    "constraints": [ ... ],
    "examples": [ ... ],
    "difficulty": "Medium",
    "function_template": {
      "python": "def solution(nums, target):\n    pass",
      "java": "class Solution { ... }",
      "cpp": "class Solution { ... }"
    },
    "stdin_wrappers": {
      "python": "...",
      "java": "...",
      "cpp": "..."
    }
  },
  "error": null,
  "metadata": {
    "source": "generated",
    "base_question_id": "uuid-string",
    "generation_time_ms": 2340
  }
}
```

**Difficulty Selection Logic:**

- Average rating < 1200 → Easy
- Average rating 1200-1600 → Medium
- Average rating ≥ 1600 → Hard

---

## Rate Limits

| Scope                 | Limit                  |
| --------------------- | ---------------------- |
| Per IP                | 60 requests/minute     |
| Per User (generation) | 5 generations/minute   |
| Global (generation)   | 100 generations/minute |

---

## Supported Languages

| Language | Version | Filename  |
| -------- | ------- | --------- |
| Python   | 3.10.0  | main.py   |
| Java     | 15.0.2  | Main.java |
| C++      | 10.2.0  | main.cpp  |

---

## Important Notes

- **Never commit `.env` to git!** Add it to `.gitignore`
- The service prewarms Java JVM and C++ compiler on startup to avoid cold start timeouts
- Variant generation requires a valid OpenAI API key (or Azure AI Foundry)
- PISTON API must be running and accessible for code execution features

---

## License

MIT
