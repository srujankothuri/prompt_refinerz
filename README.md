# Prompt Refiner

A production-style FastAPI application that improves user prompts by generating multiple high-quality alternatives with different LLM providers, scoring each variation, and selecting the strongest candidate.

Prompt Refiner is designed for teams building AI features who need **consistent, structured prompt quality** before sending prompts into downstream workflows.

---

## Overview

Given an input prompt, the system:

1. Generates multiple prompt variations using one or more providers (Anthropic, Together, Gemini).
2. Evaluates each variation against a rubric (clarity, specificity, context, structure, constraints, examples, and overall improvement).
3. Returns scored results plus the best-performing prompt.

The app includes:
- A web UI for interactive usage.
- A REST API for programmatic integration.
- A modular service layer for provider calls and scoring.

---

## Tech Stack

- **Backend:** FastAPI, Pydantic, Uvicorn
- **LLM Providers:** Anthropic Claude, Together AI, Google Gemini
- **Scoring:** Anthropic-based structured evaluation
- **Frontend:** HTML templates + vanilla JavaScript + CSS (with charts for score visualization)
- **Config:** `python-dotenv` + environment variables

---

## Project Structure

```text
.
├── main.py                  # FastAPI app, routes, app bootstrap
├── models.py                # Request/response and domain models
├── services/
│   ├── llm_service.py       # Multi-provider prompt generation
│   └── scorer.py            # Prompt scoring and best-prompt selection
├── templates/
│   └── index.html           # Main UI template
├── static/
│   ├── css/styles.css       # UI styling
│   └── js/main.js           # Client-side behavior, charts, rendering
├── requirements.txt
└── package.json             # Deployment/runtime metadata
```

---

## Features

- Multi-provider prompt generation in parallel.
- Configurable number of variations (bounded to safe limits via schema validation).
- Context-aware refinement for different output intents:
  - General
  - Summarization
  - Code Generation
  - Data Analysis
  - Technical Explanation
  - Creative Writing
- Structured scoring with strengths/weaknesses per variation.
- Automatic best-prompt selection.
- Health check endpoint for ops monitoring.

---

## Quick Start

### 1) Clone and enter project

```bash
git clone <your-repo-url>
cd prompt_refinerz
```

### 2) Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure environment variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_key
TOGETHER_API_KEY=your_together_key
GEMINI_API_KEY=your_gemini_key
```

> Notes:
> - Generation uses selected providers. Missing keys will cause provider-specific generation failures.
> - Scoring currently relies on Anthropic.

### 5) Run the app

```bash
python main.py
```

App will be available at:
- `http://localhost:8000`

---

## API

### Health Check

```http
GET /api/health
```

Sample response:

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Refine Prompt

```http
POST /api/refine
Content-Type: application/json
```

Request body:

```json
{
  "original_prompt": "Write a blog post about edge AI.",
  "desired_output_type": "General",
  "context": "Audience is senior ML engineers",
  "providers": ["anthropic", "together", "gemini"],
  "num_variations": 5
}
```

Response includes:
- `variations`: scored variations with strengths and weaknesses.
- `best_prompt`: highest scoring variation.
- `processing_time`: total request processing time in seconds.

---

## Architecture Notes

- `LLMService` handles provider orchestration and parallel generation.
- `ScoringService` evaluates all candidates concurrently and annotates each with rubric-based feedback.
- Best prompt is selected via descending `quality_score` sort.

This architecture keeps generation and scoring concerns isolated, making it easier to:
- Add new providers.
- Swap scoring backends.
- Evolve scoring rubric independently.

---

## Validation and Constraints

Schema-level validation (Pydantic):
- `num_variations`: min 3, max 10.
- Providers are enum-constrained (`anthropic`, `together`, `gemini`).

Fallback behavior:
- If generation/scoring fails, the system returns safe defaults instead of crashing, allowing the UI/API contract to stay stable.

---


## Vercel Deployment

This project deploys on Vercel using the Python runtime with `main.py` as the single server entrypoint.

### Required settings

- Set the following environment variables in Vercel Project Settings:
  - `ANTHROPIC_API_KEY`
  - `TOGETHER_API_KEY`
  - `GEMINI_API_KEY`
- Ensure build output uses the repository root where `main.py`, `templates/`, and `static/` are present.

### Why this works

All routes are now forwarded to `main.py`, so FastAPI handles:
- `GET /` (Jinja template rendering)
- `/static/*` (mounted static files)
- `/api/*` endpoints

This avoids broken deployments caused by routing `/` to a non-existent `static/index.html`.

---

## Deployment

A `package.json` with deployment routing is included for platform workflows.

For production hardening, consider:
- Adding request rate limiting.
- Adding API authentication.
- Moving API keys to managed secret stores.
- Instrumenting with structured logs + tracing.
- Pinning dependency versions.

---

## Development Backlog (Suggested)

- Add automated tests for service-layer behavior and API contracts.
- Introduce provider-level timeouts and retries with circuit breaking.
- Add per-provider score analytics over time.
- Implement persistence for prompt history and team collaboration.
- Support custom scoring rubrics from UI.

---

## License

Add your preferred license (MIT/Apache-2.0/etc.) before public distribution.
