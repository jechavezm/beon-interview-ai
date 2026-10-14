# BEON Interview AI

A Python/FastAPI starter for a one-hour FDE AI interview exercise. A simple POST
endpoint works locally without an AI service. An optional LangChain endpoint
uses Ollama, Claude, or OpenAI and validates model output with Pydantic.

## Run locally

Use Python 3.13 or newer and uv:

```bash
uv sync --locked
uv run uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs to try the endpoints interactively.
Run these commands from the repository root.

## Basic POST endpoint

`POST /api/v1/echo` accepts a nonempty message and returns the same JSON:

```json
{"message": "Ready for the interview"}
```

Try it from a terminal with:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/echo \
  -H "Content-Type: application/json" \
  -d '{"message": "Ready for the interview"}'
```

This endpoint requires no credentials, model downloads, or external services.
It is a starting point to adapt after the interviewer explains the exercise.

## Optional AI endpoint

Copy `.env.example` to `.env` and configure only the provider you intend to use.
For Ollama, select a model already installed locally and start the Ollama server.
Claude requires `ANTHROPIC_API_KEY`; `ANTHROPIC_MODEL` defaults to
`claude-haiku-5-5`. OpenAI requires `OPENAI_API_KEY`.

The AI prompt covers LangChain concepts, implementation, debugging, RAG, tools,
agents, and version compatibility.

Send a request to `POST /api/v1/generate`:

```json
{
  "prompt": "What is LCEL in LangChain?",
  "provider": "ollama"
}
```

`provider` defaults to `ollama`; `claude` and `openai` are also supported.
The response now contains a structured object:

```json
{
  "response": {
    "answer": "LCEL composes LangChain Runnables into reusable pipelines.",
    "steps": [],
    "code_examples": [],
    "caveats": []
  },
  "provider_used": "ollama",
  "status": "success"
}
```

Each code example contains `language`, `code`, and `explanation`. All output fields
are required; lists may be empty. Unexpected output fields are rejected. Invalid
model output returns HTTP 502.

Clients that previously consumed `response` as a string should use
`response.answer` and the additional structured fields.

## Tests

Run the offline tests with:

```bash
uv run python -m unittest discover -s tests -v
```
