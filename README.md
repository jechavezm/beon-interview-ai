# BEON Interview AI

A Python/FastAPI app with a LangChain question endpoint that connects to a local
Ollama server and returns JSON validated by Pydantic. Claude and OpenAI are also
available through the existing provider contract.

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

## Ask a question with Ollama

Set these values in `.env` in the repository root (existing shell environment
variables take precedence):

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

`OLLAMA_MODEL` is required. `OLLAMA_BASE_URL` defaults to
`http://localhost:11434`. Select a model installed on your Ollama server. If
needed, download it and start the server in a separate terminal:

```bash
ollama pull llama3.2:3b
ollama serve
```

If the Ollama desktop app already runs the server, skip `ollama serve`.

Start FastAPI using the commands above, then send any question to
`POST /api/v1/generate`:

```json
{
  "question": "What is LCEL in LangChain?"
}
```

```bash
curl -X POST http://127.0.0.1:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LCEL in LangChain?"}'
```

`provider` defaults to `ollama`. The legacy field `prompt` is also accepted in
place of `question`; send exactly one of these fields. Missing, empty, or
whitespace-only questions return HTTP 422.

The response contains a structured object, not an encoded JSON string:

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
are required; lists may be empty. Ollama receives the output's JSON schema through
LangChain's `ChatOllama(format=...)`, and the complete JSON response is validated
before returning it. Unexpected output fields and malformed or truncated JSON
are rejected.

| HTTP status | Meaning |
| --- | --- |
| 400 | Missing or invalid provider configuration |
| 422 | Invalid request body |
| 502 | Provider error or invalid structured model output |
| 503 | AI service is unreachable |
| 504 | AI request timed out (120-second limit) |

Model calls use `ainvoke` so they do not block FastAPI's event loop. `/echo` and
the API docs work without model configuration or a running Ollama server.

## Other providers

Set `provider` to `claude` or `openai` in the same request. Claude requires both
`ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`; OpenAI requires both `OPENAI_API_KEY`
and `OPENAI_MODEL`. These providers use the same JSON instructions and output
validation. Only the selected provider is initialized.

## Tests

Run the offline tests with:

```bash
uv run python -m unittest discover -s tests -v
```
