import json
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import main


VALID_OUTPUT = {
    "answer": "Compose the prompt, model, and parser using LCEL.",
    "steps": ["Create a prompt template.", "Invoke the composed chain asynchronously."],
    "code_examples": [
        {
            "language": "python",
            "code": "result = await chain.ainvoke({'query': 'Explain LCEL'})",
            "explanation": "ainvoke avoids blocking an asynchronous FastAPI endpoint.",
        }
    ],
    "caveats": ["Use integrations compatible with your LangChain version."],
}


def fake_llm(content, captured_prompts=None):
    """Return a runnable chat response without creating a provider client."""
    async def respond(prompt_value):
        if captured_prompts is not None:
            captured_prompts.append(prompt_value)
        return AIMessage(content=content)

    return RunnableLambda(respond)


class GenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_structured_response_for_each_provider(self):
        for provider in ("ollama", "claude", "openai"):
            with self.subTest(provider=provider):
                with patch(
                    "main.get_llm", return_value=fake_llm(json.dumps(VALID_OUTPUT))
                ) as get_llm:
                    result = await main.generate_ai_response(
                        main.AIRequest(prompt="How do I compose an LCEL chain?", provider=provider)
                    )

                get_llm.assert_called_once_with(provider)
                self.assertIsInstance(result.response, main.AIOutputPrompt)
                self.assertEqual(result.response.model_dump(), VALID_OUTPUT)
                self.assertEqual(result.provider_used, provider)
                self.assertEqual(result.status, "success")

    async def test_prompt_keeps_question_as_a_human_message(self):
        captured_prompts = []
        question = 'Why does chain.ainvoke({"query": "hello"}) fail?'
        with patch(
            "main.get_llm",
            return_value=fake_llm(json.dumps(VALID_OUTPUT), captured_prompts),
        ):
            await main.generate_ai_response(main.AIRequest(prompt=question))

        self.assertEqual(len(captured_prompts), 1)
        messages = captured_prompts[0].to_messages()
        self.assertEqual([message.type for message in messages], ["system", "human"])
        self.assertEqual(messages[1].content, question)
        self.assertIn("LangChain", messages[0].content)
        self.assertIn(main.output_parser.get_format_instructions(), messages[0].content)

    async def test_malformed_json_returns_502(self):
        with patch("main.get_llm", return_value=fake_llm("not JSON")):
            with self.assertRaises(HTTPException) as error:
                await main.generate_ai_response(main.AIRequest(prompt="Explain LCEL"))

        self.assertEqual(error.exception.status_code, 502)

    async def test_schema_invalid_output_returns_502(self):
        missing_answer = {key: value for key, value in VALID_OUTPUT.items() if key != "answer"}
        invalid_outputs = {
            "missing required field": missing_answer,
            "wrong field type": {**VALID_OUTPUT, "steps": "Create a prompt template."},
            "unexpected field": {**VALID_OUTPUT, "unsupported_field": "extra"},
            "empty answer": {**VALID_OUTPUT, "answer": ""},
            "missing code field": {
                **VALID_OUTPUT,
                "code_examples": [{"language": "python", "explanation": "Example"}],
            },
            "unexpected code field": {
                **VALID_OUTPUT,
                "code_examples": [{**VALID_OUTPUT["code_examples"][0], "unsupported_field": "extra"}],
            },
        }
        for case, payload in invalid_outputs.items():
            with self.subTest(case=case):
                with patch("main.get_llm", return_value=fake_llm(json.dumps(payload))):
                    with self.assertRaises(HTTPException) as error:
                        await main.generate_ai_response(main.AIRequest(prompt="Explain LCEL"))

                self.assertEqual(error.exception.status_code, 502)

    async def test_configuration_error_still_returns_400(self):
        message = "Missing required environment variable: OLLAMA_MODEL"
        with patch("main.get_llm", side_effect=ValueError(message)):
            with self.assertRaises(HTTPException) as error:
                await main.generate_ai_response(main.AIRequest(prompt="Explain LCEL"))

        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(error.exception.detail, message)


class APITests(unittest.TestCase):
    def test_echo_succeeds_without_model_or_configuration_access(self):
        payload = {"message": "Interview setup ready."}
        with patch("main.get_llm") as get_llm, patch("main._required_env") as required_env:
            with TestClient(main.app) as client:
                response = client.post("/api/v1/echo", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)
        get_llm.assert_not_called()
        required_env.assert_not_called()

    def test_echo_validates_missing_and_empty_message(self):
        with TestClient(main.app) as client:
            for payload in ({}, {"message": ""}):
                with self.subTest(payload=payload):
                    response = client.post("/api/v1/echo", json=payload)

                    self.assertEqual(response.status_code, 422)
                    self.assertEqual(response.json()["detail"][0]["loc"], ["body", "message"])

    def test_api_serializes_nested_structured_output(self):
        with patch("main.get_llm", return_value=fake_llm(json.dumps(VALID_OUTPUT))):
            with TestClient(main.app) as client:
                response = client.post(
                    "/api/v1/generate",
                    json={"prompt": "Explain LCEL", "provider": "openai"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"response": VALID_OUTPUT, "provider_used": "openai", "status": "success"},
        )

    def test_openapi_describes_nested_response(self):
        with TestClient(main.app) as client:
            response = client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        schemas = response.json()["components"]["schemas"]
        self.assertEqual(
            schemas["AIResponse"]["properties"]["response"]["$ref"],
            "#/components/schemas/AIOutputPrompt",
        )
        self.assertEqual(
            schemas["AIOutputPrompt"]["properties"]["code_examples"]["items"]["$ref"],
            "#/components/schemas/CodeExample",
        )
        self.assertFalse(schemas["AIOutputPrompt"]["additionalProperties"])
        self.assertFalse(schemas["CodeExample"]["additionalProperties"])

    def test_invalid_provider_returns_422_before_generation(self):
        with patch("main.get_llm") as get_llm:
            with TestClient(main.app) as client:
                response = client.post(
                    "/api/v1/generate",
                    json={"prompt": "Explain LCEL", "provider": "unsupported"},
                )

        self.assertEqual(response.status_code, 422)
        get_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
