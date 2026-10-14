import logging
import os
from pathlib import Path
from typing import Literal

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError


from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
import uvicorn

from dotenv import load_dotenv

# Langchain imports
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.exceptions import OutputParserException

load_dotenv()  # Load environment variables from .env file


app = FastAPI(title="AI Forward Deployed API (Multi-Provider)")

# --- data models (PYDANTIC) ---
class EchoPayload(BaseModel):
    message: str = Field(min_length=1)

class AIRequest(BaseModel):
    model: Literal["ollama3.2:3b"] = Field(..., description="The AI model to use (currently only 'ollama' is supported).")
    prompt: str = Field(..., min_length=1, description="The prompt to send to the AI model.")

class AIResponse(BaseModel):
    response: str = Field(..., description="The response from the AI model.")
    status: str = Field(..., description="The status of the AI request (e.g., 'success' or 'error').")


PROMPT = """
Intention: Response user query about BEON.tech mission, services, and values.
Details: 
- BEON.tech mission is to place the brightest tech talent in the most
disruptive and innovative U.S. companies.
- BEON.tech offer IT staff augmentation services for every modern tech need,
from backend and frontend to AI, machine learning, DevOps and QA.
- At BEON.tech, building software means far more than just filling roles — it's
about creating strong relationships, empowering careers and helping people
grow.
Examples:
- User: What is BEON.tech's mission?
  Response: BEON.tech's mission is to place the brightest tech talent in the most disruptive and innovative U.S. companies.
- User: What services does BEON.tech offer?
  Response: BEON.tech offers IT staff augmentation services for every modern tech need, from backend and frontend to AI, machine learning, DevOps, and QA.
- User: What does BEON.tech value in building software?
  Response: At BEON.tech, building software means far more than just filling roles — it's about creating strong relationships, empowering careers, and helping people grow.
Action: Provide a clear and concise response to the user's query based on the information provided.
Limitations:
- Do not provide information outside of the provided details about BEON.tech.
Format: Respond in JSON format with a professional tone and informative manner, using complete sentences. 
Validation: Ensure that the response is accurate, relevant, and adheres to the provided details about BEON.tech.
Role: You are an AI assistant knowledgeable about BEON.tech's mission, services, and values.
"""

# Create the route base
router = APIRouter(
    prefix="/api/v1",
    tags=["AI Forward Deployed API"],
    responses={404: {"description": "Not found"}},
)
app.include_router(router)

# --- Echo Endpoint ---
@router.post("/echo", response_model=EchoPayload)
async def echo(payload: EchoPayload) -> EchoPayload:
    return payload


# --- AI Endpoint ---
@router.post("/ai", response_model=AIResponse)
async def ai_endpoint(request: AIRequest) -> AIResponse:
    if request.model == "ollama":
        try:
            # Initialize the Ollama chat model
            chat_model = ChatOllama(model_name="llama3.2:3b")

            # Create a chat prompt template with the provided prompt            
            prompt_template = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="history"),
                ("system", PROMPT),
                ("user", request.prompt)
            ])

            # Generate the response from the AI model
            response = chat_model.predict_messages(prompt_template.format_prompt(history=[]).to_messages())

            return AIResponse(response=response[0].content, status="success")
        except OutputParserException as e:
            raise HTTPException(status_code=400, detail=f"Output parsing error: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported model specified. Currently, only 'ollama' is supported.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
