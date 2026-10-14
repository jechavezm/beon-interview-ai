from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
import uvicorn
import os

app = FastAPI(title="AI Forward Deployed API (Multi-Provider)")

# --- data models (PYDANTIC) ---
class EchoPayload(BaseModel):
    message: str = Field(min_length=1)

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
