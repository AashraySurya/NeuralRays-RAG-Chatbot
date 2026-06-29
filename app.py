"""
FastAPI app for the Neural Rays RAG chatbot.

This file exposes the chatbot through an API and serves a simple web interface.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chatbot import answer_question


app = FastAPI(
    title="Neural Rays RAG Chatbot",
    description="A RAG chatbot prototype that answers questions using Neural Rays website content.",
    version="1.0.0",
)

# Allow the frontend to call the API during local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve files from the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    """
    Request body for chatbot questions.
    """

    question: str


class ChatResponse(BaseModel):
    """
    Response body returned by the chatbot API.
    """

    question: str
    answer: str
    sources: list[str]


@app.get("/")
def home() -> FileResponse:
    """
    Serve the chatbot web interface.
    """

    return FileResponse("static/index.html")


@app.get("/health")
def health_check() -> dict[str, str]:
    """
    Simple health check endpoint.
    """

    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Answer a user question using the chatbot retrieval pipeline.
    """

    result = answer_question(request.question)

    return ChatResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
    )