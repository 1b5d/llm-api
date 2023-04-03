"""
Main entry point for LLM-api
"""
from logging.config import dictConfig
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # pylint: disable=no-name-in-module
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.llms import get_model_class

log_config = uvicorn.config.LOGGING_CONFIG
log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
log_config["formatters"]["default"][
    "fmt"
] = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
log_config["loggers"]["llm-api"] = {
    "handlers": ["default"],
    "level": uvicorn.config.LOG_LEVELS[settings.log_level],
}

dictConfig(log_config)

app = FastAPI(title="llm-api", version="0.0.1")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):  # pylint: disable=too-few-public-methods
    """
    A general generate request representation
    """

    prompt: str
    params: Optional[Dict[str, Any]]


class EmbeddingsRequest(BaseModel):  # pylint: disable=too-few-public-methods
    """
    A general embeddings request representation
    """

    text: str


ModelClass = get_model_class(settings.model_family)

llm = ModelClass(params=settings.model_params)


@app.post("/generate")
def generate(request: GenerateRequest):
    """
    Generate text based on a text prompt
    """
    return llm.generate(prompt=request.prompt, params=request.params)


@app.post("/agenerate")
def agenerate(request: GenerateRequest):
    """
    Generate a stream of text based on a text prompt
    """
    return EventSourceResponse(
        token
        async for token in llm.agenerate(prompt=request.prompt, params=request.params)
    )


@app.post("/embeddings")
def embeddings(request: EmbeddingsRequest):
    """
    Generate embeddings for a text input
    """
    return llm.embeddings(request.text)


@app.get("/check")
def check():
    """
    Status check
    """
    return "Ok"


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_config=log_config,
        log_level=settings.log_level,
    )
