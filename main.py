import asyncio
import logging
import math
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import AsyncInferenceClient
from pydantic import BaseModel

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful AI coding assistant for the andriod development tasks. "
    "Don't give any messages just give code snippets."
)
MODEL = "Qwen/Qwen3-Coder-Next:fastest"


def _setting(name: str, default: str, *, integer: bool = False, minimum: float = 0):
    """Read and validate concurrency settings at startup."""
    raw = os.environ.get(name, default)
    try:
        value = int(raw) if integer else float(raw)
        if not math.isfinite(value) or value <= minimum:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: must be {'an integer' if integer else 'a number'} greater than {minimum}") from exc
    return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    # These limits are per worker, not global across multiple processes.
    max_concurrent = _setting("MAX_CONCURRENT_REQUESTS", "10", integer=True)
    max_queued = _setting("MAX_QUEUED_REQUESTS", "50", integer=True, minimum=-1)
    app.state.queue_timeout = _setting("QUEUE_TIMEOUT_SECONDS", "5")
    app.state.inference_timeout = _setting("INFERENCE_TIMEOUT_SECONDS", "120")
    app.state.capacity = max_concurrent + max_queued
    app.state.admitted = 0
    app.state.slots = asyncio.Semaphore(max_concurrent)
    app.state.client = AsyncInferenceClient(api_key=os.environ.get("HF_TOKEN"))
    try:
        yield
    finally:
        await app.state.client.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    state = request.app.state
    # There is no await between checking and incrementing: both operations run
    # together on the worker's event loop. This bounds active + waiting calls.
    if state.admitted >= state.capacity:
        raise HTTPException(status_code=503, detail="Server is busy. Please retry.", headers={"Retry-After": "1"})
    state.admitted += 1
    acquired = False
    try:
        try:
            await asyncio.wait_for(state.slots.acquire(), timeout=state.queue_timeout)
            acquired = True
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="Request queue timed out. Please retry.", headers={"Retry-After": "1"})

        try:
            completion = await asyncio.wait_for(
                state.client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": req.message},
                    ],
                ),
                timeout=state.inference_timeout,
            )
            return ChatResponse(response=completion.choices[0].message.content)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Inference request timed out.")
        except Exception:
            logger.exception("Inference request failed")
            raise HTTPException(status_code=502, detail="Inference service request failed.")
    finally:
        if acquired:
            state.slots.release()
        state.admitted -= 1
