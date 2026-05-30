"""
Chat API route — POST /api/chat with Server-Sent Events streaming.
"""

import asyncio
import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.agents.graph import build_graph, run_graph_streaming
from backend.core.config import Settings, get_settings
from backend.core.logging import get_logger
from backend.core.security import hash_session_id, sanitize_user_input, rate_limit
from backend.models.schemas import ChatRequest, ChatResponse, StreamEvent

logger = get_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post(
    "",
    summary="Theological Q&A with SSE streaming",
    response_description="Server-Sent Events stream of tokens",
    dependencies=[Depends(rate_limit)],
)
async def chat_endpoint(
    request: Request,
    payload: ChatRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """
    Main chat endpoint. Streams response tokens via Server-Sent Events.

    The LangGraph multi-agent pipeline runs behind this endpoint:
      1. Input Guardrail (jailbreak detection)
      2. Intent Router
      3. Retriever (Pinecone hybrid search)
      4. Generator (Claude 3.5 Sonnet)
      5. Citation Verifier (deterministic SQLite check)
      6. Output Guardrail (toxicity check)
    """
    # Sanitize input
    clean_query = sanitize_user_input(payload.query)
    if not clean_query:
        raise HTTPException(status_code=422, detail="Query cannot be empty after sanitization.")

    safe_session = hash_session_id(payload.session_id)
    logger.info(
        "Chat request received",
        extra={
            "extra": {
                "session_id_hash": safe_session,
                "denomination": payload.denomination,
                "query_length": len(clean_query),
            }
        },
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        """Generate SSE events from the LangGraph pipeline."""
        try:
            async for event in run_graph_streaming(
                query=clean_query,
                denomination=payload.denomination,
                session_id=payload.session_id,
                history=payload.history,
                settings=settings,
            ):
                yield f"data: {json.dumps(event.model_dump())}\n\n"
                await asyncio.sleep(0)  # Yield control to event loop

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            error_event = StreamEvent(type="error", data={"message": _map_error(e)})
            yield f"data: {json.dumps(error_event.model_dump())}\n\n"
        finally:
            done_event = StreamEvent(type="done")
            yield f"data: {json.dumps(done_event.model_dump())}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )


@router.post(
    "/sync",
    summary="Synchronous chat (for testing only)",
    response_model=ChatResponse,
    include_in_schema=False,  # Hidden from Swagger docs in prod
)
async def chat_sync_endpoint(
    payload: ChatRequest,
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    """Non-streaming endpoint for automated testing and evals."""
    from backend.agents.graph import run_graph_sync

    clean_query = sanitize_user_input(payload.query)
    result = await run_graph_sync(
        query=clean_query,
        denomination=payload.denomination,
        session_id=payload.session_id,
        history=payload.history,
        settings=settings,
    )
    return result


def _map_error(error: Exception) -> str:
    """Map technical errors to user-friendly messages."""
    error_str = str(error).lower()
    if "rate_limit" in error_str or "ratelimit" in error_str:
        return "The system is experiencing high prayer volume. Please wait a moment."
    if "safety" in error_str or "content_policy" in error_str:
        return "I'm unable to fulfill this request as it falls outside my operational guidelines."
    if "timeout" in error_str:
        return "The response took too long. Please try a simpler question."
    return "An unexpected error occurred. Please try again."
