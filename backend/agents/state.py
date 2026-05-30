"""
LangGraph Agent State definition.
All nodes read from and write to this shared state object.
TypedDict with Annotated fields for proper LangGraph message reduction.
"""

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from backend.models.schemas import (
    CitationVerificationResult,
    Denomination,
    IntentClassification,
    RetrievedChunk,
    SafetyCheckResult,
)


class AgentState(TypedDict):
    """
    Shared state machine for the multi-agent LangGraph pipeline.

    Flow:
        input_guard → router → retriever → generator → verifier → output_guard → END

    With a cycle: verifier can route back to generator if hallucination detected.
    """

    # ── Input ───────────────────────────────────────────────────────────────
    query: str
    """Original user query (sanitized)."""

    denomination: Denomination
    """User's denominational preference."""

    session_id: str
    """Session identifier for memory management."""

    messages: Annotated[list[Any], add_messages]
    """Full conversation history (LangGraph message reducer)."""

    # ── Safety ─────────────────────────────────────────────────────────────
    input_safety: Optional[SafetyCheckResult]
    """Result of pre-LLM input moderation."""

    output_safety: Optional[SafetyCheckResult]
    """Result of post-LLM output moderation."""

    safety_failure_count: int
    """Number of consecutive safety failures (triggers hard fallback at 3)."""

    # ── Routing ────────────────────────────────────────────────────────────
    intent: Optional[IntentClassification]
    """Classified intent: theological_qa, image_generation, etc."""

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk]
    """Verse chunks retrieved from Pinecone."""

    context_string: str
    """Formatted context string injected into the generator prompt."""

    # ── Generation ─────────────────────────────────────────────────────────
    draft_response: str
    """LLM-generated draft (before verification)."""

    final_response: str
    """Verified, safe response ready for streaming to the user."""

    # ── Verification ───────────────────────────────────────────────────────
    verification: Optional[CitationVerificationResult]
    """Result of deterministic citation verification."""

    verifier_retry_count: int
    """Number of generator→verifier cycles completed."""

    # ── Error Handling ─────────────────────────────────────────────────────
    error: Optional[str]
    """Error message if a node failed."""

    should_abort: bool
    """Set to True to skip remaining nodes and return a safe fallback."""
