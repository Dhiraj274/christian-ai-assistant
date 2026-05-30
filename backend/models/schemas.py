"""
All Pydantic request/response models for the Christianity-Focused AI Assistant.
Every API boundary is strictly typed — no raw dicts pass through the system.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ── Denomination Enum ──────────────────────────────────────────────────────────

Denomination = Literal["Catholic", "Protestant", "Orthodox", "General"]

# ── Chat Models ────────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """A single turn in the conversation history."""

    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1, max_length=10_000)


class ChatRequest(BaseModel):
    """
    Incoming chat request payload.

    Example:
        {
            "query": "What does Matthew 5:3 say about humility?",
            "denomination": "Catholic",
            "session_id": "uuid-here",
            "history": [...]
        }
    """

    query: str = Field(..., min_length=1, max_length=2000, description="User's question")
    denomination: Denomination = Field(
        default="General",
        description="User's denominational preference for contextualized responses",
    )
    session_id: str = Field(
        default="default",
        max_length=128,
        description="Session identifier for conversation memory",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=20,
        description="Recent conversation history (last N turns)",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class RetrievedChunk(BaseModel):
    """A single retrieved Bible verse or commentary chunk."""

    text: str
    book: str
    chapter: int
    verse: int
    translation: str = "KJV"
    source_type: Literal["scripture", "commentary"] = "scripture"
    score: float = Field(ge=0.0, le=1.0)


class CitationVerificationResult(BaseModel):
    """Result of the deterministic citation verifier node."""

    is_accurate: bool
    hallucinated_citations: list[str] = Field(default_factory=list)
    verified_citations: list[str] = Field(default_factory=list)
    feedback: Optional[str] = None
    retry_count: int = 0


class SafetyCheckResult(BaseModel):
    """Result of input/output safety moderation."""

    is_safe: bool
    reason: Optional[str] = None
    threat_category: Optional[
        Literal["jailbreak", "hate_speech", "scripture_manipulation", "image_policy"]
    ] = None

    @field_validator("threat_category", mode="before")
    @classmethod
    def normalize_threat_category(cls, v: Any) -> Optional[str]:
        """Normalize LLM output to snake_case before Pydantic validates the Literal."""
        if v is None:
            return None
        
        # Map various input formats to target Literal values
        mapping = {
            "jailbreak": "jailbreak",
            "hate_speech": "hate_speech",
            "hate": "hate_speech",
            "scripture_manipulation": "scripture_manipulation",
            "manipulation": "scripture_manipulation",
            "image_policy": "image_policy",
        }
        
        normalized = str(v).strip().lower().replace(" ", "_")
        return mapping.get(normalized)


class IntentClassification(BaseModel):
    """Result of the intent router node."""

    intent: Literal[
        "theological_qa",
        "image_generation",
        "general_conversation",
        "unsafe",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    explicit_denomination: Optional[Denomination] = None


class ChatResponse(BaseModel):
    """Non-streaming chat response (used for testing)."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    denomination_used: Denomination = "General"
    verification_passed: bool = True
    session_id: str = "default"
    intent: str = "theological_qa"
    retrieved_chunks_count: int = 0


class StreamEvent(BaseModel):
    """A single SSE event payload."""

    type: Literal["token", "metadata", "citation", "verification", "error", "done", "process"]
    data: Any = None


# ── Image Models ───────────────────────────────────────────────────────────────


class ImageRequest(BaseModel):
    """Incoming image generation request."""

    prompt: str = Field(..., min_length=5, max_length=1000)
    denomination: Denomination = "General"

    @field_validator("prompt")
    @classmethod
    def strip_prompt(cls, v: str) -> str:
        return v.strip()


class ImageResponse(BaseModel):
    """Response from the image generation pipeline."""

    url: str
    revised_prompt: str
    original_prompt: str
    safety_rewritten: bool = False


# ── Health Check ───────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str = "1.0.0"
    services: dict[str, bool] = Field(default_factory=dict)
