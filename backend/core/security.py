"""
Security utilities: API key validation, rate limiting context, sanitization.
"""

import hashlib
import re
from typing import Any


def sanitize_user_input(text: str, max_length: int = 2000) -> str:
    """
    Basic input sanitization:
    - Truncates to max_length
    - Strips null bytes and control characters
    - Collapses excessive whitespace

    Args:
        text: Raw user input string.
        max_length: Maximum allowed character count.

    Returns:
        Sanitized string.
    """
    # Remove null bytes
    text = text.replace("\x00", "")
    # Remove non-printable control characters (keep newlines and tabs)
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse excessive whitespace
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    # Truncate
    return text[:max_length].strip()


def hash_session_id(session_id: str) -> str:
    """
    One-way hash a session ID for safe logging (avoid PII in logs).

    Args:
        session_id: Raw session identifier.

    Returns:
        Short hex digest safe for logging.
    """
    return hashlib.sha256(session_id.encode()).hexdigest()[:12]


def mask_api_key(key: str) -> str:
    """
    Mask API key for display in logs/UI.
    Returns first 8 + last 4 chars separated by ***.

    Args:
        key: Full API key string.

    Returns:
        Masked representation.
    """
    if len(key) < 16:
        return "***"
    return f"{key[:8]}***{key[-4:]}"


def extract_citations(text: str) -> list[dict[str, Any]]:
    """
    Extract all scripture citations from a text string using regex.
    Matches patterns like [John 3:16] or [Romans 8:28-30].

    Args:
        text: LLM-generated response text.

    Returns:
        List of dicts with keys: raw, book, chapter, verse.
    """
    pattern = r"\[([A-Za-z\s]+)\s+(\d+):(\d+(?:-\d+)?)\]"
    matches = re.finditer(pattern, text)
    citations = []
    for match in matches:
        citations.append(
            {
                "raw": match.group(0),
                "book": match.group(1).strip(),
                "chapter": int(match.group(2)),
                "verse": match.group(3),
            }
        )
    return citations


import time
from fastapi import Request, HTTPException, Depends
from backend.core.config import Settings, get_settings

# Simple in-memory token bucket for rate limiting per IP
_RATE_LIMITS: dict[str, list[float]] = {}

def rate_limit(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """
    Enforce a simple sliding-window rate limit per IP.
    """
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    
    if client_ip not in _RATE_LIMITS:
        _RATE_LIMITS[client_ip] = []
        
    # Filter out requests older than 60 seconds
    _RATE_LIMITS[client_ip] = [t for t in _RATE_LIMITS[client_ip] if now - t < 60]
    
    if len(_RATE_LIMITS[client_ip]) >= settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Please wait a minute before sending more requests."
        )
        
    _RATE_LIMITS[client_ip].append(now)

