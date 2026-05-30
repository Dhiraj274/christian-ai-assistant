"""
Image generation API route — POST /api/image
Includes prompt rewriting for safety and reverence before DALL-E 3.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from langfuse.openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import Settings, get_settings
from backend.core.logging import get_logger
from backend.core.security import rate_limit
from backend.models.schemas import ImageRequest, ImageResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/api/image", tags=["image"])

# ── Banned content patterns ───────────────────────────────────────────────────

BANNED_IMAGE_PATTERNS = [
    "upside down cross",
    "inverted cross",
    "burning cross",
    "satanic",
    "demonic",
    "666",
    "antichrist",
    "blasphemy",
    "naked",
    "sexual",
]

# Denomination-specific iconography restrictions
DEITY_DEPICTION_RESTRICTIONS = {
    "Orthodox": "Avoid direct anthropomorphic depictions of God the Father; use light/symbol.",
    "Protestant": "Keep imagery non-idolatrous; avoid halos unless clearly artistic.",
    "Catholic": "Traditional iconographic style is acceptable.",
    "General": "Use respectful, non-denominational imagery.",
}


@router.post(
    "",
    summary="Generate a reverent Christian-themed image",
    response_model=ImageResponse,
    dependencies=[Depends(rate_limit)],
)
async def image_endpoint(
    request: Request,
    payload: ImageRequest,
    settings: Settings = Depends(get_settings),
) -> ImageResponse:
    """
    Image generation pipeline:
      1. Safety pre-screen (block banned patterns).
      2. Prompt rewriter (add reverence guidelines).
      3. DALL-E 3 API call.
      4. Return image URL.
    """
    # Step 1: Pre-screen for banned content
    prompt_lower = payload.prompt.lower()
    for pattern in BANNED_IMAGE_PATTERNS:
        if pattern in prompt_lower:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"This image request contains content ('{pattern}') that violates "
                    "our respectful imagery guidelines. Please rephrase your request."
                ),
            )

    # Step 2: Rewrite prompt for safety and reverence
    safety_rewritten = False
    revised_prompt = await _rewrite_image_prompt(
        payload.prompt, payload.denomination, settings
    )
    if revised_prompt != payload.prompt:
        safety_rewritten = True
        logger.info(f"Image prompt rewritten for denomination={payload.denomination}")

    # Step 3: Generate image
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    image_url = await _generate_image(client, revised_prompt)

    return ImageResponse(
        url=image_url,
        revised_prompt=revised_prompt,
        original_prompt=payload.prompt,
        safety_rewritten=safety_rewritten,
    )


async def _rewrite_image_prompt(
    original_prompt: str,
    denomination: str,
    settings: Settings,
) -> str:
    """
    Use GPT-4o-mini to rewrite the image prompt for safety and reverence.
    """
    from langfuse.openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    denomination_note = DEITY_DEPICTION_RESTRICTIONS.get(denomination, DEITY_DEPICTION_RESTRICTIONS["General"])

    system_prompt = f"""You are an image prompt safety editor for a Christian AI assistant.
Your task is to rewrite image prompts to ensure they are:
1. Respectful and reverent of Christian traditions
2. Free from offensive, blasphemous, or politically charged imagery
3. Appropriate for all ages
4. Aligned with this denomination's guidelines: {denomination_note}

If the prompt is already safe and reverent, return it unchanged.
Return ONLY the rewritten prompt, no explanation."""

    response = await client.chat.completions.create(
        model=settings.fast_llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Original prompt: {original_prompt}"},
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return response.choices[0].message.content or original_prompt


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _generate_image(client: AsyncOpenAI, prompt: str) -> str:
    """Generate image via GPT Image API with retry logic."""
    response = await client.images.generate(
        model="gpt-image-1-mini",
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    import logging
    logging.getLogger("agents.image").info(f"API Response received.")
    
    # Modern GPT Image API models return base64 by default
    b64_data = response.data[0].b64_json
    if b64_data:
        return f"data:image/png;base64,{b64_data}"
        
    url = response.data[0].url
    if not url and not b64_data:
        raise ValueError(f"GPT Image API returned no image data. Response: {response}")
    return url
