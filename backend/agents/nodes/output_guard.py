"""
Node 6: Output Guardrail
Post-generation toxicity and safety check.
Runs AFTER the generator but BEFORE streaming to the user.
Also injects denominational disclaimers for contested theological topics.
"""

import json
from backend.agents.state import AgentState
from backend.core.config import Settings
from backend.core.logging import get_logger
from backend.models.schemas import SafetyCheckResult

logger = get_logger("agents.output_guard")

# Topics requiring a pluralism disclaimer
CONTESTED_TOPICS = [
    "purgatory", "predestination", "rapture", "end times", "speaking in tongues",
    "once saved always saved", "free will", "baptism", "eucharist", "communion",
    "mary", "saints", "pope", "papal infallibility", "sola scriptura",
]

DISCLAIMER = (
    "\n\n> **Theological Note:** Theological interpretations vary across Christian traditions. "
    "This response reflects the perspective of your selected denomination. "
    "Other traditions may hold different, equally sincere views."
)

OUTPUT_GUARD_PROMPT = """You are a safety reviewer for a Christian theological AI assistant.
Review the AI's response for policy violations. Return JSON only.

VIOLATIONS (classify as unsafe):
1. Extreme toxicity or hate speech targeting any group
2. Content that promotes violence or self-harm
3. Content that is sexually explicit
4. Content that falsely claims to be divine revelation

SAFE CONTENT: Theological explanations, scripture quotations, historical information,
denominational perspectives, pastoral guidance.

Return ONLY: {"safe": true/false, "reason": "brief or null"}"""


async def output_guard_node(state: AgentState, settings: Settings) -> dict:
    """Post-generation safety and quality check."""
    if state.get("should_abort"):
        return {}

    response = state.get("final_response", "")
    if not response:
        return {}

    try:
        from langfuse.openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)

        check_response = await client.chat.completions.create(
            model=settings.fast_llm_model,
            messages=[
                {"role": "system", "content": OUTPUT_GUARD_PROMPT},
                {"role": "user", "content": f"AI response to review:\n\n{response[:2000]}"},
            ],
            response_format={"type": "json_object"},
            max_tokens=100,
            temperature=0.0,
        )
        raw = check_response.choices[0].message.content or '{"safe": true}'
        data = json.loads(raw)

        safety_result = SafetyCheckResult(
            is_safe=bool(data.get("safe", True)),
            reason=data.get("reason"),
        )

        if not safety_result.is_safe:
            logger.warning(f"Output guard FAILED: {safety_result.reason}")
            return {
                "output_safety": safety_result,
                "final_response": (
                    "I'm not able to provide this response as it may contain "
                    "inappropriate content. Please try rephrasing your question."
                ),
            }

        # Inject disclaimer for contested topics
        final_response = response
        query_lower = state["query"].lower()
        if any(topic in query_lower or topic in response.lower() for topic in CONTESTED_TOPICS):
            final_response = response + DISCLAIMER
            logger.info("Appended denominational pluralism disclaimer")

        return {
            "output_safety": safety_result,
            "final_response": final_response,
        }

    except Exception as e:
        logger.error(f"Output guard error: {e}", exc_info=True)
        # Fail open — don't block the user due to guard infra issues
        return {
            "output_safety": SafetyCheckResult(is_safe=True, reason="Guard unavailable"),
        }
