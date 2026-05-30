"""
Node 4: Generator Node
Uses OpenAI gpt-4o-mini to generate theologically grounded responses.
gpt-4o-mini chosen over gpt-5-nano (not a real model) for:
  - Proven theological nuance and instruction-following
  - $0.15/1M input tokens (extremely cost-effective)
  - 128k context window — fits full Bible passage retrieval
  - Same API as router/safety nodes (single SDK, zero extra dependencies)

Injects retrieved context and denomination preference into the master system prompt.
"""

import re
from langfuse.openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.agents.state import AgentState
from backend.core.config import Settings
from backend.core.logging import get_logger

logger = get_logger("agents.generator")

MASTER_SYSTEM_PROMPT = """You are an expert, respectful, and highly precise Christian theological assistant.
Your goal is to provide historically accurate, contextually grounded answers to questions regarding Christianity.

CURRENT USER DENOMINATION PREFERENCE: {denomination}

CORE DIRECTIVES:
1. SCRIPTURE GROUNDING: You MUST ONLY cite scripture provided in your <retrieved_context> below.
   Do NOT invent, hallucinate, or recall verses from memory. If a verse is not in the context, say so.
2. TONE: Maintain a pastoral, objective, and respectful tone. Do not proselytize or pressure.
3. DENOMINATION AWARENESS: When a topic has varying interpretations (e.g., Eucharist, Baptism,
   Purgatory, Predestination), explicitly acknowledge the perspectives of Protestant, Catholic, and
   Orthodox traditions, while heavily weighting the user's preferred denomination.
4. CITATION FORMAT (CRITICAL): You MUST wrap EVERY scripture reference in square brackets. 
   Do NOT write "Galatians 5:22". You MUST write "[Galatians 5:22]". 
   If you fail to use brackets, the system will break. Examples: [John 3:16], [1 Corinthians 13:4].
5. HONESTY: If a question is beyond the scope of what the retrieved context supports, say so clearly.
   Do not confabulate theological positions.

6. NO META-SUMMARIES: Do not use horizontal rules (---) or write concluding summaries about your own response (e.g., 'This response provides an overview...'). Just answer the question directly.

{verifier_feedback}

<retrieved_context>
{context}
</retrieved_context>

Think step-by-step in a <thought_process> block before answering to ensure you accurately synthesize
the context and respect denominational nuances. Keep the thought process brief (2-3 sentences).
Then provide your answer outside the <thought_process> block."""


async def generator_node(state: AgentState, settings: Settings) -> dict:
    """
    Primary response generation node using OpenAI gpt-4o-mini.
    Can be called multiple times (on verifier retry cycles).
    """
    if state.get("should_abort"):
        return {}

    query = state["query"]
    denomination = state["denomination"]
    context = state.get("context_string", "No context available.")
    messages = state.get("messages", [])
    retry_count = state.get("verifier_retry_count", 0)

    # Inject verifier feedback on retry cycles
    verifier_feedback = ""
    verification = state.get("verification")
    if verification and not verification.is_accurate and retry_count > 0:
        hallucinated = ", ".join(verification.hallucinated_citations)
        verifier_feedback = (
            f"\n⚠️ CORRECTION REQUIRED: Your previous response contained hallucinated "
            f"citations: {hallucinated}. These verses do not exist or were misquoted. "
            f"You MUST only use verses from the <retrieved_context> above. Rewrite your answer."
        )

    system_prompt = MASTER_SYSTEM_PROMPT.format(
        denomination=denomination,
        context=context,
        verifier_feedback=verifier_feedback,
    )

    # Build OpenAI-format messages (system prompt as first message)
    openai_messages = _build_messages(system_prompt, messages, query, retry_count)

    try:
        draft = await _generate(openai_messages, settings)
        
        # Post-process: Auto-wrap any unbracketed verses the LLM missed
        # e.g., turns "Galatians 5:22-23" into "[Galatians 5:22-23]"
        pattern = r"(?<!\[)\b((?:[123]\s)?[A-Z][a-z]+(?:\s(?:of\s)?[A-Z][a-z]+)?)\s+(\d+):(\d+(?:-\d+)?)\b(?!\])"
        draft = re.sub(pattern, r"[\1 \2:\3]", draft)

        logger.info(f"Generator produced {len(draft)} char response (retry={retry_count})")
        return {"draft_response": draft}

    except Exception as e:
        logger.error(f"Generator error: {e}", exc_info=True)
        return {
            "draft_response": "",
            "error": str(e),
            "should_abort": True,
            "final_response": "I encountered an error generating your response. Please try again.",
        }


def _build_messages(
    system_prompt: str,
    history: list,
    current_query: str,
    retry_count: int,
) -> list[dict]:
    """
    Build OpenAI chat message format.
    Handles two message formats that may appear in conversation history:
      1. Plain dicts: {"role": "user", "content": "..."}
      2. LangChain objects: HumanMessage / AIMessage (Pydantic — no .get(), no .role)
         These use .type ("human" | "ai") and .content
    """
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    for msg in history[-10:]:
        if isinstance(msg, dict):
            # Standard dict format
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            # LangChain HumanMessage / AIMessage
            # .type returns "human" or "ai" — map to OpenAI roles
            msg_type = getattr(msg, "type", "human")
            role = "user" if msg_type == "human" else "assistant"
            content = getattr(msg, "content", "")

        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    # Add current query
    if retry_count > 0:
        messages.append({
            "role": "user",
            "content": f"{current_query}\n\n[Please correct the citation errors noted above.]",
        })
    else:
        messages.append({"role": "user", "content": current_query})

    return messages


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _generate(messages: list[dict], settings: Settings) -> str:
    """Call OpenAI chat completions API with retry logic."""
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    response = await client.chat.completions.create(
        model=settings.primary_llm_model,  # gpt-4o-mini
        messages=messages,
        max_tokens=2048,
        temperature=0.3,  # Low temperature for factual/theological accuracy
    )

    content = response.choices[0].message.content
    return content or ""
