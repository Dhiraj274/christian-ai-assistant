"""
Node 2: Intent Router
Classifies the user's query into one of several intents.
Routes the pipeline to the correct downstream nodes.
"""

import json
from backend.agents.state import AgentState
from backend.core.config import Settings
from backend.core.logging import get_logger
from backend.models.schemas import IntentClassification

logger = get_logger("agents.router")

ROUTER_PROMPT = """You are an intent classifier for a Christian theological AI assistant.
Classify the user's query into exactly one intent category, and detect if the user explicitly asks for a specific denominational perspective.

CATEGORIES:
- theological_qa: Questions about Bible, theology, church history, prayer, Christian living, denominations
- image_generation: Requests to generate, draw, create, or show an image
- general_conversation: Greetings, thank-yous, general chat not requiring theological knowledge
- unsafe: Should not reach this node (handled by input guard)

DENOMINATION OVERRIDE:
If the user explicitly asks for the perspective of a specific denomination (e.g. "Catholic view", "Protestant view", "Orthodox view"), extract it. Otherwise return null. Allowed values: "Catholic", "Protestant", "Orthodox".

Return ONLY valid JSON: {"intent": "category", "confidence": 0.0-1.0, "reasoning": "brief", "explicit_denomination": "Catholic" | "Protestant" | "Orthodox" | null}"""


async def router_node(state: AgentState, settings: Settings) -> dict:
    """Intent classification routing node."""
    if state.get("should_abort"):
        return {}

    from langfuse.openai import AsyncOpenAI

    query = state["query"]

    try:
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model=settings.fast_llm_model,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": f"Query: {query}"},
            ],
            response_format={"type": "json_object"},
            max_tokens=100,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        classification = IntentClassification(
            intent=data.get("intent", "theological_qa"),
            confidence=float(data.get("confidence", 0.8)),
            reasoning=data.get("reasoning"),
            explicit_denomination=data.get("explicit_denomination")
        )
        logger.info(f"Intent classified: {classification.intent} (conf={classification.confidence:.2f})")
        
        updates = {"intent": classification}
        if classification.explicit_denomination:
            logger.info(f"Overriding UI denomination with explicitly requested: {classification.explicit_denomination}")
            updates["denomination"] = classification.explicit_denomination
            
        return updates

    except Exception as e:
        logger.error(f"Router error: {e}", exc_info=True)
        # Default to theological_qa on error
        return {"intent": IntentClassification(intent="theological_qa", confidence=0.5)}
