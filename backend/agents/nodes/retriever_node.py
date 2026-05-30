"""
Node 3: Retriever Node
Fetches relevant Bible verses from Pinecone using hybrid search.
Formats retrieved chunks into a clean context string for the generator.
"""

from backend.agents.state import AgentState
from backend.core.config import Settings
from backend.core.logging import get_logger
from backend.retrievers.bible_retriever import get_retriever

logger = get_logger("agents.retriever")


async def retriever_node(state: AgentState, settings: Settings) -> dict:
    """Hybrid retrieval node — fetches Bible verses for the generator."""
    if state.get("should_abort"):
        return {}

    intent = state.get("intent")
    if intent and intent.intent not in ("theological_qa", "general_conversation"):
        return {"retrieved_chunks": [], "context_string": ""}

    query = state["query"]
    denomination = state["denomination"]

    try:
        retriever = get_retriever(settings)
        chunks = await retriever.retrieve(
            query=query,
            denomination=denomination,
            k=settings.max_retriever_results,
        )

        # Format context for injection into the generator prompt
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] {chunk.book} {chunk.chapter}:{chunk.verse} (KJV)\n{chunk.text}"
            )

        context_string = "\n\n".join(context_parts) if context_parts else (
            "No specific verses were retrieved for this query. "
            "Please answer from general theological knowledge and clearly note this."
        )

        logger.info(f"Retrieved {len(chunks)} chunks for '{query[:50]}'")
        return {"retrieved_chunks": chunks, "context_string": context_string}

    except Exception as e:
        logger.error(f"Retriever error: {e}", exc_info=True)
        return {
            "retrieved_chunks": [],
            "context_string": "Retrieval service temporarily unavailable.",
            "error": str(e),
        }
