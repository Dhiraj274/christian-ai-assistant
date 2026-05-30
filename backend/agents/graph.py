"""
LangGraph Graph Assembly — The Multi-Agent Orchestrator.

Graph topology:
  START
    → input_guard (jailbreak detection)
    → [if abort] END
    → router (intent classification)
    → [if image] END (handled separately)
    → retriever (Pinecone hybrid search)
    → generator (Claude 3.5 Sonnet)
    → verifier (deterministic citation check)
    → [if hallucination & retry < max] → generator (CYCLE)
    → [if verified] → output_guard (toxicity check)
    → END

This cyclic generator→verifier→generator loop is the hallucination prevention system.
"""

import functools
from typing import AsyncGenerator, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import os

from backend.agents.state import AgentState
from backend.agents.nodes.input_guard import input_guard_node
from backend.agents.nodes.router import router_node
from backend.agents.nodes.retriever_node import retriever_node
from backend.agents.nodes.generator import generator_node
from backend.agents.nodes.verifier import verifier_node
from backend.agents.nodes.output_guard import output_guard_node
from backend.core.config import Settings
from backend.core.logging import get_logger
from backend.models.schemas import (
    ChatMessage,
    ChatResponse,
    Denomination,
    StreamEvent,
)

logger = get_logger("agents.graph")


# ── Node Wrappers (inject settings via closure) ───────────────────────────────

def _make_node(fn, settings: Settings):
    """Wrap an async node function to inject settings."""
    @functools.wraps(fn)
    async def wrapped(state: AgentState) -> dict:
        return await fn(state, settings)
    return wrapped


# ── Conditional Edges ─────────────────────────────────────────────────────────

def should_abort(state: AgentState) -> str:
    """After input_guard: abort if unsafe, else route to router."""
    return "end" if state.get("should_abort") else "router"


def route_by_intent(state: AgentState) -> str:
    """After router: route based on classified intent."""
    if state.get("should_abort"):
        return "end"
    intent = state.get("intent")
    if not intent:
        return "retriever"
    if intent.intent == "image_generation":
        return "end"  # Image handled by separate API route
    return "retriever"


def verify_or_continue(state: AgentState) -> str:
    """After verifier: cycle back to generator or proceed to output guard."""
    if state.get("should_abort"):
        return "end"

    verification = state.get("verification")
    retry_count = state.get("verifier_retry_count", 0)
    max_retries = 2  # Hardcoded here; settings injected at build time

    if verification and not verification.is_accurate and retry_count <= max_retries:
        logger.info(f"Routing back to generator (retry {retry_count})")
        return "generator"  # CYCLE: hallucination correction loop

    return "output_guard"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(settings: Settings) -> StateGraph:
    """
    Assemble and compile the LangGraph state machine.

    Args:
        settings: Application settings for dependency injection.

    Returns:
        Compiled LangGraph graph with memory checkpointing.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("input_guard", _make_node(input_guard_node, settings))
    graph.add_node("router", _make_node(router_node, settings))
    graph.add_node("retriever", _make_node(retriever_node, settings))
    graph.add_node("generator", _make_node(generator_node, settings))
    graph.add_node("verifier", _make_node(verifier_node, settings))
    graph.add_node("output_guard", _make_node(output_guard_node, settings))

    # Entry point
    graph.set_entry_point("input_guard")

    # Conditional edges
    graph.add_conditional_edges(
        "input_guard",
        should_abort,
        {"end": END, "router": "router"},
    )
    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {"end": END, "retriever": "retriever"},
    )

    # Linear edges
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "verifier")

    # Conditional: verifier can cycle back to generator
    graph.add_conditional_edges(
        "verifier",
        verify_or_continue,
        {
            "generator": "generator",   # Hallucination detected: retry
            "output_guard": "output_guard",  # Clean: proceed
            "end": END,
        },
    )

    graph.add_edge("output_guard", END)

    # Return uncompiled StateGraph
    return graph


# ── Execution Helpers ─────────────────────────────────────────────────────────

_uncompiled_graph = None
_graph_settings = None


def _get_uncompiled_graph(settings: Settings):
    """Get or create the uncompiled graph singleton."""
    global _uncompiled_graph, _graph_settings
    if _uncompiled_graph is None or _graph_settings != settings:
        _uncompiled_graph = build_graph(settings)
        _graph_settings = settings
    return _uncompiled_graph


def _build_initial_state(
    query: str,
    denomination: Denomination,
    session_id: str,
    history: list[ChatMessage],
) -> AgentState:
    """Build the initial agent state for a new query."""
    return {
        "query": query,
        "denomination": denomination,
        "session_id": session_id,
        "messages": [{"role": m.role, "content": m.content} for m in history],
        "input_safety": None,
        "output_safety": None,
        "safety_failure_count": 0,
        "intent": None,
        "retrieved_chunks": [],
        "context_string": "",
        "draft_response": "",
        "final_response": "",
        "verification": None,
        "verifier_retry_count": 0,
        "error": None,
        "should_abort": False,
    }


async def run_graph_streaming(
    query: str,
    denomination: Denomination,
    session_id: str,
    history: list[ChatMessage],
    settings: Settings,
) -> AsyncGenerator[StreamEvent, None]:
    """
    Execute the LangGraph pipeline and yield StreamEvents for SSE.
    """
    uncompiled_graph = _get_uncompiled_graph(settings)
    initial_state = _build_initial_state(query, denomination, session_id, history)
    config = {"configurable": {"thread_id": session_id}}

    db_path = settings.sqlite_db_path.replace("bible.db", "memory.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    async with AsyncSqliteSaver.from_conn_string(db_path) as memory:
        await memory.setup()
        graph = uncompiled_graph.compile(checkpointer=memory)
        
        final_state: AgentState | None = None

        # Run the graph and collect final state
        async for chunk in graph.astream(initial_state, config=config):
            for node_name, node_output in chunk.items():
                if node_name == "__end__":
                    continue

                # Map the completed node to the *next* process step to show the user
                process_msg = None
                if node_name == "input_guard":
                    process_msg = "Classifying theological intent..."
                elif node_name == "router":
                    process_msg = "Retrieving scripture and context..."
                elif node_name == "retriever":
                    process_msg = "Generating response..."
                elif node_name == "generator":
                    process_msg = "Verifying citations deterministically..."
                elif node_name == "verifier":
                    process_msg = "Performing final safety checks..."
                
                if process_msg:
                    yield StreamEvent(type="process", data={"step": process_msg})

                # Emit metadata events as nodes complete
                if node_name == "verifier":
                    verification = node_output.get("verification")
                    if verification:
                        yield StreamEvent(
                            type="verification",
                            data={
                                "passed": verification.is_accurate,
                                "hallucinated": verification.hallucinated_citations,
                                "verified": verification.verified_citations,
                            },
                        )

                if node_name == "output_guard":
                    final_response = node_output.get("final_response", "")
                    if final_response:
                        final_state = node_output

        # After graph completes, stream the final response token by token
        # Get the final state from graph
        try:
            state = await graph.aget_state(config)
            final_response = state.values.get("final_response", "")
            verification = state.values.get("verification")
            intent = state.values.get("intent")

            if not final_response:
                final_response = (
                    "I was unable to generate a response. Please try rephrasing your question."
                )

            # Emit citations metadata
            if verification and verification.verified_citations:
                yield StreamEvent(type="citation", data={"citations": verification.verified_citations})

            # Stream response tokens (simulate streaming from full text)
            words = final_response.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                yield StreamEvent(type="token", data={"content": token})

            # Emit final metadata
            yield StreamEvent(
                type="metadata",
                data={
                    "intent": intent.intent if intent else "unknown",
                    "denomination": denomination,
                    "verification_passed": verification.is_accurate if verification else True,
                    "retrieved_count": len(state.values.get("retrieved_chunks", [])),
                },
            )

        except Exception as e:
            logger.error(f"Error streaming final response: {e}", exc_info=True)
            yield StreamEvent(type="error", data={"message": "Error retrieving response"})


async def run_graph_sync(
    query: str,
    denomination: Denomination,
    session_id: str,
    history: list[ChatMessage],
    settings: Settings,
) -> ChatResponse:
    """
    Non-streaming execution for testing and evaluation.
    """
    uncompiled_graph = _get_uncompiled_graph(settings)
    initial_state = _build_initial_state(query, denomination, session_id, history)
    config = {"configurable": {"thread_id": session_id}}

    db_path = settings.sqlite_db_path.replace("bible.db", "memory.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    async with AsyncSqliteSaver.from_conn_string(db_path) as memory:
        await memory.setup()
        graph = uncompiled_graph.compile(checkpointer=memory)
        final_state = await graph.ainvoke(initial_state, config=config)

    verification = final_state.get("verification")
    intent = final_state.get("intent")

    return ChatResponse(
        answer=final_state.get("final_response", "No response generated"),
        citations=verification.verified_citations if verification else [],
        denomination_used=denomination,
        verification_passed=verification.is_accurate if verification else True,
        session_id=session_id,
        intent=intent.intent if intent else "theological_qa",
        retrieved_chunks_count=len(final_state.get("retrieved_chunks", [])),
    )
