"""
Hybrid Bible Retriever — BM25 sparse + dense vector search.
Denomination-aware metadata filtering for contextualized retrieval.

BM25 captures exact keyword matches (e.g., "propitiation", "sanctification").
Dense vectors capture semantic meaning (e.g., "forgiveness" → related verses).
"""

from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

from backend.core.config import Settings
from backend.core.logging import get_logger
from backend.models.schemas import Denomination, RetrievedChunk

logger = get_logger(__name__)

# Denomination → Pinecone metadata filter mapping
DENOMINATION_FILTERS: dict[str, list[str]] = {
    "Catholic": ["universal", "catholic"],
    "Protestant": ["universal", "protestant"],
    "Orthodox": ["universal", "orthodox"],
    "General": ["universal"],
}


class BibleRetriever:
    """
    Hybrid retriever combining BM25 (sparse) + Pinecone dense search.
    Falls back to dense-only if BM25 index is not loaded.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._vector_store: PineconeVectorStore | None = None
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[dict[str, Any]] = []

    def _get_vector_store(self) -> PineconeVectorStore:
        """Lazy-initialize Pinecone vector store."""
        if self._vector_store is None:
            pc = Pinecone(api_key=self.settings.pinecone_api_key)
            embeddings = OpenAIEmbeddings(
                api_key=self.settings.openai_api_key,
                model=self.settings.embedding_model,
            )
            self._vector_store = PineconeVectorStore(
                index=pc.Index(self.settings.pinecone_index_name),
                embedding=embeddings,
                text_key="text",
            )
        return self._vector_store

    async def retrieve(
        self,
        query: str,
        denomination: Denomination,
        k: int = 8,
    ) -> list[RetrievedChunk]:
        """
        Execute hybrid search and return top-k deduplicated results.

        Args:
            query: The user's theological query.
            denomination: User's denominational preference.
            k: Number of results to return.

        Returns:
            List of RetrievedChunk objects ranked by relevance.
        """
        denomination_list = DENOMINATION_FILTERS.get(denomination, ["universal"])
        metadata_filter = {"denomination": {"$in": denomination_list}}

        # Dense retrieval
        dense_results = await self._dense_search(query, metadata_filter, k=k)

        # BM25 re-ranking (if we have a local corpus loaded)
        if self._bm25 and self._bm25_corpus:
            results = self._bm25_rerank(query, dense_results, k=k)
        else:
            results = dense_results[:k]

        logger.info(
            f"Retrieved {len(results)} chunks",
            extra={"extra": {"denomination": denomination, "query_preview": query[:50]}},
        )
        return results

    async def _dense_search(
        self,
        query: str,
        metadata_filter: dict,
        k: int,
    ) -> list[RetrievedChunk]:
        """Semantic vector search via Pinecone (Optimized: Concurrent execution)."""
        try:
            import asyncio
            store = self._get_vector_store()
            
            # Step 1: Start the step-back query expansion in the background
            expansion_task = asyncio.create_task(self._step_back_query(query))
            
            # Step 2: Immediately search Pinecone with the original query
            original_search_task = store.asimilarity_search_with_score(
                query,
                k=k,
                filter=metadata_filter,
            )
            
            # Wait for both original search and expansion to finish
            original_docs, expanded_query = await asyncio.gather(
                original_search_task,
                expansion_task
            )
            
            # Step 3: Search with the expanded query (if it changed)
            expanded_docs = []
            if expanded_query != query:
                expanded_docs = await store.asimilarity_search_with_score(
                    expanded_query,
                    k=k,
                    filter=metadata_filter,
                )
                
            # Combine and deduplicate
            seen_ids = set()
            docs_and_scores = []
            
            for doc, score in (original_docs + expanded_docs):
                doc_id = f"{doc.metadata.get('book')}_{doc.metadata.get('chapter')}_{doc.metadata.get('verse')}"
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    docs_and_scores.append((doc, score))
                    
            # Sort by score and take top k*2 for BM25 re-ranking
            docs_and_scores.sort(key=lambda x: x[1], reverse=True)
            docs_and_scores = docs_and_scores[:k * 2]

            chunks = []
            for doc, score in docs_and_scores:
                meta = doc.metadata
                chunks.append(
                    RetrievedChunk(
                        text=doc.page_content,
                        book=meta.get("book", "Unknown"),
                        chapter=int(meta.get("chapter", 0)),
                        verse=int(meta.get("verse", 0)),
                        translation=meta.get("translation", "KJV"),
                        source_type=meta.get("source_type", "scripture"),
                        score=min(float(score), 1.0),
                    )
                )
            return chunks

        except Exception as e:
            logger.error(f"Dense search failed: {e}", exc_info=True)
            return []

    def _bm25_rerank(
        self,
        query: str,
        dense_results: list[RetrievedChunk],
        k: int,
    ) -> list[RetrievedChunk]:
        """Re-rank dense results using BM25 scores for hybrid scoring."""
        if not dense_results:
            return []

        query_tokens = query.lower().split()
        candidate_texts = [r.text.lower().split() for r in dense_results]
        bm25 = BM25Okapi(candidate_texts)
        bm25_scores = bm25.get_scores(query_tokens)

        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        normalized_bm25 = [s / max_bm25 for s in bm25_scores]

        # Hybrid score: 70% dense + 30% BM25
        scored = [
            (r, 0.7 * r.score + 0.3 * normalized_bm25[i])
            for i, r in enumerate(dense_results)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [r for r, _ in scored[:k]]

    async def _step_back_query(self, query: str) -> str:
        """
        Expand specific queries to more general theological concepts
        for better vector search coverage.

        Example: "Why did Jesus wash feet?" →
                 "Theological significance of servant leadership in Christianity"
        """
        # Only expand queries that seem specific/narrow
        if len(query.split()) < 5 or "?" not in query:
            return query

        try:
            from langfuse.openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            response = await client.chat.completions.create(
                model=self.settings.fast_llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a theological search query expander. "
                            "Rewrite the user's question as a broader theological concept "
                            "that would help retrieve relevant Bible passages. "
                            "Return ONLY the expanded query, nothing else. "
                            "Keep it under 20 words."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                max_tokens=50,
                temperature=0.1,
            )
            expanded = response.choices[0].message.content or query
            logger.info(f"Step-back expansion: '{query}' → '{expanded}'")
            return expanded
        except Exception:
            return query


# ── Module-level singleton ────────────────────────────────────────────────────

_retriever_instance: BibleRetriever | None = None


def get_retriever(settings: Settings) -> BibleRetriever:
    """Get or create the module-level retriever singleton."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = BibleRetriever(settings)
    return _retriever_instance
