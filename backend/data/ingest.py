"""
Bible data ingestion pipeline.
Downloads the public domain KJV Bible from github.com/aruljohn/Bible-kjv
(one JSON file per book) and upserts all 31,102 verses to Pinecone
with semantic embeddings, chunked at the verse level for maximum retrieval precision.

Run: python -m backend.data.ingest
"""

import json
import sys
from pathlib import Path
from typing import Any

import requests
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

from backend.core.config import get_settings
from backend.core.logging import get_logger, setup_logging

setup_logging("INFO")
logger = get_logger("ingest")

BATCH_SIZE = 100

# ── KJV Source: aruljohn/Bible-kjv (one file per book) ───────────────────────
# The repo does NOT have a single Bible.json — each book is a separate file.
# Book filenames as they appear in the repo (no spaces, capitalised).
KJV_BOOKS = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth", "1Samuel", "2Samuel",
    "1Kings", "2Kings", "1Chronicles", "2Chronicles", "Ezra",
    "Nehemiah", "Esther", "Job", "Psalms", "Proverbs",
    "Ecclesiastes", "SongofSolomon", "Isaiah", "Jeremiah", "Lamentations",
    "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk",
    "Zephaniah", "Haggai", "Zechariah", "Malachi",
    "Matthew", "Mark", "Luke", "John", "Acts",
    "Romans", "1Corinthians", "2Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1Thessalonians", "2Thessalonians",
    "1Timothy", "2Timothy", "Titus", "Philemon", "Hebrews",
    "James", "1Peter", "2Peter", "1John", "2John",
    "3John", "Jude", "Revelation",
]

# Display-name mapping (repo filename → canonical book name)
# Needed for books with no spaces in the filename but spaces in the display name.
BOOK_DISPLAY_NAMES: dict[str, str] = {
    "1Samuel": "1 Samuel", "2Samuel": "2 Samuel",
    "1Kings": "1 Kings", "2Kings": "2 Kings",
    "1Chronicles": "1 Chronicles", "2Chronicles": "2 Chronicles",
    "SongofSolomon": "Song of Solomon",
    "1Corinthians": "1 Corinthians", "2Corinthians": "2 Corinthians",
    "1Thessalonians": "1 Thessalonians", "2Thessalonians": "2 Thessalonians",
    "1Timothy": "1 Timothy", "2Timothy": "2 Timothy",
    "1Peter": "1 Peter", "2Peter": "2 Peter",
    "1John": "1 John", "2John": "2 John", "3John": "3 John",
}

RAW_BASE = "https://raw.githubusercontent.com/aruljohn/Bible-kjv/master"

NEW_TESTAMENT = {
    "Matthew", "Mark", "Luke", "John", "Acts", "Romans",
    "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews",
    "James", "1 Peter", "2 Peter", "1 John", "2 John",
    "3 John", "Jude", "Revelation",
}


# ── Download ──────────────────────────────────────────────────────────────────

def download_kjv_bible(save_path: str) -> list[dict[str, Any]]:
    """
    Download all 66 KJV books from aruljohn/Bible-kjv and merge into one list.
    Each element: {"book": "Genesis", "chapters": [...]}

    Format per book file:
    {
      "book": "Genesis",
      "chapters": [
        {"chapter": "1", "verses": [{"verse": "1", "text": "..."}, ...]},
        ...
      ]
    }
    """
    path = Path(save_path)
    if path.exists():
        logger.info(f"KJV Bible already cached at {save_path}")
        with open(save_path, encoding="utf-8") as f:
            return json.load(f)

    path.parent.mkdir(parents=True, exist_ok=True)
    all_books: list[dict[str, Any]] = []
    failed: list[str] = []

    for book_file in KJV_BOOKS:
        url = f"{RAW_BASE}/{book_file}.json"
        display = BOOK_DISPLAY_NAMES.get(book_file, book_file)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # Normalise: ensure "book" key uses display name
            data["book"] = display
            all_books.append(data)
            logger.info(f"  ✓ {display} ({len(data.get('chapters', []))} chapters)")
        except Exception as e:
            logger.warning(f"  ✗ Failed to download {display}: {e}")
            failed.append(book_file)

    if failed:
        logger.error(f"Failed to download {len(failed)} books: {failed}")
        if len(failed) > 5:
            raise RuntimeError("Too many download failures — check internet connection.")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_books, f)
    logger.info(f"KJV Bible saved to {save_path} ({len(all_books)} books)")
    return all_books


# ── Chunking ──────────────────────────────────────────────────────────────────

def prepare_verse_chunks(bible_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert raw Bible JSON into verse-level chunks for embedding."""
    chunks = []

    for book_data in bible_data:
        book_name = book_data.get("book", "Unknown")
        testament = "new" if book_name in NEW_TESTAMENT else "old"

        for chap in book_data.get("chapters", []):
            # chapter key may be int or string depending on the file
            chap_num = int(chap.get("chapter", 0))
            for v in chap.get("verses", []):
                verse_num = int(v.get("verse", 0))
                text = v.get("text", "").strip()
                if text:
                    chunks.append(_make_chunk(book_name, chap_num, verse_num, text, testament))

    logger.info(f"Prepared {len(chunks)} verse chunks")
    return chunks


def _make_chunk(book: str, chapter: int, verse: int, text: str, testament: str) -> dict[str, Any]:
    verse_id = f"kjv_{book.lower().replace(' ', '_')}_{chapter}_{verse}"
    return {
        "id": verse_id,
        "text": f"{book} {chapter}:{verse} — {text}",
        "raw_text": text,
        "metadata": {
            "book": book, "chapter": chapter, "verse": verse,
            "translation": "KJV", "testament": testament,
            "reference": f"{book} {chapter}:{verse}",
            "source_type": "scripture",
            "denomination": ["universal"],
        },
    }


# ── Pinecone Upsert ───────────────────────────────────────────────────────────

def get_embeddings(texts: list[str], client: OpenAI, model: str) -> list[list[float]]:
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def ingest_to_pinecone(chunks: list[dict[str, Any]], settings: Any, start_from: int = 0) -> None:
    """Upsert all verse chunks to Pinecone with embeddings."""
    pc = Pinecone(api_key=settings.pinecone_api_key)
    openai_client = OpenAI(api_key=settings.openai_api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        logger.info(f"Creating Pinecone index: {settings.pinecone_index_name}")
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(settings.pinecone_index_name)
    to_process = chunks[start_from:]

    for i in range(0, len(to_process), BATCH_SIZE):
        batch = to_process[i: i + BATCH_SIZE]
        embeddings = get_embeddings([c["text"] for c in batch], openai_client, settings.embedding_model)
        vectors = [
            {"id": c["id"], "values": e, "metadata": {**c["metadata"], "text": c["raw_text"]}}
            for c, e in zip(batch, embeddings)
        ]
        index.upsert(vectors=vectors)
        done = start_from + i + len(batch)
        pct = (done / len(chunks)) * 100
        logger.info(f"Progress: {done}/{len(chunks)} ({pct:.1f}%)")

    logger.info("Ingestion complete!")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Ingest KJV Bible into Pinecone")
    parser.add_argument("--start-from", type=int, default=0, help="Resume from this chunk index")
    parser.add_argument("--bible-path", type=str, default="./backend/data/kjv.json", help="Local cache path")
    parser.add_argument("--dry-run", action="store_true", help="Download and parse only — skip Pinecone")
    args = parser.parse_args()

    settings = get_settings()
    if not args.dry_run and (not settings.pinecone_api_key or not settings.openai_api_key):
        logger.error("Missing PINECONE_API_KEY or OPENAI_API_KEY in .env")
        sys.exit(1)

    logger.info("Downloading KJV Bible (66 books, one file each)...")
    bible_data = download_kjv_bible(args.bible_path)
    chunks = prepare_verse_chunks(bible_data)
    logger.info(f"Total verses: {len(chunks)}")

    if args.dry_run:
        logger.info("Dry-run complete — skipping Pinecone upload.")
        return

    ingest_to_pinecone(chunks, settings, start_from=args.start_from)


if __name__ == "__main__":
    main()
