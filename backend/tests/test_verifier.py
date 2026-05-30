"""
Unit tests for the Citation Verifier — the most critical component.
Tests deterministic hallucination detection without making LLM calls.
"""

import pytest

from backend.core.security import extract_citations
from backend.data.bible_sqlite import (
    VALID_BOOK_NAMES,
    _resolve_book_name,
    validate_book_exists,
    validate_chapter_exists,
)


# ── Citation Extraction Tests ─────────────────────────────────────────────────

class TestCitationExtraction:
    def test_extracts_simple_citation(self):
        text = "As stated in [John 3:16], God loved the world."
        citations = extract_citations(text)
        assert len(citations) == 1
        assert citations[0]["book"] == "John"
        assert citations[0]["chapter"] == 3
        assert citations[0]["verse"] == "16"

    def test_extracts_multiple_citations(self):
        text = "See [Romans 8:28] and [Philippians 4:13] for encouragement."
        citations = extract_citations(text)
        assert len(citations) == 2

    def test_extracts_verse_range(self):
        text = "The Beatitudes are found in [Matthew 5:3-12]."
        citations = extract_citations(text)
        assert len(citations) == 1
        assert citations[0]["verse"] == "3-12"

    def test_no_citations_returns_empty(self):
        text = "God is love. Faith is important."
        citations = extract_citations(text)
        assert citations == []

    def test_ignores_partial_brackets(self):
        text = "See [John] for context and [5:3 partial]"
        citations = extract_citations(text)
        # Should not match malformed patterns
        assert len(citations) == 0


# ── Book Validation Tests ─────────────────────────────────────────────────────

class TestBookValidation:
    def test_valid_books_are_recognized(self):
        valid_books = ["John", "Romans", "Genesis", "Revelation", "Matthew"]
        for book in valid_books:
            assert book.lower() in VALID_BOOK_NAMES, f"{book} not found"

    def test_fake_books_not_recognized(self):
        fake_books = ["Hezekiah", "3 Corinthians", "Maccabees", "Enoch"]
        for book in fake_books:
            assert book.lower() not in VALID_BOOK_NAMES, f"{book} falsely valid"

    def test_book_alias_resolution(self):
        assert _resolve_book_name("Matt") == "Matthew"
        assert _resolve_book_name("Rom") == "Romans"
        assert _resolve_book_name("Gen") == "Genesis"
        assert _resolve_book_name("Rev") == "Revelation"

    def test_case_insensitive_resolution(self):
        assert _resolve_book_name("JOHN").lower() == "john"
        assert _resolve_book_name("genesis").lower() == "genesis"


# ── Chapter Validation Tests ─────────────────────────────────────────────────

@pytest.fixture
def tmp_db_path(tmp_path):
    """Create a temporary test database."""
    return str(tmp_path / "test_bible.db")


class TestChapterValidation:
    @pytest.mark.asyncio
    async def test_valid_chapter_passes(self, tmp_db_path):
        from backend.data.bible_sqlite import init_bible_db
        await init_bible_db(tmp_db_path)
        assert await validate_chapter_exists("Romans", 8, tmp_db_path)
        assert await validate_chapter_exists("John", 3, tmp_db_path)
        assert await validate_chapter_exists("Revelation", 22, tmp_db_path)

    @pytest.mark.asyncio
    async def test_invalid_chapter_fails(self, tmp_db_path):
        from backend.data.bible_sqlite import init_bible_db
        await init_bible_db(tmp_db_path)
        # Romans has 16 chapters
        assert not await validate_chapter_exists("Romans", 19, tmp_db_path)
        # John has 21 chapters
        assert not await validate_chapter_exists("John", 25, tmp_db_path)
        # Revelation has 22 chapters
        assert not await validate_chapter_exists("Revelation", 23, tmp_db_path)

    @pytest.mark.asyncio
    async def test_verse_lookup_works(self, tmp_db_path):
        from backend.data.bible_sqlite import init_bible_db, lookup_verse
        await init_bible_db(tmp_db_path)
        # Seed data includes John 3:16
        verse = await lookup_verse("John", 3, 16, tmp_db_path)
        assert verse is not None
        assert "God" in verse["text"]

    @pytest.mark.asyncio
    async def test_fake_book_fails_validation(self, tmp_db_path):
        from backend.data.bible_sqlite import init_bible_db
        await init_bible_db(tmp_db_path)
        assert not await validate_book_exists("Hezekiah", tmp_db_path)
        assert not await validate_book_exists("3 Corinthians", tmp_db_path)
