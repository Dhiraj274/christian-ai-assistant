"""
Deterministic Bible SQLite Database.
This is the "ground truth" source that the Citation Verifier node queries
to confirm that an LLM-claimed verse actually exists and matches exactly.

The verifier checks LLM output against this DB — NOT against the LLM's memory.
This breaks the hallucination loop deterministically.
"""

import aiosqlite
import json
import os
from pathlib import Path

from backend.core.logging import get_logger

logger = get_logger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

CREATE_VERSES_TABLE = """
CREATE TABLE IF NOT EXISTS verses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    book        TEXT NOT NULL,
    book_abbr   TEXT NOT NULL,
    chapter     INTEGER NOT NULL,
    verse       INTEGER NOT NULL,
    text        TEXT NOT NULL,
    translation TEXT NOT NULL DEFAULT 'KJV',
    UNIQUE(book, chapter, verse, translation)
);
"""

CREATE_BOOKS_TABLE = """
CREATE TABLE IF NOT EXISTS books (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    abbr       TEXT NOT NULL,
    testament  TEXT NOT NULL CHECK(testament IN ('old', 'new')),
    chapters   INTEGER NOT NULL
);
"""

CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_book_chapter_verse
ON verses (book, chapter, verse, translation);
"""

# ── KJV Book Metadata ─────────────────────────────────────────────────────────

KJV_BOOKS = [
    # (name, abbreviation, testament, chapter_count)
    ("Genesis", "Gen", "old", 50), ("Exodus", "Exo", "old", 40),
    ("Leviticus", "Lev", "old", 27), ("Numbers", "Num", "old", 36),
    ("Deuteronomy", "Deu", "old", 34), ("Joshua", "Jos", "old", 24),
    ("Judges", "Jdg", "old", 21), ("Ruth", "Rut", "old", 4),
    ("1 Samuel", "1Sa", "old", 31), ("2 Samuel", "2Sa", "old", 24),
    ("1 Kings", "1Ki", "old", 22), ("2 Kings", "2Ki", "old", 25),
    ("1 Chronicles", "1Ch", "old", 29), ("2 Chronicles", "2Ch", "old", 36),
    ("Ezra", "Ezr", "old", 10), ("Nehemiah", "Neh", "old", 13),
    ("Esther", "Est", "old", 10), ("Job", "Job", "old", 42),
    ("Psalms", "Psa", "old", 150), ("Proverbs", "Pro", "old", 31),
    ("Ecclesiastes", "Ecc", "old", 12), ("Song of Solomon", "Son", "old", 8),
    ("Isaiah", "Isa", "old", 66), ("Jeremiah", "Jer", "old", 52),
    ("Lamentations", "Lam", "old", 5), ("Ezekiel", "Eze", "old", 48),
    ("Daniel", "Dan", "old", 12), ("Hosea", "Hos", "old", 14),
    ("Joel", "Joe", "old", 3), ("Amos", "Amo", "old", 9),
    ("Obadiah", "Oba", "old", 1), ("Jonah", "Jon", "old", 4),
    ("Micah", "Mic", "old", 7), ("Nahum", "Nah", "old", 3),
    ("Habakkuk", "Hab", "old", 3), ("Zephaniah", "Zep", "old", 3),
    ("Haggai", "Hag", "old", 2), ("Zechariah", "Zec", "old", 14),
    ("Malachi", "Mal", "old", 4),
    ("Matthew", "Mat", "new", 28), ("Mark", "Mar", "new", 16),
    ("Luke", "Luk", "new", 24), ("John", "Joh", "new", 21),
    ("Acts", "Act", "new", 28), ("Romans", "Rom", "new", 16),
    ("1 Corinthians", "1Co", "new", 16), ("2 Corinthians", "2Co", "new", 13),
    ("Galatians", "Gal", "new", 6), ("Ephesians", "Eph", "new", 6),
    ("Philippians", "Phi", "new", 4), ("Colossians", "Col", "new", 4),
    ("1 Thessalonians", "1Th", "new", 5), ("2 Thessalonians", "2Th", "new", 3),
    ("1 Timothy", "1Ti", "new", 6), ("2 Timothy", "2Ti", "new", 4),
    ("Titus", "Tit", "new", 3), ("Philemon", "Phm", "new", 1),
    ("Hebrews", "Heb", "new", 13), ("James", "Jam", "new", 5),
    ("1 Peter", "1Pe", "new", 5), ("2 Peter", "2Pe", "new", 3),
    ("1 John", "1Jo", "new", 5), ("2 John", "2Jo", "new", 1),
    ("3 John", "3Jo", "new", 1), ("Jude", "Jud", "new", 1),
    ("Revelation", "Rev", "new", 22),
]

# Maps common abbreviations / alternate names -> canonical book name
BOOK_NAME_ALIASES: dict[str, str] = {
    "gen": "Genesis", "ex": "Exodus", "exod": "Exodus",
    "lev": "Leviticus", "num": "Numbers", "deut": "Deuteronomy",
    "josh": "Joshua", "judg": "Judges", "1 sam": "1 Samuel",
    "2 sam": "2 Samuel", "1 kgs": "1 Kings", "2 kgs": "2 Kings",
    "ps": "Psalms", "psalm": "Psalms", "prov": "Proverbs",
    "eccl": "Ecclesiastes", "eccles": "Ecclesiastes", "sol": "Song of Solomon",
    "isa": "Isaiah", "jer": "Jeremiah", "lam": "Lamentations",
    "ezek": "Ezekiel", "dan": "Daniel", "hos": "Hosea",
    "matt": "Matthew", "mk": "Mark", "lk": "Luke", "jn": "John",
    "rom": "Romans", "1 cor": "1 Corinthians", "2 cor": "2 Corinthians",
    "gal": "Galatians", "eph": "Ephesians", "phil": "Philippians",
    "col": "Colossians", "1 thess": "1 Thessalonians",
    "2 thess": "2 Thessalonians", "1 tim": "1 Timothy", "2 tim": "2 Timothy",
    "tit": "Titus", "phlm": "Philemon", "heb": "Hebrews", "jas": "James",
    "1 pet": "1 Peter", "2 pet": "2 Peter", "1 jn": "1 John",
    "2 jn": "2 John", "3 jn": "3 John", "jude": "Jude", "rev": "Revelation",
    "apoc": "Revelation",
}

VALID_BOOK_NAMES: set[str] = {book[0].lower() for book in KJV_BOOKS}


async def init_bible_db(db_path: str) -> None:
    """
    Initialize the SQLite Bible database.
    Creates tables and loads verse data from a JSON file if available.
    Falls back to a curated seed dataset for demo purposes.
    """
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    async with aiosqlite.connect(db_path) as db:
        await db.execute(CREATE_BOOKS_TABLE)
        await db.execute(CREATE_VERSES_TABLE)
        await db.execute(CREATE_INDEX)
        await db.commit()

        # Check if already populated
        cursor = await db.execute("SELECT COUNT(*) FROM books")
        count = (await cursor.fetchone())[0]
        if count > 0:
            logger.info(f"Bible DB already populated with {count} books")
            return

        # Populate books table
        await db.executemany(
            "INSERT OR IGNORE INTO books (name, abbr, testament, chapters) VALUES (?, ?, ?, ?)",
            KJV_BOOKS,
        )

        # Load verses from JSON if available
        json_path = Path(db_path).parent / "kjv.json"
        if json_path.exists():
            await _load_from_json(db, json_path)
        else:
            # Load curated demo seed (key verses)
            await _load_seed_verses(db)

        await db.commit()
        cursor = await db.execute("SELECT COUNT(*) FROM verses")
        verse_count = (await cursor.fetchone())[0]
        logger.info(f"Bible DB initialized: {verse_count} verses loaded")


async def _load_from_json(db: aiosqlite.Connection, json_path: Path) -> None:
    """
    Load all verses from a KJV JSON file.

    Supports the format produced by ingest.py:
    [
      {
        "book": "Genesis",
        "chapters": [
          {"chapter": "1", "verses": [{"verse": "1", "text": "..."}, ...]},
          ...
        ]
      },
      ...
    ]
    """
    logger.info(f"Loading Bible verses from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    verses = []

    if isinstance(data, list):
        # Format from ingest.py: list of book objects
        for book_obj in data:
            book_name = book_obj.get("book", "Unknown")
            book_abbr = next((b[1] for b in KJV_BOOKS if b[0] == book_name), book_name[:3])
            for chap in book_obj.get("chapters", []):
                chap_num = int(chap.get("chapter", 0))
                for v in chap.get("verses", []):
                    verse_num = int(v.get("verse", 0))
                    text = v.get("text", "").strip()
                    if text:
                        verses.append((book_name, book_abbr, chap_num, verse_num, text, "KJV"))
    else:
        # Legacy dict format: {"Genesis": {"1": {"1": "text"}}}
        for book_name, chapters in data.items():
            book_abbr = next((b[1] for b in KJV_BOOKS if b[0] == book_name), book_name[:3])
            for chapter_num, chapter_verses in chapters.items():
                for verse_num, verse_text in chapter_verses.items():
                    if verse_text.strip():
                        verses.append((
                            book_name, book_abbr, int(chapter_num),
                            int(verse_num), verse_text.strip(), "KJV"
                        ))

    await db.executemany(
        "INSERT OR IGNORE INTO verses (book, book_abbr, chapter, verse, text, translation) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        verses,
    )
    logger.info(f"Loaded {len(verses)} verses from JSON")


async def _load_seed_verses(db: aiosqlite.Connection) -> None:
    """Load a curated set of key Bible verses for demo purposes."""
    SEED_VERSES = [
        ("John", "Joh", 3, 16, "For God so loved the world, that he gave his only begotten Son, that whosoever believeth in him should not perish, but have everlasting life.", "KJV"),
        ("John", "Joh", 3, 17, "For God sent not his Son into the world to condemn the world; but that the world through him might be saved.", "KJV"),
        ("Romans", "Rom", 8, 28, "And we know that all things work together for good to them that love God, to them who are the called according to his purpose.", "KJV"),
        ("Romans", "Rom", 8, 38, "For I am persuaded, that neither death, nor life, nor angels, nor principalities, nor powers, nor things present, nor things to come,", "KJV"),
        ("Romans", "Rom", 8, 39, "Nor height, nor depth, nor any other creature, shall be able to separate us from the love of God, which is in Christ Jesus our Lord.", "KJV"),
        ("Matthew", "Mat", 5, 3, "Blessed are the poor in spirit: for theirs is the kingdom of heaven.", "KJV"),
        ("Matthew", "Mat", 5, 4, "Blessed are they that mourn: for they shall be comforted.", "KJV"),
        ("Matthew", "Mat", 5, 5, "Blessed are the meek: for they shall inherit the earth.", "KJV"),
        ("Matthew", "Mat", 5, 9, "Blessed are the peacemakers: for they shall be called the children of God.", "KJV"),
        ("Matthew", "Mat", 6, 9, "After this manner therefore pray ye: Our Father which art in heaven, Hallowed be thy name.", "KJV"),
        ("Matthew", "Mat", 6, 10, "Thy kingdom come, Thy will be done in earth, as it is in heaven.", "KJV"),
        ("Matthew", "Mat", 28, 19, "Go ye therefore, and teach all nations, baptizing them in the name of the Father, and of the Son, and of the Holy Ghost:", "KJV"),
        ("Psalms", "Psa", 23, 1, "The LORD is my shepherd; I shall not want.", "KJV"),
        ("Psalms", "Psa", 23, 4, "Yea, though I walk through the valley of the shadow of death, I will fear no evil: for thou art with me; thy rod and thy staff they comfort me.", "KJV"),
        ("Psalms", "Psa", 119, 105, "Thy word is a lamp unto my feet, and a light unto my path.", "KJV"),
        ("Genesis", "Gen", 1, 1, "In the beginning God created the heaven and the earth.", "KJV"),
        ("Genesis", "Gen", 1, 26, "And God said, Let us make man in our image, after our likeness: and let them have dominion over the fish of the sea, and over the fowl of the air, and over the cattle, and over all the earth, and over every creeping thing that creepeth upon the earth.", "KJV"),
        ("Isaiah", "Isa", 53, 5, "But he was wounded for our transgressions, he was bruised for our iniquities: the chastisement of our peace was upon him; and with his stripes we are healed.", "KJV"),
        ("Proverbs", "Pro", 3, 5, "Trust in the LORD with all thine heart; and lean not unto thine own understanding.", "KJV"),
        ("Proverbs", "Pro", 3, 6, "In all thy ways acknowledge him, and he shall direct thy paths.", "KJV"),
        ("Philippians", "Phi", 4, 13, "I can do all things through Christ which strengtheneth me.", "KJV"),
        ("Ephesians", "Eph", 2, 8, "For by grace are ye saved through faith; and that not of yourselves: it is the gift of God:", "KJV"),
        ("Ephesians", "Eph", 2, 9, "Not of works, lest any man should boast.", "KJV"),
        ("Hebrews", "Heb", 11, 1, "Now faith is the substance of things hoped for, the evidence of things not seen.", "KJV"),
        ("1 Corinthians", "1Co", 13, 4, "Charity suffereth long, and is kind; charity envieth not; charity vaunteth not itself, is not puffed up,", "KJV"),
        ("1 Corinthians", "1Co", 13, 13, "And now abideth faith, hope, charity, these three; but the greatest of these is charity.", "KJV"),
        ("Revelation", "Rev", 22, 21, "The grace of our Lord Jesus Christ be with you all. Amen.", "KJV"),
        ("James", "Jam", 2, 17, "Even so faith, if it hath not works, is dead, being alone.", "KJV"),
        ("1 John", "1Jo", 4, 8, "He that loveth not knoweth not God; for God is love.", "KJV"),
        ("Galatians", "Gal", 5, 22, "But the fruit of the Spirit is love, joy, peace, longsuffering, gentleness, goodness, faith,", "KJV"),
        ("Galatians", "Gal", 5, 23, "Meekness, temperance: against such there is no law.", "KJV"),
    ]
    await db.executemany(
        "INSERT OR IGNORE INTO verses (book, book_abbr, chapter, verse, text, translation) VALUES (?, ?, ?, ?, ?, ?)",
        SEED_VERSES,
    )
    logger.info(f"Seed data loaded: {len(SEED_VERSES)} key verses")


async def lookup_verse(
    book: str,
    chapter: int,
    verse: int,
    db_path: str,
    translation: str = "KJV",
) -> dict | None:
    """
    Deterministic verse lookup — the core of the Citation Verifier.

    Args:
        book: Book name (canonical or alias).
        chapter: Chapter number.
        verse: Verse number.
        db_path: Path to SQLite database.
        translation: Bible translation (default: KJV).

    Returns:
        Dict with verse data, or None if not found.
    """
    # Normalize book name
    canonical_book = _resolve_book_name(book)

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM verses WHERE book = ? AND chapter = ? AND verse = ? AND translation = ?",
            (canonical_book, chapter, verse, translation),
        )
        row = await cursor.fetchone()

    if row:
        return dict(row)
    return None


async def validate_book_exists(book: str, db_path: str) -> bool:
    """Check if a book name is valid (exists in our canonical list)."""
    canonical = _resolve_book_name(book)
    return canonical.lower() in VALID_BOOK_NAMES


async def validate_chapter_exists(book: str, chapter: int, db_path: str) -> bool:
    """Check if a chapter number is valid for the given book."""
    canonical = _resolve_book_name(book)
    book_data = next((b for b in KJV_BOOKS if b[0].lower() == canonical.lower()), None)
    if not book_data:
        return False
    return 1 <= chapter <= book_data[3]


def _resolve_book_name(book: str) -> str:
    """Resolve a book name or abbreviation to its canonical form."""
    normalized = book.strip().lower()
    # Direct match (case-insensitive)
    for canonical_name, *_ in KJV_BOOKS:
        if canonical_name.lower() == normalized:
            return canonical_name
    # Alias match
    if normalized in BOOK_NAME_ALIASES:
        return BOOK_NAME_ALIASES[normalized]
    # Abbreviation match
    for canonical_name, abbr, *_ in KJV_BOOKS:
        if abbr.lower() == normalized:
            return canonical_name
    # Return as-is (verifier will catch invalid books)
    return book.strip()
