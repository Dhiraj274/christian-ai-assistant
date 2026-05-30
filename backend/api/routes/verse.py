from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from backend.data.bible_sqlite import lookup_verse
from backend.core.config import get_settings

router = APIRouter(tags=["verse"])
settings = get_settings()

class VerseResponse(BaseModel):
    book: str
    chapter: int
    verse: str
    text: str
    translation: str

@router.get("/api/verse", response_model=VerseResponse)
async def get_verse(
    book: str = Query(..., description="Book name"),
    chapter: int = Query(..., description="Chapter number"),
    verse: str = Query(..., description="Verse number or range (e.g., 22-23)"),
    translation: Optional[str] = Query("KJV", description="Translation")
):
    """
    Fetch the exact text of a Bible verse from the local SQLite database.
    Used by the frontend to display tooltips when clicking on citations.
    Supports single verses (e.g., "3") or ranges (e.g., "22-23").
    """
    start_verse = 0
    end_verse = 0
    
    if "-" in verse:
        parts = verse.split("-")
        try:
            start_verse = int(parts[0])
            end_verse = int(parts[1])
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid verse range format.")
    else:
        try:
            start_verse = int(verse)
            end_verse = start_verse
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid verse format.")

    combined_text = []
    
    for v in range(start_verse, end_verse + 1):
        verse_data = await lookup_verse(
            book=book,
            chapter=chapter,
            verse=v,
            db_path=settings.sqlite_db_path,
            translation=translation
        )
        if verse_data:
            combined_text.append(verse_data["text"])
            
    if not combined_text:
        raise HTTPException(status_code=404, detail="Verse not found in database.")
        
    return VerseResponse(
        book=book,
        chapter=chapter,
        verse=verse,
        text=" ".join(combined_text),
        translation=translation
    )
