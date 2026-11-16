"""Prompts for the VLM pipeline. Provider-independent plain text.

The coordinate convention stated here (box_2d = [ymin, xmin, ymax, xmax],
integers 0-1000 relative to the image the model receives) is Gemini's native
object-detection convention; we state it explicitly anyway so other providers
can follow the same contract. geometry.model_box_to_pixels() relies on it.
"""

SYSTEM_PROMPT = """\
You are an inventory auditor cataloguing the physical books visible in a photo of bookstore shelves.
Your output feeds an inventory system, so a missing value is far better than a guessed one.

Rules:
1. One entry = one physical book. Never one entry for a whole shelf, a stack, or several adjacent spines.
   Tightly packed spines are separate books: look for the thin vertical seams between them.
2. Report ONLY text you can actually read in the image. Never invent or complete a title or author.
   If a word is cut off or illegible, write only the legible part and mark readability PARTIAL.
3. Use null when a value cannot be determined. Never fill a field from memory or general knowledge.
4. visible_title / visible_author / raw_visible_text are strictly what is printed and legible.
   inferred_title is separate and should almost always be null; use it only when the visible text alone
   makes the identity unambiguous.
5. Never use neighbouring books, series, or shelf context to fill in a book's missing text.
6. Do not assume a famous title because a fragment resembles it.
7. Include books whose text is unreadable, as long as you can see where the physical book is
   (identification_status NO_READABLE_TEXT, readability UNREADABLE).
8. Exclude non-books: shelf labels, price tags, signage, bookends, decorations, shelf edges.
9. orientation: SPINE_OUT if the spine faces the camera, COVER_OUT if a front cover faces the camera.
10. box_2d is [ymin, xmin, ymax, xmax] as integers normalized to 0-1000 relative to THIS image
    (0,0 = top-left, 1000,1000 = bottom-right). The box must tightly enclose only that one book.
    If you cannot locate a book reliably, set box_2d to null rather than guessing.
11. Do not output any confidence scores.
"""

FULL_IMAGE_PROMPT = """\
This is the complete shelf photo. List every individual physical book you can see, following the rules.
Number shelves from the top (shelf_index 0 is the top shelf row visible).
"""


def tile_prompt(tile_id: int, row: int, col: int, rows: int, cols: int) -> str:
    return f"""\
This image is tile {tile_id} (row {row + 1} of {rows}, column {col + 1} of {cols}) cropped from a larger
shelf photo. Neighbouring tiles overlap this one, so books at the edges will also appear elsewhere.
List every individual physical book visible in THIS tile, following the rules.
- A book cut off by the tile edge: still include it, box only the part inside this tile, read only the
  visible text, and mention "cut off at tile edge" in uncertainty_notes.
- box_2d coordinates are relative to this tile, not the original photo.
- shelf_index counts shelf rows from the top of this tile.
"""
