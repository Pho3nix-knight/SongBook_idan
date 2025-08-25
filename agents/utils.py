
from typing import List


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """Split ``text`` into overlapping chunks.

    Args:
        text: The source text to split.
        chunk_size: Maximum size of each chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        A list containing the text chunks.

    Raises:
        ValueError: If ``overlap`` is not smaller than ``chunk_size``.
    """

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    parts = []
    i = 0
    step = chunk_size - overlap
    while i < len(text):
        parts.append(text[i:i + chunk_size])
        i += step
    return parts
