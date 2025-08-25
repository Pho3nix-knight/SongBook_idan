
from typing import List

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    parts = []
    i = 0
    while i < len(text):
        parts.append(text[i:i+chunk_size])
        i += chunk_size - overlap
        if i < 0:
            break
    return parts
