from typing import List

def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int
) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks