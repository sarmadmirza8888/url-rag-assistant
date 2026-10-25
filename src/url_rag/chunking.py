"""Split long text into overlapping chunks for retrieval."""


def chunk_text(text, chunk_size=800, overlap=150):
    if not text or not text.strip():
        return []

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "chunk_id": len(chunks),
                "text": chunk
            })

        if end >= len(text):
            break

        start = end - overlap

    return chunks
