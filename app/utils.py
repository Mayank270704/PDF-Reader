def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into overlapping chunks.

    chunk_size: number of characters per chunk
    overlap: number of characters shared between chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
