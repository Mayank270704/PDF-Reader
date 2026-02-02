from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore
from app.utils import chunk_text

# Initialize components
embedder = EmbeddingModel()
vector_store = VectorStore(dim=384)

# Sample document
document = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.
Machine learning is a subset of AI that enables systems to learn from data.
Deep learning is a further subset that uses neural networks with many layers.
"""

# Chunk document
chunks = chunk_text(document)

# Embed chunks
embeddings = embedder.embed(chunks)

# Store vectors
vector_store.add(embeddings, chunks)

# Query
query = "What is machine learning?"
query_embedding = embedder.embed([query])[0]

# Retrieve
results = vector_store.search(query_embedding)

print("Retrieved Chunks:")
for r in results:
    print("-", r.strip())
