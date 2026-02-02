from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore
from app.utils import chunk_text
from app.rag_pipeline import RAGPipeline

# Setup
embedder = EmbeddingModel()
store = VectorStore(dim=384)
pipeline = RAGPipeline(embedder, store)

# Document
document = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.
Machine learning is a subset of AI that enables systems to learn from data.
Deep learning uses neural networks with many layers.
"""

# Indexing
chunks = chunk_text(document)
embeddings = embedder.embed(chunks)
store.add(embeddings, chunks)

# Query
query = "Explain machine learning"
contexts = pipeline.retrieve(query)
response = pipeline.generate(query, contexts)

print(response)
