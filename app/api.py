from fastapi import APIRouter, UploadFile, File
from uuid import uuid4
import os

from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore
from app.rag_pipeline import RAGPipeline
from app.utils import chunk_text

# -----------------------------
# Router
# -----------------------------
router = APIRouter()

# -----------------------------
# Paths & Directories
# -----------------------------
UPLOAD_DIR = "data/uploads"
INDEX_PATH = "data/index.faiss"
META_PATH = "data/meta.pkl"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# -----------------------------
# Core Components (PERSISTENT)
# -----------------------------
embedder = EmbeddingModel()

vector_store = VectorStore(
    dim=384,
    index_path=INDEX_PATH,
    meta_path=META_PATH
)

rag_pipeline = RAGPipeline(embedder, vector_store)

# -----------------------------
# Health Check / Root
# -----------------------------
@router.get("/")
def root():
    return {"message": "AI Document Intelligence System is running"}

# -----------------------------
# Upload & Index Document
# -----------------------------
@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_id = str(uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    # Read uploaded file
    content = await file.read()
    text = content.decode("utf-8")

    # Save document to disk (persistence)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Chunk, embed, and store
    chunks = chunk_text(text)
    embeddings = embedder.embed(chunks)
    vector_store.add(embeddings, chunks)
    vector_store.save()

    return {
        "message": "Document uploaded and indexed successfully",
        "file_id": file_id,
        "chunks_indexed": len(chunks)
    }

# -----------------------------
# Query Documents
# -----------------------------
@router.post("/query")
async def query_document(query: str):
    contexts = rag_pipeline.retrieve(query)
    answer = rag_pipeline.generate(query, contexts)

    return {
        "query": query,
        "answer": answer,
        "sources": contexts
    }
