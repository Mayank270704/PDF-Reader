import faiss
import numpy as np
import os
import pickle

class VectorStore:
    def __init__(self, dim, index_path="data/index.faiss", meta_path="data/meta.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path

        # Load existing index if present
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.texts = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.texts = []

    def add(self, embeddings, texts):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.texts.extend(texts)

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.texts, f)

    def search(self, query_embedding, k=3):
        query_embedding = np.array([query_embedding]).astype("float32")
        _, indices = self.index.search(query_embedding, k)
        return [self.texts[i] for i in indices[0]]
