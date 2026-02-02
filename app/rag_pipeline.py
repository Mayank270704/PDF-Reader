class RAGPipeline:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.store = vector_store

    def retrieve(self, query, k=3):
        q_emb = self.embedder.embed([query])[0]
        return self.store.search(q_emb, k=k)

    def generate(self, query, contexts):
        if not contexts:
            return "I could not find relevant information in the uploaded documents."

        # Simple, deterministic synthesis (no LLM yet)
        unique_lines = []
        seen = set()

        for ctx in contexts:
            for line in ctx.split("."):
                line = line.strip()
                if line and line.lower() not in seen:
                    seen.add(line.lower())
                    unique_lines.append(line)

        # Keep it concise
        summary = ". ".join(unique_lines[:3])
        return summary + "."
