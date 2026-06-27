from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.rag.document_processor import load_chunks


class RAGRetriever:
    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        print("Initializing RAG retriever...")
        self.chunks = load_chunks()
        texts = [c.get("text", "") for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self._initialized = True
        print(f"RAG ready: {len(self.chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not self._initialized:
            self.initialize()
        if not self.vectorizer or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results: list[dict] = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score <= 0.05:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "text": chunk.get("text", ""),
                    "source": chunk.get("source", ""),
                    "score": score,
                }
            )
        return results


_retriever: RAGRetriever | None = None


def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
        _retriever.initialize()
    return _retriever

