from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.rag.document_processor import load_chunks


def filter_by_stage(docs: list[dict], patient_stage: str) -> list[dict]:
    stage_num = {
        "G2": 2,
        "G3a": 3,
        "G3b": 3,
        "G4": 4,
    }.get(patient_stage, 3)

    filtered: list[dict] = []
    for doc in docs:
        source = str(doc.get("source", "")).lower()
        has_stage_marker = any(f"g{i}" in source for i in [2, 3, 4, 5])
        if not has_stage_marker:
            filtered.append(doc)
            continue
        if f"g{stage_num}" in source or patient_stage.lower() in source:
            filtered.append(doc)
    return filtered


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

    def _expand_query(self, query: str) -> str:
        """
        Add clinical keywords to improve TF-IDF matching on patient-friendly questions.
        """
        query_lower = query.lower()
        expansions: list[str] = []

        if any(w in query_lower for w in ["breakfast", "morning", "eat"]):
            expansions.append("dietary recommendations food intake nutrition")

        if any(w in query_lower for w in ["potassium", "phosphorus", "protein", "sodium"]):
            expansions.append("CKD nutrient restriction kidney disease dietary")

        if any(w in query_lower for w in ["safe", "avoid", "eat", "drink"]):
            expansions.append("food safety CKD stage dietary guidance")

        if expansions:
            return query + " " + " ".join(expansions)
        return query

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        patient_stage: str | None = None,
    ) -> list[dict]:
        if not self._initialized:
            self.initialize()
        if not self.vectorizer or self.tfidf_matrix is None:
            return []

        expanded = self._expand_query(query)
        query_vec = self.vectorizer.transform([expanded])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        threshold = 0.03 if len(query.split()) < 6 else 0.05
        top_indices = np.argsort(similarities)[::-1]

        results: list[dict] = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < threshold:
                continue
            chunk = self.chunks[int(idx)]
            candidate = {
                "text": chunk.get("text", ""),
                "source": chunk.get("source", ""),
                "score": score,
            }
            if patient_stage:
                if not filter_by_stage([candidate], patient_stage):
                    continue
            results.append(candidate)
            if len(results) >= top_k:
                break
        return results


_retriever: RAGRetriever | None = None


def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
        _retriever.initialize()
    return _retriever
