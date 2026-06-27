from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

RAG_DOCS_DIR = Path("backend/data/rag_documents")
CHUNKS_CACHE = Path("backend/data/rag_chunks.json")

CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 100


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append((page.extract_text() or "") + "\n")
        except Exception:
            parts.append("\n")
    return "".join(parts)


def chunk_text(text: str, source: str) -> list[dict]:
    chunks: list[dict] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if len(chunk.strip()) > 50:
            chunks.append({"text": chunk.strip(), "source": source, "start": start})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _stage_num(stage: str) -> int:
    s = stage.strip()
    if s.startswith("G"):
        s = s[1:]
    s = s.replace("a", "").replace("b", "")
    try:
        return int(s)
    except ValueError:
        return 3


def _is_stage_safe(ckd_stage_safe: str, stage_number: int) -> bool:
    if not ckd_stage_safe or str(ckd_stage_safe).lower() == "nan":
        return False
    raw = str(ckd_stage_safe).strip()
    if "-" in raw:
        parts = raw.split("-")
        try:
            low, high = int(parts[0]), int(parts[1])
            return low <= stage_number <= high
        except Exception:
            return False
    try:
        return int(raw) == stage_number
    except Exception:
        return False


def process_all_documents() -> list[dict]:
    all_chunks: list[dict] = []

    if RAG_DOCS_DIR.exists():
        for pdf_file in sorted(RAG_DOCS_DIR.glob("*.pdf")):
            print(f"Processing {pdf_file.name}...")
            text = extract_text_from_pdf(pdf_file)
            chunks = chunk_text(text, pdf_file.name)
            all_chunks.extend(chunks)
            print(f"  → {len(chunks)} chunks")

    food_db_path = Path("backend/data/food_database.csv")
    if food_db_path.exists():
        df = pd.read_csv(food_db_path)

        for c in ["english", "potassium_mg", "phosphorus_mg", "protein_g", "sodium_mg", "ckd_stage_safe", "category"]:
            if c not in df.columns:
                df[c] = None

        for stage in ["G2", "G3a", "G3b", "G4"]:
            stage_number = _stage_num(stage)
            safe = df[df["ckd_stage_safe"].apply(lambda x: _is_stage_safe(str(x), stage_number))]
            safe = safe.sort_values("potassium_mg").head(120)

            summary_lines = [f"CKD safe foods for {stage} (low potassium first):"]
            for _, row in safe.iterrows():
                summary_lines.append(
                    f"- {row['english']}: K {row.get('potassium_mg', 'N/A')}mg, "
                    f"P {row.get('phosphorus_mg', 'N/A')}mg, "
                    f"Protein {row.get('protein_g', 'N/A')}g, "
                    f"Na {row.get('sodium_mg', 'N/A')}mg per 100g"
                )
            all_chunks.extend(chunk_text("\n".join(summary_lines), f"food_database_safe_{stage}.txt"))

        try:
            high_k = df[df["potassium_mg"] > 300].nlargest(200, "potassium_mg")
            low_k = df[df["potassium_mg"] < 100].nsmallest(200, "potassium_mg")
        except Exception:
            high_k = df.head(0)
            low_k = df.head(0)

        if not high_k.empty:
            text = "HIGH POTASSIUM FOODS TO LIMIT:\n" + "\n".join(
                [f"- {r['english']}: {r['potassium_mg']}mg K per 100g" for _, r in high_k.iterrows()]
            )
            all_chunks.extend(chunk_text(text, "food_database_high_potassium.txt"))

        if not low_k.empty:
            text = "LOW POTASSIUM OPTIONS:\n" + "\n".join(
                [f"- {r['english']}: {r['potassium_mg']}mg K per 100g" for _, r in low_k.iterrows()]
            )
            all_chunks.extend(chunk_text(text, "food_database_low_potassium.txt"))

    CHUNKS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_CACHE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def load_chunks() -> list[dict]:
    if CHUNKS_CACHE.exists():
        with open(CHUNKS_CACHE, encoding="utf-8") as f:
            return json.load(f)
    return process_all_documents()

