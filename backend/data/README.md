# backend/data/

## Live (production)

| File | Role |
|---|---|
| `food_database.csv` | Full food catalog loaded by the API / codegen |
| `rag_chunks.json` | RAG retrieval chunks |
| `rag_documents/` | Source PDFs for RAG |
| `nhanes/` | Symlinks / copies of NHANES files for local ML (not deployed runtime) |

Do **not** hand-edit `food_database.csv` without regenerating `frontend` food TS via `scripts/generate_food_database_ts.py`.

## Archive (not loaded at runtime)

| File | Role |
|---|---|
| `archive/food_database_50_original.csv` | Original 50-food subset |
| `archive/Rwandan_food_database - Sheet1.csv` | Spreadsheet import source |

## NHANES Files
Place NHANES XPT files in `backend/data/nhanes/` (often symlinked from `data/raw/nhanes/`).

Required for training notebooks: DR1TOT_J, DR2TOT_J, DR1IFF_J, DR2IFF_J, BIOPRO_J, DEMO_J, BMX_J.
