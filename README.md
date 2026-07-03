# GuidaPlate

AI-Powered Dietary Decision Support for CKD Patients in Rwanda

BSc Software Engineering Capstone  
African Leadership University · July 2026

## Links

| | |
|---|---|
| 🎥 Demo Video | *(add YouTube link before submission)* |
| 🌐 Live App | [guidaplate.vercel.app](https://guidaplate.vercel.app) |
| 🔌 Backend API | [guidaplate-production.up.railway.app](https://guidaplate-production.up.railway.app) |
| 📁 Repository | https://github.com/Jade-Isimbi/GUIDAPLATE |

## Key Results

| Model | Accuracy | F1 Macro | MOD Recall |
|---|---|---|---|
| Rule-Based Baseline | 75.0% | 0.718 | 0.357 |
| XGBoost v1 (leakage) | 75.3% | 0.723 | 0.367 |
| XGBoost v3 (production) | 99.0% | 0.985 | 0.969 |
| LSTM v1 (original) | 81.4% | 0.765 | 0.357 |
| LSTM v3 (production) | 91.8% | 0.915 | 0.908 |
| Weekly RF Baseline (rule) | — | 0.646 | 0.351 |
| Weekly RF + CW MOD=3 (production) | 0.836 | 0.822 | 0.903 |
| HMM Supervised | 67.8% | 0.670 | 0.602 |

McNemar (Baseline vs XGBoost v3): p<0.0001  
McNemar (Baseline vs LSTM v3): p<0.000001  
McNemar (Rule vs Weekly RF): p=6.07e-08  
Weekly RF AUC: 0.947 · CV F1: 0.808 ± 0.028

## Analysis of Results

The original project proposal set two clear technical targets for the machine learning models: an AUC-ROC score above 0.90 and a sensitivity of at least 0.85 for HIGH-risk cases. The proposal also asked four research questions: can a trained classifier beat a simple rule-based check, can a sequence model catch short-term risk patterns across meals, which nutrients matter most to the model's decisions, and can the system give stage-specific food advice.

**What was achieved**

- All three tiers beat both technical targets. XGBoost v3: AUC 0.997, 100% HIGH sensitivity. LSTM v3: AUC 0.984, 88.4% HIGH sensitivity. Weekly RF: AUC 0.947, 86.0% HIGH sensitivity.
- Improvements over the rule-based baseline are statistically significant (McNemar p < 0.0001 for all three comparisons).
- **RQ1** — XGBoost accuracy rose from 75.0% to 99.0%; HIGH sensitivity maxed at 100% on the test set.
- **RQ2** — LSTM reads the last six meals and classifies trend as escalating, stable, or improving.
- **RQ3** — SHAP explanations run live in the app for every risk check.
- **RQ4** — Stage-specific advice enforced: G3b patients never see clinically forbidden foods (e.g. beans, banana); meal plans prioritize protein at Lunch/Dinner with Rwandan foods only.

**Where it fell short or grew beyond the plan**

- A third weekly tier was added beyond the original two-model proposal. Edge-case testing (`docs/testing/06_edge_case_testing`) shows it defaults to MODERATE on some extreme inputs (4–6 of 9 cases).
- Weekly LOW-risk recall is 61.7% — conservative but more false alarms than ideal.
- Models are trained on NHANES (US data); Rwandan food thresholds are applied at entry. Local patient validation remains future work.

**Bottom line:** both proposal targets (AUC > 0.90, HIGH sensitivity ≥ 0.85) were met at every tier. The weekly layer is the main open item.

## Discussion

The three-tier design mirrors clinical workflow: assess each meal (Tier 1), detect short-term patterns across six meals (Tier 2), then evaluate weekly dietary risk (Tier 3). Class-weighting MOD=3 on the weekly model reflects a safety-first approach — missing a moderate-risk week is clinically costlier than a false alarm.

The RAG meal planner grounds LLM responses in KDOQI/KDIGO guidelines rather than generating nutrient values from memory. Stage-aware forbidden-food filters add a clinical safety layer on top of the database's `ckd_stage_safe` range check.

## Recommendations

- **Community:** Deploy in low-bandwidth environments; partner with Rwandan nephrology clinics for food-database validation and clinician review of meal plans.
- **Future work:** Collect local Rwandan patient dietary data; improve weekly-tier edge cases; add Kinyarwanda UI; offline mode for rural connectivity gaps.

## Stack

- Frontend: React + TypeScript (Vite)
- Backend: FastAPI + SQLite
- ML: XGBoost v3, LSTM v3, Weekly RF
- RAG: KDOQI 2020 + KDIGO 2024 (5,874 chunks)
- LLM: Llama-3.1-8B via Groq
- Training data: NHANES 2017–2018 (1,862 CKD patients)
- Food database: 386 foods (50 Rwandan, trilingual)

## Project Structure

```
GUIDAPLATE/
├── backend/           # FastAPI API, ML inference, RAG meal planner
├── frontend/          # React + Vite UI
├── models/            # Production artifacts (xgboost, LSTM, weekly RF)
├── notebooks/         # Training & evaluation pipeline (v3)
├── docs/testing/      # Testing evidence screenshots (10 strategies)
├── scripts/           # Evidence generation & utilities
└── verify_tier3.py    # Tier 3 integration check (9/9)
```

## Setup

### Backend

```bash
git clone https://github.com/Jade-Isimbi/GUIDAPLATE
cd GUIDAPLATE
cp .env.example .env
# Fill in .env values (see below)
python3 -m venv venv311
source venv311/bin/activate   # Windows: venv311\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/api/health`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — the app calls `http://localhost:8000` by default.

To point at the deployed backend locally, create `frontend/.env.local`:

```
VITE_API_URL=https://guidaplate-production.up.railway.app
```

## Environment Variables

Copy `.env.example` to `.env` and fill in these values:

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | Groq API key for Llama 3.1 (console.groq.com) | ✅ Yes |
| `JWT_SECRET` | Secret key for auth tokens (min 32 chars) | ✅ Yes |
| `SENDGRID_API_KEY` | SendGrid API key for password reset emails | ✅ Yes |
| `SENDGRID_FROM_EMAIL` | Verified sender email in SendGrid | ✅ Yes |
| `RESET_BASE_URL` | Frontend URL for password reset links | ✅ Yes |
| `ALLOWED_ORIGINS` | Comma-separated allowed CORS origins | ✅ Yes |
| `DATABASE_URL` | SQLite DB path (defaults to project root) | ❌ Optional |

Example `.env` (local development):

```
GROQ_API_KEY=gsk_...
JWT_SECRET=your-random-secret-here
SENDGRID_API_KEY=SG....
SENDGRID_FROM_EMAIL=noreply@yourdomain.com
RESET_BASE_URL=http://localhost:5173
ALLOWED_ORIGINS=http://localhost:5173
```

### Production deployment

| Component | Platform | Notes |
|---|---|---|
| Backend | [Railway](https://railway.com) | `Procfile` / `railway.toml` in repo root |
| Frontend | [Vercel](https://vercel.com) | Root dir `frontend/`, framework Vite |

**Railway:** set all backend env vars above; `ALLOWED_ORIGINS` and `RESET_BASE_URL` → `https://guidaplate.vercel.app`

**Vercel:** set `VITE_API_URL` → `https://guidaplate-production.up.railway.app`

Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

## Notebooks (run in order)

Production pipeline (v3):

1. `01_data_exploration.ipynb`
2. `03c_labels_v3_clinical_score.ipynb`
3. `04c_xgboost_v3_raw_features.ipynb`
4. `05c_lstm_v3_improved.ipynb`
5. `06_model_comparison.ipynb`
6. `11_weekly_tier3.ipynb`

Supporting research: `03_statistical_analysis.ipynb`, `03b_labels_v2_sequence_aware.ipynb`, `04_xgboost_training.ipynb`, `04b_xgboost_v2_improved.ipynb`, `05_lstm_training.ipynb`, `05b_lstm_v2_improved.ipynb`

Archived (superseded): `notebooks/archive/`

## Architecture

### Three-Tier ML System

| Tier | Model | Input | Output | MOD Recall |
|---|---|---|---|---|
| Tier 1 | XGBoost v3 | Single meal nutrients | LOW/MODERATE/HIGH | 0.969 |
| Tier 2 | LSTM v3 | Last 6 meals (hidden state) | Pattern + trend direction | 0.908 |
| Tier 3 | Random Forest + CW MOD=3 | 7-day XGBoost probability sequence | Weekly risk | 0.903 |

### LSTM Trend Detection

LSTM v3 extracts 64-dimensional hidden states from layer 1 (`return_sequences=True`) to classify whether dietary risk is escalating, stable, or improving across the last six logged meals.

### RAG Meal Planner

5,874 chunks from KDOQI 2020, KDIGO 2024, and Kenya FCT 2018. Stage-aware retrieval filters documents to the patient's CKD stage. Clinical forbidden-food lists per stage (G2/G3a/G3b/G4) filter suggestions beyond the database safety range. LLM: Llama-3.1-8B-Instant via Groq.

## Testing

| Strategy | Description | Evidence |
|---|---|---|
| McNemar tests | Statistical significance across model comparisons | `docs/testing/01_mcnemar_tests/` |
| 5-fold CV | Stratified cross-validation | `docs/testing/02_cross_validation/` |
| Overfitting analysis | Train/test gap < 2% | `docs/testing/03_overfitting_analysis/` |
| Confusion matrices | Per-class error analysis (7 models) | `docs/testing/04_confusion_matrices/` |
| Per-stage breakdown | G2/G3a/G3b/G4 performance | `docs/testing/05_per_stage_breakdown/` |
| Edge case testing | 9 synthetic boundary cases | `docs/testing/06_edge_case_testing/` |
| Model comparison | Full comparison table | `docs/testing/07_model_comparison/` |
| Hyperparameter sweep | 15 RF combinations | `docs/testing/08_hyperparameter_sweep/` |
| Calibration | Platt vs Isotonic | `docs/testing/09_calibration/` |
| Integration testing | Tier 3 wiring (9 checks) | `docs/testing/10_integration_verification/` |

Regenerate evidence images:

```bash
source venv311/bin/activate
python scripts/generate_testing_evidence.py
python verify_tier3.py   # expect 9/9 passed
```

## Features

- Meal risk assessment with SHAP explanation
- Daily nutrient budget (KDOQI limits)
- Weekly dietary trend analysis (Tier 3)
- AI meal planner — RAG + Groq, Rwandan foods, stage forbidden lists
- Forgot password (SendGrid)
- Voice food input (Web Speech API)
- Chat session history (SQLite)
