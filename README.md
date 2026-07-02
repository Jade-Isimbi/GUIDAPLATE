# GuidaPlate

AI-Powered Dietary Decision Support for CKD Patients in Rwanda

BSc Software Engineering Capstone  
African Leadership University · July 2026

## Links
| | |
|---|---|
| 🎥 Demo Video | [Watch on YouTube](#) |
| 🌐 Live App | [guidaplate.vercel.app](#) |
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

## Stack

- Frontend: React + TypeScript (Vite)
- Backend: FastAPI + SQLite
- ML: XGBoost v3, LSTM v3, HMM
- RAG: KDOQI 2020 + KDIGO 2024 (5874 chunks)
- LLM: Llama-3.1-8B via Groq (sub-second response time)
- Training data: NHANES 2017-2018 (1,862 CKD patients)
- Food database: 386 foods (trilingual)

## Setup

### Backend

```bash
git clone https://github.com/Jade-Isimbi/GUIDAPLATE
cd GUIDAPLATE
cp .env.example .env
# Fill in .env values
python3 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Environment Variables

Copy `.env.example` to `.env` and fill in these values:

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | Groq API key for Llama 3.1 (console.groq.com) | ✅ Yes |
| `JWT_SECRET` | Secret key for auth tokens (any random string) | ✅ Yes |
| `SENDGRID_API_KEY` | SendGrid API key for password reset emails | ✅ Yes |
| `SENDGRID_FROM_EMAIL` | Verified sender email in SendGrid | ✅ Yes |
| `RESET_BASE_URL` | Frontend URL for password reset links | ✅ Yes |
| `ALLOWED_ORIGINS` | Comma-separated allowed CORS origins | ✅ Yes |
| `DATABASE_URL` | SQLite DB path (optional — defaults to project root) | ❌ Optional |

Example `.env`:

```
GROQ_API_KEY=gsk_...
JWT_SECRET=your-random-secret-here
SENDGRID_API_KEY=SG....
SENDGRID_FROM_EMAIL=noreply@yourdomain.com
RESET_BASE_URL=http://localhost:5173
ALLOWED_ORIGINS=http://localhost:5173
```

## Notebooks (run in order)
Production pipeline (v3):
1. `01_data_exploration.ipynb`
2. `03c_labels_v3_clinical_score.ipynb`
3. `04c_xgboost_v3_raw_features.ipynb`
4. `05c_lstm_v3_improved.ipynb`
5. `06_model_comparison.ipynb`
6. `11_weekly_tier3.ipynb`

Supporting research:
- `03_statistical_analysis.ipynb`
  — Kruskal-Wallis, Spearman tests
- `03b_labels_v2_sequence_aware.ipynb`
  — v2 label pipeline (ablation base)
- `04_xgboost_training.ipynb`
  — XGBoost v1 baseline
- `04b_xgboost_v2_improved.ipynb`
  — XGBoost v2 comparison
- `05_lstm_training.ipynb`
  — LSTM v1 baseline
- `05b_lstm_v2_improved.ipynb`
  — LSTM v2 + ablation study

Archived notebooks (superseded):
  `notebooks/archive/`

## Architecture

### Three-Tier ML System

| Tier | Model | Input | Output | MOD Recall |
|---|---|---|---|---|
| Tier 1 | XGBoost v3 | Single meal nutrients | LOW/MODERATE/HIGH | 0.969 |
| Tier 2 | LSTM v3 | Last 6 meals (hidden state) | Pattern + trend direction | 0.908 |
| Tier 3 | Random Forest + CW MOD=3 | 7-day XGBoost probability sequence | Weekly risk | 0.903 |

### LSTM Trend Detection

The LSTM v3 uses an intermediate Keras model extracting 64-dimensional hidden states from layer 1 (`return_sequences=True`) to detect whether the patient's dietary risk is escalating, stable, or improving across their last 6 logged meals.

### RAG Meal Planner

5,874 chunks indexed from KDOQI 2020, KDIGO 2024, and Kenya FCT 2018. Stage-aware retrieval filters documents to the patient's CKD stage. LLM: Llama-3.1-8B-Instant via Groq.

## Testing

| Strategy | Description |
|---|---|
| McNemar tests | Statistical significance across 6 model comparisons |
| 5-fold CV | Stratified cross-validation on all production models |
| Overfitting analysis | Train/test gap < 2% for all models |
| Per-stage breakdown | Performance verified across G2/G3a/G3b/G4 |
| Edge case testing | 9 synthetic boundary cases per model |
| Calibration testing | Platt scaling vs Isotonic regression comparison |
| Integration testing | 9-point wiring verification for Tier 3 |
| Functional testing | All 33 API endpoints registered and tested |

Testing evidence: `docs/testing/`

## Features

- Meal risk assessment with SHAP explanation
- Daily nutrient budget (KDOQI limits)
- Weekly dietary trend analysis
- AI meal planner (RAG + Llama-3.1)
- Forgot password (SendGrid)
- Voice food input (Web Speech API)
- Chat session history (SQLite)
