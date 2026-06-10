# GuidaPlate

An AI-powered dietary decision-support system
for CKD patients in Rwanda.

## GitHub Repository

https://github.com/Jade-Isimbi/GUIDAPLATE

## Project Status
Currently in active development.
ML models trained and evaluated.
React MVP running with 50-food
Rwanda database connected.

## What This System Does
GuidaPlate helps CKD patients in Rwanda
manage their diet safely by:
- Predicting dietary risk (HIGH/MODERATE/LOW)
- Detecting dangerous eating patterns over time
- Recommending safer Rwandan food alternatives

All recommendations are grounded in
KDOQI 2020 and KDIGO 2024 clinical guidelines.

## Tech Stack
- Backend: FastAPI (Python)
- Frontend: React (Vercel) — pending
- Database: SQLite
- ML Models: XGBoost + LSTM (TensorFlow)
- Explainability: SHAP TreeExplainer
- Food Data: Kenya FCT 2018 + USDA FDC

## Setup

### 1. Install dependencies
pip install -r requirements.txt

### 2. Download NHANES data
See backend/data/nhanes/README.md

### 3. Place food database
Place food_database.csv in backend/data/

### 4. Run the React demo
cd frontend
npm install
npm run dev
Open http://localhost:5173

Note: FastAPI backend is under
development (stubs only).
Full backend integration planned
for July 2026.

## Project Structure
See folder structure in documentation.

## Data Sources
- NHANES 2017-2018 (CDC)
- Kenya Food Composition Tables 2018
- USDA FoodData Central
- Rwanda National Food Balance Sheet

## ML Model Results

| Model | Accuracy | AUC-ROC | HIGH RISK Sensitivity |
|---|---|---|---|
| XGBoost v1 | 99.7%* | 1.000* | 100%* |
| LSTM v1 | 89.6% | 0.9818 ✅ | 93.6% ✅ |

*XGBoost metrics reflect feature-label
alignment in initial version.
LSTM metrics reflect genuine
learned performance on meal sequences.

Target: AUC-ROC > 0.90 ✅
Target: HIGH RISK Sensitivity > 0.85 ✅

## Statistical Analysis Results

Five tests run on 1,862 NHANES
CKD patients at α = 0.05:

| Test | Type | Result |
|---|---|---|
| Descriptive Statistics | Descriptive | Cohort characterized |
| Spearman Correlation | Inference | All 4 nutrients significant |
| Exceedance Rate Analysis | Descriptive | G4: 28% exceed K limit |
| Kruskal-Wallis | Inference | All 4 nutrients p < 0.001 |
| McNemar Test | Inference | Infrastructure ready |

Key finding: 66-75% of CKD patients
exceed phosphorus limits regardless
of stage. Phosphorus is the primary
dietary risk driver in the cohort.

## Deployment Plan

### Current State
React MVP running locally
at http://localhost:5173
FastAPI backend: stubs only
(full implementation July 2026)

### Target Architecture
Frontend: React → Vercel
Backend: FastAPI → Render
Database: SQLite (file-based)
Models: XGBoost + LSTM via
FastAPI inference endpoints

### API Endpoints (planned)
GET  /api/foods
     Returns all 50 Rwandan foods
POST /api/predict/risk
     XGBoost dietary risk prediction
POST /api/predict/pattern
     LSTM meal pattern analysis
GET  /api/recommendations
     KDOQI-grounded food substitutions

### Performance Targets — MET
AUC-ROC > 0.90:
  ✅ LSTM achieved 0.9818
HIGH RISK Sensitivity > 0.85:
  ✅ LSTM achieved 93.6%

### Timeline
June 2026: ML models + stats ✅
July Week 1-2: FastAPI + SQLite
July Week 3: React integration
July 15: Final submission

## Designs

### Live React Frontend
The GuidaPlate React frontend runs
at http://localhost:5173

Built with React TypeScript, Shadcn UI,
Tailwind CSS, and Recharts.

### Dashboard
![Dashboard](outputs/screenshots/01_dashboard.png)

The dashboard shows the ML architecture
components (XGBoost, LSTM, SHAP,
Recommender) and key system metrics:
50 Rwandan foods, 4 CKD stages,
1,862 NHANES training patients.

### Food Explorer
![Food Explorer](outputs/screenshots/02_food_explorer.png)

Browse and search 50 verified Rwandan
foods in English, French, and Kinyarwanda.
Color-coded potassium safety ratings
based on KDOQI 2020 thresholds.

### Risk Assessment
![Risk Assessment](outputs/screenshots/03_risk_assessment.png)

Log foods eaten by gram weight.
The system calculates total nutrient
intake and classifies dietary risk
as HIGH, MODERATE, or LOW based on
the patient's CKD stage and KDOQI limits.

### Risk Result with Recommendations
![Risk Result](outputs/screenshots/04_risk_result.png)

After assessment the system shows
which nutrients are at risk and
recommends safer Rwandan food
alternatives within the same
food category.

### Daily Meal Tracking
![Daily Tracking](outputs/screenshots/05_daily_tracking.png)

Track multiple meals across the day.
Running daily totals show cumulative
nutrient intake against safe limits
with color-coded progress indicators.

### Data Visualizations
Key findings from the NHANES CKD
cohort analysis:

![CKD Stage Distribution](outputs/figures/01_ckd_stage_distribution.png)
![Nutrient Intake by Stage](outputs/figures/03_nutrient_intake_by_stage.png)
![Spearman Correlation](outputs/figures/08_spearman_correlation.png)
![Exceedance Rates](outputs/figures/09_exceedance_rates.png)

## Author
ISIMBI TUZINDE Jade Keslie
BSc Software Engineering
African Leadership University
Supervisor: Emmanuel Adjei

## Clinical Disclaimer

GuidaPlate is a proof-of-concept
research system developed as a
BSc Software Engineering capstone
project at African Leadership University. It does not diagnose
kidney disease or prescribe
medical treatment.

All dietary suggestions are grounded
in published KDOQI 2020 and KDIGO
2024 clinical guidelines. Nutrient
values are sourced from Kenya FCT
2018 and USDA FoodData Central.

Always consult a qualified
nephrologist or registered dietitian
before making dietary changes.
