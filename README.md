# GuidaPlate

An AI-powered dietary decision-support system
for CKD patients in Rwanda.

## Project Status
Currently in active development.
Week 2 of implementation phase.

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

### 4. Run backend
uvicorn backend.main:app --reload

## Project Structure
See folder structure in documentation.

## Data Sources
- NHANES 2017-2018 (CDC)
- Kenya Food Composition Tables 2018
- USDA FoodData Central
- Rwanda National Food Balance Sheet

## Author
ISIMBI TUZINDE Jade Keslie
BSc Software Engineering
African Leadership University
Supervisor: Emmanuel Adjei
