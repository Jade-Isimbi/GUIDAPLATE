# data/

This folder contains raw and processed data
for the GuidaPlate system.

## Structure

data/
├── processed/          ← cleaned NHANES cohort CSVs
│   ├── ckd_patients.csv
│   ├── ckd_patients_clean.csv
│   └── food_nutrients_clean.csv (legacy USDA table)
└── raw/
    ├── nhanes/         ← NHANES 2017-2018 XPT files
    └── usda/           ← USDA FoodData Central files

## NHANES Files Required

Place these in data/raw/nhanes/:
- DR1TOT_J.XPT  — Day 1 total nutrient intake
- DR2TOT_J.XPT  — Day 2 total nutrient intake
- DR1IFF_J.XPT  — Day 1 individual food items 
- DR2IFF_J.XPT  — Day 2 individual food items 
- BIOPRO_J.XPT  — Laboratory biochemistry
- DEMO_J.XPT    — Demographics

Download from CDC NHANES:
https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Dietary&CycleBeginYear=2017

## Food Database

The verified 50-food Rwanda database is at:
backend/data/food_database.csv

Do NOT use data/processed/food_nutrients_clean.csv
for the GuidaPlate system. That is a legacy
USDA file from the old prototype.
