# backend/data/

## Food Database
food_database.csv is present.
50 verified Rwandan foods.
15 columns.
Sources: Kenya FCT 2018 + USDA FDC.
Do NOT modify this file.

Columns:
food_id, english, french,
kinyarwanda, category, meal_type,
protein_g, potassium_mg,
phosphorus_mg, sodium_mg,
energy_kcal, preparation_method,
source, ckd_stage_safe, notes

## NHANES Files
Place NHANES XPT files in:
backend/data/nhanes/

These are symlinked from:
data/raw/nhanes/

Required files:
DR1TOT_J.xpt — Day 1 totals
DR2TOT_J.xpt — Day 2 totals
DR1IFF_J.xpt — Day 1 individual foods
DR2IFF_J.xpt — Day 2 individual foods
BIOPRO_J.xpt — Blood biochemistry
DEMO_J.xpt   — Demographics
BMX_J.xpt    — Body measures

All 7 files present. ✅
