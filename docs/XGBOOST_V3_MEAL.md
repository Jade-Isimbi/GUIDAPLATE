# GuidaPlate — Meal-Level XGBoost v3

**Historical artifact (offline only):** `models/xgboost_v3_meal.pkl`  
**Live meal model:** `models/xgboost_v3_meal_noscore.pkl` (occasion + meal caps; no `clinical_score` feature).  
**Fallback:** Model failure uses the same-scale exceeded-count rule. Day and legacy meal pickles are research/evaluation only and are never live fallbacks.  
**Protected day-model hash (must never change during meal work):**  
`0c31b13c74fd49b63e7d4ce750fdcf897c850410438b99e8f27d364d17b679f5`

---

## 1. Purpose

GuidaPlate’s Meal Check classifies a **single eating occasion** (Breakfast, Lunch, Dinner, or Snack) as **LOW**, **MODERATE**, or **HIGH** dietary risk for a patient with CKD stages G2–G4.

The original production classifier (`xgboost_v3.pkl`) was trained on **daily** NHANES nutrient totals and daily KDOQI limits. At inference, Meal Check sends **meal** totals. That is a train/serve scale mismatch: a normal dinner can look “HIGH” against a full-day budget, or a heavy snack can look safer than it is relative to a snack budget.

**Meal-level XGBoost v3** closes that gap. It is the same model family, same nine features, same clinical-score weights and label thresholds — but every label and the `clinical_score` feature are computed against **occasion-fraction × daily KDOQI** caps (the same fractions GuidaPlate already uses in `OCCASION_RULES`).

In one sentence: *it is an honest meal-scale calibrator of the v3 risk rule, not a replacement of the clinical logic.*

---

## 2. Why a second model (and why not overwrite v3)

| Concern | Decision |
|--------|----------|
| Day-v3 AUC 0.9975 is a published / proposal result | Keep `xgboost_v3.pkl` frozen |
| Meal Check needs meal-scale alignment | Train a **parallel** artifact |
| “Fake” fix (rescale inputs at serve time only) | Rejected — labels would still be day-trained |
| Promote only if meal eval is equal-or-better on meal labels | Met; promotion is a product choice, not automatic |

Training and tests hash-check the day pickle before and after so meal work cannot silently replace production.

---

## 3. Clinical label design

### 3.1 Daily KDOQI limits (unchanged)

Stage-specific daily caps (mg for K/P/Na; g/kg for protein):

| Stage | Potassium | Phosphorus | Protein (g/kg) | Sodium |
|-------|-----------|------------|----------------|--------|
| G2 | 3500 | 1000 | 0.8 | 2300 |
| G3a / G3b | 3000 | 800 | 0.6 | 2300 |
| G4 | 2500 | 700 | 0.55 | 2300 |

### 3.2 Occasion fractions (from `OCCASION_RULES`)

| Occasion | K | P | Protein | Na |
|----------|---|---|---------|-----|
| Breakfast | 0.25 | 0.25 | 0.30 | 0.25 |
| Lunch | 0.40 | 0.40 | 0.40 | 0.40 |
| Dinner | 0.40 | 0.40 | 0.40 | 0.40 |
| Snack | 0.15 | 0.15 | 0.10 | 0.15 |

Meal cap = daily limit × fraction. Example: G3a Dinner potassium cap = 3000 × 0.40 = **1200 mg**.

### 3.3 Clinical score (same weights as day v3)

Severity weights: potassium **0.35**, phosphorus **0.30**, protein **0.25**, sodium **0.10**.

For each nutrient:

- ratio = intake / meal_cap  
- if ratio ≤ 1: contribute `weight × ratio`  
- if ratio > 1: contribute `weight × (1 + (ratio − 1) × 2)` (overshoot penalized)

### 3.4 Risk labels (same thresholds as day v3)

| Score | Label |
|-------|--------|
| &lt; 0.7 | LOW |
| 0.7 – &lt; 1.2 | MODERATE |
| ≥ 1.2 | HIGH |

So the **decision language** of the product stays consistent; only the **denominator** moves from day to meal.

---

## 4. Data pipeline

### 4.1 Sources

- Cohort: `data/processed/ckd_cohort_final.csv` (CKD stages G2–G4)
- Food-level intake: NHANES 2017–2018 `DR1IFF_J.xpt`, `DR2IFF_J.xpt`
- Eating occasion name: **`DR1_030Z` / `DR2_030Z`** (not `DR*_020`, which in cycle J is time-of-day in seconds)

### 4.2 Occasion mapping (CDC codes → GuidaPlate)

| Codes (selected) | GuidaPlate occasion |
|------------------|---------------------|
| 1, 5, 10 (Breakfast / Brunch / Desayuno) | Breakfast |
| 2, 11 (Lunch / Almuerzo) | Lunch |
| 3, 4, 12, 13 (Dinner / Supper / Comida / Cena) | Dinner |
| 6–9, 14–19, 91, 99 (Snack, drink, Spanish snack terms, other) | Snack |

If `030Z` is missing, time-of-day fallback maps to Breakfast / Lunch / Dinner only.

### 4.3 Aggregation

1. Filter IFF rows to cohort `SEQN`s.  
2. Sum K, P, protein, Na by `(SEQN, day, occasion)`.  
3. Join body weight → `protein_per_kg = protein / weight_kg`.  
4. Drop empty meals.  
5. Compute meal `clinical_score` and `risk_label`.

**Resulting dataset:** **10,765** meal rows  
(`outputs/stats/05_risk_labels_v3_meal.csv`)

| Occasion | Count |
|----------|------:|
| Snack | 2924 |
| Dinner | 2875 |
| Breakfast | 2761 |
| Lunch | 2205 |

| Label | Count | Share |
|-------|------:|------:|
| HIGH | 4797 | 44.6% |
| LOW | 3891 | 36.1% |
| MODERATE | 2077 | 19.3% |

Stage mix is skewed toward G2 (typical of the NHANES CKD cohort); G4 is sparse (~175 meals).

---

## 5. Features (identical set to day v3)

Order must match `backend/models/xgboost_model.py` / notebook `04c`:

1. `potassium`  
2. `phosphorus`  
3. `protein_per_kg`  
4. `sodium`  
5. `ckd_stage_encoded` (G2=1 … G4=4)  
6. `stage_numeric` (G2=2; G3a/G3b=3; G4=4)  
7. `k_p_product` = (K × P) / 1e6  
8. `protein_sodium_ratio` = protein_per_kg / (sodium/1000 + ε)  
9. `clinical_score` — **meal-scale** (the only semantic change)

---

## 6. Training recipe

Script: `scripts/train_xgboost_v3_meal.py`

| Setting | Value |
|---------|--------|
| Algorithm | `XGBClassifier`, `multi:softprob`, 3 classes |
| Split | 80/20 stratified (`random_state=42`) → 8,612 train / 2,153 test |
| Class weights | HIGH:1, **MODERATE:4**, LOW:1 (same minority boost as day v3) |
| Search | `RandomizedSearchCV`, 50 iters, 5-fold stratified, score=`f1_macro` |
| Best CV F1 macro | **0.9982** |
| Selected params (this run) | `n_estimators=200`, `max_depth=4`, `learning_rate=0.01`, `subsample=0.7`, `colsample_bytree=0.9`, `min_child_weight=5`, `gamma=0.2`, `reg_alpha=0`, `reg_lambda=2.0` |

Outputs:

- `models/xgboost_v3_meal.pkl`  
- `outputs/stats/10_xgboost_v3_meal_metrics.csv`  
- `outputs/stats/05_risk_labels_v3_meal.csv`

---

## 7. Held-out performance

### 7.1 Primary metrics (meal-labeled test)

| Metric | Meal XGBoost v3 |
|--------|----------------:|
| Accuracy | 0.9986 |
| F1 weighted | 0.9986 |
| F1 macro | 0.9982 |
| AUC-ROC (weighted OvR) | 1.0000 |
| HIGH sensitivity | 1.0000 |
| MODERATE sensitivity | 0.9952 |

Proposal targets (AUC &gt; 0.90, sensitivity ≥ 0.85): **met**.

### 7.2 Confusion matrix (test)

| True \\ Pred | LOW | MOD | HIGH |
|--------------|----:|----:|-----:|
| LOW | 777 | 1 | 0 |
| MODERATE | 2 | 414 | 0 |
| HIGH | 0 | 0 | 959 |

**Zero HIGH false negatives.** The three errors are LOW↔MODERATE swaps within ~0.003 of the 0.7 score boundary, at lower confidence than correct predictions.

### 7.3 Comparison (read carefully)

| System | Label scale | Accuracy | HIGH sens | MOD sens |
|--------|-------------|----------|-----------|----------|
| Meal XGB (new) | Meal | 0.9986 | 1.000 | 0.995 |
| Day XGB on **meal** test | Meal | 0.9944 | 0.995 | 0.983 |
| Day XGB (published) | **Day** | 0.9899 | 1.000 | 0.969 |
| Rule baseline (# caps exceeded) | Meal | 0.834 | 0.975 | 0.469 |
| Threshold on clinical_score alone | Meal | 1.000 | 1.000 | 1.000 |

**Fair takeaway:** on the same meal-labeled holdout, the meal model slightly beats day-v3 (especially MOD sensitivity). Day-v3’s published 0.9975 AUC is a **different task** (day labels) and must not be “beaten” by quoting meal AUC out of context.

McNemar (meal vs rule): p ≈ 4×10⁻⁷⁸ (meal corrects 355 cases rule misses; rule corrects 1 meal misses).  
McNemar (meal vs day on meal labels): meal wins 10 disagreements, day wins 1 (p ≈ 0.016).

### 7.4 Cross-validation & overfitting

- 5-fold stratified accuracy: **0.9978 ± 0.0011**  
- 5-fold F1 macro: **0.9971 ± 0.0015**  
- Train − test accuracy gap: **≈ 0** (−0.01 pp) — no overfitting signal

### 7.5 Slices

Stable across Breakfast / Lunch / Dinner / Snack and G2–G4 (G4 test n=24 is small but clean on this holdout).

---

## 8. What the high AUC really means

Labels are defined by thresholding `clinical_score`, and `clinical_score` is also a feature. Ablation evidence:

| Feature set | Accuracy | F1 macro | AUC |
|-------------|----------|----------|-----|
| Full 9 features (saved model) | 0.9986 | 0.9982 | ~1.00 |
| Drop `clinical_score` | ~0.775 | ~0.760 | ~0.95 |
| Raw nutrients + stage only | ~0.771 | ~0.758 | ~0.95 |

Permutation importance: shuffling `clinical_score` drops F1 macro by ~**0.66**; other features are near zero once the score is present.

**Honest interpretation**

- The model is a **soft, probabilistic thresholder** of the meal clinical score (plus minor smoothing near boundaries).  
- Near-perfect AUC is **expected** under this design — the same pattern exists for day v3.  
- It does **not** prove independent discovery of CKD risk beyond the engineered clinical rule.  
- Its value is **scale honesty**: Meal Check budgets and labels speak the same language.

A pure threshold on `clinical_score` is a perfect oracle *by construction*. The XGBoost wrapper still adds: class probabilities for UX/confidence, MOD oversampling behavior, and a drop-in compatible interface with the existing SHAP / API stack.

---

## 9. Testing evidence

Automated battery: `scripts/test_xgboost_v3_meal.py`  
Latest run: **58/58 PASS**  
Report: `outputs/stats/11_xgboost_v3_meal_test_report.md`  
Folder: `docs/testing/11_meal_xgboost_v3/`

Strategies covered (aligned with project testing docs):

0. Integrity / day-v3 SHA protection  
1. McNemar vs rule and vs day  
2. 5-fold CV  
3. Overfitting gap  
4. Confusion + proposal floors  
5. Per-stage + per-occasion  
6. Nine synthetic edge cases (empty meal, boundary, extremes)  
7. Model comparison floors  
8. Hyperparameter sanity  
9. Calibration (Brier for HIGH; confidence on errors)  
10. Integration (shapes, occasion caps, dataset occasions, day model still loads)

---

## 10. How this fits GuidaPlate architecture

```
Meal Check intake (meal totals + stage + occasion)
        │
        ├─ live: xgboost_v3_meal_noscore.pkl
        │         (raw nutrients + occasion + meal caps; no clinical_score feature)
        │
        └─ model failure: transparent meal-scale exceeded-count rule
```

**Live in the API.** The current path:

1. Loads `xgboost_v3_meal_noscore.pkl`.  
2. Builds raw nutrient, occasion, and meal-cap features.  
3. Keeps SHAP / risk UI on the same response shape.  
4. Falls back to the evaluated meal count-rule if model inference fails.

Legacy `xgboost_v3_meal.pkl` remains on disk for offline experiments only.

---

## 11. Limitations

1. **Rule circularity** — performance largely recovers the labeling function via `clinical_score`.  
2. **NHANES (US) data** — not Rwandan patient meals; local validation remains future work (same as day v3).  
3. **G4 sparsity** — few severe-stage meal rows.  
4. **Occasion coding** — relies on NHANES `030Z` mapping; some codes collapse into Snack.  
5. **Legacy meal pickle** — `xgboost_v3_meal.pkl` retains the circular `clinical_score` feature and is not loaded by the live API.

---

## 12. Reproducibility

```bash
# Train (never overwrites xgboost_v3.pkl)
./venv311/bin/python3 scripts/train_xgboost_v3_meal.py

# Full test battery
./venv311/bin/python3 scripts/test_xgboost_v3_meal.py
```

| File | Role |
|------|------|
| `scripts/train_xgboost_v3_meal.py` | Dataset build + train + metrics |
| `scripts/test_xgboost_v3_meal.py` | 58-check evaluation suite |
| `models/xgboost_v3_meal.pkl` | Meal classifier artifact |
| `models/xgboost_v3.pkl` | Production day classifier (protected) |
| `outputs/stats/05_risk_labels_v3_meal.csv` | Meal labels |
| `outputs/stats/10_xgboost_v3_meal_metrics.csv` | Holdout metrics |
| `outputs/stats/10_xgboost_v3_meal_deep_eval.json` | Ablation / importance / errors |
| `outputs/stats/11_xgboost_v3_meal_test_report.md` | Test report |

---

## 13. Bottom line for the thesis / demo

**What we built:** a meal-occasion XGBoost classifier that uses the same clinical scoring philosophy as production v3, applied at GuidaPlate’s real serving unit (one meal).

**What we proved:** on meal-labeled NHANES holdout data it clears proposal floors, beats a simple nutrient-exceedance rule with statistical significance, matches or exceeds day-v3 when both are scored at meal scale, and passes a full automated test battery — without touching the published day model.

**What we do not claim:** that AUC ≈ 1.0 is novel clinical intelligence beyond the KDOQI-weighted score. The scientific contribution is **correct calibration of Tier-1 risk to meal budgets**, removing a train/serve mismatch that would otherwise undermine Meal Check credibility.
