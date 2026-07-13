# Meal XGBoost — clinical guidelines compliance

**Result: ALL PASSED** (97/97)

## Verdict

Meal XGBoost v3 aligns with GuidaPlate's KDOQI/KDIGO-derived clinical constants and OCCASION_RULES meal operationalization. Severity weights/thresholds are author-derived (documented). Forbidden-food guidelines are enforced outside this model.

| Section | Check | Result | Guideline ref | Detail |
|---|---|---|---|---|
| A_constants | sodium_2300_G2 | PASS | KDOQI 2020 Guideline 5.1 | Na=2300.0 |
| A_constants | sodium_2300_G3a | PASS | KDOQI 2020 Guideline 5.1 | Na=2300.0 |
| A_constants | sodium_2300_G3b | PASS | KDOQI 2020 Guideline 5.1 | Na=2300.0 |
| A_constants | sodium_2300_G4 | PASS | KDOQI 2020 Guideline 5.1 | Na=2300.0 |
| A_constants | protein_kdoqi_G2 | PASS | KDOQI 2020 Table 11 | protein_per_kg=0.8 |
| A_constants | protein_kdoqi_G3a | PASS | KDOQI 2020 Table 11 | protein_per_kg=0.6 |
| A_constants | protein_kdoqi_G3b | PASS | KDOQI 2020 Table 11 | protein_per_kg=0.6 |
| A_constants | protein_kdoqi_G4 | PASS | KDOQI 2020 Table 11 | protein_per_kg=0.55 |
| A_constants | potassium_monotone_by_stage | PASS | KDIGO stage severity + product KDOQI table | G2=3500.0 G3a/b=3000.0 G4=2500.0 |
| A_constants | phosphorus_monotone_by_stage | PASS | KDIGO stage severity + product KDOQI table | G2=1000.0 G3=800.0 G4=700.0 |
| A_constants | train_kdoqi_matches_backend_G2 | PASS | clinical_constants.py source of truth | train={'potassium': 3500.0, 'phosphorus': 1000.0, 'protein_per_kg': 0.8, 'sodium': 2300.0} backend={'potassium': 3500.0, 'phosphorus': 1000.0, 'protein_per_kg': 0.8, 'sodium': 2300.0} |
| A_constants | train_kdoqi_matches_backend_G3a | PASS | clinical_constants.py source of truth | train={'potassium': 3000.0, 'phosphorus': 800.0, 'protein_per_kg': 0.6, 'sodium': 2300.0} backend={'potassium': 3000.0, 'phosphorus': 800.0, 'protein_per_kg': 0.6, 'sodium': 2300.0} |
| A_constants | train_kdoqi_matches_backend_G3b | PASS | clinical_constants.py source of truth | train={'potassium': 3000.0, 'phosphorus': 800.0, 'protein_per_kg': 0.6, 'sodium': 2300.0} backend={'potassium': 3000.0, 'phosphorus': 800.0, 'protein_per_kg': 0.6, 'sodium': 2300.0} |
| A_constants | train_kdoqi_matches_backend_G4 | PASS | clinical_constants.py source of truth | train={'potassium': 2500.0, 'phosphorus': 700.0, 'protein_per_kg': 0.55, 'sodium': 2300.0} backend={'potassium': 2500.0, 'phosphorus': 700.0, 'protein_per_kg': 0.55, 'sodium': 2300.0} |
| A_constants | weights_match_backend | PASS | clinical_constants CLINICAL_SEVERITY_WEIGHTS | train={'potassium': 0.35, 'phosphorus': 0.3, 'protein': 0.25, 'sodium': 0.1} backend={'potassium': 0.35, 'phosphorus': 0.3, 'protein': 0.25, 'sodium': 0.1} |
| A_constants | weights_sum_to_one | PASS | author-derived severity weights | sum=0.9999999999999999 |
| A_constants | weight_priority_K_gt_P_gt_pro_gt_Na | PASS | author-derived clinical priority (documented in clinical_constants) | K=0.35 P=0.3 pro=0.25 Na=0.1 |
| A_constants | thresholds_match_backend | PASS | clinical_constants SEVERITY_THRESHOLDS | {'HIGH': 1.2, 'MODERATE': 0.7} |
| A_constants | assign_label_matches_thresholds | PASS | SEVERITY_THRESHOLDS | boundaries 0.7 / 1.2 |
| A_constants | occasion_frac_Breakfast | PASS | meal_planner.OCCASION_RULES | OCCASION_RULES=(0.25, 0.25, 0.3, 0.25) train=(0.25, 0.25, 0.3, 0.25) |
| A_constants | occasion_frac_Lunch | PASS | meal_planner.OCCASION_RULES | OCCASION_RULES=(0.4, 0.4, 0.4, 0.4) train=(0.4, 0.4, 0.4, 0.4) |
| A_constants | occasion_frac_Dinner | PASS | meal_planner.OCCASION_RULES | OCCASION_RULES=(0.4, 0.4, 0.4, 0.4) train=(0.4, 0.4, 0.4, 0.4) |
| A_constants | occasion_frac_Snack | PASS | meal_planner.OCCASION_RULES | OCCASION_RULES=(0.15, 0.15, 0.1, 0.15) train=(0.15, 0.15, 0.1, 0.15) |
| A_constants | egfr_range_G2 | PASS | KDIGO 2024 Chapter 1 (product table) | 60–89 |
| A_constants | egfr_range_G3a | PASS | KDIGO 2024 Chapter 1 (product table) | 45–59 |
| A_constants | egfr_range_G3b | PASS | KDIGO 2024 Chapter 1 (product table) | 30–44 |
| A_constants | egfr_range_G4 | PASS | KDIGO 2024 Chapter 1 (product table) | 15–29 |
| B_meal_caps | caps_G2_Breakfast | PASS | KDOQI daily × OCCASION_RULES | {'potassium': 875.0, 'phosphorus': 250.0, 'protein_per_kg': 0.24, 'sodium': 575.0} |
| B_meal_caps | caps_G2_Lunch | PASS | KDOQI daily × OCCASION_RULES | {'potassium': 1400.0, 'phosphorus': 400.0, 'protein_per_kg': 0.32000000000000006, 'sodium': 920.0} |
| B_meal_caps | caps_G2_Dinner | PASS | KDOQI daily × OCCASION_RULES | {'potassium': 1400.0, 'phosphorus': 400.0, 'protein_per_kg': 0.32000000000000006, 'sodium': 920.0} |
| B_meal_caps | caps_G2_Snack | PASS | KDOQI daily × OCCASION_RULES | {'potassium': 525.0, 'phosphorus': 150.0, 'protein_per_kg': 0.08000000000000002, 'sodium': 345.0} |
| B_meal_caps | caps_G3a_Breakfast | PASS | KDOQI daily × OCCASION_RULES | {'potassium': 750.0, 'phosphorus': 200.0, 'protein_per_kg': 0.18, 'sodium': 575.0} |
| B_meal_caps | caps_G3a_Lunch | PASS | KDOQI daily × OCCASION_RULES | {'potassium': 1200.0, 'phosphorus': 320.0, 'protein_per_kg': 0.24, 'sodium': 920.0} |
| B_meal_caps | caps_G3a_Dinner | PASS | KDOQI daily × OCCASION_RULES | {'potassium': 1200.0, 'phosphorus': 320.0, 'protein_per_kg': 0.24, 'sodium': 920.0} |
| B_meal_caps | caps_G3a_Snack | PASS | KDOQI daily × OCCASION_RULES | {'potassium': 450.0, 'phosphorus': 120.0, 'protein_per_kg': 0.06, 'sodium': 345.0} |
| B_meal_caps | caps_G3b_Breakfast | PASS | KDOQI daily × OCCASION_RULES | {'potassium': 750.0, 'phosphorus': 200.0, 'protein_per_kg': 0.18, 'sodium': 575.0} |
| B_meal_caps | caps_G3b_Lunch | PASS | KDOQI daily × OCCASION_RULES | {'potassium': 1200.0, 'phosphorus': 320.0, 'protein_per_kg': 0.24, 'sodium': 920.0} |
| B_meal_caps | caps_G3b_Dinner | PASS | KDOQI daily × OCCASION_RULES | {'potassium': 1200.0, 'phosphorus': 320.0, 'protein_per_kg': 0.24, 'sodium': 920.0} |
| B_meal_caps | caps_G3b_Snack | PASS | KDOQI daily × OCCASION_RULES | {'potassium': 450.0, 'phosphorus': 120.0, 'protein_per_kg': 0.06, 'sodium': 345.0} |
| B_meal_caps | caps_G4_Breakfast | PASS | KDOQI daily × OCCASION_RULES | {'potassium': 625.0, 'phosphorus': 175.0, 'protein_per_kg': 0.165, 'sodium': 575.0} |
| B_meal_caps | caps_G4_Lunch | PASS | KDOQI daily × OCCASION_RULES | {'potassium': 1000.0, 'phosphorus': 280.0, 'protein_per_kg': 0.22000000000000003, 'sodium': 920.0} |
| B_meal_caps | caps_G4_Dinner | PASS | KDOQI daily × OCCASION_RULES | {'potassium': 1000.0, 'phosphorus': 280.0, 'protein_per_kg': 0.22000000000000003, 'sodium': 920.0} |
| B_meal_caps | caps_G4_Snack | PASS | KDOQI daily × OCCASION_RULES | {'potassium': 375.0, 'phosphorus': 105.0, 'protein_per_kg': 0.05500000000000001, 'sodium': 345.0} |
| B_meal_caps | lunch_equals_dinner_caps | PASS | OCCASION_RULES | Lunch={'potassium': 1200.0, 'phosphorus': 320.0, 'protein_per_kg': 0.24, 'sodium': 920.0} Dinner={'potassium': 1200.0, 'phosphorus': 320.0, 'protein_per_kg': 0.24, 'sodium': 920.0} |
| B_meal_caps | snack_tightest_potassium | PASS | OCCASION_RULES nutrient_caps | Snack=450.0 BF=750.0 Lunch=1200.0 |
| B_meal_caps | daily_fraction_sum_documented | PASS | OCCASION_RULES operationalization | sum_K_fracs=1.20 (allows ~120% if all meals maxed — product choice, not KDOQI text) |
| B_meal_caps | train_helper_matches_G2_Breakfast | PASS | train_xgboost_v3_meal.meal_caps | train={'potassium': 875.0, 'phosphorus': 250.0, 'protein_per_kg': 0.24, 'sodium': 575.0} backend={'potassium': 875.0, 'phosphorus': 250.0, 'protein_per_kg': 0.24, 'sodium': 575.0} |
| B_meal_caps | train_helper_matches_G2_Snack | PASS | train_xgboost_v3_meal.meal_caps | train={'potassium': 525.0, 'phosphorus': 150.0, 'protein_per_kg': 0.08000000000000002, 'sodium': 345.0} backend={'potassium': 525.0, 'phosphorus': 150.0, 'protein_per_kg': 0.08000000000000002, 'sodium': 345.0} |
| B_meal_caps | train_helper_matches_G4_Breakfast | PASS | train_xgboost_v3_meal.meal_caps | train={'potassium': 625.0, 'phosphorus': 175.0, 'protein_per_kg': 0.165, 'sodium': 575.0} backend={'potassium': 625.0, 'phosphorus': 175.0, 'protein_per_kg': 0.165, 'sodium': 575.0} |
| B_meal_caps | train_helper_matches_G4_Snack | PASS | train_xgboost_v3_meal.meal_caps | train={'potassium': 375.0, 'phosphorus': 105.0, 'protein_per_kg': 0.05500000000000001, 'sodium': 345.0} backend={'potassium': 375.0, 'phosphorus': 105.0, 'protein_per_kg': 0.05500000000000001, 'sodium': 345.0} |
| C_scenarios | under40_pct_caps_G2_Breakfast | PASS | KDOQI meal-cap operationalization | pred=LOW exp=LOW score=0.400 conf=0.85 |
| C_scenarios | under40_pct_caps_G2_Lunch | PASS | KDOQI meal-cap operationalization | pred=LOW exp=LOW score=0.400 conf=0.84 |
| C_scenarios | under40_pct_caps_G2_Dinner | PASS | KDOQI meal-cap operationalization | pred=LOW exp=LOW score=0.400 conf=0.84 |
| C_scenarios | under40_pct_caps_G2_Snack | PASS | KDOQI meal-cap operationalization | pred=LOW exp=LOW score=0.400 conf=0.87 |
| C_scenarios | under40_pct_caps_G3a_Breakfast | PASS | KDOQI meal-cap operationalization | pred=LOW exp=LOW score=0.400 conf=0.86 |
| C_scenarios | under40_pct_caps_G3a_Lunch | PASS | KDOQI meal-cap operationalization | pred=LOW exp=LOW score=0.400 conf=0.84 |
| C_scenarios | under40_pct_caps_G3a_Dinner | PASS | KDOQI meal-cap operationalization | pred=LOW exp=LOW score=0.400 conf=0.84 |
| C_scenarios | under40_pct_caps_G3a_Snack | PASS | KDOQI meal-cap operationalization | pred=LOW exp=LOW score=0.400 conf=0.87 |
| C_scenarios | under40_pct_caps_G3b_Breakfast | PASS | KDOQI meal-cap operationalization | pred=LOW exp=LOW score=0.400 conf=0.86 |
| C_scenarios | under40_pct_caps_G3b_Lunch | PASS | KDOQI meal-cap operationalization | pred=LOW exp=LOW score=0.400 conf=0.84 |
| C_scenarios | under40_pct_caps_G3b_Dinner | PASS | KDOQI meal-cap operationalization | pred=LOW exp=LOW score=0.400 conf=0.84 |
| C_scenarios | under40_pct_caps_G3b_Snack | PASS | KDOQI meal-cap operationalization | pred=LOW exp=LOW score=0.400 conf=0.87 |
| C_scenarios | under40_pct_caps_G4_Breakfast | PASS | KDOQI meal-cap operationalization | pred=LOW exp=LOW score=0.400 conf=0.87 |
| C_scenarios | under40_pct_caps_G4_Lunch | PASS | KDOQI meal-cap operationalization | pred=LOW exp=LOW score=0.400 conf=0.84 |
| C_scenarios | under40_pct_caps_G4_Dinner | PASS | KDOQI meal-cap operationalization | pred=LOW exp=LOW score=0.400 conf=0.84 |
| C_scenarios | under40_pct_caps_G4_Snack | PASS | KDOQI meal-cap operationalization | pred=LOW exp=LOW score=0.400 conf=0.87 |
| C_scenarios | at_100pct_caps_G3a_Lunch | PASS | score=1.0 → MODERATE (0.7–1.2) | pred=MODERATE exp=MODERATE score=1.0000 |
| C_scenarios | at_100pct_caps_G3a_Snack | PASS | score=1.0 → MODERATE (0.7–1.2) | pred=MODERATE exp=MODERATE score=1.0000 |
| C_scenarios | at_100pct_caps_G4_Lunch | PASS | score=1.0 → MODERATE (0.7–1.2) | pred=MODERATE exp=MODERATE score=1.0000 |
| C_scenarios | at_100pct_caps_G4_Snack | PASS | score=1.0 → MODERATE (0.7–1.2) | pred=MODERATE exp=MODERATE score=1.0000 |
| C_scenarios | double_all_caps_G3a_Dinner | PASS | KDOQI exceedance → HIGH | pred=HIGH exp=HIGH score=3.000 |
| C_scenarios | double_all_caps_G3a_Snack | PASS | KDOQI exceedance → HIGH | pred=HIGH exp=HIGH score=3.000 |
| C_scenarios | double_all_caps_G4_Dinner | PASS | KDOQI exceedance → HIGH | pred=HIGH exp=HIGH score=3.000 |
| C_scenarios | double_all_caps_G4_Snack | PASS | KDOQI exceedance → HIGH | pred=HIGH exp=HIGH score=3.000 |
| C_scenarios | potassium_overshoot_score_gt_sodium_overshoot | PASS | CLINICAL_SEVERITY_WEIGHTS K=0.35 > Na=0.10 | 2×K score=1.375 (HIGH) vs 2×Na score=0.750 (MODERATE) |
| C_scenarios | same_dinner_risk_nondecreasing_by_stage | PASS | tighter KDOQI caps at later stages | G2:MODERATE(0.76) → G3a:MODERATE(0.97) → G3b:MODERATE(0.97) → G4:MODERATE(1.16) |
| C_scenarios | same_intake_snack_ge_dinner_G3a | PASS | OCCASION_RULES tighter snack caps | Snack HIGH(2.26) vs Dinner LOW(0.53) |
| C_scenarios | same_intake_snack_ge_dinner_G4 | PASS | OCCASION_RULES tighter snack caps | Snack HIGH(2.67) vs Dinner LOW(0.60) |
| D_monotonicity | score_monotone_increasing_potassium | PASS | clinical_score definition | scores=[0.5, 0.675, 1.025, 1.375, 2.075] |
| D_monotonicity | label_nondecreasing_potassium | PASS | guideline-aligned severity direction | labels=['LOW', 'LOW', 'MODERATE', 'HIGH', 'HIGH'] |
| D_monotonicity | score_monotone_increasing_phosphorus | PASS | clinical_score definition | scores=[0.5, 0.65, 0.95, 1.25, 1.85] |
| D_monotonicity | label_nondecreasing_phosphorus | PASS | guideline-aligned severity direction | labels=['LOW', 'LOW', 'MODERATE', 'HIGH', 'HIGH'] |
| D_monotonicity | score_monotone_increasing_protein_per_kg | PASS | clinical_score definition | scores=[0.5, 0.625, 0.875, 1.125, 1.625] |
| D_monotonicity | label_nondecreasing_protein_per_kg | PASS | guideline-aligned severity direction | labels=['LOW', 'LOW', 'MODERATE', 'MODERATE', 'HIGH'] |
| D_monotonicity | score_monotone_increasing_sodium | PASS | clinical_score definition | scores=[0.5, 0.55, 0.65, 0.75, 0.95] |
| D_monotonicity | label_nondecreasing_sodium | PASS | guideline-aligned severity direction | labels=['LOW', 'LOW', 'LOW', 'MODERATE', 'MODERATE'] |
| D_monotonicity | no_false_HIGH_when_all_le_50pct_meal_cap | PASS | clinical safety property | false_HIGH=0/16 |
| D_monotonicity | no_false_LOW_when_all_ge_2x_meal_cap | PASS | clinical safety property | false_LOW=0/16 |
| E_holdout | dataset_labels_match_backend_oracle | PASS | KDOQI×OCCASION_RULES labeling | agreement=1.0000 n=10765 |
| E_holdout | model_agrees_with_guideline_oracle | PASS | SEVERITY_THRESHOLDS on meal clinical_score | agreement=0.9985 |
| E_holdout | zero_HIGH_misses_on_holdout | PASS | clinical safety on unseen meals | HIGH_misses_holdout=0/959 |
| E_holdout | never_predict_LOW_when_oracle_HIGH | PASS | clinical safety: no class-skip under-call | HIGH→LOW=0 |
| E_holdout | fullset_HIGH_misses_only_near_1_2_boundary | PASS | boundary soft errors only (score < 1.205 → MOD not LOW) | HIGH_misses_full=10/4797; score_max=1.2021151227590559; all_pred_MODERATE=True |
| F_scope | forbidden_foods_exist_separate_from_xgb | PASS | Meal planner / recommender layer (not XGBoost) | stages=['G3A', 'G3B', 'G4'] |
| F_scope | xgb_does_not_encode_food_identity | PASS | architecture boundary | XGBoost inputs are nutrient totals + stage only — food bans are enforced elsewhere |
| F_scope | weights_thresholds_are_author_derived | PASS | clinical_constants.py honesty note | CLINICAL_SEVERITY_WEIGHTS and SEVERITY_THRESHOLDS are product-calibrated (documented in clinical_constants) — not a published KDOQI formula |
| F_scope | day_v3_still_protected | PASS | artifact safety | production day model untouched |
