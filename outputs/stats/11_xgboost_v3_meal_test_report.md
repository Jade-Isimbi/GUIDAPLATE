# Meal XGBoost v3 — test report

**Result: ALL PASSED** (58/58 checks)

| Strategy | Check | Result | Detail |
|---|---|---|---|
| 00_integrity | day_v3_sha256_unchanged | PASS | sha=0c31b13c74fd49b6... |
| 00_integrity | meal_artifact_exists | PASS | models/xgboost_v3_meal.pkl |
| 00_integrity | meal_path_not_day_path | PASS | meal=xgboost_v3_meal.pkl day=xgboost_v3.pkl |
| 00_integrity | meal_sha_differs_from_day | PASS | meal=564c1cd5e4c735c4... day=0c31b13c74fd49b6... |
| 00_integrity | metrics_csv_exists | PASS | 10_xgboost_v3_meal_metrics.csv |
| 00_integrity | dataset_csv_exists | PASS | 05_risk_labels_v3_meal.csv |
| 01_mcnemar | meal_beats_rule_mcnemar | PASS | p=4.184e-78 meal_only=355 rule_only=1 |
| 01_mcnemar | meal_not_worse_than_day | PASS | p=0.01586 meal_only=10 day_only=1 |
| 02_cross_validation | cv_accuracy_above_proposal | PASS | acc=0.9978 ± 0.0011 folds=[0.9986, 0.9967, 0.9995, 0.9967, 0.9972] |
| 02_cross_validation | cv_f1_macro_stable | PASS | f1_macro=0.9971 ± 0.0015 |
| 03_overfitting | train_test_gap_under_2pp | PASS | train=0.9985 test=0.9986 gap=-0.0001 |
| 03_overfitting | test_not_collapsed | PASS | test_acc=0.9986 |
| 04_confusion | proposal_auc_floor | PASS | auc=1.0000 (floor 0.9) |
| 04_confusion | proposal_high_sensitivity | PASS | HIGH sens=1.0000 (floor 0.85) |
| 04_confusion | proposal_mod_sensitivity | PASS | MOD sens=0.9952 (floor 0.85) |
| 04_confusion | zero_high_false_negatives | PASS | HIGH→not-HIGH=0 cm=[[777, 1, 0], [2, 414, 0], [0, 0, 959]] |
| 05_per_stage | stage_G2_performance | PASS | n=1716 acc=0.9983 HIGH_sens=1.0000 MOD_sens=0.9942 |
| 05_per_stage | stage_G3a_performance | PASS | n=300 acc=1.0000 HIGH_sens=1.0000 MOD_sens=1.0000 |
| 05_per_stage | stage_G3b_performance | PASS | n=113 acc=1.0000 HIGH_sens=1.0000 MOD_sens=1.0000 |
| 05_per_stage | stage_G4_performance | PASS | n=24 acc=1.0000 HIGH_sens=1.0000 MOD_sens=1.0000 |
| 05b_per_occasion | occasion_Breakfast_present | PASS | n=551 |
| 05b_per_occasion | occasion_Breakfast_performance | PASS | acc=1.0000 HIGH_sens=1.0000 MOD_sens=1.0000 |
| 05b_per_occasion | occasion_Lunch_present | PASS | n=439 |
| 05b_per_occasion | occasion_Lunch_performance | PASS | acc=1.0000 HIGH_sens=1.0000 MOD_sens=1.0000 |
| 05b_per_occasion | occasion_Dinner_present | PASS | n=571 |
| 05b_per_occasion | occasion_Dinner_performance | PASS | acc=0.9982 HIGH_sens=1.0000 MOD_sens=0.9930 |
| 05b_per_occasion | occasion_Snack_present | PASS | n=592 |
| 05b_per_occasion | occasion_Snack_performance | PASS | acc=0.9966 HIGH_sens=1.0000 MOD_sens=0.9857 |
| 06_edge_cases | edge_empty_meal | PASS | G3a/Breakfast score=0.000 exp=LOW got=LOW |
| 06_edge_cases | edge_tiny_snack | PASS | G3a/Snack score=0.184 exp=LOW got=LOW |
| 06_edge_cases | edge_at_low_mod_boundary | PASS | G2/Lunch score=0.700 exp=LOW got=LOW |
| 06_edge_cases | edge_just_above_high | PASS | G3b/Dinner score=1.500 exp=HIGH got=HIGH |
| 06_edge_cases | edge_extreme_high_K | PASS | G4/Dinner score=3.521 exp=HIGH got=HIGH |
| 06_edge_cases | edge_extreme_high_P | PASS | G4/Lunch score=6.356 exp=HIGH got=HIGH |
| 06_edge_cases | edge_high_protein | PASS | G3a/Dinner score=3.327 exp=HIGH got=HIGH |
| 06_edge_cases | edge_high_sodium | PASS | G2/Breakfast score=1.655 exp=HIGH got=HIGH |
| 06_edge_cases | edge_balanced_safe | PASS | G3a/Lunch score=0.399 exp=LOW got=LOW |
| 06_edge_cases | edge_all_pass | PASS | 9/9 edge cases matched label oracle |
| 07_comparison | meal_accuracy_ge_day_on_meal | PASS | meal=0.9986 day_on_meal=0.9944 |
| 07_comparison | meal_beats_rule_accuracy | PASS | meal=0.9986 rule=0.8342 |
| 07_comparison | meal_mod_sens_ge_day | PASS | meal=0.9952 day=0.9832 |
| 08_hyperparams | n_estimators_positive | PASS | n_estimators=200 |
| 08_hyperparams | max_depth_reasonable | PASS | max_depth=4 |
| 08_hyperparams | learning_rate_positive | PASS | learning_rate=0.01 |
| 08_hyperparams | subsample_valid | PASS | subsample=0.7 |
| 09_calibration | high_brier_reasonable | PASS | Brier(HIGH)=0.0097 |
| 09_calibration | correct_more_confident_than_errors | PASS | conf_correct=0.870 conf_wrong=0.585 n_err=3 |
| 10_integration | joblib_load_meal | PASS | model loaded |
| 10_integration | predict_proba_shape | PASS | shape=(1, 3) |
| 10_integration | class_order_012 | PASS | classes=[0, 1, 2] |
| 10_integration | n_features_is_9 | PASS | n_features_in_=9 |
| 10_integration | breakfast_caps_lt_dinner | PASS | BF K=750.0 Dinner K=1200.0 |
| 10_integration | snack_caps_tightest | PASS | Snack K=450.0 |
| 10_integration | same_nutrients_higher_risk_as_snack | PASS | Breakfast score=1.411 Snack score=3.759 |
| 10_integration | day_model_still_loads | PASS | xgboost_v3.pkl OK |
| 10_integration | dataset_has_four_occasions | PASS | occasions=['Breakfast', 'Dinner', 'Lunch', 'Snack'] |
| 10_integration | published_metrics_sane | PASS | pub auc=1.0 HIGH=1.0 MOD=0.9952 |
| 10_integration | day_v3_untouched_after_tests | PASS | sha match |
