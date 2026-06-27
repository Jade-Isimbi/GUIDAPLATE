"""Patch 03b and 05b notebooks for LSTM v2 pipeline."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

V2_LABELS_CELL = r'''print('=' * 50)
print('FIX 1 — SEQUENCE-AWARE LABELS (v2)')
print('=' * 50)

import datetime


def load_iff(path: Path) -> pd.DataFrame:
    try:
        import pyreadstat
        df, _ = pyreadstat.read_xpt(str(path))
        return df
    except Exception:
        return pd.read_sas(path, format='xport')


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    for c in candidates:
        alt = c.replace('.', '_')
        if alt in df.columns:
            return alt
    matches = [col for col in df.columns if any(c.replace('.', '_') in str(col) for c in candidates)]
    if matches:
        return matches[0]
    raise KeyError(f'None of {candidates} found.')


def standardize_iff(df: pd.DataFrame, day: int) -> pd.DataFrame:
    prefix = f'DR{day}'
    rename = {
        pick_col(df, [f'{prefix}.020', f'{prefix}_020']): 'meal_code',
        pick_col(df, [f'{prefix}IPOTA']): 'potassium',
        pick_col(df, [f'{prefix}IPHOS']): 'phosphorus',
        pick_col(df, [f'{prefix}IPROT']): 'protein',
        pick_col(df, [f'{prefix}ISODI']): 'sodium',
    }
    out = df.rename(columns=rename)
    return out[['SEQN', 'meal_code', 'potassium', 'phosphorus', 'protein', 'sodium']].copy()


def map_meal_slot(day, meal_code):
    if pd.isna(meal_code):
        return 2 if day == 1 else 5
    if isinstance(meal_code, datetime.time):
        code = meal_code.hour * 3600 + meal_code.minute * 60 + meal_code.second
    else:
        try:
            code = float(meal_code)
        except (TypeError, ValueError):
            return 2 if day == 1 else 5
    if code <= 10:
        if code == 1:
            return 0 if day == 1 else 3
        if code == 2:
            return 1 if day == 1 else 4
        return 2 if day == 1 else 5
    if code < 39600:
        return 0 if day == 1 else 3
    if code < 61200:
        return 1 if day == 1 else 4
    return 2 if day == 1 else 5


def count_escalating(meal_seq: np.ndarray, limits: dict) -> int:
    """Count nutrients with positive slope and final slot > 50% of KDOQI limit."""
    slots = np.arange(6)
    k, p, pr, na = meal_seq[:, 0], meal_seq[:, 1], meal_seq[:, 2], meal_seq[:, 3]
    slopes = [
        np.polyfit(slots, k, 1)[0],
        np.polyfit(slots, p, 1)[0],
        np.polyfit(slots, pr, 1)[0],
        np.polyfit(slots, na, 1)[0],
    ]
    finals = [k[5], p[5], pr[5], na[5]]
    limit_vals = [
        limits['potassium'],
        limits['phosphorus'],
        limits['protein_per_kg'],
        limits['sodium'],
    ]
    escalating = 0
    for slope, final, limit in zip(slopes, finals, limit_vals):
        if slope > 0 and final > 0.5 * limit:
            escalating += 1
    return escalating


def assign_risk_label_v2(row, meal_seq: np.ndarray) -> str | None:
    primary = assign_risk_label(row)
    if primary is None:
        return None
    stage = row['ckd_stage']
    if stage not in KDOQI:
        return primary
    limits = KDOQI[stage]
    escalating = count_escalating(meal_seq, limits)
    if primary == 'LOW' and escalating >= 2:
        return 'MODERATE'
    if primary == 'MODERATE' and escalating >= 3:
        return 'HIGH'
    return primary


# Build per-patient 6-slot meal sequences for escalation analysis
iff1 = load_iff(ROOT / 'data' / 'raw' / 'nhanes' / 'DR1IFF_J.xpt')
iff2 = load_iff(ROOT / 'data' / 'raw' / 'nhanes' / 'DR2IFF_J.xpt')
cohort_seqns = set(df['SEQN'])

iff1_ckd = standardize_iff(iff1[iff1['SEQN'].isin(cohort_seqns)].copy(), day=1)
iff1_ckd['meal_slot'] = iff1_ckd['meal_code'].apply(lambda x: map_meal_slot(1, x))
iff2_ckd = standardize_iff(iff2[iff2['SEQN'].isin(cohort_seqns)].copy(), day=2)
iff2_ckd['meal_slot'] = iff2_ckd['meal_code'].apply(lambda x: map_meal_slot(2, x))

all_foods = pd.concat([
    iff1_ckd[['SEQN', 'meal_slot', 'potassium', 'phosphorus', 'protein', 'sodium']],
    iff2_ckd[['SEQN', 'meal_slot', 'potassium', 'phosphorus', 'protein', 'sodium']],
], ignore_index=True)
for col in ['potassium', 'phosphorus', 'protein', 'sodium']:
    all_foods[col] = pd.to_numeric(all_foods[col], errors='coerce').fillna(0)

meal_nutrients = all_foods.groupby(['SEQN', 'meal_slot'])[
    ['potassium', 'phosphorus', 'protein', 'sodium']
].sum().reset_index()

meal_seq_by_seqn: dict[int, np.ndarray] = {}
for seqn in cohort_seqns:
    weight = df.loc[df['SEQN'] == seqn, 'weight_kg']
    if weight.empty or pd.isna(weight.iloc[0]) or weight.iloc[0] <= 0:
        continue
    w = float(weight.iloc[0])
    seq = np.zeros((6, 4))
    patient_meals = meal_nutrients[meal_nutrients['SEQN'] == seqn]
    for _, meal in patient_meals.iterrows():
        slot = int(meal['meal_slot'])
        if 0 <= slot <= 5:
            seq[slot, 0] = meal['potassium']
            seq[slot, 1] = meal['phosphorus']
            seq[slot, 2] = meal['protein'] / w
            seq[slot, 3] = meal['sodium']
    meal_seq_by_seqn[int(seqn)] = seq

# Apply v2 labels
v2_labels = []
for _, row in df.iterrows():
    seqn = int(row['SEQN'])
    meal_seq = meal_seq_by_seqn.get(seqn, np.zeros((6, 4)))
    v2_labels.append(assign_risk_label_v2(row, meal_seq))

df['risk_label_v2'] = v2_labels

# Comparison
orig = df['risk_label']
v2 = df['risk_label_v2']
changed = (orig != v2) & orig.notna() & v2.notna()
n_changed = int(changed.sum())
n_valid = int(orig.notna().sum())
pct_changed = n_changed / n_valid * 100 if n_valid else 0

print('Label distribution — Original:')
print(orig.value_counts(dropna=False))
print()
print('Label distribution — v2:')
print(v2.value_counts(dropna=False))
print()
print('Changes by original class:')
for cls in ['LOW', 'MODERATE', 'HIGH']:
    mask = orig == cls
    n_cls = int(mask.sum())
    n_up = int((mask & changed).sum())
    print(f'  {cls}: {n_up}/{n_cls} changed ({n_up / n_cls * 100 if n_cls else 0:.1f}%)')

print()
print(f'Total labels changed: {n_changed}/{n_valid} ({pct_changed:.1f}%)')
if pct_changed > 20:
    print('⚠ WARNING: More than 20% of labels changed — review escalation thresholds.')

risk_export_v2 = df[
    ['SEQN', 'ckd_stage', 'potassium', 'phosphorus', 'protein_per_kg', 'sodium', 'risk_label_v2']
].rename(columns={'risk_label_v2': 'risk_label'})
risk_export_v2.to_csv(STATS_DIR / '05_risk_labels_v2.csv', index=False)
print(f'Saved: {STATS_DIR / "05_risk_labels_v2.csv"}')
'''

def patch_03b(nb):
    nb['cells'].insert(14, {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '## Fix 1 — Sequence-aware labels (v2)\n\n'
            'Primary signal: same KDOQI count rule as `assign_risk_label`.\n'
            'Secondary escalation: positive nutrient slope across 6 meal slots '
            'with final slot > 50% of stage limit upgrades borderline cases.'
        ],
    })
    nb['cells'].insert(15, {
        'cell_type': 'code',
        'metadata': {},
        'source': [line + '\n' for line in V2_LABELS_CELL.split('\n')],
        'outputs': [],
        'execution_count': None,
    })
    # Update summary cell references if needed - leave as is
    return nb


def patch_05b(nb):
    # Cell 0 - title
    nb['cells'][0]['source'] = [
        '# GuidaPlate — LSTM Dietary Pattern Analyzer (v2)\n',
        '## Improved pipeline: sequence-aware labels + occasion feature + Masking\n\n',
        '**Notebook 05b** — does not modify notebook 05 or production artifacts.\n\n',
        '### Input Sequence (v2)\n',
        'Each patient = 6 meal steps: Day 1 Breakfast → Lunch → Dinner; Day 2 Breakfast → Lunch → Dinner.\n\n',
        'Each step = 5 features: [potassium, phosphorus, protein_per_kg, sodium, occasion_encoded]\n\n',
        'Input shape: (n_patients, 6, 5)\n\n',
        'Labels: `outputs/stats/05_risk_labels_v2.csv`\n',
        'Cache: `models/lstm_sequences_cache_v2.npz`\n',
    ]

    # Cell 1 - imports
    src1 = ''.join(nb['cells'][1]['source'])
    src1 = src1.replace(
        'from tensorflow.keras.layers import LSTM, Dense, Dropout',
        'from tensorflow.keras.layers import Masking, LSTM, Dense, Dropout',
    )
    src1 = src1.replace(
        'from sklearn.model_selection import train_test_split',
        'from sklearn.model_selection import train_test_split\n'
        'from sklearn.utils.class_weight import compute_class_weight',
    )
    nb['cells'][1]['source'] = [line + '\n' for line in src1.split('\n') if line or src1.endswith('\n')]

    # Cell 3 - load v2 labels
    src3 = ''.join(nb['cells'][3]['source'])
    src3 = src3.replace(
        "labels = pd.read_csv(ROOT / 'outputs' / 'stats' / '05_risk_labels.csv')",
        "labels = pd.read_csv(ROOT / 'outputs' / 'stats' / '05_risk_labels_v2.csv')",
    )
    nb['cells'][3]['source'] = [line + '\n' for line in src3.split('\n')]

    # Cell 4 markdown
    nb['cells'][4]['source'] = [
        '## Section 4 — Build Meal Sequences (v2)\n\n',
        'Fix 2: append deterministic `occasion_encoded` as 5th feature per slot.\n',
        'Slots 0/3 → 0.00 (Breakfast), 1/4 → 0.33 (Lunch), 2/5 → 0.67 (Dinner).\n',
    ]

    # Cell 5 - sequence build v2
    nb['cells'][5]['source'] = [line + '\n' for line in '''ckd_seqns = set(cohort['SEQN'])
print(f"CKD patients to process: {len(ckd_seqns)}")

OCCASION_BY_SLOT = {0: 0.00, 1: 0.33, 2: 0.67, 3: 0.00, 4: 0.33, 5: 0.67}

import datetime

def map_meal_slot(day, meal_code):
    if pd.isna(meal_code):
        return 2 if day == 1 else 5
    if isinstance(meal_code, datetime.time):
        code = meal_code.hour * 3600 + meal_code.minute * 60 + meal_code.second
    else:
        try:
            code = float(meal_code)
        except (TypeError, ValueError):
            return 2 if day == 1 else 5
    if code <= 10:
        if code == 1:
            return 0 if day == 1 else 3
        if code == 2:
            return 1 if day == 1 else 4
        return 2 if day == 1 else 5
    if code < 39600:
        return 0 if day == 1 else 3
    if code < 61200:
        return 1 if day == 1 else 4
    return 2 if day == 1 else 5

print("Processing Day 1 foods...")
iff1_ckd = standardize_iff(iff1[iff1['SEQN'].isin(ckd_seqns)].copy(), day=1)
iff1_ckd['meal_slot'] = iff1_ckd['meal_code'].apply(lambda x: map_meal_slot(1, x))
print(f"Day 1 food records for CKD patients: {len(iff1_ckd)}")

print("Processing Day 2 foods...")
iff2_ckd = standardize_iff(iff2[iff2['SEQN'].isin(ckd_seqns)].copy(), day=2)
iff2_ckd['meal_slot'] = iff2_ckd['meal_code'].apply(lambda x: map_meal_slot(2, x))
print(f"Day 2 food records for CKD patients: {len(iff2_ckd)}")

all_foods = pd.concat([
    iff1_ckd[['SEQN', 'meal_slot', 'potassium', 'phosphorus', 'protein', 'sodium']],
    iff2_ckd[['SEQN', 'meal_slot', 'potassium', 'phosphorus', 'protein', 'sodium']],
], ignore_index=True)

for col in ['potassium', 'phosphorus', 'protein', 'sodium']:
    all_foods[col] = pd.to_numeric(all_foods[col], errors='coerce').fillna(0)

meal_nutrients = all_foods.groupby(['SEQN', 'meal_slot'])[['potassium', 'phosphorus', 'protein', 'sodium']].sum().reset_index()
print(f"Meal nutrient records: {len(meal_nutrients)}")

sequence_data = []
sequence_labels = []
sequence_seqns = []
cohort_weights = cohort[['SEQN', 'weight_kg', 'ckd_stage', 'risk_label']].copy()
n_processed = 0
n_skipped = 0

for _, patient in cohort_weights.iterrows():
    seqn = patient['SEQN']
    weight = patient['weight_kg']
    risk = patient['risk_label']
    if pd.isna(weight) or weight <= 0:
        n_skipped += 1
        continue
    if risk not in RISK_CLASSES:
        n_skipped += 1
        continue

    seq = np.zeros((6, 5))
    for slot in range(6):
        seq[slot, 4] = OCCASION_BY_SLOT[slot]

    patient_meals = meal_nutrients[meal_nutrients['SEQN'] == seqn]
    for _, meal in patient_meals.iterrows():
        slot = int(meal['meal_slot'])
        if 0 <= slot <= 5:
            seq[slot, 0] = meal['potassium']
            seq[slot, 1] = meal['phosphorus']
            seq[slot, 2] = meal['protein'] / weight
            seq[slot, 3] = meal['sodium']
            seq[slot, 4] = OCCASION_BY_SLOT[slot]

    sequence_data.append(seq)
    sequence_labels.append(risk)
    sequence_seqns.append(seqn)
    n_processed += 1

X_seq = np.array(sequence_data)
y_seq = np.array(sequence_labels)

print(f"Sequences built: {n_processed}")
print(f"Skipped: {n_skipped}")
print(f"Sequence array shape: {X_seq.shape}")
assert X_seq.shape[1:] == (6, 5), f"Expected (6, 5), got {X_seq.shape[1:]}"

print("Risk label distribution:")
unique, counts = np.unique(y_seq, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u}: {c} ({c/len(y_seq)*100:.1f}%)")

np.savez(
    MODEL_DIR / 'lstm_sequences_cache_v2.npz',
    sequences=X_seq,
    labels=y_seq,
    patient_ids=np.array(sequence_seqns),
)
print(f"Cached {len(X_seq)} sequences to models/lstm_sequences_cache_v2.npz")
'''.split('\n')]

    # Cell 7 markdown
    nb['cells'][7]['source'] = [
        '## Section 5b — Train/Test Split + Truncated Augmentation (v2)\n\n',
        'Fix 3: train on **raw** sequences with `Masking(mask_value=0.0)`.\n',
        'Padding = all four nutrient features are 0 (occasion alone 0 does not mask).\n',
        'Truncated augmentation zero-pads all 5 features in future slots.\n',
    ]

    # Cell 8 - split and augment (raw, no scaler before model)
    nb['cells'][8]['source'] = [line + '\n' for line in '''y_encoded = np.array([RISK_ENCODE[r] for r in y_seq])

print("Label encoding:")
for i, cls in enumerate(RISK_CLASSES):
    print(f"  {i} = {cls}")

n_patients, n_steps, n_features = X_seq.shape
print(f"Sequence shape (raw): {X_seq.shape}")

# Scaler for artifact export only (nutrients); model trains on raw values with Masking
scaler = StandardScaler()
nutrient_mask = X_seq.reshape(-1, n_features)[:, :4].any(axis=1)
scaler.fit(X_seq.reshape(-1, n_features)[nutrient_mask][:, :4])

X_train_raw, X_test_raw, y_train_enc, y_test_enc, idx_train, idx_test = train_test_split(
    X_seq, y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded,
)

print(f"Training sequences (raw): {len(X_train_raw)}")
print(f"Test sequences (raw): {len(X_test_raw)}")


def augment_with_truncated_sequences(X, y, n_steps=6):
    X_augmented = [X.copy()]
    y_augmented = [y.copy()]
    for cutoff in range(1, n_steps):
        X_truncated = X.copy()
        X_truncated[:, cutoff:, :] = 0
        X_augmented.append(X_truncated)
        y_augmented.append(y.copy())
    return np.concatenate(X_augmented, axis=0), np.concatenate(y_augmented, axis=0)


X_train_aug_raw, y_train_aug = augment_with_truncated_sequences(
    X_train_raw, y_train_enc, n_steps=n_steps
)
print(f"Augmented training sequences (raw): {X_train_aug_raw.shape}")

# Class weights for imbalanced labels
classes = np.unique(y_train_aug)
class_weights_arr = compute_class_weight('balanced', classes=classes, y=y_train_aug)
class_weight = {int(c): float(w) for c, w in zip(classes, class_weights_arr)}
print("Class weights:", class_weight)

X_train = X_train_aug_raw
X_test = X_test_raw
y_train = y_train_aug
y_test = y_test_enc
'''.split('\n')]

    # Cells 9-11 - architecture / skip tuning
    nb['cells'][9]['source'] = [
        '## Section 6 — LSTM v2 Architecture (Masking)\n\n',
        '| Layer | Type | Units |\n|---|---|---|\n',
        '| 1 | Masking | mask_value=0.0 |\n',
        '| 2 | LSTM | 64 |\n',
        '| 3 | Dropout | 0.3 |\n',
        '| 4 | LSTM | 32 |\n',
        '| 5 | Dropout | 0.3 |\n',
        '| 6 | Dense | 3 (softmax) |\n',
    ]
    nb['cells'][10]['source'] = [
        '## Section 6b — Skipped Hyperparameter Tuning\n\n',
        'v2 uses the fixed architecture from the improvement spec (Masking + 64/32 LSTM).\n',
    ]
    nb['cells'][11]['source'] = ['print("Skipping hyperparameter tuning for v2 — using fixed Masking architecture.")\n']

    # Cell 13 - train v2 model
    nb['cells'][13]['source'] = [line + '\n' for line in '''print("Training LSTM v2 with Masking layer")
print("=" * 45)

tf.random.set_seed(RANDOM_STATE)

model = Sequential([
    Masking(mask_value=0.0, input_shape=(n_steps, n_features)),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(3, activation='softmax'),
], name='GuidaPlate_LSTM_v2')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()
print(f"Total parameters: {model.count_params():,}")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=str(MODEL_DIR / 'lstm_v2_best.keras'), monitor='val_loss', save_best_only=True, verbose=0),
]

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1,
)

best_epoch = int(np.argmin(history.history['val_loss']) + 1)
print("Training complete.")
print(f"Best epoch: {best_epoch}")
'''.split('\n')]

    # Cell 14 markdown - simplify
    nb['cells'][14]['source'] = [
        '## Section 7b — v2 Evaluation Report\n\n',
        'Full metrics on held-out test set vs original LSTM benchmarks.\n',
    ]

    # Cell 15 - skip old model comparison, just note
    nb['cells'][15]['source'] = ['print("v2 evaluation in Section 8 below.")\n']

    # Cell 17 - full evaluation
    nb['cells'][17]['source'] = [line + '\n' for line in '''from sklearn.metrics import classification_report

y_pred_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

v2_accuracy = accuracy_score(y_test, y_pred)
v2_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
v2_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr', average='macro')

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
print("Confusion matrix:")
print(pd.DataFrame(cm, index=RISK_CLASSES, columns=RISK_CLASSES))
print()
print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=RISK_CLASSES, zero_division=0))
print(f"ROC-AUC (macro, ovr): {v2_auc:.4f}")

high_idx = RISK_ENCODE['HIGH']
mod_idx = RISK_ENCODE['MODERATE']
high_tp = ((y_test == high_idx) & (y_pred == high_idx)).sum()
high_fn = ((y_test == high_idx) & (y_pred != high_idx)).sum()
v2_high_sensitivity = high_tp / (high_tp + high_fn) if (high_tp + high_fn) > 0 else 0.0

mod_tp = ((y_test == mod_idx) & (y_pred == mod_idx)).sum()
mod_fn = ((y_test == mod_idx) & (y_pred != mod_idx)).sum()
v2_mod_sensitivity = mod_tp / (mod_tp + mod_fn) if (mod_tp + mod_fn) > 0 else 0.0

print(f"HIGH sensitivity: {v2_high_sensitivity:.4f} ({v2_high_sensitivity*100:.2f}%)")
print(f"MODERATE sensitivity: {v2_mod_sensitivity:.4f} ({v2_mod_sensitivity*100:.2f}%)")

ORIGINAL = {
    'accuracy': 0.9071,
    'f1': 0.9052,
    'auc': 0.9825,
    'high_sensitivity': 0.9360,
    'mod_sensitivity': None,
}

print()
print("=" * 55)
print("METRIC COMPARISON — Original vs V2")
print("=" * 55)
print(f"{'Metric':<18} | {'Original':>10} | {'V2':>10}")
print("-" * 45)
print(f"{'Accuracy':<18} | {ORIGINAL['accuracy']*100:>9.2f}% | {v2_accuracy*100:>9.2f}%")
print(f"{'F1 (weighted)':<18} | {ORIGINAL['f1']:>10.4f} | {v2_f1:>10.4f}")
print(f"{'AUC':<18} | {ORIGINAL['auc']:>10.4f} | {v2_auc:>10.4f}")
print(f"{'HIGH sensitivity':<18} | {ORIGINAL['high_sensitivity']*100:>9.2f}% | {v2_high_sensitivity*100:>9.2f}%")
orig_mod = ORIGINAL['mod_sensitivity']
orig_mod_str = f"{orig_mod*100:.2f}%" if orig_mod is not None else "N/A"
print(f"{'MOD sensitivity':<18} | {orig_mod_str:>10} | {v2_mod_sensitivity*100:>9.2f}%")
'''.split('\n')]

    # Cell 19 - update viz for 5 features (nutrients only)
    src19 = ''.join(nb['cells'][19]['source'])
    src19 = src19.replace("nutrient_names = ['Potassium (mg)', 'Phosphorus (mg)', 'Protein (g/kg)', 'Sodium (mg)']",
                          "nutrient_names = ['Potassium (mg)', 'Phosphorus (mg)', 'Protein (g/kg)', 'Sodium (mg)']")
    src19 = src19.replace("FIG_DIR / '13_lstm_training_history.png'", "FIG_DIR / '13_lstm_v2_training_history.png'")
    src19 = src19.replace("FIG_DIR / '14_lstm_confusion_matrix.png'", "FIG_DIR / '14_lstm_v2_confusion_matrix.png'")
    src19 = src19.replace("FIG_DIR / '15_lstm_meal_patterns.png'", "FIG_DIR / '15_lstm_v2_meal_patterns.png'")
    nb['cells'][19]['source'] = [line + '\n' for line in src19.split('\n')]

    # Cell 20 markdown
    nb['cells'][20]['source'] = ['## Section 10 — Save v2 Artifacts (if metrics >= original)\n']

    # Cell 21 - decision cell
    nb['cells'][21]['source'] = [line + '\n' for line in '''label_encoder = {'classes': RISK_CLASSES, 'encode': RISK_ENCODE}

v2_better = (
    v2_accuracy >= ORIGINAL['accuracy'] and
    v2_f1 >= ORIGINAL['f1'] and
    v2_auc >= ORIGINAL['auc'] and
    v2_high_sensitivity >= ORIGINAL['high_sensitivity']
)

if v2_better:
    model.save(MODEL_DIR / 'lstm_v2.keras')
    joblib.dump(scaler, MODEL_DIR / 'lstm_v2_scaler.pkl')
    joblib.dump(label_encoder, MODEL_DIR / 'lstm_v2_label_encoder.pkl')
    print("✅ V2 is better — artifacts saved")
    print("Next step: update backend/config.py to point to v2 artifacts")
else:
    print("❌ V2 did not improve on original")
    print("Keeping original model in production")
    print("Document v2 attempt in Chapter 5")

metrics_df = pd.DataFrame([{
    'model': 'LSTM v2',
    'accuracy': round(v2_accuracy, 4),
    'f1_score': round(v2_f1, 4),
    'auc_roc': round(v2_auc, 4),
    'high_risk_sensitivity': round(v2_high_sensitivity, 4),
    'moderate_sensitivity': round(v2_mod_sensitivity, 4),
    'sequence_length': n_steps,
    'features_per_step': n_features,
}])
metrics_df.to_csv(STATS_DIR / '07_lstm_v2_metrics.csv', index=False)
print(f"Metrics saved: {STATS_DIR / '07_lstm_v2_metrics.csv'}")
print("Original models/lstm_final.keras untouched.")
'''.split('\n')]

    return nb


def main():
    p03 = ROOT / 'notebooks' / '03b_labels_v2_sequence_aware.ipynb'
    p05 = ROOT / 'notebooks' / '05b_lstm_v2_improved.ipynb'

    nb03 = json.loads(p03.read_text())
    nb05 = json.loads(p05.read_text())

    patch_03b(nb03)
    patch_05b(nb05)

    p03.write_text(json.dumps(nb03, indent=1))
    p05.write_text(json.dumps(nb05, indent=1))
    print('Patched notebooks OK')


if __name__ == '__main__':
    main()
