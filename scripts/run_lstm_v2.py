#!/usr/bin/env python3
"""Run Fix 2+3 from notebook 05b — build cache v2 and train LSTM v2."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking
from tensorflow.keras.models import Sequential

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / 'models'
STATS_DIR = ROOT / 'outputs' / 'stats'

RANDOM_STATE = 42
TEST_SIZE = 0.2
RISK_CLASSES = ['LOW', 'MODERATE', 'HIGH']
RISK_ENCODE = {c: i for i, c in enumerate(RISK_CLASSES)}
OCCASION_BY_SLOT = {0: 0.00, 1: 0.33, 2: 0.67, 3: 0.00, 4: 0.33, 5: 0.67}

ORIGINAL = {
    'accuracy': 0.9071,
    'f1': 0.9052,
    'auc': 0.9825,
    'high_sensitivity': 0.9360,
}


def load_iff(path: Path) -> pd.DataFrame:
    try:
        import pyreadstat
        df, _ = pyreadstat.read_xpt(str(path))
        return df
    except Exception:
        return pd.read_sas(path, format='xport')


def pick_col(df: pd.DataFrame, candidates):
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


def build_sequences():
    cohort = pd.read_csv(ROOT / 'data' / 'processed' / 'ckd_cohort_final.csv')
    labels = pd.read_csv(STATS_DIR / '05_risk_labels_v2.csv')
    cohort = cohort.merge(labels[['SEQN', 'risk_label']], on='SEQN', how='inner')
    ckd_seqns = set(cohort['SEQN'])

    iff1 = load_iff(ROOT / 'data' / 'raw' / 'nhanes' / 'DR1IFF_J.xpt')
    iff2 = load_iff(ROOT / 'data' / 'raw' / 'nhanes' / 'DR2IFF_J.xpt')

    iff1_ckd = standardize_iff(iff1[iff1['SEQN'].isin(ckd_seqns)].copy(), day=1)
    iff1_ckd['meal_slot'] = iff1_ckd['meal_code'].apply(lambda x: map_meal_slot(1, x))
    iff2_ckd = standardize_iff(iff2[iff2['SEQN'].isin(ckd_seqns)].copy(), day=2)
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

    sequence_data, sequence_labels, sequence_seqns = [], [], []
    for _, patient in cohort[['SEQN', 'weight_kg', 'risk_label']].iterrows():
        seqn = patient['SEQN']
        weight = patient['weight_kg']
        risk = patient['risk_label']
        if pd.isna(weight) or weight <= 0 or risk not in RISK_CLASSES:
            continue
        seq = np.zeros((6, 5))
        for slot in range(6):
            seq[slot, 4] = OCCASION_BY_SLOT[slot]
        for _, meal in meal_nutrients[meal_nutrients['SEQN'] == seqn].iterrows():
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

    X_seq = np.array(sequence_data)
    y_seq = np.array(sequence_labels)
    assert X_seq.shape[1:] == (6, 5), X_seq.shape
    np.savez(
        MODEL_DIR / 'lstm_sequences_cache_v2.npz',
        sequences=X_seq,
        labels=y_seq,
        patient_ids=np.array(sequence_seqns),
    )
    print(f'Cache shape: {X_seq.shape}')
    return X_seq, y_seq


def augment_with_truncated_sequences(X, y, n_steps=6):
    X_augmented, y_augmented = [X.copy()], [y.copy()]
    for cutoff in range(1, n_steps):
        X_truncated = X.copy()
        X_truncated[:, cutoff:, :] = 0
        X_augmented.append(X_truncated)
        y_augmented.append(y.copy())
    return np.concatenate(X_augmented, axis=0), np.concatenate(y_augmented, axis=0)


def train_and_evaluate(X_seq, y_seq):
    y_encoded = np.array([RISK_ENCODE[r] for r in y_seq])
    n_steps, n_features = X_seq.shape[1], X_seq.shape[2]

    scaler = StandardScaler()
    nutrient_mask = X_seq.reshape(-1, n_features)[:, :4].any(axis=1)
    scaler.fit(X_seq.reshape(-1, n_features)[nutrient_mask][:, :4])

    X_train_raw, X_test_raw, y_train_enc, y_test_enc = train_test_split(
        X_seq, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded,
    )
    X_train, y_train = augment_with_truncated_sequences(X_train_raw, y_train_enc, n_steps)
    X_test, y_test = X_test_raw, y_test_enc

    classes = np.unique(y_train)
    class_weights_arr = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, class_weights_arr)}

    tf.random.set_seed(RANDOM_STATE)
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(n_steps, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(3, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=str(MODEL_DIR / 'lstm_v2_best.keras'),
            monitor='val_loss', save_best_only=True, verbose=0,
        ),
    ]
    model.fit(
        X_train, y_train,
        epochs=50, batch_size=32, validation_split=0.2,
        callbacks=callbacks, class_weight=class_weight, verbose=1,
    )

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    v2_accuracy = accuracy_score(y_test, y_pred)
    v2_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    v2_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr', average='macro')

    high_idx, mod_idx = RISK_ENCODE['HIGH'], RISK_ENCODE['MODERATE']
    high_tp = ((y_test == high_idx) & (y_pred == high_idx)).sum()
    high_fn = ((y_test == high_idx) & (y_pred != high_idx)).sum()
    v2_high_sensitivity = high_tp / (high_tp + high_fn) if (high_tp + high_fn) > 0 else 0.0
    mod_tp = ((y_test == mod_idx) & (y_pred == mod_idx)).sum()
    mod_fn = ((y_test == mod_idx) & (y_pred != mod_idx)).sum()
    v2_mod_sensitivity = mod_tp / (mod_tp + mod_fn) if (mod_tp + mod_fn) > 0 else 0.0

    print('\nConfusion matrix:')
    print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]), index=RISK_CLASSES, columns=RISK_CLASSES))
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=RISK_CLASSES, zero_division=0))
    print(f'ROC-AUC (macro): {v2_auc:.4f}')
    print(f'HIGH sensitivity: {v2_high_sensitivity*100:.2f}%')
    print(f'MOD sensitivity: {v2_mod_sensitivity*100:.2f}%')

    print('\n' + '=' * 55)
    print('METRIC COMPARISON — Original vs V2')
    print('=' * 55)
    print(f"{'Metric':<18} | {'Original':>10} | {'V2':>10}")
    print('-' * 45)
    print(f"{'Accuracy':<18} | {ORIGINAL['accuracy']*100:>9.2f}% | {v2_accuracy*100:>9.2f}%")
    print(f"{'F1 (weighted)':<18} | {ORIGINAL['f1']:>10.4f} | {v2_f1:>10.4f}")
    print(f"{'AUC':<18} | {ORIGINAL['auc']:>10.4f} | {v2_auc:>10.4f}")
    print(f"{'HIGH sensitivity':<18} | {ORIGINAL['high_sensitivity']*100:>9.2f}% | {v2_high_sensitivity*100:>9.2f}%")
    print(f"{'MOD sensitivity':<18} | {'N/A':>10} | {v2_mod_sensitivity*100:>9.2f}%")

    v2_better = (
        v2_accuracy >= ORIGINAL['accuracy'] and
        v2_f1 >= ORIGINAL['f1'] and
        v2_auc >= ORIGINAL['auc'] and
        v2_high_sensitivity >= ORIGINAL['high_sensitivity']
    )

    label_encoder = {'classes': RISK_CLASSES, 'encode': RISK_ENCODE}
    if v2_better:
        model.save(MODEL_DIR / 'lstm_v2.keras')
        joblib.dump(scaler, MODEL_DIR / 'lstm_v2_scaler.pkl')
        joblib.dump(label_encoder, MODEL_DIR / 'lstm_v2_label_encoder.pkl')
        print('\n✅ V2 is better — artifacts saved')
    else:
        print('\n❌ V2 did not improve on original')
        print('Keeping original model in production')

    pd.DataFrame([{
        'model': 'LSTM v2',
        'accuracy': round(v2_accuracy, 4),
        'f1_score': round(v2_f1, 4),
        'auc_roc': round(v2_auc, 4),
        'high_risk_sensitivity': round(v2_high_sensitivity, 4),
        'moderate_sensitivity': round(v2_mod_sensitivity, 4),
    }]).to_csv(STATS_DIR / '07_lstm_v2_metrics.csv', index=False)


def main():
    X_seq, y_seq = build_sequences()
    train_and_evaluate(X_seq, y_seq)


if __name__ == '__main__':
    main()
