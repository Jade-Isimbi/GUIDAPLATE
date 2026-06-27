#!/usr/bin/env python3
"""Run controlled LSTM ablations A2, B2, C2."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / 'models'
STATS_DIR = ROOT / 'outputs' / 'stats'

RANDOM_STATE = 42
RISK_CLASSES = ['LOW', 'MODERATE', 'HIGH']
RISK_ENCODE = {c: i for i, c in enumerate(RISK_CLASSES)}
N_STEPS = 6
OCCASION_BY_SLOT = {0: 0.00, 1: 0.33, 2: 0.67, 3: 0.00, 4: 0.33, 5: 0.67}

ORIGINAL = {
    'accuracy': 0.9071, 'f1': 0.9052, 'auc': 0.9825, 'high_sensitivity': 0.9360,
}
ALL3 = {
    'accuracy': 0.7213, 'f1': 0.7494, 'auc': 0.8499,
    'high_sensitivity': 0.8000, 'mod_sensitivity': 0.4359,
}


def augment_with_truncated_sequences(X, y, n_steps=6):
    X_augmented, y_augmented = [X.copy()], [y.copy()]
    for cutoff in range(1, n_steps):
        X_truncated = X.copy()
        X_truncated[:, cutoff:, :] = 0
        X_augmented.append(X_truncated)
        y_augmented.append(y.copy())
    return np.concatenate(X_augmented, axis=0), np.concatenate(y_augmented, axis=0)


def scale_sequences(X, scaler, n_features, n_steps=6):
    n = X.shape[0]
    return scaler.transform(X.reshape(-1, n_features)).reshape(n, n_steps, n_features)


def zero_out_scaled_padding(X_scaled, X_raw):
    X_out = X_scaled.copy()
    pad_mask = (X_raw == 0).all(axis=-1)
    for i in range(X_out.shape[0]):
        X_out[i, pad_mask[i], :] = 0.0
    return X_out


def controlled_preprocess(X_seq, y_labels, n_features, apply_mask_zero=False):
    y_encoded = np.array([RISK_ENCODE[r] for r in y_labels])
    y_cat = to_categorical(y_encoded, num_classes=3)
    scaler = StandardScaler()
    scaler.fit(X_seq.reshape(-1, n_features))
    X_train_raw, X_test_raw, y_train_cat, _, y_train_enc, y_test_enc = train_test_split(
        X_seq, y_cat, y_encoded,
        test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded,
    )
    X_train_aug_raw, y_train_aug_cat = augment_with_truncated_sequences(
        X_train_raw, y_train_cat, n_steps=N_STEPS,
    )
    X_train_scaled = scale_sequences(X_train_aug_raw, scaler, n_features)
    X_test_scaled = scale_sequences(X_test_raw, scaler, n_features)
    if apply_mask_zero:
        X_train_scaled = zero_out_scaled_padding(X_train_scaled, X_train_aug_raw)
        X_test_scaled = zero_out_scaled_padding(X_test_scaled, X_test_raw)
    y_train_int = np.argmax(y_train_aug_cat, axis=1)
    return X_train_scaled, X_test_scaled, y_train_int, y_test_enc


def evaluate_metrics(y_true, y_pred, y_prob):
    high_idx, mod_idx = RISK_ENCODE['HIGH'], RISK_ENCODE['MODERATE']
    high_tp = ((y_true == high_idx) & (y_pred == high_idx)).sum()
    high_fn = ((y_true == high_idx) & (y_pred != high_idx)).sum()
    mod_tp = ((y_true == mod_idx) & (y_pred == mod_idx)).sum()
    mod_fn = ((y_true == mod_idx) & (y_pred != mod_idx)).sum()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'auc': roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'),
        'high_sensitivity': high_tp / (high_tp + high_fn) if (high_tp + high_fn) else 0.0,
        'mod_sensitivity': mod_tp / (mod_tp + mod_fn) if (mod_tp + mod_fn) else 0.0,
        '_y_test': y_true, '_y_pred': y_pred,
    }


def build_model(n_features, use_masking=False):
    if use_masking:
        return Sequential([
            Masking(mask_value=0.0, input_shape=(N_STEPS, n_features)),
            LSTM(64, return_sequences=True), Dropout(0.3),
            LSTM(32), Dropout(0.3), Dense(3, activation='softmax'),
        ])
    return Sequential([
        LSTM(64, return_sequences=True, input_shape=(N_STEPS, n_features)),
        Dropout(0.3), LSTM(32), Dropout(0.3), Dense(3, activation='softmax'),
    ])


def train_ablation(X_train, X_test, y_train, y_test, model, save_name):
    tf.random.set_seed(RANDOM_STATE)
    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy', metrics=['accuracy'],
    )
    model.fit(
        X_train, y_train, epochs=50, batch_size=32, validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)],
        class_weight=class_weight, verbose=1,
    )
    model.save(MODEL_DIR / save_name)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    return evaluate_metrics(y_test, y_pred, y_prob)


def decision(name, m):
    ok = all(m[k] >= ORIGINAL[k] for k in ['accuracy', 'f1', 'auc', 'high_sensitivity'])
    tradeoff = m['mod_sensitivity'] > 0.20 and (ORIGINAL['accuracy'] - m['accuracy']) < 0.05
    if ok:
        print(f"✅ {name} improves or matches original — production candidate")
        return 'PASS'
    print(f"❌ {name} does not improve original")
    for k, label, pct in [
        ('accuracy', 'accuracy', True), ('f1', 'F1', False), ('auc', 'AUC', False),
        ('high_sensitivity', 'HIGH sens', True),
    ]:
        if m[k] < ORIGINAL[k]:
            if pct:
                print(f"   ↓ {label} {m[k]*100:.2f}% vs {ORIGINAL[k]*100:.2f}% ({(m[k]-ORIGINAL[k])*100:+.2f}pp)")
            else:
                print(f"   ↓ {label} {m[k]:.4f} vs {ORIGINAL[k]:.4f} ({m[k]-ORIGINAL[k]:+.4f})")
    if tradeoff:
        print(f"⚠ TRADE-OFF — review manually: MOD sens {m['mod_sensitivity']*100:.1f}%")
        return 'TRADE-OFF'
    return 'FAIL'


def main():
    cache = np.load(MODEL_DIR / 'lstm_sequences_cache.npz', allow_pickle=True)
    X_orig, patient_ids = cache['sequences'], cache['patient_ids']
    labels_orig = pd.read_csv(STATS_DIR / '05_risk_labels.csv')
    labels_v2 = pd.read_csv(STATS_DIR / '05_risk_labels_v2.csv')
    y_orig = np.array([dict(zip(labels_orig.SEQN, labels_orig.risk_label))[int(s)] for s in patient_ids])
    y_v2 = np.array([dict(zip(labels_v2.SEQN, labels_v2.risk_label))[int(s)] for s in patient_ids])

    X_occ = np.zeros((len(X_orig), N_STEPS, 5))
    X_occ[:, :, :4] = X_orig
    for slot, val in OCCASION_BY_SLOT.items():
        X_occ[:, slot, 4] = val

    print('=== A2 ===')
    Xt, Xe, yt, ye = controlled_preprocess(X_orig, y_orig, 4, apply_mask_zero=True)
    m_a2 = train_ablation(Xt, Xe, yt, ye, build_model(4, True), 'lstm_ablation_a2.keras')
    d_a2 = decision('A2', m_a2)

    print('=== B2 ===')
    Xt, Xe, yt, ye = controlled_preprocess(X_occ, y_orig, 5, apply_mask_zero=False)
    m_b2 = train_ablation(Xt, Xe, yt, ye, build_model(5, False), 'lstm_ablation_b2.keras')
    d_b2 = decision('B2', m_b2)

    print('=== C2 ===')
    Xt, Xe, yt, ye = controlled_preprocess(X_orig, y_v2, 4, apply_mask_zero=False)
    m_c2 = train_ablation(Xt, Xe, yt, ye, build_model(4, False), 'lstm_ablation_c2.keras')
    d_c2 = decision('C2', m_c2)

    rows = [
        {'experiment': 'Original', **ORIGINAL, 'mod_sensitivity': None, 'decision': 'baseline'},
        {'experiment': 'A2-Mask', **{k: m_a2[k] for k in ['accuracy','f1','auc','high_sensitivity','mod_sensitivity']}, 'decision': d_a2},
        {'experiment': 'B2-Occ', **{k: m_b2[k] for k in ['accuracy','f1','auc','high_sensitivity','mod_sensitivity']}, 'decision': d_b2},
        {'experiment': 'C2-Label', **{k: m_c2[k] for k in ['accuracy','f1','auc','high_sensitivity','mod_sensitivity']}, 'decision': d_c2},
        {'experiment': 'All-3', **ALL3, 'decision': 'FAIL'},
    ]
    pd.DataFrame(rows).to_csv(STATS_DIR / '08_lstm_ablation_controlled.csv', index=False)

    print('\nCONTROLLED ABLATION COMPARISON TABLE')
    print(f"{'Metric':<12} | {'Original':>9} | {'A2-Mask':>7} | {'B2-Occ':>7} | {'C2-Label':>8} | {'All-3':>7}")
    for label, key, pct in [
        ('Accuracy', 'accuracy', True), ('F1', 'f1', False), ('AUC', 'auc', False),
        ('HIGH sens', 'high_sensitivity', True), ('MOD sens', 'mod_sensitivity', True),
    ]:
        o = ORIGINAL.get(key, 0)
        vals = [m_a2[key], m_b2[key], m_c2[key], ALL3[key]]
        if pct:
            o_s = '~0%' if key == 'mod_sensitivity' else f"{o*100:.2f}%"
            print(f"{label:<12} | {o_s:>9} | " + ' | '.join(f"{v*100:>6.2f}%" for v in vals))
        else:
            print(f"{label:<12} | {o:>9.4f} | " + ' | '.join(f"{v:>7.4f}" for v in vals))
    print(f"\nSaved {STATS_DIR / '08_lstm_ablation_controlled.csv'}")


if __name__ == '__main__':
    main()
