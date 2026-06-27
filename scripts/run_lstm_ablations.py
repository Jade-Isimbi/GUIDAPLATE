#!/usr/bin/env python3
"""Run LSTM ablation experiments A, B, C."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking
from tensorflow.keras.models import Sequential

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / 'models'
STATS_DIR = ROOT / 'outputs' / 'stats'

RANDOM_STATE = 42
RISK_CLASSES = ['LOW', 'MODERATE', 'HIGH']
RISK_ENCODE = {c: i for i, c in enumerate(RISK_CLASSES)}

ORIGINAL = {
    'accuracy': 0.9071,
    'f1': 0.9052,
    'auc': 0.9825,
    'high_sensitivity': 0.9360,
}
ALL3 = {
    'accuracy': 0.7213,
    'f1': 0.7494,
    'auc': 0.8499,
    'high_sensitivity': 0.8000,
    'mod_sensitivity': 0.4359,
}
OCCASION_BY_SLOT = {0: 0.00, 1: 0.33, 2: 0.67, 3: 0.00, 4: 0.33, 5: 0.67}


def evaluate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    high_idx = RISK_ENCODE['HIGH']
    mod_idx = RISK_ENCODE['MODERATE']
    high_tp = ((y_true == high_idx) & (y_pred == high_idx)).sum()
    high_fn = ((y_true == high_idx) & (y_pred != high_idx)).sum()
    high_sens = high_tp / (high_tp + high_fn) if (high_tp + high_fn) > 0 else 0.0
    mod_tp = ((y_true == mod_idx) & (y_pred == mod_idx)).sum()
    mod_fn = ((y_true == mod_idx) & (y_pred != mod_idx)).sum()
    mod_sens = mod_tp / (mod_tp + mod_fn) if (mod_tp + mod_fn) > 0 else 0.0
    return {
        'accuracy': acc, 'f1': f1, 'auc': auc,
        'high_sensitivity': high_sens, 'mod_sensitivity': mod_sens,
        '_y_test': y_true, '_y_pred': y_pred,
    }


def print_ablation_metrics(label, metrics):
    print(f"\n{'='*55}")
    print(label)
    print('='*55)
    print(f"Accuracy:         {metrics['accuracy']*100:.2f}%")
    print(f"F1 (weighted):    {metrics['f1']:.4f}")
    print(f"AUC (macro):      {metrics['auc']:.4f}")
    print(f"HIGH sensitivity: {metrics['high_sensitivity']*100:.2f}%")
    print(f"MOD sensitivity:  {metrics['mod_sensitivity']*100:.2f}%")
    print(classification_report(
        metrics['_y_test'], metrics['_y_pred'],
        labels=[0, 1, 2], target_names=RISK_CLASSES, zero_division=0,
    ))


def ablation_decision(name, metrics):
    ok = (
        metrics['accuracy'] >= ORIGINAL['accuracy'] and
        metrics['f1'] >= ORIGINAL['f1'] and
        metrics['auc'] >= ORIGINAL['auc'] and
        metrics['high_sensitivity'] >= ORIGINAL['high_sensitivity']
    )
    if ok:
        print(f"✅ {name} improves or matches original")
    else:
        print(f"❌ {name} does not improve original")
        if metrics['accuracy'] < ORIGINAL['accuracy']:
            print(f"   ↓ accuracy {metrics['accuracy']*100:.2f}% vs {ORIGINAL['accuracy']*100:.2f}% ({(metrics['accuracy']-ORIGINAL['accuracy'])*100:+.2f}pp)")
        if metrics['f1'] < ORIGINAL['f1']:
            print(f"   ↓ F1 {metrics['f1']:.4f} vs {ORIGINAL['f1']:.4f} ({metrics['f1']-ORIGINAL['f1']:+.4f})")
        if metrics['auc'] < ORIGINAL['auc']:
            print(f"   ↓ AUC {metrics['auc']:.4f} vs {ORIGINAL['auc']:.4f} ({metrics['auc']-ORIGINAL['auc']:+.4f})")
        if metrics['high_sensitivity'] < ORIGINAL['high_sensitivity']:
            print(
                f"   ↓ HIGH sens {metrics['high_sensitivity']*100:.2f}% vs "
                f"{ORIGINAL['high_sensitivity']*100:.2f}% "
                f"({(metrics['high_sensitivity']-ORIGINAL['high_sensitivity'])*100:+.2f}pp)"
            )


def train_ablation(X, y_labels, model, save_name, seed=42):
    y_encoded = np.array([RISK_ENCODE[r] for r in y_labels])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded,
    )
    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    tf.random.set_seed(seed)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    model.fit(
        X_train, y_train, epochs=50, batch_size=32, validation_split=0.2,
        callbacks=[early_stop], class_weight=class_weight, verbose=1,
    )
    model.save(MODEL_DIR / save_name)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    return evaluate_metrics(y_test, y_pred, y_prob)


def main():
    cache_orig = np.load(MODEL_DIR / 'lstm_sequences_cache.npz', allow_pickle=True)
    X_orig = cache_orig['sequences']
    patient_ids = cache_orig['patient_ids']

    labels_orig_df = pd.read_csv(STATS_DIR / '05_risk_labels.csv')
    y_orig = np.array([dict(zip(labels_orig_df['SEQN'], labels_orig_df['risk_label']))[int(s)] for s in patient_ids])

    labels_v2_df = pd.read_csv(STATS_DIR / '05_risk_labels_v2.csv')
    y_v2 = np.array([dict(zip(labels_v2_df['SEQN'], labels_v2_df['risk_label']))[int(s)] for s in patient_ids])

    print(f"Original cache shape: {X_orig.shape}")

    model_a = Sequential([
        Masking(mask_value=0.0, input_shape=(6, 4)),
        LSTM(64, return_sequences=True), Dropout(0.3), LSTM(32), Dropout(0.3),
        Dense(3, activation='softmax'),
    ])
    metrics_a = train_ablation(X_orig, y_orig, model_a, 'lstm_ablation_a.keras')
    print_ablation_metrics('A — Masking only', metrics_a)
    ablation_decision('A', metrics_a)

    X_occ = np.zeros((len(X_orig), 6, 5))
    X_occ[:, :, :4] = X_orig
    for slot, val in OCCASION_BY_SLOT.items():
        X_occ[:, slot, 4] = val

    model_b = Sequential([
        LSTM(64, return_sequences=True, input_shape=(6, 5)),
        Dropout(0.3), LSTM(32), Dropout(0.3), Dense(3, activation='softmax'),
    ])
    metrics_b = train_ablation(X_occ, y_orig, model_b, 'lstm_ablation_b.keras')
    print_ablation_metrics('B — Occasion feature only', metrics_b)
    ablation_decision('B', metrics_b)

    model_c = Sequential([
        LSTM(64, return_sequences=True, input_shape=(6, 4)),
        Dropout(0.3), LSTM(32), Dropout(0.3), Dense(3, activation='softmax'),
    ])
    metrics_c = train_ablation(X_orig, y_v2, model_c, 'lstm_ablation_c.keras')
    print_ablation_metrics('C — Sequence labels only', metrics_c)
    ablation_decision('C', metrics_c)

    rows = [
        {'experiment': 'Original', 'accuracy': ORIGINAL['accuracy'], 'f1': ORIGINAL['f1'],
         'auc': ORIGINAL['auc'], 'high_sensitivity': ORIGINAL['high_sensitivity'], 'mod_sensitivity': None},
        {'experiment': 'A-Mask', **{k: metrics_a[k] for k in ['accuracy', 'f1', 'auc', 'high_sensitivity', 'mod_sensitivity']}},
        {'experiment': 'B-Occ', **{k: metrics_b[k] for k in ['accuracy', 'f1', 'auc', 'high_sensitivity', 'mod_sensitivity']}},
        {'experiment': 'C-Label', **{k: metrics_c[k] for k in ['accuracy', 'f1', 'auc', 'high_sensitivity', 'mod_sensitivity']}},
        {'experiment': 'All-3', **{k: ALL3[k] for k in ['accuracy', 'f1', 'auc', 'high_sensitivity', 'mod_sensitivity']}},
    ]
    ablation_df = pd.DataFrame(rows)

    print('\n' + '=' * 72)
    print('ABLATION COMPARISON TABLE')
    print('=' * 72)
    print(f"{'Metric':<14} | {'Original':>9} | {'A-Mask':>7} | {'B-Occ':>7} | {'C-Label':>7} | {'All-3':>7}")
    print('-' * 72)
    for label, key, pct in [
        ('Accuracy', 'accuracy', True),
        ('F1 weighted', 'f1', False),
        ('AUC', 'auc', False),
        ('HIGH sens', 'high_sensitivity', True),
        ('MOD sens', 'mod_sensitivity', True),
    ]:
        o = ORIGINAL.get(key, 0) if key != 'mod_sensitivity' else 0
        a, b, c, d = metrics_a[key], metrics_b[key], metrics_c[key], ALL3[key]
        if pct:
            o_s = f"{o*100:.2f}%" if key != 'mod_sensitivity' else '~0%'
            print(f"{label:<14} | {o_s:>9} | {a*100:>6.2f}% | {b*100:>6.2f}% | {c*100:>6.2f}% | {d*100:>6.2f}%")
        else:
            print(f"{label:<14} | {o:>9.4f} | {a:>7.4f} | {b:>7.4f} | {c:>7.4f} | {d:>7.4f}")

    ablation_df.to_csv(STATS_DIR / '08_lstm_ablation_results.csv', index=False)
    print(f"\nSaved: {STATS_DIR / '08_lstm_ablation_results.csv'}")


if __name__ == '__main__':
    main()
