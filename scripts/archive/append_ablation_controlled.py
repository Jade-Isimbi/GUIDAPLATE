#!/usr/bin/env python3
"""Append Ablation Study v2 — Controlled section to 05b notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB_PATH = ROOT / 'notebooks' / '05b_lstm_v2_improved.ipynb'

MD = '''## Ablation Study v2 — Controlled

Reruns A/B/C ablations with **the same preprocessing as notebook 05** (StandardScaler + truncated-sequence augmentation on train only).

### Reference — exact cells from notebook 05

**Scaler + split + augmentation + scale** (notebook 05, Section 5b):

```python
y_encoded = np.array([RISK_ENCODE[r] for r in y_seq])
y_cat = to_categorical(y_encoded, num_classes=3)

n_patients, n_steps, n_features = X_seq.shape
X_flat = X_seq.reshape(-1, n_features)
scaler = StandardScaler()
scaler.fit(X_flat)  # notebook 05 fits on ALL sequences before split

# Split on RAW sequences; scale after augmentation
X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
    X_seq, y_cat, y_encoded,
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded,
)

def augment_with_truncated_sequences(X, y, n_steps=6):
    """For each sequence, add truncated+padded versions (cutoffs 1..5), same label."""
    X_augmented = [X.copy()]
    y_augmented = [y.copy()]
    for cutoff in range(1, n_steps):
        X_truncated = X.copy()
        X_truncated[:, cutoff:, :] = 0
        X_augmented.append(X_truncated)
        y_augmented.append(y.copy())
    return np.concatenate(X_augmented, axis=0), np.concatenate(y_augmented, axis=0)

X_train_aug_raw, y_train_aug = augment_with_truncated_sequences(
    X_train_raw, y_train, n_steps=n_steps
)

def scale_sequences(X, scaler):
    n = X.shape[0]
    flat = X.reshape(-1, n_features)
    return scaler.transform(flat).reshape(n, n_steps, n_features)

X_train = scale_sequences(X_train_aug_raw, scaler)
X_test = scale_sequences(X_test_raw, scaler)
y_train = y_train_aug
```

**Note:** Notebook 05 fits `StandardScaler` on the **full** `X_seq` flat array before the train/test split (not train-only). Controlled ablations replicate this exactly.

### A2 / B2 / C2 changes

| Exp | Input | Labels | Model change |
|-----|-------|--------|--------------|
| A2 | (n,6,4) | original | + Masking after explicit zero-out of padded scaled slots |
| B2 | (n,6,5) | original | 5 features, no masking |
| C2 | (n,6,4) | v2 | new labels only |
'''

CODE = r'''from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Masking
from tensorflow.keras.utils import to_categorical

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
N_STEPS = 6


def augment_with_truncated_sequences(X, y, n_steps=6):
    """Exact copy from notebook 05 — truncated+padded versions, same label."""
    X_augmented = [X.copy()]
    y_augmented = [y.copy()]
    for cutoff in range(1, n_steps):
        X_truncated = X.copy()
        X_truncated[:, cutoff:, :] = 0
        X_augmented.append(X_truncated)
        y_augmented.append(y.copy())
    return np.concatenate(X_augmented, axis=0), np.concatenate(y_augmented, axis=0)


def scale_sequences(X, scaler, n_features, n_steps=6):
    """Exact copy from notebook 05."""
    n = X.shape[0]
    flat = X.reshape(-1, n_features)
    return scaler.transform(flat).reshape(n, n_steps, n_features)


def zero_out_scaled_padding(X_scaled, X_raw):
    """A2: after scaling, force padded timesteps back to 0.0 for Masking layer."""
    X_out = X_scaled.copy()
    pad_mask = (X_raw == 0).all(axis=-1)
    for i in range(X_out.shape[0]):
        X_out[i, pad_mask[i], :] = 0.0
    return X_out


def controlled_preprocess(X_seq, y_labels, n_features, apply_mask_zero=False):
    """Notebook-05 preprocessing: scaler on full X_seq, split, augment train, scale."""
    y_encoded = np.array([RISK_ENCODE[r] for r in y_labels])
    y_cat = to_categorical(y_encoded, num_classes=3)

    scaler = StandardScaler()
    scaler.fit(X_seq.reshape(-1, n_features))

    X_train_raw, X_test_raw, y_train_cat, y_test_cat, y_train_enc, y_test_enc = train_test_split(
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
    classes = np.unique(y_train_int)
    cw = compute_class_weight('balanced', classes=classes, y=y_train_int)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    return X_train_scaled, X_test_scaled, y_train_int, y_test_enc, class_weight


def evaluate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    high_idx, mod_idx = RISK_ENCODE['HIGH'], RISK_ENCODE['MODERATE']
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


def build_model(n_features, use_masking=False):
    layers = []
    if use_masking:
        layers.append(Masking(mask_value=0.0, input_shape=(N_STEPS, n_features)))
    else:
        layers.append(LSTM(64, return_sequences=True, input_shape=(N_STEPS, n_features)))
        layers += [Dropout(0.3), LSTM(32), Dropout(0.3), Dense(3, activation='softmax')]
        return Sequential(layers)
    layers += [
        LSTM(64, return_sequences=True), Dropout(0.3),
        LSTM(32), Dropout(0.3), Dense(3, activation='softmax'),
    ]
    return Sequential(layers)


def train_ablation_controlled(X_train, X_test, y_train, y_test, model, save_name):
    tf.random.set_seed(RANDOM_STATE)
    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    early_stop = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=0,
    )
    model.fit(
        X_train, y_train, epochs=50, batch_size=32, validation_split=0.2,
        callbacks=[early_stop], class_weight=class_weight, verbose=0,
    )
    model.save(MODEL_DIR / save_name)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    return evaluate_metrics(y_test, y_pred, y_prob)


def ablation_decision_controlled(name, metrics):
    ok = (
        metrics['accuracy'] >= ORIGINAL['accuracy'] and
        metrics['f1'] >= ORIGINAL['f1'] and
        metrics['auc'] >= ORIGINAL['auc'] and
        metrics['high_sensitivity'] >= ORIGINAL['high_sensitivity']
    )
    tradeoff = (
        metrics['mod_sensitivity'] > 0.20 and
        (ORIGINAL['accuracy'] - metrics['accuracy']) < 0.05
    )
    if ok:
        print(f"✅ {name} improves or matches original — production candidate")
        return 'PASS'
    print(f"❌ {name} does not improve original")
    if metrics['accuracy'] < ORIGINAL['accuracy']:
        print(f"   ↓ accuracy {metrics['accuracy']*100:.2f}% vs {ORIGINAL['accuracy']*100:.2f}% ({(metrics['accuracy']-ORIGINAL['accuracy'])*100:+.2f}pp)")
    if metrics['f1'] < ORIGINAL['f1']:
        print(f"   ↓ F1 {metrics['f1']:.4f} vs {ORIGINAL['f1']:.4f} ({metrics['f1']-ORIGINAL['f1']:+.4f})")
    if metrics['auc'] < ORIGINAL['auc']:
        print(f"   ↓ AUC {metrics['auc']:.4f} vs {ORIGINAL['auc']:.4f} ({metrics['auc']-ORIGINAL['auc']:+.4f})")
    if metrics['high_sensitivity'] < ORIGINAL['high_sensitivity']:
        print(f"   ↓ HIGH sens {metrics['high_sensitivity']*100:.2f}% vs {ORIGINAL['high_sensitivity']*100:.2f}% ({(metrics['high_sensitivity']-ORIGINAL['high_sensitivity'])*100:+.2f}pp)")
    if tradeoff:
        print(f"⚠ TRADE-OFF — review manually: MOD sens {metrics['mod_sensitivity']*100:.1f}% with accuracy drop {(ORIGINAL['accuracy']-metrics['accuracy'])*100:.1f}pp")
        return 'TRADE-OFF'
    return 'FAIL'


def print_metrics(label, m):
    print(f"\n{'='*55}\n{label}\n{'='*55}")
    print(f"Accuracy: {m['accuracy']*100:.2f}%  F1: {m['f1']:.4f}  AUC: {m['auc']:.4f}")
    print(f"HIGH sens: {m['high_sensitivity']*100:.2f}%  MOD sens: {m['mod_sensitivity']*100:.2f}%")
    print(classification_report(m['_y_test'], m['_y_pred'], labels=[0,1,2], target_names=RISK_CLASSES, zero_division=0))


# ── Load data ─────────────────────────────────────────────────────────────
cache = np.load(MODEL_DIR / 'lstm_sequences_cache.npz', allow_pickle=True)
X_orig = cache['sequences']
patient_ids = cache['patient_ids']

labels_orig = pd.read_csv(STATS_DIR / '05_risk_labels.csv')
labels_v2 = pd.read_csv(STATS_DIR / '05_risk_labels_v2.csv')
map_orig = dict(zip(labels_orig['SEQN'], labels_orig['risk_label']))
map_v2 = dict(zip(labels_v2['SEQN'], labels_v2['risk_label']))
y_orig = np.array([map_orig[int(s)] for s in patient_ids])
y_v2 = np.array([map_v2[int(s)] for s in patient_ids])

X_occ = np.zeros((len(X_orig), N_STEPS, 5))
X_occ[:, :, :4] = X_orig
for slot, val in OCCASION_BY_SLOT.items():
    X_occ[:, slot, 4] = val

print(f"Original cache: {X_orig.shape}  |  Occasion cache: {X_occ.shape}")

# ── A2: Masking + controlled preprocessing ────────────────────────────────
Xt, Xe, yt, ye, _ = controlled_preprocess(X_orig, y_orig, n_features=4, apply_mask_zero=True)
m_a2 = train_ablation_controlled(Xt, Xe, yt, ye, build_model(4, use_masking=True), 'lstm_ablation_a2.keras')
print_metrics('A2 — Masking + controlled preprocessing', m_a2)
d_a2 = ablation_decision_controlled('A2', m_a2)

# ── B2: Occasion + controlled preprocessing ───────────────────────────────
Xt, Xe, yt, ye, _ = controlled_preprocess(X_occ, y_orig, n_features=5, apply_mask_zero=False)
m_b2 = train_ablation_controlled(Xt, Xe, yt, ye, build_model(5, use_masking=False), 'lstm_ablation_b2.keras')
print_metrics('B2 — Occasion + controlled preprocessing', m_b2)
d_b2 = ablation_decision_controlled('B2', m_b2)

# ── C2: v2 labels + controlled preprocessing ──────────────────────────────
Xt, Xe, yt, ye, _ = controlled_preprocess(X_orig, y_v2, n_features=4, apply_mask_zero=False)
m_c2 = train_ablation_controlled(Xt, Xe, yt, ye, build_model(4, use_masking=False), 'lstm_ablation_c2.keras')
print_metrics('C2 — Sequence labels + controlled preprocessing', m_c2)
d_c2 = ablation_decision_controlled('C2', m_c2)

# ── Comparison table ────────────────────────────────────────────────────
rows = [
    {'experiment': 'Original', 'accuracy': ORIGINAL['accuracy'], 'f1': ORIGINAL['f1'],
     'auc': ORIGINAL['auc'], 'high_sensitivity': ORIGINAL['high_sensitivity'],
     'mod_sensitivity': None, 'decision': 'baseline'},
    {'experiment': 'A2-Mask', **{k: m_a2[k] for k in ['accuracy','f1','auc','high_sensitivity','mod_sensitivity']}, 'decision': d_a2},
    {'experiment': 'B2-Occ', **{k: m_b2[k] for k in ['accuracy','f1','auc','high_sensitivity','mod_sensitivity']}, 'decision': d_b2},
    {'experiment': 'C2-Label', **{k: m_c2[k] for k in ['accuracy','f1','auc','high_sensitivity','mod_sensitivity']}, 'decision': d_c2},
    {'experiment': 'All-3', **{k: ALL3[k] for k in ['accuracy','f1','auc','high_sensitivity','mod_sensitivity']}, 'decision': 'FAIL'},
]
df_ctrl = pd.DataFrame(rows)

print('\n' + '=' * 72)
print('CONTROLLED ABLATION COMPARISON TABLE')
print('=' * 72)
print(f"{'Metric':<12} | {'Original':>9} | {'A2-Mask':>7} | {'B2-Occ':>7} | {'C2-Label':>8} | {'All-3':>7}")
print('-' * 72)
print(f"{'Accuracy':<12} | {ORIGINAL['accuracy']*100:>8.2f}% | {m_a2['accuracy']*100:>6.2f}% | {m_b2['accuracy']*100:>6.2f}% | {m_c2['accuracy']*100:>7.2f}% | {ALL3['accuracy']*100:>6.2f}%")
print(f"{'F1':<12} | {ORIGINAL['f1']:>9.4f} | {m_a2['f1']:>7.4f} | {m_b2['f1']:>7.4f} | {m_c2['f1']:>8.4f} | {ALL3['f1']:>7.4f}")
print(f"{'AUC':<12} | {ORIGINAL['auc']:>9.4f} | {m_a2['auc']:>7.4f} | {m_b2['auc']:>7.4f} | {m_c2['auc']:>8.4f} | {ALL3['auc']:>7.4f}")
print(f"{'HIGH sens':<12} | {ORIGINAL['high_sensitivity']*100:>8.2f}% | {m_a2['high_sensitivity']*100:>6.2f}% | {m_b2['high_sensitivity']*100:>6.2f}% | {m_c2['high_sensitivity']*100:>7.2f}% | {ALL3['high_sensitivity']*100:>6.2f}%")
print(f"{'MOD sens':<12} | {'~0%':>9} | {m_a2['mod_sensitivity']*100:>6.2f}% | {m_b2['mod_sensitivity']*100:>6.2f}% | {m_c2['mod_sensitivity']*100:>7.2f}% | {ALL3['mod_sensitivity']*100:>6.2f}%")

df_ctrl.to_csv(STATS_DIR / '08_lstm_ablation_controlled.csv', index=False)
print(f"\nSaved: {STATS_DIR / '08_lstm_ablation_controlled.csv'}")
print('models/lstm_final.keras untouched.')
'''


def main():
    nb = json.loads(NB_PATH.read_text())
    nb['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in MD.strip().split('\n')],
    })
    nb['cells'].append({
        'cell_type': 'code',
        'metadata': {},
        'source': [line + '\n' for line in CODE.strip().split('\n')],
        'outputs': [],
        'execution_count': None,
    })
    NB_PATH.write_text(json.dumps(nb, indent=1))
    print('Controlled ablation section appended.')


if __name__ == '__main__':
    main()
