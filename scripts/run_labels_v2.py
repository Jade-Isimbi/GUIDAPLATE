#!/usr/bin/env python3
"""Run Fix 1 from notebook 03b — generate 05_risk_labels_v2.csv."""
from pathlib import Path
import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STATS_DIR = ROOT / 'outputs' / 'stats'

STAGE_ORDER = ['G2', 'G3a', 'G3b', 'G4']
KDOQI = {
    'G2':  {'potassium': 3500, 'phosphorus': 1000, 'protein_per_kg': 0.8, 'sodium': 2300},
    'G3a': {'potassium': 3000, 'phosphorus': 800,  'protein_per_kg': 0.6, 'sodium': 2300},
    'G3b': {'potassium': 3000, 'phosphorus': 800,  'protein_per_kg': 0.6, 'sodium': 2300},
    'G4':  {'potassium': 2500, 'phosphorus': 700,  'protein_per_kg': 0.55, 'sodium': 2300},
}


def assign_risk_label(row):
    stage = row['ckd_stage']
    if stage not in KDOQI:
        return None
    limits = KDOQI[stage]
    exceeded = 0
    if pd.notna(row['potassium']) and row['potassium'] > limits['potassium']:
        exceeded += 1
    if pd.notna(row['phosphorus']) and row['phosphorus'] > limits['phosphorus']:
        exceeded += 1
    if pd.notna(row['protein_per_kg']) and row['protein_per_kg'] > limits['protein_per_kg']:
        exceeded += 1
    if pd.notna(row['sodium']) and row['sodium'] > limits['sodium']:
        exceeded += 1
    if exceeded >= 2:
        return 'HIGH'
    if exceeded == 1:
        return 'MODERATE'
    return 'LOW'


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
    slots = np.arange(6)
    k, p, pr, na = meal_seq[:, 0], meal_seq[:, 1], meal_seq[:, 2], meal_seq[:, 3]
    slopes = [np.polyfit(slots, vals, 1)[0] for vals in (k, p, pr, na)]
    finals = [k[5], p[5], pr[5], na[5]]
    limit_vals = [limits['potassium'], limits['phosphorus'], limits['protein_per_kg'], limits['sodium']]
    escalating = 0
    for slope, final, limit in zip(slopes, finals, limit_vals):
        if slope > 0 and final > 0.5 * limit:
            escalating += 1
    return escalating


def assign_risk_label_v2(row, meal_seq: np.ndarray):
    primary = assign_risk_label(row)
    if primary is None:
        return None
    stage = row['ckd_stage']
    if stage not in KDOQI:
        return primary
    escalating = count_escalating(meal_seq, KDOQI[stage])
    if primary == 'LOW' and escalating >= 2:
        return 'MODERATE'
    if primary == 'MODERATE' and escalating >= 3:
        return 'HIGH'
    return primary


def main():
    df = pd.read_csv(ROOT / 'data' / 'processed' / 'ckd_cohort_final.csv')
    df['risk_label'] = df.apply(assign_risk_label, axis=1)

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

    meal_seq_by_seqn = {}
    for seqn in cohort_seqns:
        weight = df.loc[df['SEQN'] == seqn, 'weight_kg']
        if weight.empty or pd.isna(weight.iloc[0]) or weight.iloc[0] <= 0:
            continue
        w = float(weight.iloc[0])
        seq = np.zeros((6, 4))
        for _, meal in meal_nutrients[meal_nutrients['SEQN'] == seqn].iterrows():
            slot = int(meal['meal_slot'])
            if 0 <= slot <= 5:
                seq[slot, 0] = meal['potassium']
                seq[slot, 1] = meal['phosphorus']
                seq[slot, 2] = meal['protein'] / w
                seq[slot, 3] = meal['sodium']
        meal_seq_by_seqn[int(seqn)] = seq

    v2_labels = []
    for _, row in df.iterrows():
        seqn = int(row['SEQN'])
        meal_seq = meal_seq_by_seqn.get(seqn, np.zeros((6, 4)))
        v2_labels.append(assign_risk_label_v2(row, meal_seq))
    df['risk_label_v2'] = v2_labels

    orig = df['risk_label']
    v2 = df['risk_label_v2']
    changed = (orig != v2) & orig.notna() & v2.notna()
    n_changed = int(changed.sum())
    n_valid = int(orig.notna().sum())
    pct_changed = n_changed / n_valid * 100 if n_valid else 0

    print('Original:', orig.value_counts().to_dict())
    print('v2:', v2.value_counts().to_dict())
    print(f'Changed: {n_changed}/{n_valid} ({pct_changed:.1f}%)')
    if pct_changed > 20:
        print('WARNING: >20% labels changed')

    out = df[['SEQN', 'ckd_stage', 'potassium', 'phosphorus', 'protein_per_kg', 'sodium', 'risk_label_v2']]
    out = out.rename(columns={'risk_label_v2': 'risk_label'})
    out_path = STATS_DIR / '05_risk_labels_v2.csv'
    out.to_csv(out_path, index=False)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
