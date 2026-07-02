/**
 * GuidaPlate Clinical Constants
 * Mirrors backend/clinical_constants.py
 * Source: KDOQI 2020 / KDIGO 2024
 */

export const KDOQI_DAILY_LIMITS = {
  G2: { potassium: 3500, phosphorus: 1000, protein: 0.8, sodium: 2300 },
  G3a: { potassium: 3000, phosphorus: 800, protein: 0.6, sodium: 2300 },
  G3b: { potassium: 3000, phosphorus: 800, protein: 0.6, sodium: 2300 },
  G4: { potassium: 2500, phosphorus: 700, protein: 0.55, sodium: 2300 },
} as const;

export const EGFR_RANGES = {
  G2: '60–89',
  G3a: '45–59',
  G3b: '30–44',
  G4: '15–29',
} as const;

export const STAGE_DISPLAY = {
  G2: 'Stage 2',
  G3a: 'Stage 3a',
  G3b: 'Stage 3b',
  G4: 'Stage 4',
} as const;

export type CkdStage = keyof typeof KDOQI_DAILY_LIMITS;

export function getLimits(stage: string) {
  return KDOQI_DAILY_LIMITS[stage as CkdStage] ?? KDOQI_DAILY_LIMITS.G3b;
}

export function formatLimit(value: number): string {
  return value.toLocaleString('en-US');
}
