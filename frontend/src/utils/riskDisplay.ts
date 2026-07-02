export interface RiskDisplay {
  label: string;
  sublabel: string;
  icon: string;
  color: string;
  bgColor: string;
  borderColor: string;
  advice: string;
}

export function getRiskDisplay(risk: string): RiskDisplay {
  const r = risk?.toUpperCase();

  if (r === 'HIGH') {
    return {
      label: 'Reduce Intake',
      sublabel: 'This meal exceeds your recommended limits',
      icon: '🔴',
      color: 'text-red-700',
      bgColor: 'bg-red-50',
      borderColor: 'border-red-200',
      advice:
        'Consider swapping some foods for lower-nutrient alternatives shown below.',
    };
  }

  if (r === 'MODERATE') {
    return {
      label: 'Caution',
      sublabel: 'Some nutrients are approaching your daily limit',
      icon: '🟡',
      color: 'text-amber-700',
      bgColor: 'bg-amber-50',
      borderColor: 'border-amber-200',
      advice: 'Keep your next meal light to stay within your daily limits.',
    };
  }

  return {
    label: 'Safe',
    sublabel: 'This meal is within your recommended limits',
    icon: '✅',
    color: 'text-green-700',
    bgColor: 'bg-green-50',
    borderColor: 'border-green-200',
    advice: 'Great choice for your kidney health.',
  };
}

export function getRiskColor(risk: string): string {
  const r = risk?.toUpperCase();
  if (r === 'HIGH') return 'text-red-600';
  if (r === 'MODERATE') return 'text-amber-600';
  return 'text-green-600';
}

/** Display kidney stage codes as patient-friendly labels (G3b → Stage 3b). */
export function formatStageDisplay(stage: string | null | undefined): string {
  if (!stage) return '';
  const cleaned = stage.trim();
  if (/^stage\s/i.test(cleaned)) return cleaned;
  if (/^G\d/i.test(cleaned)) {
    return `Stage ${cleaned.slice(1)}`;
  }
  return cleaned;
}

/** Weekly / timeline risk labels (High → High intake, etc.). */
export function getWeeklyRiskLabel(risk: string): string {
  const r = risk?.toUpperCase();
  if (r === 'HIGH') return 'High intake';
  if (r === 'MODERATE') return 'Caution';
  if (r === 'LOW') return 'Within limits';
  return risk;
}

/** Patient-facing nutrient status badges (internal logic unchanged). */
export function displayNutrientStatus(
  status: 'Safe' | 'Near limit' | 'Exceeded',
): string {
  if (status === 'Safe') return 'Within limit';
  if (status === 'Near limit') return 'Near limit';
  return 'Over limit';
}

export function displayBudgetStatus(status: 'over' | 'near' | 'ok'): string {
  if (status === 'over') return '↑ Over limit';
  if (status === 'near') return '~ Near limit';
  return '✓ Within limit';
}

const STAGE_DISPLAY_LABELS: Record<string, string> = {
  G2: 'Stage 2',
  G3a: 'Stage 3a',
  G3b: 'Stage 3b',
  G4: 'Stage 4',
};

export function stageOptionLabel(stageCode: string): string {
  return STAGE_DISPLAY_LABELS[stageCode] ?? formatStageDisplay(stageCode);
}

// Stage ordering for comparison
const STAGE_ORDER: Record<string, number> = {
  G2: 1,
  G3a: 2,
  G3b: 3,
  G4: 4,
  G5: 5,
};

const STAGES_BY_NUM = ['G2', 'G3a', 'G3b', 'G4', 'G5'] as const;

// Map numeric safe range to stage labels (1 = G2, 2 = G3a, 3 = G3b, …)
const RANGE_TO_STAGES: Record<string, string[]> = {
  '1': ['G2'],
  '1-2': ['G2', 'G3a'],
  '1-3': ['G2', 'G3a', 'G3b'],
  '1-4': ['G2', 'G3a', 'G3b', 'G4'],
  '1-5': ['G2', 'G3a', 'G3b', 'G4', 'G5'],
  '2-3': ['G3a', 'G3b'],
  '2-4': ['G3a', 'G3b', 'G4'],
  '3-4': ['G3b', 'G4'],
  '3-5': ['G3b', 'G4', 'G5'],
  '4-5': ['G4', 'G5'],
};

function stagesInRange(stageRange: string): string[] | null {
  const normalized = stageRange.trim();
  if (RANGE_TO_STAGES[normalized]) {
    return RANGE_TO_STAGES[normalized];
  }
  const match = normalized.match(/^(\d+)-(\d+)$/);
  if (match) {
    const low = parseInt(match[1], 10) - 1;
    const high = parseInt(match[2], 10) - 1;
    if (low >= 0 && high < STAGES_BY_NUM.length && low <= high) {
      return STAGES_BY_NUM.slice(low, high + 1);
    }
  }
  return null;
}

export interface StageSafety {
  isSafe: boolean;
  label: string;
  color: string;
  detail: string;
}

export function getStageSafety(
  stageRange: string | null,
  patientStage: string,
  potassiumMg: number,
  phosphorusMg: number,
  proteinG: number,
): StageSafety {
  const stage = patientStage || 'G3b';

  const LIMITS_PER_100G: Record<string, { k: number; p: number; pro: number }> = {
    G2: { k: 525, p: 150, pro: 13 },
    G3a: { k: 450, p: 120, pro: 9 },
    G3b: { k: 450, p: 120, pro: 9 },
    G4: { k: 375, p: 105, pro: 8 },
  };

  const limits = LIMITS_PER_100G[stage] || LIMITS_PER_100G.G3b;

  const kSafe = potassiumMg <= limits.k;
  const pSafe = phosphorusMg <= limits.p;
  const proSafe = proteinG <= limits.pro;

  let rangeSafe = true;
  if (stageRange) {
    const allowed = stagesInRange(stageRange);
    if (allowed) {
      rangeSafe = allowed.includes(stage);
    }
  }

  const overallSafe = kSafe && pSafe && proSafe && rangeSafe;

  const concerns: string[] = [];
  if (!kSafe) concerns.push('high potassium');
  if (!pSafe) concerns.push('high phosphorus');
  if (!proSafe) concerns.push('high protein');
  if (!rangeSafe) concerns.push(`not recommended for ${formatStageDisplay(stage)}`);

  const stageDisplay = formatStageDisplay(stage);

  if (overallSafe) {
    return {
      isSafe: true,
      label: `✓ Safe for ${stageDisplay}`,
      color: 'text-green-600',
      detail: `This food fits within ${stageDisplay} limits`,
    };
  }

  if (concerns.length > 0) {
    return {
      isSafe: false,
      label: `⚠ Limit for ${stageDisplay}`,
      color: 'text-amber-600',
      detail: `Use small portions — ${concerns.join(', ')}`,
    };
  }

  return {
    isSafe: false,
    label: `✗ Avoid for ${stageDisplay}`,
    color: 'text-red-600',
    detail: `Not recommended for ${stageDisplay}`,
  };
}

/** Whether a food's numeric stage range includes the patient stage (range only). */
export function isFoodInStageRange(stageRange: string | null, patientStage: string): boolean {
  if (!stageRange) return true;
  const allowed = stagesInRange(stageRange);
  if (!allowed) return true;
  return allowed.includes(patientStage || 'G3b');
}
