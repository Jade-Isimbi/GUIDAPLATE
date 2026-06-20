import { useState, useRef, useEffect } from 'react';
import { AlertTriangle, CheckCircle2, XCircle, Zap, Info, ChevronRight, RotateCcw, Search, Trash2, Plus, Minus, Mic, MicOff } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, Cell } from 'recharts';
import {
  CKDStage,
  FOODS,
  STAGE_THRESHOLDS,
  type Food,
  getDefaultGrams,
  potassiumColor,
} from '../../data/foodDatabase';

const API_BASE_URL = 'http://localhost:8000/api';

const NO_SUBSTITUTES_MSG =
  'No category-matched lower-potassium substitutes found for this meal.';

const saveFoodLog = async (food: Food) => {
  const token = localStorage.getItem('guidaplate_token');
  if (!token) return;

  try {
    await fetch(`${API_BASE_URL}/patient/food-log`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        food_name: food.english,
        category: food.category,
        stage_safe_range: food.ckd_stage_safe,
      }),
    });
  } catch (err) {
    console.error('Failed to save food log:', err);
  }
};

const saveRiskAssessment = async (
  riskLevel: string,
  riskScore: number,
  nutrientTotals: Record<string, number>,
) => {
  const token = localStorage.getItem('guidaplate_token');
  if (!token) return;

  try {
    await fetch(`${API_BASE_URL}/patient/risk-assessment`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        risk_level: riskLevel,
        risk_score: riskScore,
        nutrients_summary: JSON.stringify(nutrientTotals),
      }),
    });
  } catch (err) {
    console.error('Failed to save risk assessment:', err);
  }
};

interface RiskAssessmentProps {
  isDark: boolean;
  theme: Record<string, string>;
  initialBodyWeight?: number;
}

const STAGE_META: Record<CKDStage, { label: string; gfr: string }> = {
  G2: { label: 'Mildly decreased', gfr: '60–89' },
  G3a: { label: 'Mild to moderate decrease', gfr: '45–59' },
  G3b: { label: 'Moderate to severe decrease', gfr: '30–44' },
  G4: { label: 'Severe decrease', gfr: '15–29' },
};

function getStageLimits(stage: CKDStage, bodyWeightKg: number) {
  const t = STAGE_THRESHOLDS[stage];
  return {
    potassium: t.potassium,
    phosphorus: t.phosphorus,
    protein: t.protein * bodyWeightKg,
    sodium: t.sodium,
    label: STAGE_META[stage].label,
    gfr: STAGE_META[stage].gfr,
  };
}

const stageCovers = (ckd_stage_safe: string, stage: string): boolean => {
  const stageNum: Record<string, number> = {
    G1: 1, G2: 2, G3a: 3, G3b: 3, G4: 4, G5: 5,
  };
  const n = stageNum[stage] ?? 2;
  if (ckd_stage_safe === '1') return n === 1;
  const parts = ckd_stage_safe.split('-');
  if (parts.length !== 2) return true;
  const min = parseInt(parts[0], 10);
  const max = parseInt(parts[1], 10);
  return n >= min && n <= max;
};

function substituteCategories(category: string): string[] {
  const rules: Record<string, string[]> = {
    Starch: ['Starch'],
    Grain: ['Grain'],
    Meat: ['Meat', 'Fish'],
    Fish: ['Fish', 'Meat'],
    Vegetable: ['Vegetable'],
    Fruit: ['Fruit'],
    Legume: ['Legume'],
    Dairy: ['Dairy'],
    Egg: ['Egg', 'Dairy'],
    'Fat/Oil': ['Fat/Oil'],
    Beverage: ['Beverage'],
  };
  return rules[category] ?? [category];
}

interface MealFoodItem { food: Food; grams: number }

function getFoodRiskScore(entry: MealFoodItem, limits: ReturnType<typeof getStageLimits>): number {
  const scale = entry.grams / 100;
  return Math.max(
    (entry.food.potassium_mg * scale) / limits.potassium,
    (entry.food.phosphorus_mg * scale) / limits.phosphorus,
    (entry.food.protein_g * scale) / limits.protein,
    (entry.food.sodium_mg * scale) / limits.sodium,
  );
}

interface FoodSubstitution { riskyFood: Food; substitutes: Food[] }

function getSmartSubstitutions(
  mealFoods: MealFoodItem[],
  stage: CKDStage,
  limits: ReturnType<typeof getStageLimits>,
): FoodSubstitution[] {
  if (mealFoods.length === 0) return [];

  const mealIds = new Set(mealFoods.map((e) => e.food.id));
  const ranked = [...mealFoods].sort((a, b) => getFoodRiskScore(b, limits) - getFoodRiskScore(a, limits));
  const maxScore = getFoodRiskScore(ranked[0], limits);
  const riskyEntries = ranked.filter((e) => getFoodRiskScore(e, limits) >= maxScore * 0.25);

  const seen = new Set<number>();
  const riskyFoods: Food[] = [];
  for (const entry of riskyEntries) {
    if (!seen.has(entry.food.id)) {
      seen.add(entry.food.id);
      riskyFoods.push(entry.food);
    }
  }

  return riskyFoods
    .map((riskyFood) => {
      const allowedCats = substituteCategories(riskyFood.category);

      // "Other" category foods (336 USDA Foundation Foods imports)
      // have no clinically meaningful cross-category substitution -
      // skip substitution entirely, matching backend recommender.py
      if (riskyFood.category === 'Other') {
        return { riskyFood, substitutes: [] };
      }

      const substitutes = FOODS
        .filter((f) => f.id !== riskyFood.id)
        .filter((f) => !mealIds.has(f.id))
        .filter((f) => allowedCats.includes(f.category))
        .filter((f) => f.potassium_mg < riskyFood.potassium_mg)
        .filter((f) => stageCovers(f.ckd_stage_safe, stage))
        .sort((a, b) => a.potassium_mg - b.potassium_mg)
        .slice(0, 3);
      return { riskyFood, substitutes };
    })
    .filter((s) => s.substitutes.length > 0);
}

function sumMealNutrients(foods: MealFoodItem[]) {
  return foods.reduce(
    (acc, { food, grams }) => {
      const scale = grams / 100;
      return {
        potassium: acc.potassium + food.potassium_mg * scale,
        phosphorus: acc.phosphorus + food.phosphorus_mg * scale,
        protein: acc.protein + food.protein_g * scale,
        sodium: acc.sodium + food.sodium_mg * scale,
      };
    },
    { potassium: 0, phosphorus: 0, protein: 0, sodium: 0 },
  );
}

function lookupFoodByEnglish(name: string): Food | undefined {
  const key = name.trim().toLowerCase();
  return FOODS.find((f) => f.english.toLowerCase() === key);
}

function buildBreakdown(
  mealTotals: ReturnType<typeof sumMealNutrients>,
  limits: ReturnType<typeof getStageLimits>,
  exceeded?: string[],
  nearLimit?: string[],
): Record<string, BreakdownEntry> {
  const nutrientKey: Record<string, string> = {
    potassium: 'Potassium',
    phosphorus: 'Phosphorus',
    protein: 'Protein',
    sodium: 'Sodium',
  };

  const breakdown: Record<string, BreakdownEntry> = {
    Potassium:  { value: Math.round(mealTotals.potassium),  limit: limits.potassium,  pct: 0, status: 'Safe' },
    Phosphorus: { value: Math.round(mealTotals.phosphorus), limit: limits.phosphorus, pct: 0, status: 'Safe' },
    Protein:    { value: +mealTotals.protein.toFixed(1),    limit: limits.protein,    pct: 0, status: 'Safe' },
    Sodium:     { value: Math.round(mealTotals.sodium),     limit: limits.sodium,     pct: 0, status: 'Safe' },
  };

  for (const key of Object.keys(breakdown)) {
    const b = breakdown[key];
    b.pct = Math.min(150, (b.value / b.limit) * 100);
    b.status = b.pct > 100 ? 'Exceeded' : b.pct > 80 ? 'Near limit' : 'Safe';
  }

  if (exceeded || nearLimit) {
    for (const [apiName, displayName] of Object.entries(nutrientKey)) {
      if (exceeded?.includes(apiName)) breakdown[displayName].status = 'Exceeded';
      else if (nearLimit?.includes(apiName)) breakdown[displayName].status = 'Near limit';
    }
  }

  return breakdown;
}

function scoreFromBreakdown(breakdown: Record<string, BreakdownEntry>): number {
  const avgPct = Object.values(breakdown).reduce((a, b) => a + b.pct, 0) / 4;
  return Math.min(100, Math.round(avgPct));
}

function dailyTotalColor(value: number, limit: number): string {
  const pct = (value / limit) * 100;
  if (pct > 100) return '#E74C3C';
  if (pct >= 70) return '#F39C12';
  return '#27AE60';
}

const RECOMMENDATIONS: Record<string, string[]> = {
  HIGH:     ['Immediately reduce high-risk nutrient intake', 'Consult your nephrologist or dietitian as soon as possible', 'Leach vegetables by boiling and discarding cooking water', 'Avoid processed and canned foods (high sodium)', 'Choose lower-potassium Rwandan staples like rice or cabbage'],
  MODERATE: ['Monitor portion sizes carefully at each meal', 'Substitute high-potassium foods with lower-potassium alternatives', 'Read nutrition labels to track phosphorus additives', 'Spread protein intake evenly across all meals', 'Schedule a dietary review with your care team this week'],
  LOW:      ['Continue your current dietary pattern', 'Maintain a daily food log to track trends', 'Stay well hydrated within your fluid limit', 'Regular follow-up assessments are recommended monthly', 'Good work — consistency is key in CKD management'],
};

const NUTRIENT_COLORS: Record<string, string> = { Potassium: '#2E86AB', Phosphorus: '#F39C12', Protein: '#27AE60', Sodium: '#E74C3C' };

const RISK_CFG = {
  HIGH:     { color: '#E74C3C', bg: 'rgba(231,76,60,0.1)',  border: 'rgba(231,76,60,0.35)',  icon: XCircle,       label: 'High Risk',     desc: 'This meal significantly exceeds safe nutrient limits for your CKD stage' },
  MODERATE: { color: '#F39C12', bg: 'rgba(243,156,18,0.1)', border: 'rgba(243,156,18,0.35)', icon: AlertTriangle, label: 'Moderate Risk', desc: 'Some nutrients in this meal are near or above recommended thresholds' },
  LOW:      { color: '#27AE60', bg: 'rgba(39,174,96,0.1)',  border: 'rgba(39,174,96,0.35)',  icon: CheckCircle2,  label: 'Low Risk',      desc: 'This meal is within safe nutrient ranges for your CKD stage' },
};

type MealEntry = {
  mealType: 'Breakfast' | 'Lunch' | 'Dinner' | 'Snack';
  foods: MealFoodItem[];
  assessedAt: string;
  riskLevel: 'LOW' | 'MODERATE' | 'HIGH';
};

interface BreakdownEntry { value: number; limit: number; pct: number; status: 'Safe' | 'Near limit' | 'Exceeded' }
interface ResultState {
  level: 'LOW' | 'MODERATE' | 'HIGH';
  score: number;
  breakdown: Record<string, BreakdownEntry>;
  assessedFoods: MealFoodItem[];
  substitutions: FoodSubstitution[];
}

const MEAL_TYPES: MealEntry['mealType'][] = ['Breakfast', 'Lunch', 'Dinner', 'Snack'];

export function RiskAssessment({ isDark, theme, initialBodyWeight }: RiskAssessmentProps) {
  const [stage,           setStage]           = useState<CKDStage>('G3a');
  const [bodyWeightKg,    setBodyWeightKg]    = useState<number>(initialBodyWeight ?? 65);
  const [entries,         setEntries]         = useState<MealFoodItem[]>([]);
  const [result,          setResult]          = useState<ResultState | null>(null);
  const [error,           setError]           = useState('');
  const [search,          setSearch]          = useState('');
  const [showDrop,        setShowDrop]        = useState(false);
  const [isListening,     setIsListening]     = useState(false);
  const [voiceSupported,  setVoiceSupported]  = useState(true);
  const [dailyMeals,      setDailyMeals]      = useState<MealEntry[]>([]);
  const [currentMealType, setCurrentMealType] = useState<MealEntry['mealType']>('Breakfast');
  const [dayResetMsg,     setDayResetMsg]     = useState('');
  const [apiStatus,       setApiStatus]       = useState<'unknown' | 'connected' | 'unavailable'>('unknown');
  const [usingLiveModel,  setUsingLiveModel]  = useState<boolean>(false);
  const [modelConfidence, setModelConfidence] = useState<number | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(2000) })
      .then((res) => setApiStatus(res.ok ? 'connected' : 'unavailable'))
      .catch(() => setApiStatus('unavailable'));
  }, []);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setShowDrop(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  useEffect(() => {
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setVoiceSupported(false);
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setSearch(transcript);
      setShowDrop(true);
      setIsListening(false);
    };

    recognition.onerror = () => {
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current = recognition;

    return () => {
      recognition.abort();
    };
  }, []);

  const toggleVoiceInput = () => {
    if (!recognitionRef.current) return;

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      setSearch('');
      recognitionRef.current.start();
      setIsListening(true);
    }
  };

  const q = search.trim().toLowerCase();
  const visibleFoods = (q === ''
    ? FOODS
    : FOODS.filter((food) =>
        food.english.toLowerCase().includes(q) ||
        food.french.toLowerCase().includes(q) ||
        food.kinyarwanda.toLowerCase().includes(q)
      )
  ).filter((f) => !entries.find((e) => e.food.id === f.id));

  const addFood = (food: Food) => {
    setEntries((prev) => [...prev, { food, grams: getDefaultGrams(food.category) }]);
    setSearch('');
    setShowDrop(false);
    setResult(null);
    setError('');
    void saveFoodLog(food);
  };

  const updateGrams = (id: number, delta: number) => {
    setEntries((prev) => prev.map((e) => e.food.id === id ? { ...e, grams: Math.max(10, e.grams + delta) } : e));
    setResult(null);
  };

  const setGrams = (id: number, val: string) => {
    const n = parseInt(val, 10);
    if (!isNaN(n) && n > 0) setEntries((prev) => prev.map((e) => e.food.id === id ? { ...e, grams: n } : e));
    setResult(null);
  };

  const removeEntry = (id: number) => {
    setEntries((prev) => prev.filter((e) => e.food.id !== id));
    setResult(null);
    setError('');
  };

  // Calculate totals scaled by grams / 100
  const totals = entries.reduce(
    (acc, { food, grams }) => {
      const scale = grams / 100;
      return {
        potassium:  acc.potassium  + food.potassium_mg  * scale,
        phosphorus: acc.phosphorus + food.phosphorus_mg * scale,
        protein:    acc.protein    + food.protein_g    * scale,
        sodium:     acc.sodium     + food.sodium_mg     * scale,
      };
    },
    { potassium: 0, phosphorus: 0, protein: 0, sodium: 0 }
  );

  const thresholds = getStageLimits(stage, bodyWeightKg);

  const computeRisk = async () => {
    if (entries.length === 0) { setError('Add at least one food item to assess this meal.'); return; }
    setError('');

    const assessedFoods = [...entries];
    const mealTotals = sumMealNutrients(assessedFoods);
    const limits = getStageLimits(stage, bodyWeightKg);
    const primaryEntry = [...assessedFoods].sort(
      (a, b) => getFoodRiskScore(b, limits) - getFoodRiskScore(a, limits),
    )[0];
    const primaryFoodName = primaryEntry.food.english;

    if (apiStatus === 'connected') {
      try {
        const response = await fetch(`${API_BASE_URL}/predict/risk`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            potassium: mealTotals.potassium,
            phosphorus: mealTotals.phosphorus,
            protein_per_kg: mealTotals.protein / bodyWeightKg,
            sodium: mealTotals.sodium,
            ckd_stage: stage,
            food_name: primaryFoodName,
          }),
          signal: AbortSignal.timeout(5000),
        });

        if (!response.ok) throw new Error('API error');

        const apiResult = await response.json();
        setUsingLiveModel(true);
        setModelConfidence(typeof apiResult.confidence === 'number' ? apiResult.confidence : null);

        const breakdown = buildBreakdown(
          mealTotals,
          limits,
          apiResult.exceeded_nutrients,
          apiResult.near_limit_nutrients,
        );

        const apiSubs: Food[] = (apiResult.substitutes ?? [])
          .map((s: { english: string }) => lookupFoodByEnglish(s.english))
          .filter((f: Food | undefined): f is Food => f !== undefined);

        const substitutions: FoodSubstitution[] =
          apiSubs.length > 0
            ? [{ riskyFood: primaryEntry.food, substitutes: apiSubs }]
            : [];

        const level = apiResult.risk_label as 'LOW' | 'MODERATE' | 'HIGH';

        setResult({
          level,
          score: scoreFromBreakdown(breakdown),
          breakdown,
          assessedFoods,
          substitutions,
        });

        setDailyMeals((prev) => [...prev, {
          mealType: currentMealType,
          foods: assessedFoods,
          assessedAt: new Date().toLocaleTimeString(),
          riskLevel: level,
        }]);
        setEntries([]);
        setDayResetMsg('');
        void saveRiskAssessment(level, apiResult.confidence, mealTotals);
        return;
      } catch {
        setUsingLiveModel(false);
        setModelConfidence(null);
      }
    } else {
      setUsingLiveModel(false);
      setModelConfidence(null);
    }

    const breakdown: Record<string, BreakdownEntry> = {
      Potassium:  { value: Math.round(totals.potassium),  limit: thresholds.potassium,  pct: 0, status: 'Safe' },
      Phosphorus: { value: Math.round(totals.phosphorus), limit: thresholds.phosphorus, pct: 0, status: 'Safe' },
      Protein:    { value: +totals.protein.toFixed(1),    limit: thresholds.protein,    pct: 0, status: 'Safe' },
      Sodium:     { value: Math.round(totals.sodium),     limit: thresholds.sodium,     pct: 0, status: 'Safe' },
    };
    for (const key of Object.keys(breakdown)) {
      const b = breakdown[key];
      b.pct    = Math.min(150, (b.value / b.limit) * 100);
      b.status = b.pct > 100 ? 'Exceeded' : b.pct > 80 ? 'Near limit' : 'Safe';
    }
    const maxPct        = Math.max(...Object.values(breakdown).map((b) => b.pct));
    const exceededCount = Object.values(breakdown).filter((b) => b.pct > 100).length;
    let level: 'LOW' | 'MODERATE' | 'HIGH' = 'LOW';
    if (exceededCount >= 2 || maxPct > 130) level = 'HIGH';
    else if (exceededCount >= 1 || maxPct > 80) level = 'MODERATE';

    const fallbackScore = scoreFromBreakdown(breakdown);

    setResult({
      level,
      score: fallbackScore,
      breakdown,
      assessedFoods,
      substitutions: getSmartSubstitutions(assessedFoods, stage, thresholds),
    });

    setDailyMeals((prev) => [...prev, {
      mealType: currentMealType,
      foods: assessedFoods,
      assessedAt: new Date().toLocaleTimeString(),
      riskLevel: level,
    }]);
    setEntries([]);
    setDayResetMsg('');
    void saveRiskAssessment(level, fallbackScore / 100, mealTotals);
  };

  const reset = () => {
    setEntries([]);
    setResult(null);
    setError('');
    setSearch('');
    setUsingLiveModel(false);
    setModelConfidence(null);
  };

  const resetDay = () => {
    setDailyMeals([]);
    setDayResetMsg('Daily log cleared');
  };

  const dailyTotals = dailyMeals.reduce(
    (acc, meal) => {
      const t = sumMealNutrients(meal.foods);
      return {
        potassium: acc.potassium + t.potassium,
        phosphorus: acc.phosphorus + t.phosphorus,
        protein: acc.protein + t.protein,
        sodium: acc.sodium + t.sodium,
      };
    },
    { potassium: 0, phosphorus: 0, protein: 0, sodium: 0 },
  );

  const proteinLimitGkg = STAGE_THRESHOLDS[stage].protein;

  const nutrientSummary = [
    { label: 'Potassium',  value: Math.round(totals.potassium),  limit: thresholds.potassium,  unit: 'mg', color: '#2E86AB' },
    { label: 'Phosphorus', value: Math.round(totals.phosphorus), limit: thresholds.phosphorus, unit: 'mg', color: '#F39C12' },
    { label: 'Protein',    value: +totals.protein.toFixed(1),    limit: thresholds.protein,    unit: 'g',  color: '#27AE60' },
    { label: 'Sodium',     value: Math.round(totals.sodium),     limit: thresholds.sodium,     unit: 'mg', color: '#E74C3C' },
  ];

  return (
    <div className="space-y-5 sm:space-y-6">
      {/* Header */}
      <div>
        <div style={{ color: theme.text, fontSize: '1.4rem', fontWeight: 600 }}>Meal Assessment</div>
        <p style={{ color: theme.textSecondary, marginTop: 4, fontSize: '0.9rem' }}>
          Log what you just ate or are about to eat — the system calculates the nutrients and assesses the risk for your CKD stage
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-5 sm:gap-6">
        {/* ── Left: CKD stage + meal builder ─────────────────────── */}
        <div className="lg:col-span-2 space-y-4 sm:space-y-5">

          {/* CKD Stage */}
          <div className="p-5 sm:p-6 rounded-2xl" style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}>
            <div style={{ color: theme.text, fontWeight: 600, marginBottom: 14 }}>Your CKD Stage</div>
            <div className="grid grid-cols-3 gap-2">
              {(Object.keys(STAGE_THRESHOLDS) as CKDStage[]).map((s) => (
                <button
                  key={s}
                  onClick={() => { setStage(s); setResult(null); }}
                  className="py-2 sm:py-2.5 rounded-xl transition-all duration-150"
                  style={{
                    background: stage === s ? 'rgba(46,134,171,0.15)' : isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)',
                    border:     `1px solid ${stage === s ? 'rgba(46,134,171,0.5)' : theme.cardBorder}`,
                    color:      stage === s ? '#2E86AB' : theme.text,
                    fontWeight: stage === s ? 700 : 400,
                    fontSize: '0.875rem',
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
            <div className="mt-3 p-3 rounded-xl" style={{ background: isDark ? 'rgba(46,134,171,0.08)' : 'rgba(46,134,171,0.07)', border: '1px solid rgba(46,134,171,0.18)', marginBottom: 24 }}>
              <p style={{ color: '#2E86AB', fontWeight: 600, fontSize: '0.78rem' }}>Stage {stage} — eGFR {thresholds.gfr} mL/min/1.73m²</p>
              <p style={{ color: theme.textSecondary, fontSize: '0.72rem', marginTop: 2 }}>{thresholds.label}</p>
            </div>

            <div style={{ marginBottom: 16, marginTop: 24 }}>
              <div style={{ color: theme.text, fontWeight: 600, marginBottom: 14 }}>
                Body Weight (kg)
              </div>
              <div className="mt-3 p-3 rounded-xl" style={{
                background: isDark ? 'rgba(46,134,171,0.08)' : 'rgba(46,134,171,0.07)',
                border: '1px solid rgba(46,134,171,0.18)'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <input
                    type="number"
                    min={30}
                    max={200}
                    step={0.5}
                    value={bodyWeightKg}
                    onChange={(e) => setBodyWeightKg(parseFloat(e.target.value) || 65)}
                    style={{
                      width: 80,
                      padding: '4px 10px',
                      borderRadius: 8,
                      border: '1px solid rgba(46,134,171,0.35)',
                      background: isDark ? 'rgba(46,134,171,0.10)' : 'rgba(46,134,171,0.06)',
                      color: '#2E86AB',
                      fontWeight: 600,
                      fontSize: '0.78rem',
                      outline: 'none',
                    }}
                  />
                  <p style={{ color: theme.textSecondary, fontSize: '0.72rem', margin: 0 }}>
                    Used to calculate your protein limit
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Meal builder */}
          <div className="p-5 sm:p-6 rounded-2xl" style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}>
            <div style={{ color: theme.text, fontWeight: 600, marginBottom: 4 }}>What did you eat?</div>
            <p style={{ color: theme.textSecondary, fontSize: '0.78rem', marginBottom: 14 }}>
              Search our {FOODS.length}-food database and add each item with its weight in grams
            </p>

            <div className="mb-4">
              <p style={{ color: theme.textSecondary, fontSize: '0.72rem', marginBottom: 8 }}>Meal type</p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {MEAL_TYPES.map((type) => (
                  <button
                    key={type}
                    onClick={() => setCurrentMealType(type)}
                    className="py-2 rounded-xl transition-all duration-150"
                    style={{
                      background: currentMealType === type ? 'rgba(46,134,171,0.15)' : isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)',
                      border: `1px solid ${currentMealType === type ? 'rgba(46,134,171,0.5)' : theme.cardBorder}`,
                      color: currentMealType === type ? '#2E86AB' : theme.text,
                      fontWeight: currentMealType === type ? 700 : 400,
                      fontSize: '0.78rem',
                    }}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>

            {/* Food search */}
            <div ref={wrapRef} className="relative mb-4">
              <div
                className="flex items-center gap-2 px-3 py-2.5 rounded-xl"
                style={{ background: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)', border: `1px solid ${theme.cardBorder}` }}
              >
                <Search size={14} style={{ color: theme.textSecondary, flexShrink: 0 }} />
                <input
                  className="flex-1 bg-transparent outline-none min-w-0"
                  style={{ color: theme.text, fontSize: '0.875rem' }}
                  placeholder={isListening ? "Listening..." : "Search foods to add…"}
                  value={search}
                  onChange={(e) => { setSearch(e.target.value); setShowDrop(true); }}
                  onFocus={() => setShowDrop(true)}
                />
                {voiceSupported && (
                  <button
                    type="button"
                    onClick={toggleVoiceInput}
                    aria-label={isListening ? "Stop voice input" : "Start voice input"}
                    className="flex-shrink-0 transition-colors"
                    style={{
                      color: isListening ? '#ef4444' : theme.textSecondary,
                      animation: isListening ? 'pulse 1.5s ease-in-out infinite' : 'none',
                    }}
                  >
                    {isListening ? <MicOff size={16} /> : <Mic size={16} />}
                  </button>
                )}
              </div>

              {voiceSupported && (
                <p
                  className="mt-1.5 px-1"
                  style={{
                    fontSize: '0.75rem',
                    color: theme.textSecondary,
                    opacity: 0.8,
                  }}
                >
                  🎤 Tip: say one food at a time, e.g. "banana" or "sweet potato"
                </p>
              )}

              {showDrop && visibleFoods.length > 0 && (
                <div
                  className="absolute top-full mt-1.5 left-0 right-0 z-30 rounded-xl overflow-hidden shadow-xl"
                  style={{ background: isDark ? '#111827' : '#fff', border: `1px solid ${theme.cardBorder}`, maxHeight: '300px', overflowY: 'auto' }}
                >
                  {visibleFoods.map((f, i) => (
                    <button
                      key={f.id}
                      className="w-full text-left px-4 py-2.5 flex items-center justify-between gap-3 transition-colors"
                      style={{ borderBottom: i < visibleFoods.length - 1 ? `1px solid ${theme.cardBorder}` : 'none' }}
                      onMouseDown={() => addFood(f)}
                      onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)'; }}
                      onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = 'transparent'; }}
                    >
                      <div className="min-w-0">
                        <p style={{ color: theme.text, fontSize: '0.85rem', fontWeight: 600 }} className="truncate capitalize">{f.english}</p>
                        <p style={{ color: theme.text, fontSize: '0.78rem', marginTop: 2 }} className="truncate">{f.french}</p>
                        <p style={{ color: theme.textSecondary, fontSize: '0.72rem', marginTop: 2 }} className="truncate italic">{f.kinyarwanda}</p>
                      </div>
                      <span className="shrink-0 px-2 py-0.5 rounded-full" style={{ background: potassiumColor(f.potassium_mg) + '20', color: potassiumColor(f.potassium_mg), fontSize: '0.65rem', fontWeight: 600 }}>
                        {f.ckd_stage_safe}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Meal entries */}
            {entries.length === 0 ? (
              <div
                className="flex flex-col items-center justify-center py-8 rounded-xl"
                style={{ background: isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.03)', border: `1px dashed ${theme.cardBorder}` }}
              >
                <p style={{ color: theme.textTertiary, fontSize: '0.8rem' }}>No foods added yet</p>
                <p style={{ color: theme.textTertiary, fontSize: '0.72rem', marginTop: 4 }}>Search above to build your meal</p>
              </div>
            ) : (
              <div className="space-y-2">
                {entries.map(({ food, grams }) => (
                  <div
                    key={food.id}
                    className="rounded-xl overflow-hidden"
                    style={{ border: `1px solid ${theme.cardBorder}` }}
                  >
                    {/* Food name row */}
                    <div
                      className="flex items-center gap-2 px-3 py-2"
                      style={{ background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.03)', borderBottom: `1px solid ${theme.cardBorder}` }}
                    >
                      <div className="w-2 h-2 rounded-full shrink-0" style={{ background: potassiumColor(food.potassium_mg) }} />
                      <p className="flex-1 min-w-0 truncate" style={{ color: theme.text, fontSize: '0.82rem', fontWeight: 600 }}>{food.english}</p>
                      <span style={{ color: theme.textTertiary, fontSize: '0.68rem' }}>{food.category}</span>
                      <button onClick={() => removeEntry(food.id)} className="ml-1 transition-opacity hover:opacity-60">
                        <Trash2 size={12} style={{ color: theme.textTertiary }} />
                      </button>
                    </div>
                    {/* Gram control row */}
                    <div className="flex items-center gap-2 px-3 py-2">
                      <button
                        onClick={() => updateGrams(food.id, -10)}
                        className="w-6 h-6 rounded-md flex items-center justify-center transition-opacity hover:opacity-70"
                        style={{ background: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.07)' }}
                      >
                        <Minus size={10} style={{ color: theme.textSecondary }} />
                      </button>
                      <input
                        type="number"
                        className="bg-transparent outline-none text-center"
                        style={{ color: theme.text, fontSize: '0.82rem', fontWeight: 700, width: 52 }}
                        value={grams}
                        onChange={(e) => setGrams(food.id, e.target.value)}
                        min="10"
                      />
                      <span style={{ color: theme.textSecondary, fontSize: '0.72rem' }}>g</span>
                      <button
                        onClick={() => updateGrams(food.id, 10)}
                        className="w-6 h-6 rounded-md flex items-center justify-center transition-opacity hover:opacity-70"
                        style={{ background: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.07)' }}
                      >
                        <Plus size={10} style={{ color: theme.textSecondary }} />
                      </button>
                      {/* Inline nutrient preview */}
                      <div className="flex-1 flex justify-end gap-3">
                        {[
                          { label: 'K', value: Math.round(food.potassium_mg * grams / 100), color: '#2E86AB' },
                          { label: 'P', value: Math.round(food.phosphorus_mg * grams / 100), color: '#F39C12' },
                          { label: 'Na', value: Math.round(food.sodium_mg * grams / 100), color: '#E74C3C' },
                        ].map((n) => (
                          <span key={n.label} style={{ color: n.color, fontSize: '0.68rem', fontWeight: 600 }}>
                            {n.label} {n.value}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Nutrient totals */}
            {entries.length > 0 && (
              <div className="mt-4 p-3.5 rounded-xl" style={{ background: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.03)', border: `1px solid ${theme.cardBorder}` }}>
                <p style={{ color: theme.textSecondary, fontSize: '0.72rem', marginBottom: 10 }}>Meal totals</p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-3">
                  {nutrientSummary.map((n) => {
                    const pct = Math.min(100, (n.value / n.limit) * 100);
                    const barColor = pct > 100 ? '#E74C3C' : pct > 80 ? '#F39C12' : n.color;
                    return (
                      <div key={n.label}>
                        <div className="flex justify-between mb-1">
                          <span style={{ color: theme.textSecondary, fontSize: '0.7rem' }}>{n.label}</span>
                          <span style={{ color: barColor, fontSize: '0.7rem', fontWeight: 600 }}>{Math.round(pct)}%</span>
                        </div>
                        <p style={{ color: n.color, fontWeight: 700, fontSize: '0.82rem' }}>
                          {n.value} <span style={{ color: theme.textTertiary, fontWeight: 400 }}>{n.unit}</span>
                        </p>
                        <div className="mt-1 rounded-full overflow-hidden" style={{ height: 3, background: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)' }}>
                          <div style={{ width: `${pct}%`, height: 3, background: barColor, borderRadius: 9999 }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-center gap-2 mt-3 px-3 py-2.5 rounded-xl" style={{ background: 'rgba(231,76,60,0.1)', border: '1px solid rgba(231,76,60,0.3)' }}>
                <AlertTriangle size={13} style={{ color: '#E74C3C', flexShrink: 0 }} />
                <p style={{ color: '#E74C3C', fontSize: '0.8rem' }}>{error}</p>
              </div>
            )}

            <div className="flex gap-3 mt-4">
              <button
                onClick={computeRisk}
                className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl text-white transition-all duration-200 hover:opacity-90 active:scale-[0.98]"
                style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)', fontWeight: 600, fontSize: '0.9rem' }}
              >
                <Zap size={15} />
                Assess Meal
              </button>
              <button
                onClick={reset}
                className="p-3 rounded-xl transition-all duration-150 hover:opacity-60"
                title="Reset"
                style={{ background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.05)', border: `1px solid ${theme.cardBorder}`, color: theme.textSecondary }}
              >
                <RotateCcw size={15} />
              </button>
            </div>
          </div>

          {/* Disclaimer */}
          <div className="p-4 rounded-2xl" style={{ background: isDark ? 'rgba(46,134,171,0.06)' : 'rgba(46,134,171,0.05)', border: '1px solid rgba(46,134,171,0.15)' }}>
            <div className="flex gap-2">
              <Info size={12} style={{ color: '#2E86AB', flexShrink: 0, marginTop: 1 }} />
              <p style={{ color: theme.textSecondary, fontSize: '0.78rem', lineHeight: 1.6 }}>
                Nutrient values are per 100 g from our food database. Thresholds follow KDIGO 2024 guidelines for your CKD stage. Always consult your healthcare provider.
              </p>
            </div>
          </div>
        </div>

        {/* ── Right: Results ────────────────────────────────────── */}
        <div className="lg:col-span-3 space-y-4 sm:space-y-5">
          {!result ? (
            <div
              className="rounded-2xl flex flex-col items-center justify-center text-center"
              style={{ background: theme.cardBg, border: `2px dashed ${theme.cardBorder}`, minHeight: 420, padding: 40 }}
            >
              <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-2xl flex items-center justify-center mb-4" style={{ background: isDark ? 'rgba(46,134,171,0.1)' : 'rgba(46,134,171,0.08)' }}>
                <Zap size={26} style={{ color: '#2E86AB' }} />
              </div>
              <div style={{ color: theme.text, fontWeight: 600, fontSize: '1.05rem' }}>Ready to assess</div>
              <p style={{ color: theme.textSecondary, marginTop: 8, maxWidth: 300, lineHeight: 1.6, fontSize: '0.875rem' }}>
                Search and add the foods in your meal, adjust gram weights, then click <strong style={{ color: '#2E86AB' }}>Assess Meal</strong> to see the nutrient risk for your CKD stage
              </p>
            </div>
          ) : (
            <>
              {/* Risk banner */}
              {(() => {
                const cfg = RISK_CFG[result.level];
                const Icon = cfg.icon;
                return (
                  <div className="p-5 sm:p-6 rounded-2xl" style={{ background: cfg.bg, border: `2px solid ${cfg.border}` }}>
                    <div className="flex items-start sm:items-center gap-4">
                      <div className="w-12 h-12 sm:w-16 sm:h-16 rounded-2xl flex items-center justify-center shrink-0" style={{ background: `${cfg.color}20` }}>
                        <Icon size={28} style={{ color: cfg.color }} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 flex-wrap mb-1">
                          <span style={{ color: cfg.color, fontSize: 'clamp(1.1rem,3vw,1.4rem)', fontWeight: 700 }}>{cfg.label}</span>
                          <span className="px-2.5 py-1 rounded-full" style={{ background: `${cfg.color}20`, color: cfg.color, fontSize: '0.72rem', fontWeight: 600 }}>
                            Avg {result.score}% of limits
                          </span>
                          {usingLiveModel && (
                            <span className="px-2.5 py-1 rounded-full" style={{ background: 'rgba(46,134,171,0.12)', color: '#2E86AB', fontSize: '0.68rem', fontWeight: 600 }}>
                              Powered by trained XGBoost model
                            </span>
                          )}
                        </div>
                        {usingLiveModel && modelConfidence !== null && (
                          <p style={{ color: theme.textTertiary, fontSize: '0.72rem', marginTop: 4 }}>
                            Model confidence: {Math.round(modelConfidence * 100)}%
                          </p>
                        )}
                        <p style={{ color: theme.textSecondary, fontSize: '0.85rem' }}>{cfg.desc}</p>
                        <p style={{ color: theme.textTertiary, fontSize: '0.75rem', marginTop: 3 }}>
                          {result.assessedFoods.length} food{result.assessedFoods.length !== 1 ? 's' : ''} · {result.assessedFoods.reduce((a, e) => a + e.grams, 0)} g total
                        </p>
                      </div>
                      <div className="text-right shrink-0 hidden sm:block">
                        <p style={{ color: theme.textTertiary, fontSize: '0.72rem', marginBottom: 2 }}>CKD Stage</p>
                        <p style={{ color: '#2E86AB', fontWeight: 700, fontSize: '1.2rem' }}>{stage}</p>
                      </div>
                    </div>
                    <div className="mt-4">
                      <div className="flex justify-between mb-1.5">
                        <span style={{ color: theme.textSecondary, fontSize: '0.78rem' }}>Average intake vs. daily limit</span>
                        <span style={{ color: cfg.color, fontWeight: 600, fontSize: '0.78rem' }}>{result.score}%</span>
                      </div>
                      <div className="rounded-full overflow-hidden" style={{ height: 9, background: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)' }}>
                        <div style={{ width: `${result.score}%`, height: 9, background: `linear-gradient(90deg,${cfg.color}80,${cfg.color})`, borderRadius: 9999, transition: 'width 0.6s ease' }} />
                      </div>
                    </div>
                  </div>
                );
              })()}

              {/* Recommendations + safer foods */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-5">
                <div className="p-4 sm:p-5 rounded-2xl" style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}>
                  <div style={{ color: theme.text, fontWeight: 600, fontSize: '0.9rem', marginBottom: 12 }}>Recommendations</div>
                  <ul className="space-y-2.5">
                    {RECOMMENDATIONS[result.level].map((r, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <ChevronRight size={12} style={{ color: '#2E86AB', marginTop: 3, flexShrink: 0 }} />
                        <span style={{ color: theme.textSecondary, fontSize: '0.825rem', lineHeight: 1.5 }}>{r}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="p-4 sm:p-5 rounded-2xl" style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}>
                  <div style={{ color: theme.text, fontWeight: 600, fontSize: '0.9rem', marginBottom: 12 }}>Safer food choices</div>
                  {result.substitutions.length === 0 &&
                  (!usingLiveModel || result.level !== 'LOW') ? (
                    <p style={{ color: theme.textSecondary, fontSize: '0.825rem' }}>{NO_SUBSTITUTES_MSG}</p>
                  ) : result.substitutions.length === 0 ? null : (
                    <ul className="space-y-4">
                      {result.substitutions.map(({ riskyFood, substitutes }) => (
                        <li key={riskyFood.id}>
                          <p style={{ color: theme.text, fontWeight: 600, fontSize: '0.825rem', marginBottom: 8 }}>
                            Instead of <span className="capitalize">{riskyFood.english}</span>:
                          </p>
                          <ul className="space-y-3 pl-1">
                            {substitutes.map((sub) => (
                              <li key={sub.id} className="flex items-start gap-2">
                                <ChevronRight size={12} style={{ color: '#27AE60', marginTop: 4, flexShrink: 0 }} />
                                <div className="min-w-0">
                                  <p style={{ color: theme.text, fontWeight: 500, fontSize: '0.825rem' }} className="capitalize">
                                    {sub.english}
                                  </p>
                                  <p style={{ color: theme.textSecondary, fontSize: '0.75rem', marginTop: 2, fontStyle: 'italic' }}>
                                    ({sub.kinyarwanda})
                                  </p>
                                  <p style={{ color: '#2E86AB', fontSize: '0.75rem', marginTop: 4, fontWeight: 600 }}>
                                    K: {riskyFood.potassium_mg}mg → {sub.potassium_mg}mg
                                  </p>
                                  <span
                                    className="inline-block mt-2 px-2 py-0.5 rounded-full"
                                    style={{ background: 'rgba(39,174,96,0.12)', color: '#27AE60', fontSize: '0.65rem', fontWeight: 600 }}
                                  >
                                    {sub.ckd_stage_safe}
                                  </span>
                                </div>
                              </li>
                            ))}
                          </ul>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>

              {/* Chart */}
              <div className="p-5 sm:p-6 rounded-2xl" style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}>
                <div style={{ color: theme.text, fontWeight: 600, marginBottom: 14 }}>Nutrient breakdown vs. daily limit</div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={Object.entries(result.breakdown).map(([name, b]) => ({ name, pct: Math.round(b.pct), value: b.value, limit: b.limit }))} margin={{ top: 8, right: 8, left: -14, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.07)'} vertical={false} />
                    <XAxis dataKey="name" tick={{ fill: theme.textSecondary as string, fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: theme.textSecondary as string, fontSize: 10 }} tickFormatter={(v) => `${v}%`} domain={[0, 150]} axisLine={false} tickLine={false} />
                    <Tooltip
                      contentStyle={{ background: isDark ? '#111827' : '#fff', border: `1px solid ${theme.cardBorder}`, borderRadius: 10, color: theme.text, fontSize: '0.825rem' }}
                      formatter={(val: number, _n: string, item: { payload: { value: number; limit: number; name: string } }) => [`${val}% — ${item.payload.value} / ${item.payload.limit}`, item.payload.name]}
                    />
                    <ReferenceLine y={100} stroke="#E74C3C" strokeDasharray="5 5" strokeWidth={1.5} />
                    <ReferenceLine y={80}  stroke="#F39C12" strokeDasharray="3 3"  strokeWidth={1} />
                    <Bar dataKey="pct" radius={[5, 5, 0, 0]}>
                      {Object.entries(result.breakdown).map(([name, b]) => (
                        <Cell key={name} fill={b.pct > 100 ? '#E74C3C' : b.pct > 80 ? '#F39C12' : NUTRIENT_COLORS[name] ?? '#2E86AB'} fillOpacity={0.9} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div className="flex gap-4 mt-3">
                  {[{ label: 'Danger (100%)', color: '#E74C3C' }, { label: 'Warning (80%)', color: '#F39C12' }].map((ref) => (
                    <div key={ref.label} className="flex items-center gap-1.5">
                      <svg width={18} height={2}><line x1="0" y1="1" x2="18" y2="1" stroke={ref.color} strokeWidth={2} strokeDasharray="4 2" /></svg>
                      <span style={{ color: theme.textSecondary, fontSize: '0.72rem' }}>{ref.label}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Per-nutrient cards */}
              <div className="grid grid-cols-2 gap-3 sm:gap-4">
                {Object.entries(result.breakdown).map(([name, b]) => {
                  const statusColor = b.status === 'Exceeded' ? '#E74C3C' : b.status === 'Near limit' ? '#F39C12' : '#27AE60';
                  const StatusIcon  = b.status === 'Exceeded' ? XCircle  : b.status === 'Near limit' ? AlertTriangle : CheckCircle2;
                  const unit = name === 'Protein' ? 'g' : 'mg';
                  return (
                    <div key={name} className="p-3 sm:p-4 rounded-xl" style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}>
                      <div className="flex items-center justify-between mb-2">
                        <span style={{ color: theme.text, fontWeight: 600, fontSize: '0.85rem' }}>{name}</span>
                        <span className="flex items-center gap-1" style={{ color: statusColor, fontSize: '0.72rem', fontWeight: 600 }}>
                          <StatusIcon size={11} />
                          <span className="hidden sm:inline">{b.status}</span>
                        </span>
                      </div>
                      <div style={{ lineHeight: 1 }}>
                        <span style={{ color: NUTRIENT_COLORS[name] ?? '#2E86AB', fontWeight: 700, fontSize: '1.1rem' }}>{b.value}</span>
                        <span style={{ color: theme.textSecondary, fontSize: '0.75rem' }}> / {b.limit} {unit}</span>
                      </div>
                      <div className="mt-2 rounded-full overflow-hidden" style={{ height: 5, background: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)' }}>
                        <div style={{ width: `${Math.min(100, b.pct)}%`, height: 5, background: statusColor, borderRadius: 9999 }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}

          {/* Daily summary */}
          {(dailyMeals.length > 0 || dayResetMsg) && (
            <div className="p-5 sm:p-6 rounded-2xl" style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}>
              <div className="flex items-center justify-between gap-3 mb-4">
                <div style={{ color: theme.text, fontWeight: 600, fontSize: '0.95rem' }}>Today&apos;s Meals</div>
                <button
                  onClick={resetDay}
                  className="px-3 py-1.5 rounded-lg transition-opacity hover:opacity-70"
                  style={{ background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.05)', border: `1px solid ${theme.cardBorder}`, color: theme.textSecondary, fontSize: '0.72rem' }}
                >
                  Reset Day
                </button>
              </div>
              {dayResetMsg && (
                <p style={{ color: '#27AE60', fontSize: '0.8rem', marginBottom: 12 }}>{dayResetMsg}</p>
              )}
              {dailyMeals.length > 0 && (
                <>
                  <div className="space-y-3 mb-5">
                    {dailyMeals.map((meal, i) => {
                      const mealTotals = sumMealNutrients(meal.foods);
                      const riskCfg = RISK_CFG[meal.riskLevel];
                      return (
                        <div
                          key={`${meal.mealType}-${meal.assessedAt}-${i}`}
                          className="p-3.5 rounded-xl"
                          style={{ background: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.03)', border: `1px solid ${theme.cardBorder}` }}
                        >
                          <div className="flex items-center justify-between gap-2 flex-wrap mb-2">
                            <span className="px-2.5 py-0.5 rounded-full" style={{ background: 'rgba(46,134,171,0.12)', color: '#2E86AB', fontSize: '0.68rem', fontWeight: 600 }}>
                              {meal.mealType}
                            </span>
                            <span style={{ color: theme.textTertiary, fontSize: '0.7rem' }}>{meal.assessedAt}</span>
                            <span className="px-2.5 py-0.5 rounded-full" style={{ background: riskCfg.bg, color: riskCfg.color, fontSize: '0.68rem', fontWeight: 600 }}>
                              {meal.riskLevel}
                            </span>
                          </div>
                          <p style={{ color: theme.textSecondary, fontSize: '0.78rem' }}>
                            {meal.foods.length} food{meal.foods.length !== 1 ? 's' : ''} · Potassium {Math.round(mealTotals.potassium)} mg
                          </p>
                        </div>
                      );
                    })}
                  </div>
                  <div className="p-4 rounded-xl" style={{ background: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.03)', border: `1px solid ${theme.cardBorder}` }}>
                    <p style={{ color: theme.text, fontWeight: 600, fontSize: '0.85rem', marginBottom: 12 }}>DAILY TOTALS</p>
                    <div className="space-y-2">
                      {[
                        { label: 'Potassium', value: Math.round(dailyTotals.potassium), limit: thresholds.potassium, unit: 'mg' },
                        { label: 'Phosphorus', value: Math.round(dailyTotals.phosphorus), limit: thresholds.phosphorus, unit: 'mg' },
                        { label: 'Protein', value: +dailyTotals.protein.toFixed(1), limit: proteinLimitGkg, unit: 'g', limitSuffix: 'g/kg' },
                        { label: 'Sodium', value: Math.round(dailyTotals.sodium), limit: thresholds.sodium, unit: 'mg' },
                      ].map((n) => {
                        const compareLimit = n.label === 'Protein' ? thresholds.protein : n.limit;
                        const color = dailyTotalColor(n.value, compareLimit);
                        return (
                          <div key={n.label} className="flex justify-between items-center">
                            <span style={{ color: theme.textSecondary, fontSize: '0.8rem' }}>{n.label}</span>
                            <span style={{ color, fontSize: '0.8rem', fontWeight: 600 }}>
                              {n.value}{n.unit} / {n.limit}{n.limitSuffix ?? n.unit}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
