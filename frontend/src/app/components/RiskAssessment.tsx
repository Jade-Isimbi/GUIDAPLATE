import { useState, useRef, useEffect, Fragment } from 'react';
import { AlertTriangle, CheckCircle2, XCircle, Zap, Info, ChevronRight, RotateCcw, Search, Trash2, Plus, Minus, Mic, MicOff } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, Cell } from 'recharts';
import {
  CKDStage,
  FOODS,
  STAGE_THRESHOLDS,
  type Food,
  potassiumColor,
} from '../../data/foodDatabase';
import { foodTranslation, matchesFoodQuery } from '../../utils/foodDisplay';
import { authFetch, getAuthToken } from '../../utils/auth';
import { displayBudgetStatus, formatStageDisplay, getRiskDisplay, getWeeklyRiskLabel, isFoodInStageRange } from '../../utils/riskDisplay';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const FOOD_DB_COUNT = 386;

function mapApiFood(row: Record<string, unknown>): Food {
  return {
    id: Number(row.food_id ?? row.id ?? 0),
    english: String(row.english ?? ''),
    french: row.french ? String(row.french) : null,
    kinyarwanda: row.kinyarwanda ? String(row.kinyarwanda) : null,
    category: String(row.category ?? 'Other'),
    meal_type: String(row.meal_type ?? 'Any'),
    protein_g: Number(row.protein_g ?? 0),
    potassium_mg: Number(row.potassium_mg ?? 0),
    phosphorus_mg: Number(row.phosphorus_mg ?? 0),
    sodium_mg: Number(row.sodium_mg ?? 0),
    energy_kcal: Number(row.energy_kcal ?? 0),
    preparation_method: String(row.preparation_method ?? ''),
    source: String(row.source ?? ''),
    ckd_stage_safe: String(row.ckd_stage_safe ?? '1-5'),
    notes: String(row.notes ?? ''),
  };
}

const UNIT_MAP: Record<string, { unit: string; grams: number }> = {
  Fruit: { unit: 'piece', grams: 150 },
  'Starch/Grain': { unit: 'cup', grams: 185 },
  Starch: { unit: 'cup', grams: 185 },
  Grain: { unit: 'cup', grams: 185 },
  Vegetable: { unit: 'cup', grams: 90 },
  Meat: { unit: 'piece', grams: 85 },
  Fish: { unit: 'piece', grams: 85 },
  Legume: { unit: 'cup', grams: 180 },
  Dairy: { unit: 'cup', grams: 240 },
  Beverage: { unit: 'cup', grams: 240 },
  Egg: { unit: 'egg', grams: 50 },
  Bread: { unit: 'slice', grams: 30 },
  'Fat/Oil': { unit: 'tablespoon', grams: 14 },
  Other: { unit: 'serving', grams: 100 },
};

const FOOD_OVERRIDES: Record<string, { unit: string; grams: number }> = {
  banana: { unit: 'banana', grams: 120 },
  egg: { unit: 'egg', grams: 50 },
  avocado: { unit: 'piece', grams: 200 },
  bread: { unit: 'slice', grams: 30 },
  milk: { unit: 'glass', grams: 240 },
  tea: { unit: 'cup', grams: 240 },
  ikivuguto: { unit: 'cup', grams: 240 },
  'sweet potatoes': { unit: 'piece', grams: 200 },
  cassava: { unit: 'piece', grams: 200 },
  rice: { unit: 'cup', grams: 185 },
  ugali: { unit: 'serving', grams: 200 },
  beans: { unit: 'cup', grams: 180 },
  chicken: { unit: 'piece', grams: 85 },
  tilapia: { unit: 'piece', grams: 85 },
  oats: { unit: 'cup', grams: 80 },
  watermelon: { unit: 'slice', grams: 200 },
  pineapple: { unit: 'slice', grams: 80 },
};

function getUnitInfo(foodName: string, category: string): { unit: string; grams: number } {
  const lower = foodName.toLowerCase();
  for (const [key, val] of Object.entries(FOOD_OVERRIDES)) {
    if (lower.includes(key)) return val;
  }
  return UNIT_MAP[category] || UNIT_MAP.Other;
}

const NO_SUBSTITUTES_MSG =
  'No category-matched lower-potassium substitutes found for this meal.';

const OCCASION_ENCODING: Record<string, number> = {
  Breakfast: 0.00,
  Lunch: 0.33,
  Dinner: 0.67,
  Snack: 0.50,
};

type MealOccasion = 'Breakfast' | 'Lunch' | 'Dinner' | 'Snack';

interface FoodLog {
  log_id: string;
  food_name: string;
  category: string | null;
  stage_safe_range: string | null;
  portion_grams: number | null;
  meal_occasion: string | null;
  potassium_mg: number | null;
  phosphorus_mg: number | null;
  protein_g: number | null;
  sodium_mg: number | null;
  logged_at: string | null;
}

const EMPTY_LOGS_BY_OCCASION: Record<MealOccasion, FoodLog[]> = {
  Breakfast: [],
  Lunch: [],
  Dinner: [],
  Snack: [],
};

const MEAL_OCCASION_ICONS: Record<MealOccasion, string> = {
  Breakfast: '🍳',
  Lunch: '🍽️',
  Dinner: '🌙',
  Snack: '🍎',
};

const RESULTS_STORAGE_PREFIX = 'results_by_occasion';
const RESULTS_DATE_PREFIX = 'results_by_occasion_date';
const MEAL_OCCASION_PREFIX = 'guidaplate_meal_occasion';

function currentCacheUserId(): string | null {
  return localStorage.getItem('guidaplate_user_id');
}

function resultsStorageKey(userId: string): string {
  return `${RESULTS_STORAGE_PREFIX}:${userId}`;
}

function resultsDateKey(userId: string): string {
  return `${RESULTS_DATE_PREFIX}:${userId}`;
}

function mealOccasionStorageKey(userId: string): string {
  return `${MEAL_OCCASION_PREFIX}:${userId}`;
}

function clearLegacyUnscopedMealKeys(): void {
  localStorage.removeItem(RESULTS_STORAGE_PREFIX);
  localStorage.removeItem(RESULTS_DATE_PREFIX);
  localStorage.removeItem(MEAL_OCCASION_PREFIX);
}

const saveFoodLog = async (food: Food, portionGrams: number, mealOccasion: string): Promise<boolean> => {
  if (!getAuthToken()) return false;

  try {
    const response = await authFetch(`${API_BASE}/api/patient/food-log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        food_name: food.english,
        category: food.category,
        stage_safe_range: food.ckd_stage_safe,
        portion_grams: portionGrams,
        meal_occasion: mealOccasion,
      }),
    });
    if (response.status === 401) return false;
    return response.ok;
  } catch (err) {
    console.error('Failed to save food log:', err);
    return false;
  }
};

const saveMealFoodLogs = async (foods: MealFoodItem[], mealOccasion: string) => {
  await Promise.all(foods.map(({ food, grams }) => saveFoodLog(food, grams, mealOccasion)));
};

const saveRiskAssessment = async (
  riskLevel: string,
  riskScore: number,
  nutrientTotals: Record<string, number>,
  extras?: {
    shap_contributions?: Record<string, number> | null;
    shap_explanation?: string | null;
    ckd_stage?: string;
    bodyWeightKg?: number;
  },
) => {
  if (!getAuthToken()) return;

  const featureValues =
    extras?.ckd_stage && extras.bodyWeightKg
      ? {
          potassium: nutrientTotals.potassium,
          phosphorus: nutrientTotals.phosphorus,
          protein_per_kg: nutrientTotals.protein / extras.bodyWeightKg,
          sodium: nutrientTotals.sodium,
          ckd_stage: extras.ckd_stage,
        }
      : undefined;

  try {
    await authFetch(`${API_BASE}/api/patient/risk-assessment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        risk_level: riskLevel,
        risk_score: riskScore,
        nutrients_summary: JSON.stringify(nutrientTotals),
        shap_contributions: extras?.shap_contributions ?? null,
        shap_explanation: extras?.shap_explanation ?? null,
        feature_values: featureValues ?? null,
        ckd_stage: extras?.ckd_stage ?? null,
      }),
    });
  } catch (err) {
    console.error('Failed to save risk assessment:', err);
  }
};

interface FoodSuggestion {
  english: string;
  french: string | null;
  kinyarwanda: string | null;
  category: string | null;
  reason: string;
}

interface MealOccasionSuggestions {
  occasion: string;
  suggestions: FoodSuggestion[];
}

interface WeeklySuggestionsResponse {
  trajectory_risk: string;
  flagged_nutrient: string | null;
  flagged_reason: string;
  remaining_days: number;
  suggestions_by_meal: MealOccasionSuggestions[];
  clinical_note: string;
  analysis_available: boolean;
}

const SUGGESTION_OCCASION_LABELS: Record<string, string> = {
  breakfast: 'Breakfast',
  lunch: 'Lunch',
  dinner: 'Dinner',
  snack: 'Snack',
};

const getNextOccasion = (current: string): string => {
  const order = ['breakfast', 'lunch', 'dinner', 'snack'];
  const idx = order.indexOf((current ?? '').toLowerCase());
  if (idx === -1) return 'lunch';
  if (idx === order.length - 1) return 'breakfast';
  return order[idx + 1];
};

const getTimeBasedOccasion = (): string => {
  const hour = new Date().getHours();
  if (hour < 10) return 'breakfast';
  if (hour < 14) return 'lunch';
  if (hour < 19) return 'dinner';
  return 'snack';
};

function mealOccasionToSuggestionTab(occasion: string | null | undefined): string {
  const key = (occasion ?? '').toLowerCase();
  if (key === 'breakfast' || key === 'lunch' || key === 'dinner' || key === 'snack') {
    return key;
  }
  return getTimeBasedOccasion();
}

const SUGGESTIONS_DEFAULT_SUBTITLE =
  'Based on your recent meals, here are safe options for your next meal';

const toPlainNutrient = (key: string): string => {
  const map: Record<string, string> = {
    potassium: 'potassium',
    protein_per_kg: 'protein',
    phosphorus: 'phosphorus',
    sodium: 'sodium',
  };
  return map[key?.toLowerCase()] ?? key;
};

function getSuggestionsSubtitle(
  hasResult: boolean,
  suggestions: WeeklySuggestionsResponse | null,
): string {
  if (!hasResult) return SUGGESTIONS_DEFAULT_SUBTITLE;
  const nutrient = suggestions?.flagged_nutrient?.trim();
  if (!nutrient) return SUGGESTIONS_DEFAULT_SUBTITLE;
  return `Based on the meal you just checked, we updated your suggestions to be lower in ${toPlainNutrient(nutrient)}`;
}

interface RiskAssessmentProps {
  isDark: boolean;
  theme: Record<string, string>;
  initialBodyWeight?: number;
  initialStage?: string;
}

const STAGE_ALIASES: Record<string, CKDStage> = {
  '3': 'G3a',
  G3: 'G3a',
};

function toCKDStage(value: string | undefined | null): CKDStage {
  const cleaned = (value ?? 'G3a').trim();
  const normalized = STAGE_ALIASES[cleaned] ?? cleaned;
  if (normalized in STAGE_THRESHOLDS) return normalized as CKDStage;
  return 'G3a';
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
        .filter((f) => isFoodInStageRange(f.ckd_stage_safe, stage))
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

function logToMealFoodItem(log: FoodLog): MealFoodItem | null {
  const grams = log.portion_grams ?? 100;
  const matched = lookupFoodByEnglish(log.food_name);
  if (matched) {
    return { food: matched, grams };
  }
  if (grams <= 0) return null;
  const scale = 100 / grams;
  const fallback: Food = {
    id: Math.abs(log.log_id.split('').reduce((a, c) => a + c.charCodeAt(0), 0)),
    english: log.food_name,
    french: log.food_name,
    kinyarwanda: log.food_name,
    category: log.category ?? 'Other',
    meal_type: 'Any',
    protein_g: (log.protein_g ?? 0) * scale,
    potassium_mg: (log.potassium_mg ?? 0) * scale,
    phosphorus_mg: (log.phosphorus_mg ?? 0) * scale,
    sodium_mg: (log.sodium_mg ?? 0) * scale,
    energy_kcal: 0,
    preparation_method: '',
    source: '',
    ckd_stage_safe: log.stage_safe_range ?? '1-5',
    notes: '',
  };
  return { food: fallback, grams };
}

function groupLogsByOccasion(logs: FoodLog[]): Record<MealOccasion, FoodLog[]> {
  const grouped: Record<MealOccasion, FoodLog[]> = {
    Breakfast: [],
    Lunch: [],
    Dinner: [],
    Snack: [],
  };
  for (const log of logs) {
    const raw = String(log.meal_occasion ?? '').trim();
    const normalized =
      raw.length === 0
        ? 'Snack'
        : (raw.charAt(0).toUpperCase() + raw.slice(1).toLowerCase());
    const occasion = (normalized as MealOccasion);
    if (occasion in grouped) grouped[occasion].push(log);
  }
  return grouped;
}

function potassiumStatusColor(mg: number, limit: number): string {
  const pct = (mg / limit) * 100;
  if (pct > 100) return '#E74C3C';
  if (pct > 80) return '#F39C12';
  return '#27AE60';
}

const SPEECH_CONNECTOR_WORDS = new Set(['and', 'with', 'plus', 'also']);

const SPEECH_FILLER_PATTERN =
  /\b(i had|i ate|i have|i am eating|for lunch|for breakfast|for dinner|for snack|some|a bit of|and some|with some|also|plus|as well as)\b/gi;

interface SpeechExtractionResult {
  matched: Food[];
  unmatched: string[];
}

function findBestFoodMatch(phrase: string, foodDatabase: Food[]): Food | undefined {
  const normalized = phrase.trim().toLowerCase();
  if (normalized.length < 3) return undefined;
  if (SPEECH_CONNECTOR_WORDS.has(normalized)) return undefined;

  const exact = foodDatabase.find(
    (f) =>
      f.english.toLowerCase() === normalized ||
      f.french?.toLowerCase() === normalized ||
      f.kinyarwanda?.toLowerCase() === normalized,
  );
  if (exact) return exact;

  const partialMatches = foodDatabase.filter((f) => {
    const english = f.english.toLowerCase();
    return english.includes(normalized) || normalized.includes(english);
  });
  if (partialMatches.length > 0) {
    partialMatches.sort((a, b) => {
      const aExact = a.english.toLowerCase() === normalized ? 1 : 0;
      const bExact = b.english.toLowerCase() === normalized ? 1 : 0;
      if (aExact !== bExact) return bExact - aExact;
      return b.english.length - a.english.length;
    });
    return partialMatches[0];
  }

  const words = normalized.split(' ').filter((w) => w.length >= 3);
  for (const word of words) {
    if (SPEECH_CONNECTOR_WORDS.has(word)) continue;
    const wordMatch = foodDatabase.find((f) => f.english.toLowerCase() === word);
    if (wordMatch) return wordMatch;
  }

  return undefined;
}

function extractFoodsFromSpeech(transcript: string, foodDatabase: Food[]): SpeechExtractionResult {
  const cleaned = transcript
    .toLowerCase()
    .trim()
    .replace(SPEECH_FILLER_PATTERN, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  const separatorPattern = /,|;|\s+and\s+|\s+with\s+|\s+plus\s+|\s+also\s+/i;
  const candidates = cleaned
    .split(separatorPattern)
    .map((s) => s.trim())
    .filter((s) => s.length > 2);

  const matched: Food[] = [];
  const matchedIds = new Set<number>();

  const addMatch = (food: Food | undefined) => {
    if (food && !matchedIds.has(food.id)) {
      matched.push(food);
      matchedIds.add(food.id);
      return true;
    }
    return false;
  };

  for (const candidate of candidates) {
    addMatch(findBestFoodMatch(candidate, foodDatabase));
  }

  const words = cleaned.split(/\s+/).filter(Boolean);
  let i = 0;
  while (i < words.length) {
    if (SPEECH_CONNECTOR_WORDS.has(words[i])) {
      i += 1;
      continue;
    }

    for (const len of [3, 2, 1]) {
      if (i + len > words.length) continue;
      const phrase = words.slice(i, i + len).join(' ');
      const food = findBestFoodMatch(phrase, foodDatabase);
      if (food) {
        if (!matchedIds.has(food.id)) {
          addMatch(food);
        }
        break;
      }
    }
    i += 1;
  }

  const unmatchedWords: string[] = [];
  i = 0;
  while (i < words.length) {
    if (SPEECH_CONNECTOR_WORDS.has(words[i])) {
      i += 1;
      continue;
    }

    let matchedAtPosition = false;
    for (const len of [3, 2, 1]) {
      if (i + len > words.length) continue;
      const phrase = words.slice(i, i + len).join(' ');
      const food = findBestFoodMatch(phrase, foodDatabase);
      if (food && matchedIds.has(food.id)) {
        matchedAtPosition = true;
        break;
      }
    }

    if (!matchedAtPosition) {
      unmatchedWords.push(words[i]);
    }
    i += 1;
  }

  const unmatched = [...new Set(unmatchedWords)].filter((w) => w.length > 2);

  return { matched, unmatched };
}

interface VoiceMatchItem {
  food: Food;
  grams: number;
}

interface VoiceSpeechResult {
  transcript: string;
  matched: VoiceMatchItem[];
  unmatched: string[];
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

type RiskLevel = 'LOW' | 'MODERATE' | 'HIGH';

const EXCEEDED_BULLETS: Record<string, string> = {
  Potassium: 'Your meal is high in potassium — reduce portions of high-potassium foods like bananas, potatoes, or beans.',
  Phosphorus: 'This meal exceeded your phosphorus limit — avoid processed foods and dairy where possible.',
  Protein: 'Protein intake is over your limit — consider smaller meat or legume portions next meal.',
  Sodium: 'Sodium is high — avoid added salt and salty condiments in your next meal.',
};

const NEAR_LIMIT_BULLETS: Record<string, string> = {
  Potassium: 'Potassium is approaching your limit — watch portions of potassium-rich foods.',
  Phosphorus: 'Phosphorus is near your limit — limit dairy and processed snacks.',
  Protein: 'Protein is close to your daily limit — keep next meal portions moderate.',
  Sodium: 'Sodium is near your limit — choose low-salt options for your next meal.',
};

const LEVEL_CLOSING_BULLETS: Record<RiskLevel, string> = {
  LOW: 'Your meal looks balanced. Keep maintaining this pattern.',
  MODERATE: 'Overall risk is moderate. Small adjustments in portions can bring you within safe limits.',
  HIGH: 'This meal poses high dietary risk. Consider a lighter, lower-nutrient next meal and consult your care provider.',
};

function generateRecommendations(
  level: RiskLevel,
  breakdown: Record<string, BreakdownEntry>,
): string[] {
  const entries = Object.entries(breakdown);

  const exceeded = entries
    .filter(([, b]) => b.status === 'Exceeded')
    .sort(([, a], [, b]) => b.pct - a.pct);

  const nearLimit = entries.filter(([, b]) => b.status === 'Near limit');
  const allSafe = exceeded.length === 0 && nearLimit.length === 0;

  if (allSafe && level === 'LOW') {
    return ['Your meal is well within safe nutrient limits. Great dietary choice!'];
  }

  let nutrientBullets = [
    ...exceeded.map(([name]) => EXCEEDED_BULLETS[name]),
    ...nearLimit.map(([name]) => NEAR_LIMIT_BULLETS[name]),
  ].filter(Boolean);

  if (nutrientBullets.length > 4) {
    nutrientBullets = exceeded
      .map(([name]) => EXCEEDED_BULLETS[name])
      .filter(Boolean)
      .slice(0, 4);
  }

  return [...nutrientBullets, LEVEL_CLOSING_BULLETS[level]].slice(0, 5);
}

const NUTRIENT_COLORS: Record<string, string> = { Potassium: '#2E86AB', Phosphorus: '#F39C12', Protein: '#27AE60', Sodium: '#E74C3C' };

const BREAKDOWN_TO_DAILY_KEY: Record<string, string> = {
  Potassium: 'potassium',
  Phosphorus: 'phosphorus',
  Protein: 'protein_per_kg',
  Sodium: 'sodium',
};

function dailyNutrientBadge(percentUsed: number | undefined): {
  label: string;
  color: string;
  Icon: typeof CheckCircle2;
} {
  const pct = percentUsed ?? 0;
  if (pct > 100) {
    return { label: '⚠ Daily limit exceeded', color: '#F59E0B', Icon: AlertTriangle };
  }
  if (pct >= 80) {
    return { label: '⚠ Approaching limit', color: '#F59E0B', Icon: AlertTriangle };
  }
  return { label: '✔ Within limit', color: '#27AE60', Icon: CheckCircle2 };
}

function isDailyBudgetExceeded(dailyBudget: DailyBudgetData | null): boolean {
  return !!dailyBudget &&
    ['potassium', 'phosphorus', 'protein_per_kg', 'sodium'].some((k) => {
      const n = dailyBudget.nutrients[k];
      return typeof n?.percent_used === 'number' && n.percent_used > 100;
    });
}

function isDailyNutrientAmber(percentUsed: number | undefined): boolean {
  return (percentUsed ?? 0) >= 80;
}

function dailyNutrientCardValues(
  name: string,
  mealEntry: BreakdownEntry,
  dailyKey: string | undefined,
  dailyBudget: DailyBudgetData | null,
  bodyWeightKg: number,
): { value: number; limit: number; pct: number; sourceLabel: 'This meal' | 'Today total' } {
  const dailyData = dailyKey ? dailyBudget?.nutrients[dailyKey] : undefined;
  const dailyPct = dailyData?.percent_used;

  if (dailyData && isDailyNutrientAmber(dailyPct)) {
    if (name === 'Protein') {
      return {
        value: +(dailyData.consumed * bodyWeightKg).toFixed(1),
        limit: +(dailyData.limit * bodyWeightKg).toFixed(1),
        pct: dailyPct ?? mealEntry.pct,
        sourceLabel: 'Today total',
      };
    }
    return {
      value: Math.round(dailyData.consumed),
      limit: Math.round(dailyData.limit),
      pct: dailyPct ?? mealEntry.pct,
      sourceLabel: 'Today total',
    };
  }

  return {
    value: mealEntry.value,
    limit: mealEntry.limit,
    pct: mealEntry.pct,
    sourceLabel: 'This meal',
  };
}

const RISK_CFG = {
  HIGH:     { color: '#E74C3C', bg: 'rgba(231,76,60,0.1)',  border: 'rgba(231,76,60,0.35)',  icon: XCircle,       label: getRiskDisplay('HIGH').label,     desc: getRiskDisplay('HIGH').sublabel },
  MODERATE: { color: '#F39C12', bg: 'rgba(243,156,18,0.1)', border: 'rgba(243,156,18,0.35)', icon: AlertTriangle, label: getRiskDisplay('MODERATE').label, desc: getRiskDisplay('MODERATE').sublabel },
  LOW:      { color: '#27AE60', bg: 'rgba(39,174,96,0.1)',  border: 'rgba(39,174,96,0.35)',  icon: CheckCircle2,  label: getRiskDisplay('LOW').label,      desc: getRiskDisplay('LOW').sublabel },
};

const MEAL_TYPES: MealOccasion[] = ['Breakfast', 'Lunch', 'Dinner', 'Snack'];

interface BreakdownEntry { value: number; limit: number; pct: number; status: 'Safe' | 'Near limit' | 'Exceeded' }
interface ResultState {
  level: 'LOW' | 'MODERATE' | 'HIGH';
  score: number;
  breakdown: Record<string, BreakdownEntry>;
  assessedFoods: MealFoodItem[];
  substitutions: FoodSubstitution[];
  shap_explanation?: string | null;
  shap_contributions?: Record<string, number> | null;
  shap_dominant_nutrient?: string | null;
}

interface OccasionAssessment {
  result: ResultState;
  lstmPattern: { risk_label: string; confidence: number; trend: string } | null;
  usingLiveModel: boolean;
  modelConfidence: number | null;
}

const EMPTY_RESULTS_BY_OCCASION: Record<MealOccasion, OccasionAssessment | null> = {
  Breakfast: null,
  Lunch: null,
  Dinner: null,
  Snack: null,
};

function readInitialMealOccasion(): MealOccasion {
  const userId = currentCacheUserId();
  if (!userId) return 'Breakfast';
  clearLegacyUnscopedMealKeys();
  const stored = localStorage.getItem(mealOccasionStorageKey(userId));
  if (stored === 'Breakfast' || stored === 'Lunch' || stored === 'Dinner' || stored === 'Snack') {
    return stored;
  }
  return 'Breakfast';
}

function loadStoredResultsByOccasion(): Record<MealOccasion, OccasionAssessment | null> {
  try {
    const userId = currentCacheUserId();
    if (!userId) return { ...EMPTY_RESULTS_BY_OCCASION };
    clearLegacyUnscopedMealKeys();

    const today = new Date().toISOString().slice(0, 10);
    const storageKey = resultsStorageKey(userId);
    const dateKey = resultsDateKey(userId);
    const storedDate = localStorage.getItem(dateKey);
    if (storedDate !== today) {
      localStorage.removeItem(storageKey);
      localStorage.removeItem(dateKey);
      return { ...EMPTY_RESULTS_BY_OCCASION };
    }
    const saved = localStorage.getItem(storageKey);
    if (!saved) return { ...EMPTY_RESULTS_BY_OCCASION };
    const parsed = JSON.parse(saved) as Record<string, OccasionAssessment | null>;
    return {
      Breakfast: parsed.Breakfast ?? null,
      Lunch: parsed.Lunch ?? null,
      Dinner: parsed.Dinner ?? null,
      Snack: parsed.Snack ?? null,
    };
  } catch {
    return { ...EMPTY_RESULTS_BY_OCCASION };
  }
}

function persistResultsByOccasion(data: Record<MealOccasion, OccasionAssessment | null>) {
  const userId = currentCacheUserId();
  if (!userId) return;
  clearLegacyUnscopedMealKeys();
  localStorage.setItem(resultsStorageKey(userId), JSON.stringify(data));
  localStorage.setItem(resultsDateKey(userId), new Date().toISOString().slice(0, 10));
}

function occasionRiskBadgeLabel(level: RiskLevel): { text: string; color: string } {
  const d = getRiskDisplay(level);
  const color = level === 'LOW' ? '#27AE60' : level === 'MODERATE' ? '#F39C12' : '#E74C3C';
  return { text: `${d.icon} ${d.label}`, color };
}

function assessmentDotClass(level: RiskLevel): string {
  if (level === 'LOW') return 'bg-green-500';
  if (level === 'MODERATE') return 'bg-amber-500';
  return 'bg-red-500';
}

interface NutrientBudgetData {
  consumed: number;
  limit: number;
  percent_used: number;
  remaining: number;
}

interface DailyBudgetData {
  ckd_stage: string;
  nutrients: Record<string, NutrientBudgetData>;
  warning_level: string;
  meals_logged_today: number;
  energy_kcal_today: number;
  categories_logged_today: Record<string, number>;
  suggestion_context?: { constraint_level: string; message: string };
  next_meal_occasion?: string | null;
  balanced_suggestions?: Array<{
    food_id: number;
    english: string;
    fallback?: boolean;
    tier?: number;
    plate_role?: string | null;
    option_index?: number;
  }>;
  date: string;
}

export function RiskAssessment({ isDark, theme, initialBodyWeight, initialStage }: RiskAssessmentProps) {
  const initialOccasion = readInitialMealOccasion();
  const initialStoredResults = loadStoredResultsByOccasion();
  const initialAssessment = initialStoredResults[initialOccasion];

  const readStoredStage = () => toCKDStage(localStorage.getItem('ckd_stage') ?? initialStage);
  const readStoredWeight = () => {
    const stored = localStorage.getItem('weight_kg');
    if (stored) return parseFloat(stored) || 65;
    return initialBodyWeight ?? 65;
  };

  const [stage, setStage] = useState<CKDStage>(readStoredStage);
  const [bodyWeightKg, setBodyWeightKg] = useState<number>(readStoredWeight);
  const [entries,         setEntries]         = useState<MealFoodItem[]>([]);
  const [resultsByOccasion, setResultsByOccasion] = useState(initialStoredResults);
  const [result,          setResult]          = useState<ResultState | null>(initialAssessment?.result ?? null);
  const [error,           setError]           = useState('');
  const [search,          setSearch]          = useState('');
  const [searchResults,   setSearchResults]   = useState<Food[]>([]);
  const [showDrop,        setShowDrop]        = useState(false);
  const [selectedFood,    setSelectedFood]    = useState<Food | null>(null);
  const [foodQty,         setFoodQty]         = useState<number>(1);
  const [isListening,     setIsListening]     = useState(false);
  const [voiceSupported,  setVoiceSupported]  = useState(true);
  const [voiceSpeechResult, setVoiceSpeechResult] = useState<VoiceSpeechResult | null>(null);
  const [currentMealType, setCurrentMealType] = useState<MealOccasion>(initialOccasion);
  const [logsByOccasion, setLogsByOccasion] = useState<Record<MealOccasion, FoodLog[]>>(EMPTY_LOGS_BY_OCCASION);
  const [occasionAddMode, setOccasionAddMode] = useState(false);
  const [deletingLogId, setDeletingLogId] = useState<string | null>(null);
  const [apiStatus,       setApiStatus]       = useState<'unknown' | 'connected' | 'unavailable'>('unknown');
  const [usingLiveModel,  setUsingLiveModel]  = useState<boolean>(initialAssessment?.usingLiveModel ?? false);
  const [modelConfidence, setModelConfidence] = useState<number | null>(initialAssessment?.modelConfidence ?? null);
  const [lstmPattern, setLstmPattern] = useState<{ risk_label: string; confidence: number; trend: string } | null>(
    initialAssessment?.lstmPattern ?? null,
  );
  const [dailyBudget,     setDailyBudget]     = useState<DailyBudgetData | null>(null);
  const [dailyBudgetLoading, setDailyBudgetLoading] = useState(true);
  const [dailyBudgetError, setDailyBudgetError] = useState<string | null>(null);
  const [resettingDay,    setResettingDay]    = useState(false);
  const [suggestions, setSuggestions] = useState<WeeklySuggestionsResponse | null>(null);
  const [suggestionTab, setSuggestionTab] = useState<string>('breakfast');
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const [entryQuantities, setEntryQuantities] = useState<Record<number, number>>({});
  const wrapRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);
  const currentMealTypeRef = useRef<MealOccasion>(currentMealType);

  useEffect(() => {
    currentMealTypeRef.current = currentMealType;
  }, [currentMealType]);

  useEffect(() => {
    if (result) return;
    setSuggestionTab(mealOccasionToSuggestionTab(currentMealType));
  }, [currentMealType, result]);

  useEffect(() => {
    if (!result) return;
    const showLstmAlternatives =
      result.level === 'HIGH' ||
      result.level === 'MODERATE' ||
      isDailyBudgetExceeded(dailyBudget);
    if (showLstmAlternatives) {
      setSuggestionTab(getNextOccasion(currentMealType));
    }
  }, [result, dailyBudget, currentMealType]);

  useEffect(() => {
    if (!getAuthToken()) {
      setSuggestions(null);
      return;
    }

    let cancelled = false;
    setSuggestions(null);
    setLoadingSuggestions(true);

    authFetch(`${API_BASE}/api/next-meal/weekly-suggestions`)
      .then(async (r) => {
        if (r.status === 401 || !r.ok) return null;
        return r.json() as Promise<WeeklySuggestionsResponse>;
      })
      .then((data) => {
        if (cancelled) return;
        const hasSuggestions = data?.suggestions_by_meal?.some(
          (group) => group.suggestions.length > 0,
        );
        if (hasSuggestions) {
          setSuggestionTab(
            result
              ? getNextOccasion(currentMealTypeRef.current)
              : mealOccasionToSuggestionTab(currentMealTypeRef.current),
          );
        }
        setSuggestions(hasSuggestions ? data : null);
      })
      .catch(() => {
        if (!cancelled) setSuggestions(null);
      })
      .finally(() => {
        if (!cancelled) setLoadingSuggestions(false);
      });

    return () => {
      cancelled = true;
    };
  }, [result]);

  const searchFoods = async (query: string) => {
    const q = query.trim();
    if (q.length < 2) {
      setSearchResults([]);
      return;
    }

    try {
      const res = await authFetch(
        `${API_BASE}/api/foods/search/${encodeURIComponent(q)}`,
      );
      if (res.ok) {
        const data = await res.json() as { results?: Record<string, unknown>[]; foods?: Record<string, unknown>[] };
        const rows = data.results ?? data.foods ?? [];
        setSearchResults(rows.slice(0, 8).map(mapApiFood));
        return;
      }
    } catch (err) {
      console.error('Food search failed:', err);
    }

    setSearchResults(
      FOODS.filter((f) => matchesFoodQuery(f, q)).slice(0, 8),
    );
  };

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void searchFoods(search);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [search]);

  const fetchDailyBudget = async () => {
    if (!getAuthToken()) return;

    setDailyBudgetLoading(true);
    setDailyBudgetError(null);

    try {
      const response = await authFetch(`${API_BASE}/api/patient/daily-budget`);
      if (response.status === 401) return;
      if (!response.ok) {
        console.warn('daily-budget request failed:', response.status, await response.text());
        setDailyBudget(null);
        setDailyBudgetError(
          response.status === 400 || response.status === 404
            ? 'Complete your profile to see your daily budget'
            : "Couldn't load your budget — try refreshing",
        );
        return;
      }
      setDailyBudget(await response.json());
      setDailyBudgetError(null);
    } catch (err) {
      console.warn('daily-budget request error:', err);
      setDailyBudget(null);
      setDailyBudgetError("Couldn't load your budget — try refreshing");
    } finally {
      setDailyBudgetLoading(false);
    }
  };

  const fetchTodayLogs = async () => {
    if (!getAuthToken()) return;

    try {
      const response = await authFetch(`${API_BASE}/api/patient/food-log/history`);
      if (response.status === 401) return;
      if (!response.ok) {
        console.warn('food-log history failed:', response.status, await response.text());
        return;
      }
      const logs: FoodLog[] = await response.json();
      const now = new Date();
      const todayLogs = logs.filter((log) => {
        if (!log.logged_at) return false;
        const d = new Date(log.logged_at);
        return d.toDateString() === now.toDateString();
      });
      const grouped = groupLogsByOccasion(todayLogs);
      setLogsByOccasion(grouped);
    } catch (err) {
      console.warn('food-log history error:', err);
    }
  };

  const deleteFoodLog = async (logId: string) => {
    if (!getAuthToken()) return;

    setDeletingLogId(logId);
    try {
      const response = await authFetch(`${API_BASE}/api/patient/food-log/${logId}`, {
        method: 'DELETE',
      });
      if (response.status === 401) return;
      if (!response.ok) {
        console.warn('delete food log failed:', response.status, await response.text());
        return;
      }
      clearOccasionAssessment(currentMealType);
      await fetchTodayLogs();
      await fetchDailyBudget();
    } catch (err) {
      console.warn('delete food log error:', err);
    } finally {
      setDeletingLogId(null);
    }
  };

  const applyAssessmentToDisplay = (assessment: OccasionAssessment | null) => {
    if (assessment) {
      setResult(assessment.result);
      setLstmPattern(assessment.lstmPattern);
      setUsingLiveModel(assessment.usingLiveModel);
      setModelConfidence(assessment.modelConfidence);
    } else {
      setResult(null);
      setLstmPattern(null);
      setUsingLiveModel(false);
      setModelConfidence(null);
    }
  };

  const commitOccasionAssessment = (
    occasion: MealOccasion,
    assessment: OccasionAssessment,
  ) => {
    setResultsByOccasion((prev) => {
      const next = { ...prev, [occasion]: assessment };
      persistResultsByOccasion(next);
      return next;
    });
    if (occasion === currentMealTypeRef.current) {
      applyAssessmentToDisplay(assessment);
    }
  };

  const clearOccasionAssessment = (occasion: MealOccasion) => {
    setResultsByOccasion((prev) => {
      const next = { ...prev, [occasion]: null };
      persistResultsByOccasion(next);
      return next;
    });
    if (occasion === currentMealTypeRef.current) {
      applyAssessmentToDisplay(null);
    }
  };

  const clearAllOccasionAssessments = () => {
    setResultsByOccasion({ ...EMPTY_RESULTS_BY_OCCASION });
    const userId = currentCacheUserId();
    if (userId) {
      localStorage.removeItem(resultsStorageKey(userId));
      localStorage.removeItem(resultsDateKey(userId));
    }
    clearLegacyUnscopedMealKeys();
    applyAssessmentToDisplay(null);
  };

  const selectMealType = (type: MealOccasion) => {
    setCurrentMealType(type);
    const userId = currentCacheUserId();
    if (userId) {
      clearLegacyUnscopedMealKeys();
      localStorage.setItem(mealOccasionStorageKey(userId), type);
    }
    setOccasionAddMode(false);
    setEntries([]);
    setError('');
    setSearch('');
    applyAssessmentToDisplay(resultsByOccasion[type]);
  };

  const reAssessOccasion = () => {
    clearOccasionAssessment(currentMealType);
    setOccasionAddMode(true);
    setEntries([]);
    setError('');
    setSearch('');
  };

  const mealOccasion = currentMealType;
  const showLoggedView =
    !occasionAddMode &&
    (logsByOccasion[mealOccasion] || []).length > 0;

  const totalLoggedToday = MEAL_TYPES.reduce(
    (sum, type) => sum + (logsByOccasion[type]?.length ?? 0),
    0,
  );

  useEffect(() => {
    void fetchDailyBudget();
    void fetchTodayLogs();
  }, []);

  useEffect(() => {
    setStage(toCKDStage(initialStage ?? localStorage.getItem('ckd_stage')));
    const storedWeight = localStorage.getItem('weight_kg');
    if (initialBodyWeight) {
      setBodyWeightKg(initialBodyWeight);
    } else if (storedWeight) {
      setBodyWeightKg(parseFloat(storedWeight) || 65);
    }
  }, [initialStage, initialBodyWeight]);

  useEffect(() => {
    if (!getAuthToken()) return;

    const loadProfile = async () => {
      try {
        const response = await authFetch(`${API_BASE}/api/patient/profile`);
        if (response.status === 401 || !response.ok) return;
        const data = await response.json();
        if (data.ckd_stage) {
          const nextStage = toCKDStage(data.ckd_stage);
          setStage(nextStage);
          localStorage.setItem('ckd_stage', nextStage);
        }
        if (typeof data.weight_kg === 'number' && data.weight_kg > 0) {
          setBodyWeightKg(data.weight_kg);
          localStorage.setItem('weight_kg', data.weight_kg.toString());
        }
      } catch {
        // keep stored values
      }
    };

    void loadProfile();
  }, []);

  useEffect(() => {
    fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(2000) })
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
      const transcript = String(event.results[0][0].transcript ?? '').trim();
      const { matched, unmatched } = extractFoodsFromSpeech(transcript, FOODS);
      setVoiceSpeechResult({
        transcript,
        matched: matched.map((food) => ({
          food,
          grams: getUnitInfo(food.english, food.category).grams,
        })),
        unmatched,
      });
      setSearch('');
      setShowDrop(false);
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
      setVoiceSpeechResult(null);
      recognitionRef.current.start();
      setIsListening(true);
    }
  };

  const updateVoiceMatchGrams = (foodId: number, grams: number) => {
    setVoiceSpeechResult((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        matched: prev.matched.map((item) =>
          item.food.id === foodId ? { ...item, grams: Math.max(10, grams) } : item,
        ),
      };
    });
  };

  const confirmVoiceFoods = () => {
    if (!voiceSpeechResult) return;

    const existingIds = new Set(entries.map((e) => e.food.id));
    const toAdd = voiceSpeechResult.matched.filter((item) => !existingIds.has(item.food.id));

    if (toAdd.length > 0) {
      setEntries((prev) => [...prev, ...toAdd]);
      if (occasionAddMode) {
        setResult(null);
        setLstmPattern(null);
        setUsingLiveModel(false);
        setModelConfidence(null);
      }
    }

    setVoiceSpeechResult(null);
    setSearch('');
    setShowDrop(false);
    setError('');
  };

  const cancelVoiceConfirmation = () => {
    setVoiceSpeechResult(null);
  };

  const q = search.trim();
  const visibleFoods = searchResults.filter(
    (f) => !entries.find((e) => e.food.id === f.id),
  );

  const selectFood = (food: Food) => {
    setSelectedFood(food);
    setFoodQty(1);
    setSearch('');
    setShowDrop(false);
  };

  const addSelectedFood = () => {
    if (!selectedFood) return;
    const unitInfo = getUnitInfo(selectedFood.english, selectedFood.category || 'Other');
    const computedGrams = Math.round(foodQty * unitInfo.grams);
    setEntries((prev) => [...prev, { food: selectedFood, grams: computedGrams }]);
    setEntryQuantities((prev) => ({ ...prev, [selectedFood.id]: foodQty }));
    setSelectedFood(null);
    setFoodQty(1);
    setSearch('');
    if (occasionAddMode) {
      setResult(null);
      setLstmPattern(null);
      setUsingLiveModel(false);
      setModelConfidence(null);
    }
    setError('');
  };

  const updateGrams = (id: number, delta: number) => {
    setEntries((prev) => prev.map((e) => e.food.id === id ? { ...e, grams: Math.max(10, e.grams + delta) } : e));
    setEntryQuantities((prev) => {
      const next = { ...prev };
      const entry = entries.find((e) => e.food.id === id);
      if (!entry) return prev;
      const unitInfo = getUnitInfo(entry.food.english, entry.food.category || 'Other');
      next[id] = Math.max(0.5, Math.min(10, Math.round(((entry.grams + delta) / unitInfo.grams) * 2) / 2));
      return next;
    });
    setResult(null);
  };

  const setGrams = (id: number, val: string) => {
    const n = parseInt(val, 10);
    if (!isNaN(n) && n > 0) {
      setEntries((prev) => prev.map((e) => e.food.id === id ? { ...e, grams: n } : e));
      setEntryQuantities((prev) => {
        const next = { ...prev };
        const entry = entries.find((e) => e.food.id === id);
        if (!entry) return prev;
        const unitInfo = getUnitInfo(entry.food.english, entry.food.category || 'Other');
        next[id] = Math.max(0.5, Math.min(10, Math.round((n / unitInfo.grams) * 2) / 2));
        return next;
      });
    }
    setResult(null);
  };

  const removeEntry = (id: number) => {
    setEntries((prev) => prev.filter((e) => e.food.id !== id));
    setEntryQuantities((prev) => {
      if (!(id in prev)) return prev;
      const next = { ...prev };
      delete next[id];
      return next;
    });
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

  const computeRisk = async (foodsOverride?: MealFoodItem[], skipSave = false) => {
    const occasion = currentMealType;
    const isAddMore = occasionAddMode && !foodsOverride;

    if (isAddMore && entries.length === 0) {
      setError('Add at least one new food item.');
      return;
    }

    const existingLoggedItems = (logsByOccasion[occasion] ?? [])
      .map(logToMealFoodItem)
      .filter((item): item is MealFoodItem => item !== null);

    const foodsToAssess = isAddMore
      ? [...existingLoggedItems, ...entries]
      : (foodsOverride ?? entries);

    if (foodsToAssess.length === 0) {
      setError('Add at least one food item to assess this meal.');
      return;
    }
    setError('');

    const assessedFoods = [...foodsToAssess];
    const mealTotals = sumMealNutrients(assessedFoods);
    const limits = getStageLimits(stage, bodyWeightKg);
    const primaryEntry = [...assessedFoods].sort(
      (a, b) => getFoodRiskScore(b, limits) - getFoodRiskScore(a, limits),
    )[0];
    const primaryFoodName = primaryEntry.food.english;

    let liveModel = false;
    let confidence: number | null = null;
    let lstm: { risk_label: string; confidence: number; trend: string } | null = null;

    if (apiStatus === 'connected') {
      try {
        const response = await fetch(`${API_BASE}/api/predict/risk`, {
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
        liveModel = true;
        confidence = typeof apiResult.confidence === 'number' ? apiResult.confidence : null;

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

        const resultState: ResultState = {
          level,
          score: scoreFromBreakdown(breakdown),
          breakdown,
          assessedFoods,
          substitutions,
          shap_explanation: apiResult.shap_explanation,
          shap_contributions: apiResult.shap_contributions,
          shap_dominant_nutrient: apiResult.shap_dominant_nutrient,
        };

        const newEntriesToSave = isAddMore ? [...entries] : assessedFoods;

        if (!skipSave) {
          setEntries([]);
          setOccasionAddMode(false);
          if (newEntriesToSave.length > 0) {
            await saveMealFoodLogs(newEntriesToSave, occasion);
          }
        } else {
          setOccasionAddMode(false);
        }
        void saveRiskAssessment(level, apiResult.confidence, mealTotals, {
          shap_contributions: apiResult.shap_contributions ?? null,
          shap_explanation: apiResult.shap_explanation ?? null,
          ckd_stage: stage,
          bodyWeightKg,
        });

        try {
          const token = getAuthToken();
          if (token) {
            const patternResponse = await fetch(`${API_BASE}/api/predict/pattern`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${token}`,
              },
              body: JSON.stringify({
                meal_sequence: [
                  {
                    potassium: mealTotals.potassium,
                    phosphorus: mealTotals.phosphorus,
                    protein_per_kg: mealTotals.protein / bodyWeightKg,
                    sodium: mealTotals.sodium,
                    occasion_encoded: OCCASION_ENCODING[occasion] ?? 0.5,
                  },
                ],
                ckd_stage: stage,
              }),
              signal: AbortSignal.timeout(5000),
            });
            if (patternResponse.ok) {
              const patternJson = await patternResponse.json();
              if (patternJson?.risk_label && typeof patternJson?.confidence === 'number') {
                lstm = {
                  risk_label: String(patternJson.risk_label),
                  confidence: Number(patternJson.confidence),
                  trend: String(patternJson.trend ?? ''),
                };
              }
            }
          }
        } catch (err) {
          console.warn('LSTM pattern request failed:', err);
        }

        commitOccasionAssessment(occasion, {
          result: resultState,
          lstmPattern: lstm,
          usingLiveModel: liveModel,
          modelConfidence: confidence,
        });

        await fetchDailyBudget();
        await fetchTodayLogs();
        return;
      } catch {
        liveModel = false;
        confidence = null;
        lstm = null;
      }
    }

    const breakdown: Record<string, BreakdownEntry> = {
      Potassium:  { value: Math.round(mealTotals.potassium),  limit: limits.potassium,  pct: 0, status: 'Safe' },
      Phosphorus: { value: Math.round(mealTotals.phosphorus), limit: limits.phosphorus, pct: 0, status: 'Safe' },
      Protein:    { value: +mealTotals.protein.toFixed(1),    limit: limits.protein,    pct: 0, status: 'Safe' },
      Sodium:     { value: Math.round(mealTotals.sodium),     limit: limits.sodium,     pct: 0, status: 'Safe' },
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

    const resultState: ResultState = {
      level,
      score: fallbackScore,
      breakdown,
      assessedFoods,
      substitutions: getSmartSubstitutions(assessedFoods, stage, limits),
    };

    const newEntriesToSave = isAddMore ? [...entries] : assessedFoods;

    if (!skipSave) {
      setEntries([]);
      setOccasionAddMode(false);
      if (newEntriesToSave.length > 0) {
        await saveMealFoodLogs(newEntriesToSave, occasion);
      }
    } else {
      setOccasionAddMode(false);
    }
    void saveRiskAssessment(level, fallbackScore / 100, mealTotals, {
      ckd_stage: stage,
      bodyWeightKg,
    });

    commitOccasionAssessment(occasion, {
      result: resultState,
      lstmPattern: null,
      usingLiveModel: false,
      modelConfidence: null,
    });

    await fetchDailyBudget();
    await fetchTodayLogs();
  };

  const assessLoggedOccasion = () => {
    const logs = logsByOccasion[currentMealType] ?? [];
    const items = logs
      .map(logToMealFoodItem)
      .filter((item): item is MealFoodItem => item !== null);
    if (items.length === 0) {
      setError('No foods found for this meal occasion.');
      return;
    }
    void computeRisk(items, true);
  };

  const reset = () => {
    setEntries([]);
    setError('');
    setSearch('');
    if (occasionAddMode) {
      setOccasionAddMode(false);
    } else {
      applyAssessmentToDisplay(resultsByOccasion[currentMealType]);
    }
  };

  const resetDay = async () => {
    if (!getAuthToken()) return;
    if (!window.confirm("Reset all food logs for today? This cannot be undone.")) return;

    setResettingDay(true);
    try {
      const response = await authFetch(`${API_BASE}/api/patient/food-log/day`, {
        method: 'DELETE',
      });
      if (response.status === 401) return;
      if (!response.ok) {
        console.error('Reset failed:', await response.text());
        return;
      }

      setEntries([]);
      setSearch('');
      setError('');
      setLogsByOccasion({ ...EMPTY_LOGS_BY_OCCASION });
      clearAllOccasionAssessments();
      await fetchDailyBudget();
      await fetchTodayLogs();
    } catch (err) {
      console.error('Reset error:', err);
    } finally {
      setResettingDay(false);
    }
  };

  const hasTodayLogs = MEAL_TYPES.some(
    (occasion) => (logsByOccasion[occasion]?.length ?? 0) > 0,
  );
  const canResetToday =
    hasTodayLogs || (dailyBudget?.meals_logged_today ?? 0) > 0;

  const nutrientSummary = [
    { label: 'Potassium',  value: Math.round(totals.potassium),  limit: thresholds.potassium,  unit: 'mg', color: '#2E86AB' },
    { label: 'Phosphorus', value: Math.round(totals.phosphorus), limit: thresholds.phosphorus, unit: 'mg', color: '#F39C12' },
    { label: 'Protein',    value: +totals.protein.toFixed(1),    limit: thresholds.protein,    unit: 'g',  color: '#27AE60' },
    { label: 'Sodium',     value: Math.round(totals.sodium),     limit: thresholds.sodium,     unit: 'mg', color: '#E74C3C' },
  ];

  const renderWeeklySuggestionsPanel = (hasCheckedMeal: boolean) => {
    if (!suggestions || loadingSuggestions) return null;
    return (
      <div className="bg-teal-50 rounded-lg p-3 flex flex-col">
        <div className="text-sm font-semibold text-teal-700" style={{ marginBottom: 4 }}>
          What to eat next
        </div>
        <p className="text-xs text-muted-foreground mb-3">
          {getSuggestionsSubtitle(hasCheckedMeal, suggestions)}
        </p>

        {!suggestions.analysis_available && (
          <div className="rounded-lg bg-amber-50 border border-amber-200 p-3 mb-3 flex items-center gap-2">
            <span className="text-amber-600 text-sm font-medium">
              ⚠ Pattern analysis temporarily unavailable
            </span>
            <span className="text-amber-500 text-xs">
              Showing general safe foods for your stage
            </span>
          </div>
        )}

        <Tabs value={suggestionTab} onValueChange={setSuggestionTab}>
          <TabsList className="w-full grid grid-cols-4 h-auto mb-2">
            {suggestions.suggestions_by_meal.map((group) => (
              <TabsTrigger key={group.occasion} value={group.occasion} className="text-xs sm:text-sm">
                {SUGGESTION_OCCASION_LABELS[group.occasion] || group.occasion}
              </TabsTrigger>
            ))}
          </TabsList>

          {suggestions.suggestions_by_meal.map((group) => {
            const foods = group.suggestions.slice(0, 4);
            return (
              <TabsContent key={group.occasion} value={group.occasion} className="mt-0">
                {foods.length === 0 ? (
                  <p className="text-sm text-muted-foreground py-2">No suggestions available</p>
                ) : (
                  <ul className="grid grid-cols-2 gap-2">
                    {foods.map((food) => (
                      <li key={`${group.occasion}-${food.english}`} className="min-w-0">
                        <p className="text-sm font-semibold text-foreground capitalize leading-snug">
                          {food.english}
                        </p>
                        {food.category && (
                          <p className="text-[11px] text-muted-foreground leading-tight">{food.category}</p>
                        )}
                        <p className="text-xs text-muted-foreground mt-0.5 leading-snug">{food.reason}</p>
                      </li>
                    ))}
                  </ul>
                )}
              </TabsContent>
            );
          })}
        </Tabs>

        {suggestions.clinical_note && (
          <p className="text-xs italic text-muted-foreground mt-3 leading-relaxed">
            {suggestions.clinical_note}
          </p>
        )}
      </div>
    );
  };

  const renderDailyBudgetGrid = () => {
    if (!dailyBudget) {
      return <p className="text-sm text-muted-foreground mt-4">Loading…</p>;
    }
    return (
      <div className="grid grid-cols-[minmax(0,1fr)_auto_auto] gap-x-2 gap-y-1 mt-4">
        {(['potassium', 'phosphorus', 'protein_per_kg', 'sodium'] as const).map((nutrient) => {
          const data = dailyBudget.nutrients[nutrient];
          if (!data) return null;
          const pct = data.percent_used;
          const status = pct >= 100 ? 'over' : pct >= 70 ? 'near' : 'ok';
          const pctColor = pct >= 100 ? '#ef4444' : pct >= 70 ? '#f59e0b' : '#6b7280';
          const statusColor = pct >= 100 ? '#ef4444' : pct >= 70 ? '#f59e0b' : '#22c55e';
          const label = nutrient === 'protein_per_kg' ? 'Protein'
            : nutrient.charAt(0).toUpperCase() + nutrient.slice(1);

          return (
            <Fragment key={nutrient}>
              <span className="text-sm font-medium" style={{ color: theme.text }}>{label}</span>
              <span className="text-sm font-semibold text-right whitespace-nowrap" style={{ color: pctColor }}>
                {pct.toFixed(0)}%
              </span>
              <span className="text-sm text-right whitespace-nowrap" style={{ color: statusColor }}>
                {displayBudgetStatus(status)}
              </span>
            </Fragment>
          );
        })}
      </div>
    );
  };

  return (
    <div className="w-full min-w-0 space-y-4 lg:space-y-3">
      {/* Header */}
      <div>
        <div style={{ color: theme.text, fontSize: '1.4rem', fontWeight: 600 }}>Meal Check</div>
        <p style={{ color: theme.textSecondary, marginTop: 4, fontSize: '0.9rem' }}>
          Add the foods you just ate and get instant feedback on whether your meal is safe for your kidneys
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4 lg:gap-5 w-full min-w-0 items-start">
        {/* ── Left: meal builder ─────────────────────── */}
        <div className="w-full min-w-0 flex flex-col lg:col-span-2 lg:sticky lg:top-24 lg:self-start lg:max-h-[calc(100vh-6.5rem)]">

          {/* Meal builder */}
          <div className="flex flex-col flex-1 min-h-0 p-4 lg:p-4 rounded-2xl overflow-y-auto" style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}>
            <div className="flex-1 min-h-0">
            <div style={{ color: theme.text, fontWeight: 600, marginBottom: 4 }}>What did you eat?</div>
            <p style={{ color: theme.textSecondary, fontSize: '0.78rem', marginBottom: 14 }}>
              Search for a food and add how much you ate
            </p>

            <div className="mb-4 w-full">
              <p style={{ color: theme.textSecondary, fontSize: '0.72rem', marginBottom: 8 }}>When did you eat this?</p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5 w-full mt-1">
                {MEAL_TYPES.map((occasion) => (
                  <button
                    key={occasion}
                    onClick={() => selectMealType(occasion)}
                    className={`w-full py-2 text-sm rounded-lg border transition-colors inline-flex items-center justify-center gap-0.5 ${
                      mealOccasion === occasion
                        ? 'border-teal-600 text-teal-600 font-semibold bg-teal-50'
                        : 'border-muted text-foreground bg-background hover:border-teal-400'
                    }`}
                  >
                    {occasion}
                    {resultsByOccasion[occasion] ? (
                      <span
                        className={`shrink-0 w-2 h-2 rounded-full inline-block ${assessmentDotClass(resultsByOccasion[occasion]!.result.level)}`}
                        title={`${getRiskDisplay(resultsByOccasion[occasion]!.result.level).label} — checked`}
                      />
                    ) : logsByOccasion[occasion].length > 0 ? (
                      <span className="shrink-0 text-xs bg-teal-500 text-white rounded-full px-1 min-w-[1rem] text-center leading-4">
                        {logsByOccasion[occasion].length}
                      </span>
                    ) : null}
                  </button>
                ))}
              </div>
            </div>

            {showLoggedView ? (
              <div
                className="rounded-xl p-4 mb-4"
                style={{ background: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.03)', border: `1px solid ${theme.cardBorder}` }}
              >
                <div className="flex items-center justify-between gap-2 mb-1">
                  <div className="flex items-center gap-2 min-w-0">
                    <span style={{ fontSize: '1.1rem' }}>{MEAL_OCCASION_ICONS[currentMealType]}</span>
                    <span style={{ color: theme.text, fontWeight: 600, fontSize: '0.95rem' }}>{currentMealType}</span>
                  </div>
                  {resultsByOccasion[currentMealType] && (() => {
                    const mealLevel = resultsByOccasion[currentMealType]!.result.level;
                    if (isDailyBudgetExceeded(dailyBudget) && mealLevel === 'LOW') {
                      return (
                        <span className="shrink-0 text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full">
                          Over Budget
                        </span>
                      );
                    }
                    const badge = occasionRiskBadgeLabel(mealLevel);
                    return (
                      <span className="shrink-0 text-xs font-semibold" style={{ color: badge.color }}>
                        {badge.text}
                      </span>
                    );
                  })()}
                </div>
                <p style={{ color: theme.textSecondary, fontSize: '0.78rem', marginBottom: 14 }}>
                  {logsByOccasion[currentMealType].length} food{logsByOccasion[currentMealType].length !== 1 ? 's' : ''} logged
                </p>
                <div className="space-y-2 mb-4">
                  {logsByOccasion[currentMealType].map((log) => {
                    const kMg = Math.round(log.potassium_mg ?? 0);
                    const kColor = potassiumStatusColor(kMg, thresholds.potassium);
                    const dailyOverride = isDailyBudgetExceeded(dailyBudget);
                    const isPotassiumSafe = kColor === '#27AE60';
                    return (
                      <div
                        key={log.log_id}
                        className="flex items-center gap-2 py-2 px-2 rounded-lg"
                        style={{ borderBottom: `1px solid ${theme.cardBorder}` }}
                      >
                        <span className="flex-1 min-w-0 truncate font-medium capitalize" style={{ color: theme.text, fontSize: '0.85rem' }}>
                          {log.food_name}
                        </span>
                        <span className="text-sm shrink-0" style={{ color: theme.textSecondary }}>
                          {Math.round(log.portion_grams ?? 0)}g
                        </span>
                        {dailyOverride ? (
                          <span className="text-sm shrink-0" style={{ color: theme.textSecondary }}>
                            {kMg}mg K
                          </span>
                        ) : (
                          <span className="text-sm shrink-0 font-medium" style={{ color: isPotassiumSafe ? '#27AE60' : kColor }}>
                            {kMg}mg K
                            {isPotassiumSafe && ' ✔ Safe'}
                          </span>
                        )}
                        <button
                          type="button"
                          onClick={() => void deleteFoodLog(log.log_id)}
                          disabled={deletingLogId === log.log_id}
                          className="shrink-0 w-6 h-6 flex items-center justify-center rounded hover:opacity-70 disabled:opacity-40"
                          style={{ color: theme.textTertiary }}
                          aria-label={`Remove ${log.food_name}`}
                        >
                          ×
                        </button>
                      </div>
                    );
                  })}
                </div>
                <div className="flex flex-col gap-2">
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        setOccasionAddMode(true);
                        setEntries([]);
                        setError('');
                      }}
                      className="flex-1 py-2.5 rounded-xl text-sm font-medium transition-opacity hover:opacity-80"
                      style={{ background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.05)', border: `1px solid ${theme.cardBorder}`, color: theme.text }}
                    >
                      + Add another food
                    </button>
                    {resultsByOccasion[currentMealType] ? (
                      <button
                        type="button"
                        onClick={reAssessOccasion}
                        className="flex-1 py-2.5 rounded-xl text-sm font-medium transition-opacity hover:opacity-80"
                        style={{ background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.05)', border: `1px solid ${theme.cardBorder}`, color: theme.text }}
                      >
                        ↺ Re-check
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={() => assessLoggedOccasion()}
                        className="flex-1 flex items-center justify-center gap-1 py-2.5 rounded-xl text-white text-sm font-semibold transition-opacity hover:opacity-90"
                        style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)' }}
                      >
                        ✓ Checked
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <>
            {occasionAddMode && (logsByOccasion[currentMealType]?.length ?? 0) > 0 && (
              <div
                className="rounded-xl px-3 py-2.5 mb-4"
                style={{ background: isDark ? 'rgba(46,134,171,0.08)' : 'rgba(46,134,171,0.06)', border: '1px solid rgba(46,134,171,0.2)' }}
              >
                <p style={{ color: theme.text, fontSize: '0.78rem', fontWeight: 600 }}>
                  Adding to {currentMealType}
                </p>
                <p style={{ color: theme.textSecondary, fontSize: '0.72rem', marginTop: 4 }}>
                  {logsByOccasion[currentMealType].length} food{logsByOccasion[currentMealType].length !== 1 ? 's' : ''} already logged — search below to add more, then assess the full meal.
                </p>
              </div>
            )}
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
                  🎤 Tip: describe your full meal, e.g. &quot;I had rice, beans and chicken for lunch&quot;
                </p>
              )}

              {voiceSpeechResult && (
                <div
                  className="mt-3 p-3 rounded-xl"
                  style={{
                    background: isDark ? 'rgba(46,134,171,0.08)' : 'rgba(46,134,171,0.06)',
                    border: `1px solid ${theme.cardBorder}`,
                  }}
                >
                  <p style={{ color: theme.textSecondary, fontSize: '0.75rem', marginBottom: 10 }}>
                    I heard: &quot;{voiceSpeechResult.transcript}&quot;
                  </p>
                  <p style={{ color: theme.text, fontSize: '0.82rem', fontWeight: 600, marginBottom: 8 }}>
                    Found these foods:
                  </p>

                  {voiceSpeechResult.matched.length > 0 && (
                    <div className="space-y-2 mb-2">
                      {voiceSpeechResult.matched.map(({ food, grams }) => (
                        <div
                          key={food.id}
                          className="flex items-center gap-2 py-1.5"
                          style={{ borderBottom: `1px solid ${theme.cardBorder}` }}
                        >
                          <span style={{ color: '#27AE60', fontSize: '0.8rem', flexShrink: 0 }}>✓</span>
                          <span className="flex-1 min-w-0 truncate capitalize font-medium" style={{ color: theme.text, fontSize: '0.82rem' }}>
                            {food.english}
                          </span>
                          <input
                            type="number"
                            className="w-14 bg-transparent outline-none text-center rounded border px-1"
                            style={{ color: theme.text, fontSize: '0.78rem', borderColor: theme.cardBorder }}
                            value={grams}
                            min={10}
                            onChange={(e) => {
                              const n = parseInt(e.target.value, 10);
                              if (!isNaN(n)) updateVoiceMatchGrams(food.id, n);
                            }}
                          />
                          <span style={{ color: theme.textSecondary, fontSize: '0.72rem' }}>g</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {voiceSpeechResult.unmatched.length > 0 && (
                    <p
                      className="text-xs italic mt-2"
                      style={{ color: theme.textSecondary }}
                    >
                      Not recognized: {voiceSpeechResult.unmatched.join(', ')}
                    </p>
                  )}

                  {voiceSpeechResult.matched.length === 0 && voiceSpeechResult.unmatched.length === 0 && (
                    <p style={{ color: theme.textSecondary, fontSize: '0.78rem', marginBottom: 8 }}>
                      No foods recognized — try describing specific items from the database.
                    </p>
                  )}

                  <div className="flex gap-2 mt-3">
                    <button
                      type="button"
                      onClick={confirmVoiceFoods}
                      disabled={voiceSpeechResult.matched.length === 0}
                      className="flex-1 py-2 rounded-lg text-white text-sm font-semibold transition-opacity hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed"
                      style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)' }}
                    >
                      Add all
                    </button>
                    <button
                      type="button"
                      onClick={cancelVoiceConfirmation}
                      className="flex-1 py-2 rounded-lg text-sm font-medium transition-opacity hover:opacity-80"
                      style={{ background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.05)', border: `1px solid ${theme.cardBorder}`, color: theme.text }}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}

              {showDrop && q.length >= 2 && visibleFoods.length > 0 && !voiceSpeechResult && !selectedFood && (
                <div
                  className="absolute top-full mt-1.5 left-0 right-0 z-30 rounded-xl overflow-hidden shadow-xl"
                  style={{ background: isDark ? '#111827' : '#fff', border: `1px solid ${theme.cardBorder}`, maxHeight: '300px', overflowY: 'auto' }}
                >
                  {visibleFoods.map((f, i) => (
                    <button
                      key={f.id}
                      className="w-full text-left px-4 py-2.5 flex items-center justify-between gap-3 transition-colors"
                      style={{ borderBottom: i < visibleFoods.length - 1 ? `1px solid ${theme.cardBorder}` : 'none' }}
                      onMouseDown={() => selectFood(f)}
                      onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)'; }}
                      onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = 'transparent'; }}
                    >
                      <div className="min-w-0">
                        <p style={{ color: theme.text, fontSize: '0.85rem', fontWeight: 600 }} className="truncate capitalize">{f.english}</p>
                        {foodTranslation(f.french) && (
                          <p style={{ color: theme.text, fontSize: '0.78rem', marginTop: 2 }} className="truncate">{foodTranslation(f.french)}</p>
                        )}
                        {foodTranslation(f.kinyarwanda) && (
                          <p style={{ color: theme.textSecondary, fontSize: '0.72rem', marginTop: 2 }} className="truncate italic">{foodTranslation(f.kinyarwanda)}</p>
                        )}
                      </div>
                      <span className="shrink-0 px-2 py-0.5 rounded-full" style={{ background: potassiumColor(f.potassium_mg) + '20', color: potassiumColor(f.potassium_mg), fontSize: '0.65rem', fontWeight: 600 }}>
                        {f.category}
                      </span>
                    </button>
                  ))}
                </div>
              )}

              {selectedFood && (() => {
                const unitInfo = getUnitInfo(selectedFood.english, selectedFood.category || 'Other');
                const computedGrams = Math.round(foodQty * unitInfo.grams);
                const unitLabel = `${foodQty} ${unitInfo.unit}${foodQty !== 1 && !unitInfo.unit.endsWith('s') ? 's' : ''}`;
                return (
                  <div
                    className="flex items-center gap-2 mt-2 px-3 py-2.5 rounded-xl"
                    style={{ background: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.03)', border: `1px solid ${theme.cardBorder}` }}
                  >
                    <p className="flex-1 min-w-0 truncate capitalize" style={{ color: theme.text, fontSize: '0.85rem', fontWeight: 600 }}>
                      {selectedFood.english}
                    </p>
                    <div className="flex items-center gap-1.5 shrink-0">
                      <button
                        type="button"
                        onClick={() => setFoodQty((q) => Math.max(0.5, q - 0.5))}
                        className="w-6 h-6 rounded-full text-sm font-bold flex items-center justify-center hover:opacity-70"
                        style={{ border: `1px solid ${theme.cardBorder}`, color: theme.textSecondary }}
                      >
                        −
                      </button>
                      <span className="text-sm font-medium min-w-[70px] text-center" style={{ color: theme.text }}>
                        {unitLabel}
                      </span>
                      <button
                        type="button"
                        onClick={() => setFoodQty((q) => q + 0.5)}
                        className="w-6 h-6 rounded-full text-sm font-bold flex items-center justify-center hover:opacity-70"
                        style={{ border: `1px solid ${theme.cardBorder}`, color: theme.textSecondary }}
                      >
                        +
                      </button>
                      <span className="text-xs ml-1" style={{ color: theme.textSecondary }}>
                        ({computedGrams}g)
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={addSelectedFood}
                      className="px-3 py-1.5 rounded-lg text-white text-xs font-medium transition-opacity hover:opacity-90 shrink-0"
                      style={{ background: '#0d9488' }}
                    >
                      Add
                    </button>
                  </div>
                );
              })()}
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
                      <div className="flex-1 min-w-0 flex items-center gap-2">
                        <p className="min-w-0 truncate" style={{ color: theme.text, fontSize: '0.82rem', fontWeight: 600 }}>{food.english}</p>
                        {food.preparation_method && String(food.preparation_method).trim().length > 0 && (
                          <span className="text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded font-normal shrink-0">
                            {String(food.preparation_method)}
                          </span>
                        )}
                      </div>
                      <span style={{ color: theme.textTertiary, fontSize: '0.68rem' }}>{food.category}</span>
                      <button onClick={() => removeEntry(food.id)} className="ml-1 transition-opacity hover:opacity-60">
                        <Trash2 size={12} style={{ color: theme.textTertiary }} />
                      </button>
                    </div>
                    {/* Gram control row */}
                    <div className="flex flex-wrap items-center gap-2 px-3 py-2">
                      {(() => {
                        const unitInfo = getUnitInfo(food.english, food.category || 'Other');
                        const qtyFromGrams = unitInfo.grams > 0 ? grams / unitInfo.grams : 1;
                        const fallbackQty = Math.max(0.5, Math.min(10, Math.round(qtyFromGrams * 2) / 2));
                        const qty = entryQuantities[food.id] ?? fallbackQty;
                        const computedGrams = Math.round(qty * unitInfo.grams);

                        const dec = () => {
                          const nextQty = Math.max(0.5, Math.round((qty - 0.5) * 2) / 2);
                          setEntryQuantities((prev) => ({ ...prev, [food.id]: nextQty }));
                          setEntries((prev) => prev.map((e) => e.food.id === food.id ? { ...e, grams: Math.round(nextQty * unitInfo.grams) } : e));
                          setResult(null);
                        };

                        const inc = () => {
                          const nextQty = Math.min(10, Math.round((qty + 0.5) * 2) / 2);
                          setEntryQuantities((prev) => ({ ...prev, [food.id]: nextQty }));
                          setEntries((prev) => prev.map((e) => e.food.id === food.id ? { ...e, grams: Math.round(nextQty * unitInfo.grams) } : e));
                          setResult(null);
                        };

                        const unitLabel = `${unitInfo.unit}${qty > 1 ? 's' : ''}`;
                        const method = String(food.preparation_method ?? '').toLowerCase();
                        const category = String(food.category ?? '');
                        const showBoiledTip = method.includes('boiled') && (category === 'Starch' || category === 'Vegetable');
                        const showFriedTip = method.includes('fried');
                        const showRawMeatTip = method.includes('raw') && (category === 'Meat' || category === 'Fish');

                        return (
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <button
                                type="button"
                                onClick={dec}
                                className="w-7 h-7 rounded-full border border-gray-300 text-sm flex items-center justify-center hover:bg-gray-100"
                              >
                                −
                              </button>
                              <div className="flex flex-col items-center min-w-[90px]">
                                <span className="text-sm font-medium text-center">
                                  {qty} {unitLabel}
                                </span>
                                <span className="text-xs text-gray-400">
                                  (~{computedGrams}g)
                                </span>
                              </div>
                              <button
                                type="button"
                                onClick={inc}
                                className="w-7 h-7 rounded-full border border-gray-300 text-sm flex items-center justify-center hover:bg-gray-100"
                              >
                                +
                              </button>
                            </div>
                            {showBoiledTip && (
                              <p className="text-xs text-teal-600 mt-1">
                                Tip: boiling and discarding water reduces potassium content.
                              </p>
                            )}
                            {showFriedTip && (
                              <p className="text-xs text-amber-600 mt-1">
                                Tip: frying increases sodium and energy content.
                              </p>
                            )}
                            {showRawMeatTip && (
                              <p className="text-xs text-amber-600 mt-1">
                                Tip: cook thoroughly before eating.
                              </p>
                            )}
                          </div>
                        );
                      })()}
                      {/* Inline nutrient preview */}
                      <div className="w-full sm:w-auto sm:flex-1 flex flex-wrap gap-x-3 gap-y-1 justify-start sm:justify-end min-w-0">
                        {[
                          { label: 'Potassium', value: Math.round(food.potassium_mg * grams / 100), color: '#2E86AB' },
                          { label: 'Phosphorus', value: Math.round(food.phosphorus_mg * grams / 100), color: '#F39C12' },
                          { label: 'Sodium', value: Math.round(food.sodium_mg * grams / 100), color: '#E74C3C' },
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

            <div className="flex gap-3 mt-4">
              <button
                onClick={() => void computeRisk()}
                className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl text-white transition-all duration-200 hover:opacity-90 active:scale-[0.98]"
                style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)', fontWeight: 600, fontSize: '0.9rem' }}
              >
                <Zap size={15} />
                Check this meal
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
              </>
            )}

            {error && (
              <div className="flex items-center gap-2 mt-3 px-3 py-2.5 rounded-xl" style={{ background: 'rgba(231,76,60,0.1)', border: '1px solid rgba(231,76,60,0.3)' }}>
                <AlertTriangle size={13} style={{ color: '#E74C3C', flexShrink: 0 }} />
                <p style={{ color: '#E74C3C', fontSize: '0.8rem' }}>{error}</p>
              </div>
            )}
            </div>
          </div>

          <div className="mt-auto pt-3">
            <div className="p-3 rounded-xl" style={{ background: isDark ? 'rgba(46,134,171,0.06)' : 'rgba(46,134,171,0.05)', border: '1px solid rgba(46,134,171,0.15)' }}>
              <div className="flex gap-2">
                <Info size={12} style={{ color: '#2E86AB', flexShrink: 0, marginTop: 1 }} />
                <p style={{ color: theme.textSecondary, fontSize: '0.72rem', lineHeight: 1.5 }}>
                  Food values are estimates per 100g. Always consult your doctor or dietitian before making changes to your diet.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* ── Right: Results ────────────────────────────────────── */}
        <div className="w-full min-w-0 lg:col-span-3 space-y-2 lg:space-y-2">
          {!result ? (
            <div
              className="rounded-2xl p-5 sm:p-6 flex flex-col gap-4 min-h-[420px] sm:min-h-[480px]"
              style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}
            >
              {suggestions && !loadingSuggestions && (
                <div className="flex-[11] min-h-0">
                  {renderWeeklySuggestionsPanel(false)}
                </div>
              )}
              <div className={suggestions && !loadingSuggestions ? 'flex-[9] min-h-0' : 'flex-1'}>
                <div style={{ color: theme.text, fontWeight: 600, fontSize: '1.05rem' }}>Today&apos;s nutrient budget</div>
                <p style={{ color: theme.textSecondary, marginTop: 8, fontSize: '0.875rem', lineHeight: 1.6 }}>
                  Add a meal to see how it affects your daily limits
                </p>
                {renderDailyBudgetGrid()}
              </div>
            </div>
          ) : (
            <>
              <p className="text-xs text-muted-foreground mb-2">
                {currentMealType} ·{' '}
                {new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })}
              </p>
              {/* Risk banner */}
              {(() => {
                const dailyExceeded =
                  !!dailyBudget &&
                  ['potassium', 'phosphorus', 'protein_per_kg', 'sodium'].some((k) => {
                    const n = dailyBudget.nutrients[k];
                    return typeof n?.percent_used === 'number' && n.percent_used > 100;
                  });

                const overrideToCaution = result.level === 'LOW' && dailyExceeded;
                const cfg = overrideToCaution ? RISK_CFG.MODERATE : RISK_CFG[result.level];
                const Icon = cfg.icon;
                return (
                  <div className="p-3 rounded-2xl" style={{ background: cfg.bg, border: `2px solid ${cfg.border}` }}>
                    <div className="flex items-start sm:items-center gap-2">
                      <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-2xl flex items-center justify-center shrink-0" style={{ background: `${cfg.color}20` }}>
                        <Icon size={22} style={{ color: cfg.color }} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap mb-1">
                          <span style={{ color: cfg.color, fontSize: 'clamp(1.05rem,3vw,1.25rem)', fontWeight: 700 }}>{cfg.label}</span>
                          <span className="px-2.5 py-1 rounded-full" style={{ background: `${cfg.color}20`, color: cfg.color, fontSize: '0.7rem', fontWeight: 600 }}>
                            Using {result.score}% of your daily allowance
                          </span>
                          {usingLiveModel && (
                            <span className="px-2.5 py-1 rounded-full" style={{ background: 'rgba(46,134,171,0.12)', color: '#2E86AB', fontSize: '0.66rem', fontWeight: 600 }}>
                              Checked by AI risk assessment
                            </span>
                          )}
                        </div>
                        <p style={{ color: theme.textSecondary, fontSize: '0.78rem' }}>
                          {overrideToCaution
                            ? "This meal is safe, but your daily nutrient limit has been reached. Be careful with your next meal."
                            : cfg.desc}
                        </p>
                        <p style={{ color: theme.textTertiary, fontSize: '0.7rem', marginTop: 1 }}>
                          {result.assessedFoods.length} food{result.assessedFoods.length !== 1 ? 's' : ''} · {result.assessedFoods.reduce((a, e) => a + e.grams, 0)} g total
                        </p>
                      </div>
                      <div className="text-right shrink-0 hidden sm:block">
                        <p style={{ color: theme.textTertiary, fontSize: '0.7rem', marginBottom: 2 }}>Your Stage</p>
                        <p style={{ color: '#2E86AB', fontWeight: 700, fontSize: '1.1rem' }}>{formatStageDisplay(stage)}</p>
                      </div>
                    </div>
                    <div className="mt-1">
                      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-1 mb-1.5">
                        <span style={{ color: theme.textSecondary, fontSize: '0.74rem' }}>How much of your daily allowance this meal uses</span>
                        <span style={{ color: cfg.color, fontWeight: 600, fontSize: '0.74rem' }}>{result.score}%</span>
                      </div>
                      <div className="rounded-full overflow-hidden" style={{ height: 5, background: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)' }}>
                        <div style={{ width: `${result.score}%`, height: 5, background: `linear-gradient(90deg,${cfg.color}80,${cfg.color})`, borderRadius: 9999, transition: 'width 0.6s ease' }} />
                      </div>
                    </div>
                    {lstmPattern?.risk_label && (() => {
                      const patternLevel = lstmPattern.risk_label as RiskLevel;
                      const patternCfg = RISK_CFG[patternLevel] ?? RISK_CFG.MODERATE;
                      const patternPhrase = getWeeklyRiskLabel(lstmPattern.risk_label).toLowerCase();
                      const mealAloneNote =
                        result.level === 'LOW'
                          ? 'this meal alone looks safe'
                          : result.level === 'HIGH'
                            ? 'this meal alone also suggests reducing intake'
                            : 'this meal alone suggests caution';
                      const mealsToday = dailyBudget?.meals_logged_today ?? result.assessedFoods.length;
                      return (
                        <>
                          <div className="mt-2 pt-2" style={{ borderTop: `1px solid ${cfg.border}` }} />
                          <div
                            className="mt-2 leading-snug"
                            title="Based on all meals logged today, not just this one"
                          >
                            <p className="text-xs" style={{ color: patternCfg.color }}>
                              Today&apos;s pattern: your meals today are showing{' '}
                              <span style={{ fontWeight: 600 }}>{patternPhrase}</span>
                              <span className="text-muted-foreground font-normal">
                                {' · '}{mealAloneNote}
                              </span>
                            </p>
                            <p className="text-xs text-muted-foreground mt-1">
                              Based on {mealsToday} meal{mealsToday !== 1 ? 's' : ''} logged today
                            </p>
                            {(patternLevel === 'HIGH' || patternLevel === 'MODERATE') && (
                              <p className="text-xs text-muted-foreground mt-1">
                                Try to keep your next meal lighter to balance your intake for the day.
                              </p>
                            )}
                          </div>
                        </>
                      );
                    })()}
                  </div>
                );
              })()}

              {/* Chart + nutrient cards — side by side on large screens */}
              <div className="grid gap-3 mt-2 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)] lg:items-start">
                <div className="rounded-lg border p-2.5 lg:p-3 bg-white dark:bg-card">
                  <p className="text-sm font-semibold mb-1.5" style={{ color: theme.text }}>How much of your daily allowance this meal uses</p>
                  <ResponsiveContainer width="100%" height={100}>
                    <BarChart height={100} data={Object.entries(result.breakdown).map(([name, b]) => ({ name, pct: Math.round(b.pct), value: b.value, limit: b.limit }))} margin={{ top: 4, right: 8, left: -14, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke={isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.07)'} vertical={false} />
                      <XAxis dataKey="name" tick={{ fill: theme.textSecondary as string, fontSize: 11 }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fill: theme.textSecondary as string, fontSize: 10 }} tickFormatter={(v) => `${v}%`} domain={[0, 150]} axisLine={false} tickLine={false} />
                      <Tooltip
                        contentStyle={{ background: isDark ? '#111827' : '#fff', border: `1px solid ${theme.cardBorder}`, borderRadius: 10, color: theme.text, fontSize: '0.825rem' }}
                        formatter={(val: number, _n: string, item: any) => {
                          const p = item?.payload as { value?: number; limit?: number; name?: string } | undefined;
                          return [`${val}% — ${p?.value ?? 0} / ${p?.limit ?? 0}`, p?.name ?? ''];
                        }}
                      />
                      <ReferenceLine y={100} stroke="#EF4444" strokeWidth={2} strokeDasharray="6 3" />
                      <ReferenceLine y={80} stroke="#F59E0B" strokeWidth={1.5} strokeDasharray="4 3" />
                      <Bar dataKey="pct" radius={[5, 5, 0, 0]}>
                        {Object.entries(result.breakdown).map(([name, b]) => (
                          <Cell key={name} fill={b.pct > 100 ? '#E74C3C' : b.pct > 80 ? '#F39C12' : NUTRIENT_COLORS[name] ?? '#2E86AB'} fillOpacity={0.9} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <div className="flex gap-3 mt-2">
                    {[{ label: 'Danger (100%)', color: '#EF4444' }, { label: 'Warning (80%)', color: '#F59E0B' }].map((ref) => (
                      <div key={ref.label} className="flex items-center gap-1.5">
                        <svg width={18} height={2}><line x1="0" y1="1" x2="18" y2="1" stroke={ref.color} strokeWidth={2} strokeDasharray="4 2" /></svg>
                        <span style={{ color: theme.textSecondary, fontSize: '0.68rem' }}>{ref.label}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(result.breakdown).map(([name, b]) => {
                    const dailyKey = BREAKDOWN_TO_DAILY_KEY[name];
                    const dailyPct = dailyKey ? dailyBudget?.nutrients[dailyKey]?.percent_used : undefined;
                    const { label: badgeLabel, color: statusColor, Icon: StatusIcon } = dailyNutrientBadge(dailyPct);
                    const { value: displayValue, limit: displayLimit, pct: displayPct, sourceLabel } =
                      dailyNutrientCardValues(name, b, dailyKey, dailyBudget, bodyWeightKg);
                    const unit = name === 'Protein' ? 'g' : 'mg';
                    return (
                      <div key={name} className="rounded-lg border p-2 lg:p-2 bg-white dark:bg-card">
                        <div className="flex justify-between items-center mb-0.5">
                          <span className="text-xs lg:text-sm font-medium" style={{ color: theme.text }}>{name}</span>
                          <span className="text-[10px] lg:text-xs flex items-center gap-0.5" style={{ color: statusColor }}>
                            <StatusIcon size={10} />
                            {badgeLabel}
                          </span>
                        </div>
                        <p className="text-[10px] text-muted-foreground mb-0.5">{sourceLabel}</p>
                        <p className="text-base lg:text-lg font-bold leading-tight" style={{ color: statusColor }}>
                          {displayValue}
                          <span className="text-xs font-normal text-muted-foreground">
                            {' '}/ {displayLimit} {unit}
                          </span>
                        </p>
                        <div className="w-full bg-muted rounded-full h-1 mt-1">
                          <div
                            className="h-1 rounded-full"
                            style={{ width: `${Math.min(100, displayPct)}%`, background: statusColor }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Three-card row — 3 across on large screens */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 lg:gap-3">
                {/* Card 1 — Recommendations */}
                <div className="rounded-lg border p-3 flex flex-col bg-white dark:bg-card">
                  <div style={{ color: theme.text, fontWeight: 600, fontSize: '0.9rem', marginBottom: 8 }}>What you should do</div>

                  {(() => {
                    const inferred =
                      (result.shap_dominant_nutrient ?? '').trim()
                      || (result.shap_explanation?.match(/\b(potassium|phosphorus|protein|sodium)\b/i)?.[1] ?? '').trim();
                    const flaggedNutrient = inferred.length > 0
                      ? inferred.charAt(0).toUpperCase() + inferred.slice(1).toLowerCase()
                      : 'nutrient';

                    const actionText =
                      result.level === 'LOW'
                        ? 'Your meal is within safe limits. For your next meal, try to keep portions similar to maintain your daily balance.'
                        : result.level === 'MODERATE'
                          ? `This meal is at your limit. Keep your next meal light — choose low ${flaggedNutrient} options from the suggestions below.`
                          : 'This meal exceeds safe limits. Do not add more food today without checking it first. See safer options below.';

                    return (
                      <div className="mb-2 p-2 rounded-md bg-muted/40 border border-muted">
                        <p className="text-xs font-semibold text-muted-foreground mb-1 uppercase tracking-wide">
                          Your action
                        </p>
                        <p className="text-xs leading-relaxed">
                          {actionText}
                        </p>
                      </div>
                    );
                  })()}

                  {result.shap_contributions &&
                    Object.values(result.shap_contributions).some((v) => (v as number) > 0) && (
                    <div className="mb-2">
                      <p className="text-xs text-muted-foreground mb-1">
                        What drove this result
                      </p>
                      {Object.entries(result.shap_contributions)
                        .sort((a, b) => b[1] - a[1])
                        .map(([nutrient, pct]) => (
                          <div key={nutrient} className="flex items-center gap-2 mb-0.5">
                            <span className="text-xs w-20 capitalize">{nutrient}</span>
                            <div className="flex-1 bg-muted rounded-full h-1.5">
                              <div
                                className="h-1.5 rounded-full transition-all"
                                style={{
                                  width: `${pct}%`,
                                  backgroundColor:
                                    pct > 50 ? '#ef4444' :
                                    pct > 25 ? '#f59e0b' :
                                               '#22c55e',
                                }}
                              />
                            </div>
                            <span className="text-xs w-10 text-right">{pct}%</span>
                          </div>
                        ))}
                    </div>
                  )}
                </div>

                {/* Card 2 — Safer food choices */}
                {(() => {
                  const showLstmAlternatives =
                    result.level === 'HIGH' ||
                    result.level === 'MODERATE' ||
                    isDailyBudgetExceeded(dailyBudget);

                  const currentOccasionKey = currentMealType.toLowerCase();
                  const occasionFoods = showLstmAlternatives && suggestions
                    ? (
                        suggestions.suggestions_by_meal.find(
                          (group) => group.occasion.toLowerCase() === currentOccasionKey,
                        )?.suggestions.slice(0, 4) ?? []
                      )
                    : [];

                  const flaggedNutrient = suggestions?.flagged_nutrient?.trim();
                  const nutrientLabel = flaggedNutrient
                    ? toPlainNutrient(flaggedNutrient)
                    : 'key nutrients';

                  if (showLstmAlternatives) {
                    return (
                      <div className="rounded-lg border p-3 flex flex-col bg-white dark:bg-card">
                        <div style={{ color: theme.text, fontWeight: 600, fontSize: '0.9rem', marginBottom: 8 }}>
                          Safer {currentOccasionKey} options
                        </div>
                        <p className="text-xs text-muted-foreground mb-3">
                          Based on your recent meal pattern, these are lower in {nutrientLabel}:
                        </p>
                        {occasionFoods.length === 0 ? (
                          <p className="text-sm text-muted-foreground">
                            No alternatives available — try reducing portion size
                          </p>
                        ) : (
                          <ul className="grid grid-cols-2 gap-2">
                            {occasionFoods.map((food) => (
                              <li key={`${currentOccasionKey}-${food.english}`} className="min-w-0">
                                <p className="text-sm font-semibold text-foreground capitalize leading-snug">
                                  {food.english}
                                </p>
                                {food.category && (
                                  <p className="text-[11px] text-muted-foreground leading-tight">{food.category}</p>
                                )}
                                <p className="text-xs text-muted-foreground mt-0.5 leading-snug">{food.reason}</p>
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                    );
                  }

                  return (
                <div className="rounded-lg border p-3 flex flex-col bg-white dark:bg-card">
                  <div style={{ color: theme.text, fontWeight: 600, fontSize: '0.9rem', marginBottom: 8 }}>Better alternatives</div>
                  {result.substitutions.length === 0 ? (
                    <p className="text-sm text-center" style={{ color: theme.textSecondary }}>
                      {!usingLiveModel
                        ? 'Connect to live model for personalized swaps.'
                        : 'Your meal looks safe — no swaps needed.'}
                    </p>
                  ) : (
                    <ul className="space-y-4">
                      {result.substitutions.map(({ riskyFood, substitutes }) => (
                        <li key={riskyFood.id}>
                          <p style={{ color: theme.text, fontWeight: 600, fontSize: '0.825rem', marginBottom: 8 }}>
                            Instead of <span className="capitalize">{riskyFood.english}</span>, try:
                          </p>
                          <ul className="space-y-3 pl-1">
                            {substitutes.map((sub) => (
                              <li key={sub.id} className="flex items-start gap-2">
                                <ChevronRight size={12} style={{ color: '#27AE60', marginTop: 4, flexShrink: 0 }} />
                                <div className="min-w-0">
                                  <p style={{ color: theme.text, fontWeight: 500, fontSize: '0.825rem' }} className="capitalize">
                                    {sub.english}
                                  </p>
                                  {foodTranslation(sub.kinyarwanda) && (
                                    <p style={{ color: theme.textSecondary, fontSize: '0.75rem', marginTop: 2, fontStyle: 'italic' }}>
                                      ({foodTranslation(sub.kinyarwanda)})
                                    </p>
                                  )}
                                  <p style={{ color: '#2E86AB', fontSize: '0.75rem', marginTop: 4, fontWeight: 600 }}>
                                    Potassium: {riskyFood.potassium_mg}mg → {sub.potassium_mg}mg
                                  </p>
                                  <span
                                    className="inline-block mt-2 px-2 py-0.5 rounded-full"
                                    style={{ background: 'rgba(39,174,96,0.12)', color: '#27AE60', fontSize: '0.65rem', fontWeight: 600 }}
                                  >
                                    Suitable for your stage
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
                  );
                })()}

                {/* Card 3 — Today's Budget */}
                {dailyBudgetLoading ? (
                  <div className="rounded-lg border p-3 flex flex-col bg-white dark:bg-card">
                    <p style={{ color: theme.textTertiary, fontSize: '0.8rem' }}>Loading…</p>
                  </div>
                ) : dailyBudgetError ? (
                  <div className="rounded-lg border p-3 flex flex-col bg-white dark:bg-card">
                    <p style={{ color: theme.textTertiary, fontSize: '0.8rem' }}>{dailyBudgetError}</p>
                  </div>
                ) : dailyBudget ? (
                  <div className="rounded-lg border p-3 flex flex-col bg-white dark:bg-card">
                    {suggestions && !loadingSuggestions && (
                      <>
                        <hr className="border-gray-100 my-3" />
                        {renderWeeklySuggestionsPanel(true)}
                      </>
                    )}

                    {dailyBudget.suggestion_context && (
                      <p
                        className="text-xs italic line-clamp-2 mt-3"
                        style={{ color: theme.textSecondary }}
                        title={dailyBudget.suggestion_context.message}
                      >
                        {dailyBudget.suggestion_context.message}
                      </p>
                    )}
                  </div>
                ) : null}
              </div>

              <div className="flex justify-end">
                <button
                  onClick={() => void resetDay()}
                  disabled={resettingDay || !canResetToday}
                  className="transition-all duration-150 hover:opacity-80 disabled:opacity-40 disabled:cursor-not-allowed"
                  style={{
                    background: 'none',
                    border: 'none',
                    color: theme.textSecondary,
                    fontSize: '0.75rem',
                    fontWeight: 500,
                    padding: '4px 0',
                    cursor: resettingDay || !canResetToday ? 'not-allowed' : 'pointer',
                  }}
                  title="Clear all meals logged today"
                >
                  {resettingDay ? 'Resetting…' : "↺ Reset today's log"}
                </button>
              </div>

            </>
          )}
        </div>
      </div>
    </div>
  );
}
