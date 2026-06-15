import { useState, useRef, useEffect } from 'react';
import { AlertTriangle, CheckCircle2, XCircle, Zap, Info, ChevronRight, RotateCcw, Search, Trash2, Plus, Minus } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, Cell } from 'recharts';

interface RiskAssessmentProps {
  isDark: boolean;
  theme: Record<string, string>;
}

type CKDStage = 'G1' | 'G2' | 'G3a' | 'G3b' | 'G4' | 'G5';
type Safety = 'Safe' | 'Moderate' | 'Avoid';

// Nutrients are per 100 g
interface FoodItem {
  id: number;
  name: string;
  category: string;
  potassium: number;   // mg
  phosphorus: number;  // mg
  protein: number;     // g
  sodium: number;      // mg
  safety: Safety;
  defaultGrams: number;
}

const DB: FoodItem[] = [
  // Grains & starches
  { id:  1, name: 'White Rice (cooked)',      category: 'Grains',     potassium:  55, phosphorus:  43, protein:  2.7, sodium:   1, safety: 'Safe',     defaultGrams: 150 },
  { id:  2, name: 'Maize Meal / Ugali',       category: 'Grains',     potassium: 130, phosphorus:  89, protein:  3.7, sodium:   5, safety: 'Safe',     defaultGrams: 200 },
  { id:  3, name: 'Brown Rice (cooked)',       category: 'Grains',     potassium: 154, phosphorus: 143, protein:  2.7, sodium:   5, safety: 'Moderate', defaultGrams: 150 },
  { id:  4, name: 'White Bread',              category: 'Grains',     potassium:  96, phosphorus:  57, protein:  7.6, sodium: 477, safety: 'Moderate', defaultGrams:  60 },
  { id:  5, name: 'Sorghum Flour',            category: 'Grains',     potassium: 350, phosphorus: 287, protein: 10.6, sodium:   2, safety: 'Moderate', defaultGrams: 100 },
  { id:  6, name: 'Millet Flour',             category: 'Grains',     potassium: 195, phosphorus: 285, protein: 11.0, sodium:   5, safety: 'Moderate', defaultGrams: 100 },
  { id:  7, name: 'Wheat Flour (white)',      category: 'Grains',     potassium: 107, phosphorus:  97, protein: 10.3, sodium:   2, safety: 'Safe',     defaultGrams: 100 },
  // Tubers & roots
  { id:  8, name: 'Cassava (boiled)',          category: 'Tubers',     potassium: 271, phosphorus:  28, protein:  1.4, sodium:  14, safety: 'Safe',     defaultGrams: 150 },
  { id:  9, name: 'Sweet Potato (boiled)',     category: 'Tubers',     potassium: 475, phosphorus:  50, protein:  2.0, sodium:  36, safety: 'Moderate', defaultGrams: 150 },
  { id: 10, name: 'Irish Potato (boiled)',     category: 'Tubers',     potassium: 379, phosphorus:  44, protein:  1.9, sodium:   5, safety: 'Moderate', defaultGrams: 150 },
  // Vegetables
  { id: 11, name: 'Cabbage (raw)',             category: 'Vegetables', potassium: 170, phosphorus:  26, protein:  1.3, sodium:  18, safety: 'Safe',     defaultGrams: 100 },
  { id: 12, name: 'Carrot (raw)',              category: 'Vegetables', potassium: 320, phosphorus:  35, protein:  0.9, sodium:  69, safety: 'Safe',     defaultGrams: 100 },
  { id: 13, name: 'Tomato (raw)',              category: 'Vegetables', potassium: 237, phosphorus:  24, protein:  0.9, sodium:   5, safety: 'Safe',     defaultGrams:  80 },
  { id: 14, name: 'Onion (raw)',               category: 'Vegetables', potassium: 146, phosphorus:  29, protein:  1.1, sodium:   4, safety: 'Safe',     defaultGrams:  50 },
  { id: 15, name: 'Green Pepper',             category: 'Vegetables', potassium: 175, phosphorus:  20, protein:  0.9, sodium:   3, safety: 'Safe',     defaultGrams:  80 },
  { id: 16, name: 'Cucumber',                 category: 'Vegetables', potassium: 147, phosphorus:  24, protein:  0.6, sodium:   2, safety: 'Safe',     defaultGrams: 100 },
  { id: 17, name: 'Eggplant (cooked)',         category: 'Vegetables', potassium: 188, phosphorus:  17, protein:  0.8, sodium:   1, safety: 'Safe',     defaultGrams: 100 },
  { id: 18, name: 'Pumpkin (boiled)',          category: 'Vegetables', potassium: 230, phosphorus:  30, protein:  1.0, sodium:   1, safety: 'Safe',     defaultGrams: 100 },
  { id: 19, name: 'Green Beans',              category: 'Vegetables', potassium: 209, phosphorus:  38, protein:  1.8, sodium:   6, safety: 'Safe',     defaultGrams: 100 },
  { id: 20, name: 'Spinach (cooked)',          category: 'Vegetables', potassium: 839, phosphorus: 101, protein:  5.4, sodium: 164, safety: 'Avoid',    defaultGrams: 100 },
  { id: 21, name: 'Amaranth Leaves (cooked)', category: 'Vegetables', potassium: 611, phosphorus:  50, protein:  3.5, sodium:  20, safety: 'Avoid',    defaultGrams: 100 },
  { id: 22, name: 'Mushrooms (cooked)',        category: 'Vegetables', potassium: 356, phosphorus:  87, protein:  3.6, sodium:   9, safety: 'Moderate', defaultGrams:  80 },
  // Fruits
  { id: 23, name: 'Pineapple (fresh)',         category: 'Fruits',     potassium: 109, phosphorus:   8, protein:  0.5, sodium:   1, safety: 'Safe',     defaultGrams: 150 },
  { id: 24, name: 'Papaya (fresh)',            category: 'Fruits',     potassium: 182, phosphorus:  10, protein:  0.5, sodium:   8, safety: 'Safe',     defaultGrams: 150 },
  { id: 25, name: 'Watermelon',               category: 'Fruits',     potassium: 112, phosphorus:  11, protein:  0.6, sodium:   1, safety: 'Safe',     defaultGrams: 200 },
  { id: 26, name: 'Orange',                   category: 'Fruits',     potassium: 181, phosphorus:  14, protein:  0.9, sodium:   0, safety: 'Safe',     defaultGrams: 120 },
  { id: 27, name: 'Passion Fruit',            category: 'Fruits',     potassium: 348, phosphorus:  68, protein:  2.2, sodium:  28, safety: 'Moderate', defaultGrams: 100 },
  { id: 28, name: 'Mango (fresh)',             category: 'Fruits',     potassium: 168, phosphorus:  14, protein:  0.8, sodium:   1, safety: 'Safe',     defaultGrams: 150 },
  { id: 29, name: 'Plantain (green, cooked)', category: 'Fruits',     potassium: 499, phosphorus:  34, protein:  1.3, sodium:   4, safety: 'Moderate', defaultGrams: 150 },
  { id: 30, name: 'Banana (ripe)',             category: 'Fruits',     potassium: 358, phosphorus:  22, protein:  1.1, sodium:   1, safety: 'Avoid',    defaultGrams: 120 },
  { id: 31, name: 'Avocado',                  category: 'Fruits',     potassium: 485, phosphorus:  52, protein:  2.0, sodium:   7, safety: 'Avoid',    defaultGrams: 100 },
  // Legumes
  { id: 32, name: 'Kidney Beans (cooked)',     category: 'Legumes',    potassium: 403, phosphorus: 244, protein: 15.4, sodium:   2, safety: 'Avoid',    defaultGrams: 150 },
  { id: 33, name: 'Lentils (cooked)',          category: 'Legumes',    potassium: 369, phosphorus: 180, protein:  9.0, sodium:   2, safety: 'Moderate', defaultGrams: 150 },
  { id: 34, name: 'Groundnuts / Peanuts',     category: 'Legumes',    potassium: 705, phosphorus: 376, protein: 25.8, sodium:  18, safety: 'Avoid',    defaultGrams:  50 },
  { id: 35, name: 'Groundnut Paste',          category: 'Legumes',    potassium: 649, phosphorus: 335, protein: 25.1, sodium: 152, safety: 'Avoid',    defaultGrams:  30 },
  // Protein
  { id: 36, name: 'Egg (whole, boiled)',       category: 'Protein',    potassium: 126, phosphorus: 172, protein: 12.6, sodium: 142, safety: 'Moderate', defaultGrams:  55 },
  { id: 37, name: 'Chicken Breast (grilled)', category: 'Protein',    potassium: 220, phosphorus: 196, protein: 27.3, sodium:  74, safety: 'Moderate', defaultGrams: 100 },
  { id: 38, name: 'Beef (lean, cooked)',       category: 'Protein',    potassium: 318, phosphorus: 207, protein: 26.1, sodium:  55, safety: 'Moderate', defaultGrams: 100 },
  { id: 39, name: 'Goat Meat (cooked)',        category: 'Protein',    potassium: 385, phosphorus: 202, protein: 27.0, sodium:  82, safety: 'Moderate', defaultGrams: 100 },
  { id: 40, name: 'Pork (lean, cooked)',       category: 'Protein',    potassium: 423, phosphorus: 246, protein: 28.9, sodium:  62, safety: 'Avoid',    defaultGrams: 100 },
  // Fish
  { id: 41, name: 'Tilapia (grilled)',         category: 'Fish',       potassium: 302, phosphorus: 204, protein: 26.2, sodium:  56, safety: 'Moderate', defaultGrams: 100 },
  { id: 42, name: 'Catfish (grilled)',         category: 'Fish',       potassium: 358, phosphorus: 224, protein: 24.5, sodium:  68, safety: 'Moderate', defaultGrams: 100 },
  { id: 43, name: 'Sardines (canned)',         category: 'Fish',       potassium: 397, phosphorus: 490, protein: 24.6, sodium: 505, safety: 'Avoid',    defaultGrams:  85 },
  // Dairy
  { id: 44, name: 'Milk (whole)',              category: 'Dairy',      potassium: 150, phosphorus: 233, protein:  8.0, sodium: 107, safety: 'Avoid',    defaultGrams: 240 },
  { id: 45, name: 'Yogurt (plain)',            category: 'Dairy',      potassium: 141, phosphorus:  95, protein:  9.0, sodium:  36, safety: 'Moderate', defaultGrams: 100 },
  // Fats & condiments
  { id: 46, name: 'Cooking Oil',              category: 'Fats',       potassium:   0, phosphorus:   0, protein:  0.0, sodium:   0, safety: 'Safe',     defaultGrams:  15 },
  { id: 47, name: 'Butter',                   category: 'Fats',       potassium:  24, phosphorus:  24, protein:  0.9, sodium: 643, safety: 'Avoid',    defaultGrams:  15 },
  { id: 48, name: 'Sugar (white)',             category: 'Fats',       potassium:   2, phosphorus:   0, protein:  0.0, sodium:   1, safety: 'Safe',     defaultGrams:  10 },
  { id: 49, name: 'Salt',                     category: 'Fats',       potassium:   8, phosphorus:   0, protein:  0.0, sodium: 38758, safety: 'Avoid',  defaultGrams:   2 },
  { id: 50, name: 'Corn on the Cob (cooked)', category: 'Grains',     potassium: 270, phosphorus:  89, protein:  3.3, sodium:   1, safety: 'Moderate', defaultGrams: 150 },
];

const safetyStyle: Record<Safety, { color: string; bg: string }> = {
  Safe:     { color: '#27AE60', bg: 'rgba(39,174,96,0.12)' },
  Moderate: { color: '#F39C12', bg: 'rgba(243,156,18,0.12)' },
  Avoid:    { color: '#E74C3C', bg: 'rgba(231,76,60,0.12)' },
};

const stageThresholds: Record<CKDStage, { potassium: number; phosphorus: number; protein: number; sodium: number; label: string; gfr: string }> = {
  G1:  { potassium: 3500, phosphorus: 1250, protein: 90, sodium: 2300, label: 'Normal or high kidney function', gfr: '≥ 90' },
  G2:  { potassium: 3500, phosphorus: 1250, protein: 80, sodium: 2300, label: 'Mildly decreased',               gfr: '60–89' },
  G3a: { potassium: 3000, phosphorus: 1000, protein: 70, sodium: 2000, label: 'Mild to moderate decrease',      gfr: '45–59' },
  G3b: { potassium: 2500, phosphorus: 800,  protein: 60, sodium: 1800, label: 'Moderate to severe decrease',    gfr: '30–44' },
  G4:  { potassium: 2000, phosphorus: 700,  protein: 50, sodium: 1500, label: 'Severe decrease',                gfr: '15–29' },
  G5:  { potassium: 1500, phosphorus: 600,  protein: 40, sodium: 1200, label: 'Kidney failure',                 gfr: '< 15' },
};

const RECOMMENDATIONS: Record<string, string[]> = {
  HIGH:     ['Immediately reduce high-risk nutrient intake', 'Consult your nephrologist or dietitian as soon as possible', 'Leach vegetables by boiling and discarding cooking water', 'Avoid processed and canned foods (high sodium)', 'Choose egg whites over whole eggs to lower phosphorus'],
  MODERATE: ['Monitor portion sizes carefully at each meal', 'Substitute high-potassium foods with lower-potassium alternatives', 'Read nutrition labels to track phosphorus additives', 'Spread protein intake evenly across all meals', 'Schedule a dietary review with your care team this week'],
  LOW:      ['Continue your current dietary pattern', 'Maintain a daily food log to track trends', 'Stay well hydrated within your fluid limit', 'Regular follow-up assessments are recommended monthly', 'Good work — consistency is key in CKD management'],
};

const SAFER_FOODS: Record<string, Array<{ name: string; why: string }>> = {
  HIGH:     [{ name: 'White rice', why: 'Very low potassium & phosphorus' }, { name: 'Cabbage', why: 'Low-potassium vegetable' }, { name: 'Pineapple', why: 'Safest fruit for CKD' }, { name: 'Egg whites', why: 'High protein, minimal phosphorus' }],
  MODERATE: [{ name: 'Cassava (boiled)', why: 'Leachable potassium, low phosphorus' }, { name: 'Maize meal', why: 'Acceptable nutrient profile' }, { name: 'Carrot', why: 'Moderate potassium, nutritious' }, { name: 'Chicken breast', why: 'Lean protein, 85 g portions' }],
  LOW:      [{ name: 'Continue current choices', why: 'Your intake is well managed' }, { name: 'Add more vegetables', why: 'Variety improves nutrition' }, { name: 'Cassava or white rice', why: 'Good staple foods for CKD' }, { name: 'Pineapple or papaya', why: 'Low-potassium fruit options' }],
};

const NUTRIENT_COLORS: Record<string, string> = { Potassium: '#2E86AB', Phosphorus: '#F39C12', Protein: '#27AE60', Sodium: '#E74C3C' };

const RISK_CFG = {
  HIGH:     { color: '#E74C3C', bg: 'rgba(231,76,60,0.1)',  border: 'rgba(231,76,60,0.35)',  icon: XCircle,       label: 'High Risk',     desc: 'This meal significantly exceeds safe nutrient limits for your CKD stage' },
  MODERATE: { color: '#F39C12', bg: 'rgba(243,156,18,0.1)', border: 'rgba(243,156,18,0.35)', icon: AlertTriangle, label: 'Moderate Risk', desc: 'Some nutrients in this meal are near or above recommended thresholds' },
  LOW:      { color: '#27AE60', bg: 'rgba(39,174,96,0.1)',  border: 'rgba(39,174,96,0.35)',  icon: CheckCircle2,  label: 'Low Risk',      desc: 'This meal is within safe nutrient ranges for your CKD stage' },
};

interface MealEntry { food: FoodItem; grams: number }
interface BreakdownEntry { value: number; limit: number; pct: number; status: 'Safe' | 'Near limit' | 'Exceeded' }
interface ResultState { level: 'LOW' | 'MODERATE' | 'HIGH'; score: number; breakdown: Record<string, BreakdownEntry> }

export function RiskAssessment({ isDark, theme }: RiskAssessmentProps) {
  const [stage,    setStage]    = useState<CKDStage>('G3a');
  const [entries,  setEntries]  = useState<MealEntry[]>([]);
  const [result,   setResult]   = useState<ResultState | null>(null);
  const [error,    setError]    = useState('');
  const [search,   setSearch]   = useState('');
  const [showDrop, setShowDrop] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setShowDrop(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const suggestions = search.length >= 1
    ? DB.filter((f) => f.name.toLowerCase().includes(search.toLowerCase()) && !entries.find((e) => e.food.id === f.id)).slice(0, 7)
    : [];

  const addFood = (food: FoodItem) => {
    setEntries((prev) => [...prev, { food, grams: food.defaultGrams }]);
    setSearch('');
    setShowDrop(false);
    setResult(null);
    setError('');
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
        potassium:  acc.potassium  + food.potassium  * scale,
        phosphorus: acc.phosphorus + food.phosphorus * scale,
        protein:    acc.protein    + food.protein    * scale,
        sodium:     acc.sodium     + food.sodium     * scale,
      };
    },
    { potassium: 0, phosphorus: 0, protein: 0, sodium: 0 }
  );

  const thresholds = stageThresholds[stage];

  const computeRisk = () => {
    if (entries.length === 0) { setError('Add at least one food item to assess this meal.'); return; }
    setError('');
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
    const avgPct        = Object.values(breakdown).reduce((a, b) => a + b.pct, 0) / 4;
    const maxPct        = Math.max(...Object.values(breakdown).map((b) => b.pct));
    const exceededCount = Object.values(breakdown).filter((b) => b.pct > 100).length;
    let level: 'LOW' | 'MODERATE' | 'HIGH' = 'LOW';
    if (exceededCount >= 2 || maxPct > 130) level = 'HIGH';
    else if (exceededCount >= 1 || maxPct > 80) level = 'MODERATE';
    setResult({ level, score: Math.min(100, Math.round(avgPct)), breakdown });
  };

  const reset = () => { setEntries([]); setResult(null); setError(''); setSearch(''); };

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
              {(Object.keys(stageThresholds) as CKDStage[]).map((s) => (
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
            <div className="mt-3 p-3 rounded-xl" style={{ background: isDark ? 'rgba(46,134,171,0.08)' : 'rgba(46,134,171,0.07)', border: '1px solid rgba(46,134,171,0.18)' }}>
              <p style={{ color: '#2E86AB', fontWeight: 600, fontSize: '0.78rem' }}>Stage {stage} — eGFR {thresholds.gfr} mL/min/1.73m²</p>
              <p style={{ color: theme.textSecondary, fontSize: '0.72rem', marginTop: 2 }}>{thresholds.label}</p>
            </div>
          </div>

          {/* Meal builder */}
          <div className="p-5 sm:p-6 rounded-2xl" style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}>
            <div style={{ color: theme.text, fontWeight: 600, marginBottom: 4 }}>What did you eat?</div>
            <p style={{ color: theme.textSecondary, fontSize: '0.78rem', marginBottom: 14 }}>
              Search our {DB.length}-food database and add each item with its weight in grams
            </p>

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
                  placeholder="Search foods to add…"
                  value={search}
                  onChange={(e) => { setSearch(e.target.value); setShowDrop(true); }}
                  onFocus={() => search.length > 0 && setShowDrop(true)}
                />
              </div>

              {showDrop && suggestions.length > 0 && (
                <div
                  className="absolute top-full mt-1.5 left-0 right-0 z-30 rounded-xl overflow-hidden shadow-xl"
                  style={{ background: isDark ? '#111827' : '#fff', border: `1px solid ${theme.cardBorder}` }}
                >
                  {suggestions.map((f, i) => (
                    <button
                      key={f.id}
                      className="w-full text-left px-4 py-2.5 flex items-center justify-between gap-3 transition-colors"
                      style={{ borderBottom: i < suggestions.length - 1 ? `1px solid ${theme.cardBorder}` : 'none' }}
                      onMouseDown={() => addFood(f)}
                      onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)'; }}
                      onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = 'transparent'; }}
                    >
                      <div className="min-w-0">
                        <p style={{ color: theme.text, fontSize: '0.85rem', fontWeight: 500 }} className="truncate">{f.name}</p>
                        <p style={{ color: theme.textSecondary, fontSize: '0.7rem' }}>{f.category} · default {f.defaultGrams} g</p>
                      </div>
                      <span className="shrink-0 px-2 py-0.5 rounded-full" style={{ background: safetyStyle[f.safety].bg, color: safetyStyle[f.safety].color, fontSize: '0.65rem', fontWeight: 600 }}>
                        {f.safety}
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
                      <div className="w-2 h-2 rounded-full shrink-0" style={{ background: safetyStyle[food.safety].color }} />
                      <p className="flex-1 min-w-0 truncate" style={{ color: theme.text, fontSize: '0.82rem', fontWeight: 600 }}>{food.name}</p>
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
                          { label: 'K', value: Math.round(food.potassium * grams / 100), color: '#2E86AB' },
                          { label: 'P', value: Math.round(food.phosphorus * grams / 100), color: '#F39C12' },
                          { label: 'Na', value: Math.round(food.sodium * grams / 100), color: '#E74C3C' },
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
                        </div>
                        <p style={{ color: theme.textSecondary, fontSize: '0.85rem' }}>{cfg.desc}</p>
                        <p style={{ color: theme.textTertiary, fontSize: '0.75rem', marginTop: 3 }}>
                          {entries.length} food{entries.length !== 1 ? 's' : ''} · {entries.reduce((a, e) => a + e.grams, 0)} g total
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

              {/* 2 + 3 — Recommendations + safer foods (immediately after risk badge) */}
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
                  <ul className="space-y-3">
                    {SAFER_FOODS[result.level].map((f, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <CheckCircle2 size={12} style={{ color: '#27AE60', marginTop: 3, flexShrink: 0 }} />
                        <div>
                          <p style={{ color: theme.text, fontWeight: 500, fontSize: '0.825rem' }}>{f.name}</p>
                          <p style={{ color: theme.textSecondary, fontSize: '0.75rem', marginTop: 1 }}>{f.why}</p>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* 4 — Nutrient breakdown chart */}
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

              {/* 5 — Per-nutrient cards */}
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
        </div>
      </div>
    </div>
  );
}
