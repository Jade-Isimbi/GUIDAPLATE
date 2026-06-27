import { useEffect, useMemo, useState } from 'react';
import {
  TrendingUp,
  Droplets,
  Zap,
  Flame,
  Wind,
  Calendar,
  ChevronRight,
  type LucideIcon,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

const API_BASE_URL = 'http://localhost:8000/api';
const TEAL = '#2E86AB';

type Risk = 'High' | 'Moderate' | 'Low';

interface DayLimits {
  k: number;
  p: number;
  pro: number;
  na: number;
}

interface DayData {
  date: string;
  shortDate: string;
  dayName: string;
  meals: number;
  risk: Risk;
  potassium: number;
  phosphorus: number;
  protein: number;
  sodium: number;
  limits: DayLimits;
}

interface BackendDay {
  date: string;
  meals_count: number;
  nutrients: {
    potassium: number;
    phosphorus: number;
    protein_per_kg: number;
    sodium: number;
  };
  budget_label: string;
}

interface LstmPattern {
  risk_label: string;
  confidence: number;
  trend: string;
  days_analyzed: number;
}

interface WeeklyTrendResponse {
  days: BackendDay[];
  lstm_pattern: LstmPattern | null;
  ckd_stage: string;
  weight_kg: number;
}

interface PatientProfile {
  ckd_stage: string;
  weight_kg: number;
  name: string;
}

interface Recommendation {
  nutrient: string;
  color: string;
  icon: LucideIcon;
  what: string;
  why: string;
  action: string;
}

interface WeeklyTrendProps {
  isDark: boolean;
  theme: Record<string, string>;
  onNavigate?: (page: string) => void;
}

const RISK_STYLES: Record<Risk, { bg: string; text: string; border: string; dot: string }> = {
  High: {
    bg: 'bg-red-50 dark:bg-red-900/20',
    text: 'text-red-700 dark:text-red-400',
    border: 'border-red-200 dark:border-red-800',
    dot: 'bg-red-500',
  },
  Moderate: {
    bg: 'bg-amber-50 dark:bg-amber-900/20',
    text: 'text-amber-700 dark:text-amber-400',
    border: 'border-amber-200 dark:border-amber-800',
    dot: 'bg-amber-500',
  },
  Low: {
    bg: 'bg-green-50 dark:bg-green-900/20',
    text: 'text-green-700 dark:text-green-400',
    border: 'border-green-200 dark:border-green-800',
    dot: 'bg-green-500',
  },
};

const NUTRIENT_COLORS = {
  Potassium: '#0d9488',
  Phosphorus: '#f59e0b',
  Protein: '#16a34a',
  Sodium: '#ef4444',
};

const CARD_CLASS =
  'bg-white dark:bg-card rounded-xl border border-border/50 p-4 shadow-sm';

function mapRisk(level: string | null | undefined): Risk {
  if (!level) return 'Low';
  const l = level.toUpperCase();
  if (l === 'HIGH') return 'High';
  if (l === 'MODERATE') return 'Moderate';
  return 'Low';
}

function buildLimits(stage: string, weightKg: number): DayLimits {
  const PROTEIN_PER_KG: Record<string, number> = {
    G2: 0.8,
    G3a: 0.6,
    G3b: 0.6,
    G4: 0.55,
  };
  const K_LIMITS: Record<string, number> = {
    G2: 3500,
    G3a: 3000,
    G3b: 3000,
    G4: 2500,
  };
  const P_LIMITS: Record<string, number> = {
    G2: 1000,
    G3a: 800,
    G3b: 800,
    G4: 700,
  };
  return {
    k: K_LIMITS[stage] || 3000,
    p: P_LIMITS[stage] || 800,
    pro: Math.round(weightKg * (PROTEIN_PER_KG[stage] || 0.6)),
    na: 2300,
  };
}

function formatDateLabel(iso: string): string {
  const d = new Date(`${iso}T12:00:00Z`);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
}

function buildSevenDays(apiDays: BackendDay[], stage: string, weightKg: number): DayData[] {
  const limits = buildLimits(stage, weightKg);
  const dayMap = new Map(apiDays.map((d) => [d.date, d]));
  const slots: DayData[] = [];
  const now = new Date();

  for (let offset = 6; offset >= 0; offset -= 1) {
    const d = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() - offset));
    const iso = d.toISOString().slice(0, 10);
    const backend = dayMap.get(iso);
    const shortDate = formatDateLabel(iso);
    const dayName = d.toLocaleDateString('en-US', { weekday: 'short', timeZone: 'UTC' });

    if (backend) {
      slots.push({
        date: iso,
        shortDate,
        dayName,
        meals: backend.meals_count,
        risk: mapRisk(backend.budget_label),
        potassium: backend.nutrients.potassium,
        phosphorus: backend.nutrients.phosphorus,
        protein: backend.nutrients.protein_per_kg * weightKg,
        sodium: backend.nutrients.sodium,
        limits,
      });
    } else {
      slots.push({
        date: iso,
        shortDate,
        dayName,
        meals: 0,
        risk: 'Low',
        potassium: 0,
        phosphorus: 0,
        protein: 0,
        sodium: 0,
        limits,
      });
    }
  }

  return slots;
}

function buildRecommendations(days: DayData[]): Recommendation[] {
  const recommendations: Recommendation[] = [];
  const daysWithData = days.filter((d) => d.meals > 0);

  const kDays = daysWithData.filter((d) => d.potassium > d.limits.k).length;
  const pDays = daysWithData.filter((d) => d.phosphorus > d.limits.p).length;
  const proDays = daysWithData.filter((d) => d.protein > d.limits.pro).length;
  const naDays = daysWithData.filter((d) => d.sodium > d.limits.na).length;

  if (kDays > 0) {
    recommendations.push({
      nutrient: 'Potassium',
      color: '#2E86AB',
      icon: Droplets,
      what: `Potassium exceeded your ${days[0]?.limits.k}mg limit on ${kDays} day${kDays > 1 ? 's' : ''} this week.`,
      why: 'High potassium can cause dangerous heart rhythm problems in CKD. Your kidneys cannot remove the excess.',
      action:
        'Replace banana and avocado with pineapple or cabbage. Boil vegetables and discard cooking water to reduce potassium content.',
    });
  }

  if (pDays > 0) {
    recommendations.push({
      nutrient: 'Phosphorus',
      color: '#F39C12',
      icon: Zap,
      what: `Phosphorus exceeded ${days[0]?.limits.p}mg on ${pDays} day${pDays > 1 ? 's' : ''} this week.`,
      why: 'Elevated phosphorus weakens bones and damages blood vessels — a common complication in CKD stages G3–G5.',
      action:
        'Avoid kidney beans and whole milk. Choose egg whites over whole eggs. Avoid processed foods with phosphate additives.',
    });
  }

  if (proDays > 0) {
    recommendations.push({
      nutrient: 'Protein',
      color: '#27AE60',
      icon: Flame,
      what: `Protein intake exceeded your limit on ${proDays} day${proDays > 1 ? 's' : ''} this week.`,
      why: 'Too much protein creates waste products your kidneys struggle to filter, accelerating CKD progression.',
      action:
        'Keep meat portions palm-sized. Choose egg whites over whole eggs. Replace one meat serving with small fish portions.',
    });
  }

  if (naDays > 0) {
    recommendations.push({
      nutrient: 'Sodium',
      color: '#E74C3C',
      icon: Wind,
      what: `Sodium exceeded ${days[0]?.limits.na}mg on ${naDays} day${naDays > 1 ? 's' : ''} this week.`,
      why: 'Excess sodium raises blood pressure, accelerating CKD progression and cardiovascular risk.',
      action: 'Cook from fresh ingredients. Use herbs and lemon instead of salt. Avoid processed and packaged foods.',
    });
  }

  const loggedDays = daysWithData.length;
  if (loggedDays < 7) {
    recommendations.push({
      nutrient: 'Meal Logging',
      color: '#8B5CF6',
      icon: Calendar,
      what: `You logged meals on ${loggedDays} of 7 days this week.`,
      why: 'Missing logs make it harder to spot patterns and can hide nutrient overloads.',
      action: 'Log each meal right after eating — even a quick estimate is better than no data.',
    });
  }

  return recommendations;
}

function computeDietScore(daysWithData: DayData[]): number {
  if (daysWithData.length === 0) return 0;
  return Math.round(
    daysWithData.reduce((sum, d) => {
      const dayScore =
        100 -
        Math.max(
          0,
          ((d.potassium / d.limits.k - 1) * 30 +
            (d.phosphorus / d.limits.p - 1) * 25 +
            (d.protein / d.limits.pro - 1) * 25 +
            (d.sodium / d.limits.na - 1) * 20) *
            100,
        );
      return sum + Math.max(0, Math.min(100, dayScore));
    }, 0) / daysWithData.length,
  );
}

function NutrientBar({
  label,
  value,
  limit,
  unit,
  color,
}: {
  label: string;
  value: number;
  limit: number;
  unit: string;
  color: string;
}) {
  const pct = limit > 0 ? Math.min(100, Math.round((value / limit) * 100)) : 0;
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <span className="text-sm font-medium text-foreground">{label}</span>
        <span className="text-xs text-muted-foreground">
          {Math.round(value).toLocaleString()} / {limit.toLocaleString()} {unit}
        </span>
      </div>
      <div className="w-full bg-muted rounded-full h-1.5">
        <div className="h-1.5 rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
      <p className="text-xs text-muted-foreground mt-0.5">{pct}% of limit</p>
    </div>
  );
}

function RecommendationCard({ rec }: { rec: Recommendation }) {
  const Icon = rec.icon;
  return (
    <div className={`${CARD_CLASS} space-y-2`}>
      <div className="flex items-center gap-2">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: `${rec.color}18` }}
        >
          <Icon size={16} style={{ color: rec.color }} />
        </div>
        <span className="text-sm font-semibold text-foreground">{rec.nutrient}</span>
      </div>
      <p className="text-sm text-foreground leading-relaxed">
        <strong>What:</strong> {rec.what}
      </p>
      <p className="text-sm text-muted-foreground leading-relaxed">
        <strong className="text-foreground">Why:</strong> {rec.why}
      </p>
      <p className="text-sm text-foreground leading-relaxed">
        <strong>Action:</strong> {rec.action}
      </p>
    </div>
  );
}

export function WeeklyTrend({ theme, onNavigate }: WeeklyTrendProps) {
  const [days, setDays] = useState<DayData[]>([]);
  const [loading, setLoading] = useState(true);
  const [profile, setProfile] = useState<PatientProfile | null>(null);
  const [selectedDay, setSelectedDay] = useState<DayData | null>(null);
  const [lstmRisk, setLstmRisk] = useState<Risk>('Low');
  const [confidence, setConfidence] = useState(0);
  const [patternStable, setPatternStable] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('guidaplate_token');
    if (!token) {
      setLoading(false);
      return;
    }

    const headers = { Authorization: `Bearer ${token}` };

    Promise.all([
      fetch(`${API_BASE_URL}/patient/weekly-trend`, { headers }).then(async (r) => {
        if (!r.ok) throw new Error(await r.text());
        return r.json() as Promise<WeeklyTrendResponse>;
      }),
      fetch(`${API_BASE_URL}/patient/profile`, { headers }).then(async (r) => {
        if (!r.ok) throw new Error(await r.text());
        return r.json() as Promise<{ ckd_stage?: string; weight_kg?: number; name?: string }>;
      }),
    ])
      .then(([trendData, profileData]) => {
        const stage = profileData.ckd_stage || trendData.ckd_stage || 'G3b';
        const weightKg = profileData.weight_kg || trendData.weight_kg || 65;
        setProfile({
          ckd_stage: stage,
          weight_kg: weightKg,
          name: profileData.name || 'Patient',
        });

        const mapped = buildSevenDays(trendData.days, stage, weightKg);
        setDays(mapped);

        const withData = mapped.filter((d) => d.meals > 0);
        if (withData.length > 0) {
          setSelectedDay(withData[withData.length - 1]);
        }

        setLstmRisk(mapRisk(trendData.lstm_pattern?.risk_label));
        setConfidence(Math.round((trendData.lstm_pattern?.confidence ?? 0) * 100));
        setPatternStable(trendData.lstm_pattern?.trend !== 'escalating');
        setLoading(false);
      })
      .catch((err) => {
        console.error('Weekly trend error:', err);
        setLoading(false);
      });
  }, []);

  const daysWithData = useMemo(() => days.filter((d) => d.meals > 0), [days]);

  const chartData = useMemo(
    () =>
      days.map((d) => ({
        day: d.dayName,
        Potassium: d.meals > 0 ? Math.round((d.potassium / d.limits.k) * 100) : null,
        Phosphorus: d.meals > 0 ? Math.round((d.phosphorus / d.limits.p) * 100) : null,
        Protein: d.meals > 0 ? Math.round((d.protein / d.limits.pro) * 100) : null,
        Sodium: d.meals > 0 ? Math.round((d.sodium / d.limits.na) * 100) : null,
      })),
    [days],
  );

  const recommendations = useMemo(() => buildRecommendations(days), [days]);
  const dietScore = useMemo(() => computeDietScore(daysWithData), [daysWithData]);

  const weekRange =
    days.length > 0 ? `${days[0].shortDate} – ${days[days.length - 1].shortDate}` : 'This week';

  const riskStyle = RISK_STYLES[lstmRisk];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 rounded-full border-2 border-teal-600 border-t-transparent animate-spin" />
          <p className="text-sm text-muted-foreground">Loading weekly trend...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-6 space-y-4 bg-muted/30 dark:bg-transparent rounded-2xl">
      <div>
        <h1 className="text-2xl font-semibold text-foreground">Weekly Dietary Trend</h1>
        <p className="text-sm text-muted-foreground mt-1">
          {weekRange} · Stage {profile?.ckd_stage || 'G3b'} · {profile?.name || 'Patient'}
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className={`${CARD_CLASS} sm:col-span-1`}>
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">Diet Score</p>
          <p className="text-4xl font-bold" style={{ color: TEAL }}>
            {dietScore}
          </p>
          <p className="text-xs text-muted-foreground mt-1">— vs last week</p>
        </div>

        <div className={`${riskStyle.bg} ${riskStyle.border} border rounded-xl p-4 shadow-sm sm:col-span-2`}>
          <p className="text-xs text-muted-foreground mb-1">Your Overall Risk This Week</p>
          <p className={`text-lg font-bold ${riskStyle.text}`}>{lstmRisk} Risk</p>
          <p className="text-xs text-muted-foreground mt-2">
            AI confidence: {confidence >= 90 ? 'High' : confidence >= 70 ? 'Moderate' : 'Low'} ({confidence}%)
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            {patternStable
              ? '✓ Your eating habits were consistent this week.'
              : '↕ Your eating pattern varied significantly this week.'}
          </p>
        </div>
      </div>

      <div className={CARD_CLASS}>
        <p className="text-sm font-semibold text-foreground mb-3">7-day risk trend</p>
        <div className="grid grid-cols-7 gap-2 mb-4">
          {days.map((day) => {
            const style = day.meals > 0 ? RISK_STYLES[day.risk] : null;
            const selected = selectedDay?.date === day.date;
            return (
              <button
                key={day.date}
                type="button"
                onClick={() => day.meals > 0 && setSelectedDay(day)}
                className={`rounded-lg border p-2 text-center transition-all ${
                  selected ? 'ring-2 ring-teal-500 border-teal-400' : 'border-border/50'
                } ${day.meals === 0 ? 'opacity-60 cursor-default' : 'hover:border-teal-400 cursor-pointer'}`}
              >
                <p className="text-xs text-muted-foreground">{day.dayName}</p>
                <p className="text-xs font-medium text-foreground">{day.shortDate}</p>
                <div className="flex justify-center my-2">
                  {day.meals > 0 ? (
                    <span className={`w-3 h-3 rounded-full ${style?.dot}`} />
                  ) : (
                    <span className="w-3 h-3 rounded-full border-2 border-dashed border-muted bg-transparent" />
                  )}
                </div>
                <p className="text-[10px] text-muted-foreground">
                  {day.meals > 0 ? `${day.meals} meal${day.meals !== 1 ? 's' : ''}` : 'No data'}
                </p>
                <p className={`text-[10px] font-medium mt-0.5 ${day.meals > 0 ? style?.text : 'text-muted-foreground'}`}>
                  {day.meals > 0
                    ? day.risk === 'Low'
                      ? '🟢 Good'
                      : day.risk === 'Moderate'
                        ? '🟡 Moderate'
                        : '🔴 High'
                    : '— No data'}
                </p>
              </button>
            );
          })}
        </div>

        <p className="text-sm font-semibold text-foreground mb-3">How close you were to your daily nutrient limits</p>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="day" tick={{ fontSize: 11, fill: theme.textSecondary }} tickLine={false} />
            <YAxis
              tick={{ fontSize: 11, fill: theme.textSecondary }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => `${v}%`}
              domain={[0, 150]}
            />
            <Tooltip formatter={(value: number, name: string) => [`${value}%`, name]} />
            <ReferenceLine y={100} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={1.5} />
            <Line type="monotone" dataKey="Potassium" stroke={NUTRIENT_COLORS.Potassium} strokeWidth={2} dot={{ r: 3 }} connectNulls={false} />
            <Line type="monotone" dataKey="Phosphorus" stroke={NUTRIENT_COLORS.Phosphorus} strokeWidth={2} dot={{ r: 3 }} connectNulls={false} />
            <Line type="monotone" dataKey="Protein" stroke={NUTRIENT_COLORS.Protein} strokeWidth={2} dot={{ r: 3 }} connectNulls={false} />
            <Line type="monotone" dataKey="Sodium" stroke={NUTRIENT_COLORS.Sodium} strokeWidth={2} dot={{ r: 3 }} connectNulls={false} />
            <Legend wrapperStyle={{ fontSize: 11 }} iconType="circle" iconSize={8} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-xs text-muted-foreground mt-2">Red dashed line = your daily limit</p>
      </div>

      {selectedDay && selectedDay.meals > 0 && (
        <div className={CARD_CLASS}>
          <p className="text-sm font-semibold text-foreground mb-1">
            Daily breakdown — {selectedDay.dayName}, {selectedDay.shortDate}
          </p>
          <p className="text-xs text-muted-foreground mb-4">
            {selectedDay.meals} meal{selectedDay.meals !== 1 ? 's' : ''} logged ·{' '}
            <span className={`font-medium ${RISK_STYLES[selectedDay.risk].text}`}>{selectedDay.risk} risk</span>
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <NutrientBar label="Potassium" value={selectedDay.potassium} limit={selectedDay.limits.k} unit="mg" color={NUTRIENT_COLORS.Potassium} />
            <NutrientBar label="Phosphorus" value={selectedDay.phosphorus} limit={selectedDay.limits.p} unit="mg" color={NUTRIENT_COLORS.Phosphorus} />
            <NutrientBar label="Protein" value={selectedDay.protein} limit={selectedDay.limits.pro} unit="g" color={NUTRIENT_COLORS.Protein} />
            <NutrientBar label="Sodium" value={selectedDay.sodium} limit={selectedDay.limits.na} unit="mg" color={NUTRIENT_COLORS.Sodium} />
          </div>
        </div>
      )}

      {daysWithData.length === 0 && (
        <div className={`${CARD_CLASS} flex flex-col items-center text-center py-12`}>
          <TrendingUp size={32} className="text-muted-foreground mb-3" />
          <p className="text-sm font-semibold text-foreground mb-2">No meal history yet.</p>
          <p className="text-sm text-muted-foreground max-w-sm mb-4">
            Start logging meals in Meal Assessment to see your weekly dietary trend here.
          </p>
          {onNavigate && (
            <button
              type="button"
              onClick={() => onNavigate('assessment')}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-white text-sm font-semibold hover:opacity-90"
              style={{ background: `linear-gradient(135deg, ${TEAL} 0%, #1A5F7A 100%)` }}
            >
              Go to Meal Assessment
              <ChevronRight size={15} />
            </button>
          )}
        </div>
      )}

      {recommendations.length > 0 && (
        <div className="space-y-3">
          <p className="text-sm font-semibold text-foreground">Personalized Recommendations</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {recommendations.map((rec) => (
              <RecommendationCard key={rec.nutrient} rec={rec} />
            ))}
          </div>
        </div>
      )}

      <p className="text-xs text-center text-muted-foreground pb-2">
        This information supports — not replaces — professional medical advice. Always consult your healthcare provider
        before making dietary changes.
      </p>
    </div>
  );
}

export { WeeklyTrend as WeeklyTrendPage };
export default WeeklyTrend;
