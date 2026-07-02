import { useEffect, useMemo, useState } from 'react';
import {
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
  ReferenceArea,
} from 'recharts';
import { authFetch, getAuthToken } from '../../utils/auth';
import { formatStageDisplay, getRiskDisplay, getWeeklyRiskLabel } from '../../utils/riskDisplay';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
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

interface WeeklySummary {
  risk_label: string;
  confidence: number;
  method: string;
  days_analyzed: number;
  model_name: string;
  mod_recall: number;
}

interface WeeklyTrendResponse {
  days: BackendDay[];
  lstm_pattern: LstmPattern | null;
  weekly_summary: WeeklySummary | null;
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
      why: 'High potassium can cause dangerous heart rhythm problems when your kidneys cannot remove the excess.',
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
      why: 'Elevated phosphorus weakens bones and damages blood vessels — a common complication in advanced kidney disease.',
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
      why: 'Too much protein creates waste products your kidneys struggle to filter, which can worsen kidney disease.',
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
      why: 'Excess sodium raises blood pressure, which can further harm your kidneys and heart.',
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

function PlainNutrientBar({
  label,
  value,
  limit,
  unit,
}: {
  label: string;
  value: number;
  limit: number;
  unit: string;
}) {
  const pct = limit > 0 ? Math.min(100, Math.round((value / limit) * 100)) : 0;
  const filled = Math.min(10, Math.round(pct / 10));
  const bar = `${'█'.repeat(filled)}${'░'.repeat(10 - filled)}`;
  return (
    <div>
      <div className="flex justify-between items-baseline gap-2 text-sm">
        <span className="font-medium text-foreground">{label}</span>
        <span className="text-muted-foreground tabular-nums text-xs sm:text-sm">
          {Math.round(value).toLocaleString()} / {limit.toLocaleString()} {unit}
        </span>
      </div>
      <div className="flex items-center gap-2 mt-1 font-mono text-xs text-teal-700 dark:text-teal-400">
        <span className="tracking-tighter">{bar}</span>
        <span className="text-muted-foreground font-sans">{pct}%</span>
      </div>
    </div>
  );
}

function InsightBlock({ rec }: { rec: Recommendation }) {
  const Icon = rec.icon;
  return (
    <div className="min-w-0 p-3">
      <div className="flex items-center gap-2 mb-2">
        <Icon size={14} style={{ color: rec.color }} />
        <span className="text-base font-semibold text-foreground">{rec.nutrient}</span>
      </div>
      <p className="text-sm text-foreground leading-relaxed">{rec.what}</p>
      <p className="text-sm text-muted-foreground leading-relaxed mt-1">{rec.action}</p>
    </div>
  );
}

const trendConfig = {
  escalating: {
    label: 'Risk is increasing',
    icon: '↑',
    color: 'text-red-600',
    bg: 'bg-red-50',
  },
  stable: {
    label: 'Risk is steady',
    icon: '→',
    color: 'text-amber-500',
    bg: 'bg-amber-50',
  },
  improving: {
    label: 'Risk is improving',
    icon: '↓',
    color: 'text-green-600',
    bg: 'bg-green-50',
  },
} as const;

export function WeeklyTrend({ theme, onNavigate }: WeeklyTrendProps) {
  const [days, setDays] = useState<DayData[]>([]);
  const [loading, setLoading] = useState(true);
  const [profile, setProfile] = useState<PatientProfile | null>(null);
  const [selectedDay, setSelectedDay] = useState<DayData | null>(null);
  const [lstmRisk, setLstmRisk] = useState<Risk>('Low');
  const [confidence, setConfidence] = useState(0);
  const [, setPatternStable] = useState(true);
  const [lstmTrend, setLstmTrend] = useState<string>('stable');
  const [weeklySummary, setWeeklySummary] = useState<WeeklySummary | null>(null);
  const [lstmDaysAnalyzed, setLstmDaysAnalyzed] = useState(0);

  useEffect(() => {
    const token = getAuthToken();
    if (!token) {
      setLoading(false);
      return;
    }

    Promise.all([
      authFetch(`${API_BASE}/api/patient/weekly-trend`).then(async (r) => {
        if (r.status === 401) return null;
        if (!r.ok) throw new Error(await r.text());
        return r.json() as Promise<WeeklyTrendResponse>;
      }),
      authFetch(`${API_BASE}/api/patient/profile`).then(async (r) => {
        if (r.status === 401) return null;
        if (!r.ok) throw new Error(await r.text());
        return r.json() as Promise<{ ckd_stage?: string; weight_kg?: number; name?: string }>;
      }),
    ])
      .then(([trendData, profileData]) => {
        if (!trendData || !profileData) return;
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
        setLstmTrend(trendData.lstm_pattern?.trend ?? 'stable');
        setLstmDaysAnalyzed(trendData.lstm_pattern?.days_analyzed ?? 0);
        setWeeklySummary(trendData.weekly_summary ?? null);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Weekly trend error:', err);
        setLoading(false);
      });
  }, []);

  const daysWithData = useMemo(() => days.filter((d) => d.meals > 0), [days]);
  const loggedDaysCount = daysWithData.length;

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

  const weekRange =
    days.length > 0 ? `${days[0].shortDate} – ${days[days.length - 1].shortDate}` : 'This week';

  const statusRiskKey = weeklySummary?.risk_label
    ?? (lstmRisk === 'High' ? 'HIGH' : lstmRisk === 'Moderate' ? 'MODERATE' : 'LOW');
  const statusRisk = getRiskDisplay(statusRiskKey);
  const statusLabel = weeklySummary
    ? getRiskDisplay(weeklySummary.risk_label).label
    : getWeeklyRiskLabel(statusRiskKey);

  const trendKey = (lstmTrend in trendConfig ? lstmTrend : 'stable') as keyof typeof trendConfig;
  const trendCfg =
    lstmRisk === 'High' && lstmTrend === 'stable'
      ? { ...trendConfig.stable, label: 'Consistently high risk', color: 'text-red-600', bg: 'bg-red-50' }
      : trendConfig[trendKey];
  const lstmRiskStyle = RISK_STYLES[lstmRisk];

  const selectedDayExceeded =
    selectedDay != null &&
    selectedDay.meals > 0 &&
    (selectedDay.potassium > selectedDay.limits.k ||
      selectedDay.phosphorus > selectedDay.limits.p ||
      selectedDay.protein > selectedDay.limits.pro ||
      selectedDay.sodium > selectedDay.limits.na);

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
    <div className="max-w-[1600px] mx-auto w-full min-w-0 px-4 py-6 space-y-5">
      {/* ROW 1 — header + weekly risk summary */}
      <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
        <div className="space-y-1">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <div className="min-w-0">
              <h1 className="text-2xl font-semibold text-foreground">Your Dietary Pattern</h1>
              <p className="text-sm text-muted-foreground mt-1">
                AI-powered analysis of whether your risk is escalating, stable, or improving
              </p>
            </div>
            <span className="text-muted-foreground hidden sm:inline">·</span>
            <span className={`text-lg font-semibold ${statusRisk.color}`}>
              {statusRisk.icon} {statusLabel}
            </span>
            {weeklySummary && (
              <span className="text-xs bg-teal-50 text-teal-700 dark:bg-teal-900/30 dark:text-teal-300 px-2 py-0.5 rounded-full font-medium">
                AI Pattern Analysis
              </span>
            )}
          </div>
          <p className="text-sm text-muted-foreground">
            {weekRange} · {formatStageDisplay(profile?.ckd_stage || 'G3b')} · {profile?.name || 'Patient'}
          </p>
        </div>

        {loggedDaysCount > 0 && (
          <div className="mt-4 pt-4 border-t border-border">
            <p className="text-sm font-semibold text-foreground">Recent Meal Pattern</p>
            <div className="flex flex-wrap items-center gap-2 mt-2">
              <span
                className={`inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full border ${lstmRiskStyle.bg} ${lstmRiskStyle.text} ${lstmRiskStyle.border}`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${lstmRiskStyle.dot}`} />
                {lstmRisk} risk · {confidence}% confidence
              </span>
              <span
                className={`inline-flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-full ${trendCfg.bg} ${trendCfg.color}`}
              >
                {trendCfg.icon} {trendCfg.label}
              </span>
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Based on your last {lstmDaysAnalyzed} meals
            </p>
          </div>
        )}
      </div>

      <hr className="border-border" />

      {/* ROW 2 — timeline + chart | day detail */}
      {loggedDaysCount === 0 ? (
        <div className="py-8 text-center">
          <p className="text-sm font-medium text-foreground mb-1">No meals logged this week</p>
          <p className="text-xs text-muted-foreground">
            Log meals in Meal Check to see your weekly dietary pattern
          </p>
          {onNavigate && (
            <button
              type="button"
              onClick={() => onNavigate('assessment')}
              className="inline-flex items-center gap-2 mt-4 px-5 py-2.5 rounded-xl text-white text-sm font-semibold hover:opacity-90"
              style={{ background: `linear-gradient(135deg, ${TEAL} 0%, #1A5F7A 100%)` }}
            >
              Go to Meal Check
              <ChevronRight size={15} />
            </button>
          )}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-[60%_40%] gap-6 lg:gap-8 items-start">
            <div className="min-w-0 rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
              <div className="flex justify-between w-full mb-2 pr-3">
                {days.map((day) => {
                  const style = day.meals > 0 ? RISK_STYLES[day.risk] : null;
                  const selected = selectedDay?.date === day.date;
                  return (
                    <button
                      key={day.date}
                      type="button"
                      onClick={() => day.meals > 0 && setSelectedDay(day)}
                      className={`flex-1 min-w-0 text-center py-1 px-0.5 rounded transition-colors ${
                        selected ? 'bg-teal-50 dark:bg-teal-900/20 ring-2 ring-teal-500 ring-offset-1 shadow-md' : ''
                      } ${day.meals === 0 ? 'opacity-50 cursor-default' : 'hover:bg-muted/50 cursor-pointer'}`}
                    >
                      <p className="text-[10px] sm:text-xs text-muted-foreground truncate">{day.dayName}</p>
                      <p className="text-[10px] sm:text-xs font-medium text-foreground truncate">{day.shortDate}</p>
                      <div className="flex justify-center my-1">
                        {day.meals > 0 ? (
                          <span className={`w-2 h-2 rounded-full ${style?.dot}`} />
                        ) : (
                          <span className="w-2 h-2 rounded-full border border-dashed border-muted" />
                        )}
                      </div>
                      <p className="text-[9px] sm:text-[10px] text-muted-foreground">
                        {day.meals > 0 ? `${day.meals} meal${day.meals !== 1 ? 's' : ''}` : '—'}
                      </p>
                    </button>
                  );
                })}
              </div>
              <div className="h-[200px] lg:h-[220px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 5, right: 12, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    {selectedDay && (
                      <ReferenceArea
                        x1={selectedDay.dayName}
                        x2={selectedDay.dayName}
                        fill={TEAL}
                        fillOpacity={0.1}
                        strokeOpacity={0}
                      />
                    )}
                    <XAxis dataKey="day" tick={{ fontSize: 10, fill: theme.textSecondary }} tickLine={false} />
                    <YAxis
                      tick={{ fontSize: 10, fill: theme.textSecondary }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(v) => `${v}%`}
                      domain={[0, 150]}
                    />
                    <Tooltip formatter={(value: number, name: string) => [`${value}%`, name]} />
                    <ReferenceLine y={100} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={1.5} />
                    <Line type="monotone" dataKey="Potassium" stroke={NUTRIENT_COLORS.Potassium} strokeWidth={2} dot={{ r: 3 }} connectNulls={true} strokeDasharray="4 2" />
                    <Line type="monotone" dataKey="Phosphorus" stroke={NUTRIENT_COLORS.Phosphorus} strokeWidth={2} dot={{ r: 3 }} connectNulls={true} strokeDasharray="4 2" />
                    <Line type="monotone" dataKey="Protein" stroke={NUTRIENT_COLORS.Protein} strokeWidth={2} dot={{ r: 3 }} connectNulls={true} strokeDasharray="4 2" />
                    <Line type="monotone" dataKey="Sodium" stroke={NUTRIENT_COLORS.Sodium} strokeWidth={2} dot={{ r: 3 }} connectNulls={true} strokeDasharray="4 2" />
                    <Legend wrapperStyle={{ fontSize: 10 }} iconType="circle" iconSize={7} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Red dashed line = your daily allowance limit
              </p>
            </div>

            <div className="min-w-0 rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
              {selectedDay && selectedDay.meals > 0 ? (
                <>
                  {selectedDayExceeded && (
                    <div className="rounded-lg bg-red-50 border border-red-200 p-2 mb-3 flex items-center gap-2">
                      <span className="text-red-600 font-semibold text-sm">⚠ Daily limit exceeded</span>
                      <span className="text-red-500 text-xs">on this day — see details below</span>
                    </div>
                  )}
                  <p className="text-sm font-medium text-foreground">
                    {selectedDay.dayName} {selectedDay.shortDate} · {selectedDay.meals} meal
                    {selectedDay.meals !== 1 ? 's' : ''} logged
                  </p>
                  <hr className="my-3 border-border" />
                  <div className="space-y-4">
                    <PlainNutrientBar label="Potassium" value={selectedDay.potassium} limit={selectedDay.limits.k} unit="mg" />
                    <PlainNutrientBar label="Phosphorus" value={selectedDay.phosphorus} limit={selectedDay.limits.p} unit="mg" />
                    <PlainNutrientBar label="Protein" value={selectedDay.protein} limit={selectedDay.limits.pro} unit="g" />
                    <PlainNutrientBar label="Sodium" value={selectedDay.sodium} limit={selectedDay.limits.na} unit="mg" />
                  </div>
                </>
              ) : (
                <p className="text-sm text-muted-foreground">Select a logged day to view nutrient details.</p>
              )}
            </div>
          </div>

          <hr className="border-border" />
        </>
      )}

      {/* ROW 3 — what this means for you */}
      {loggedDaysCount > 0 && recommendations.length > 0 && (
        <div className="rounded-xl border border-gray-200 bg-white p-3 shadow-sm">
          <p className="text-sm font-semibold text-foreground mb-3">What this means for you</p>
          <div className="grid grid-cols-2 gap-3">
            {recommendations.map((rec) => (
              <InsightBlock key={rec.nutrient} rec={rec} />
            ))}
          </div>
        </div>
      )}

      <p className="text-xs text-center text-muted-foreground pb-2 pt-2">
        This is for information only. Always follow your doctor&apos;s or dietitian&apos;s advice.
      </p>
    </div>
  );
}

export { WeeklyTrend as WeeklyTrendPage };
export default WeeklyTrend;
