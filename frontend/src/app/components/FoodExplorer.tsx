import { useMemo, useState } from 'react';
import { Search, Filter, ChevronDown, Info, BarChart2, X } from 'lucide-react';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer } from 'recharts';
import { CATEGORIES, FOODS, type Food, potassiumColor } from '../../data/foodDatabase';
import { foodTranslation, matchesFoodQuery } from '../../utils/foodDisplay';
import { formatStageDisplay, getStageSafety } from '../../utils/riskDisplay';

interface FoodExplorerProps {
  isDark: boolean;
  theme: Record<string, string>;
  patientStage?: string;
}

const STAGE_FILTER_OPTIONS = [
  { label: 'All', value: 'all' as const },
  { label: 'Safe for Stage 2', value: 'G2' as const },
  { label: 'Safe for Stage 3a', value: 'G3a' as const },
  { label: 'Safe for Stage 3b', value: 'G3b' as const },
  { label: 'Safe for Stage 4', value: 'G4' as const },
];

function foodStageSafety(food: Food, patientStage: string) {
  return getStageSafety(
    food.ckd_stage_safe,
    patientStage,
    food.potassium_mg,
    food.phosphorus_mg ?? 0,
    food.protein_g ?? 0,
  );
}

const STAGE_NUTRIENT_LIMITS: Record<string, { k: number; p: number; pro: number; na: number }> = {
  G2: { k: 525, p: 150, pro: 13, na: 230 },
  G3a: { k: 450, p: 120, pro: 9, na: 230 },
  G3b: { k: 450, p: 120, pro: 9, na: 230 },
  G4: { k: 375, p: 105, pro: 8, na: 230 },
};

function getHighestNutrientName(food: Food, patientStage: string): string {
  const limits = STAGE_NUTRIENT_LIMITS[patientStage] || STAGE_NUTRIENT_LIMITS.G3b;
  const nutrients = [
    { name: 'potassium', ratio: food.potassium_mg / limits.k },
    { name: 'phosphorus', ratio: food.phosphorus_mg / limits.p },
    { name: 'protein', ratio: food.protein_g / limits.pro },
    { name: 'sodium', ratio: food.sodium_mg / limits.na },
  ];
  nutrients.sort((a, b) => b.ratio - a.ratio);
  return nutrients[0].name;
}

function getTableSafetyReason(food: Food, patientStage: string): string {
  const safety = foodStageSafety(food, patientStage);
  const stageDisplay = formatStageDisplay(patientStage);

  if (safety.isSafe) {
    return `This food fits within ${stageDisplay} limits`;
  }

  const highest = getHighestNutrientName(food, patientStage);

  if (safety.label.includes('Avoid')) {
    return `Not recommended for ${stageDisplay} — too high in ${highest}`;
  }

  return `Use small portions — ${highest} is elevated`;
}

function generateClinicalNote(food: Food, _patientStage: string): string {
  const warnings: string[] = [];

  if (food.potassium_mg >= 400) {
    warnings.push(
      `Very high potassium (${Math.round(food.potassium_mg)}mg) — avoid at Stage 3b and above`,
    );
  } else if (food.potassium_mg >= 250) {
    warnings.push(
      `High potassium (${Math.round(food.potassium_mg)}mg) — limit portions at Stage 3b and above`,
    );
  } else if (food.potassium_mg >= 150) {
    warnings.push(
      `Moderate potassium (${Math.round(food.potassium_mg)}mg) — watch portions`,
    );
  }

  if (food.phosphorus_mg >= 300) {
    warnings.push(
      `High phosphorus (${Math.round(food.phosphorus_mg)}mg) — limit at Stage 3 and above`,
    );
  } else if (food.phosphorus_mg >= 150) {
    warnings.push(
      `Moderate phosphorus (${Math.round(food.phosphorus_mg)}mg) — watch portions`,
    );
  }

  if (food.protein_g >= 20) {
    warnings.push(`High protein (${food.protein_g}g) — limit portions`);
  }

  if (food.sodium_mg >= 400) {
    warnings.push(
      `High sodium (${Math.round(food.sodium_mg)}mg) — restrict at all CKD stages`,
    );
  }

  if (warnings.length === 0) {
    return 'Low in key nutrients — safe for most CKD stages in normal portions.';
  }

  return `${warnings.join('. ')}.`;
}

export function FoodExplorer({ isDark, theme, patientStage = 'G3b' }: FoodExplorerProps) {
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState('All');
  const [stageFilter, setStageFilter] = useState<(typeof STAGE_FILTER_OPTIONS)[number]['value']>('all');
  const [showSafeOnly, setShowSafeOnly] = useState(false);
  const [kRange, setKRange] = useState<[number, number]>([0, 1800]);
  const [selected, setSelected] = useState<Food | null>(null);
  const [showCatMenu, setShowCatMenu] = useState(false);

  const stageColumnHeader = `Safe for ${formatStageDisplay(patientStage)}?`;

  const thStyle = {
    color: theme.textSecondary,
    fontSize: '0.75rem',
    fontWeight: 600,
  } as const;

  const TABLE_MIN_WIDTH = 960;

  const filtered = useMemo(() => {
    return FOODS.filter((f) => {
      const matchSearch = matchesFoodQuery(f, search);
      const matchCat = category === 'All' || f.category === category;
      const safety = foodStageSafety(f, patientStage);
      const matchStage =
        stageFilter === 'all' ||
        getStageSafety(f.ckd_stage_safe, stageFilter, f.potassium_mg, f.phosphorus_mg ?? 0, f.protein_g ?? 0).isSafe;
      const matchSafeOnly = !showSafeOnly || safety.isSafe;
      const matchK = f.potassium_mg >= kRange[0] && f.potassium_mg <= kRange[1];
      return matchSearch && matchCat && matchStage && matchSafeOnly && matchK;
    });
  }, [search, category, stageFilter, showSafeOnly, kRange, patientStage]);

  const radarData = selected
    ? [
        { nutrient: 'Potassium', value: Math.min(100, (selected.potassium_mg / 3000) * 100) },
        { nutrient: 'Phosphorus', value: Math.min(100, (selected.phosphorus_mg / 1000) * 100) },
        { nutrient: 'Protein', value: Math.min(100, (selected.protein_g / 75) * 100) },
        { nutrient: 'Sodium', value: Math.min(100, (selected.sodium_mg / 2300) * 100) },
      ]
    : [];

  const nutrientBars = selected
    ? [
        { name: 'Potassium', value: selected.potassium_mg, limit: 3000, unit: 'mg', color: '#2E86AB' },
        { name: 'Phosphorus', value: selected.phosphorus_mg, limit: 1000, unit: 'mg', color: '#F39C12' },
        { name: 'Protein', value: selected.protein_g, limit: 75, unit: 'g', color: '#27AE60' },
        { name: 'Sodium', value: selected.sodium_mg, limit: 2300, unit: 'mg', color: '#E74C3C' },
      ]
    : [];

  return (
    <div className="space-y-4 lg:space-y-3 min-w-0">
      <div>
        <div style={{ color: theme.text, fontSize: '1.4rem', fontWeight: 600 }}>Kidney Health Food Explorer</div>
        <p style={{ color: theme.textSecondary, marginTop: 4, fontSize: '0.9rem' }}>
          Explore foods and their kidney-health safety ratings — {FOODS.length} foods in database
        </p>
      </div>

      <div className="flex gap-2 sm:gap-3 flex-wrap items-center">
        <div
          className="flex items-center gap-2 px-3 sm:px-4 py-2.5 rounded-xl"
          style={{ flex: '1 1 160px', minWidth: 0, background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}
        >
          <Search size={14} style={{ color: theme.textSecondary, flexShrink: 0 }} />
          <input
            className="flex-1 bg-transparent outline-none min-w-0"
            style={{ color: theme.text, fontSize: '0.875rem' }}
            placeholder="Search English, French, or Kinyarwanda..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          {search && (
            <button onClick={() => setSearch('')}>
              <X size={13} style={{ color: theme.textSecondary }} />
            </button>
          )}
        </div>

        <div className="relative">
          <button
            onClick={() => setShowCatMenu(!showCatMenu)}
            className="flex items-center gap-2 px-3 sm:px-4 py-2.5 rounded-xl"
            style={{
              background: theme.cardBg,
              border: `1px solid ${category !== 'All' ? 'rgba(46,134,171,0.5)' : theme.cardBorder}`,
              color: category !== 'All' ? '#2E86AB' : theme.text,
              fontSize: '0.875rem',
              whiteSpace: 'nowrap',
            }}
          >
            <Filter size={13} />
            <span>{category === 'All' ? 'Category' : category}</span>
            <ChevronDown size={12} style={{ color: theme.textSecondary }} />
          </button>
          {showCatMenu && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setShowCatMenu(false)} />
              <div
                className="absolute top-full mt-1.5 left-0 z-20 rounded-xl overflow-hidden shadow-xl max-h-64 overflow-y-auto"
                style={{ background: isDark ? '#111827' : '#fff', border: `1px solid ${theme.cardBorder}`, minWidth: 160 }}
              >
                {CATEGORIES.map((c) => (
                  <button
                    key={c}
                    className="w-full text-left px-4 py-2.5"
                    style={{
                      color: category === c ? '#2E86AB' : theme.text,
                      background: category === c ? 'rgba(46,134,171,0.1)' : 'transparent',
                      fontSize: '0.875rem',
                    }}
                    onClick={() => { setCategory(c); setShowCatMenu(false); }}
                  >
                    {c}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>

        <div className="flex gap-1.5 flex-wrap">
          {STAGE_FILTER_OPTIONS.map((opt) => {
            const active = stageFilter === opt.value;
            return (
              <button
                key={opt.value}
                onClick={() => setStageFilter(opt.value)}
                className="px-3 py-2 rounded-xl text-sm"
                style={{
                  background: active ? 'rgba(46,134,171,0.12)' : theme.cardBg,
                  border: `1px solid ${active ? 'rgba(46,134,171,0.5)' : theme.cardBorder}`,
                  color: active ? '#2E86AB' : theme.textSecondary,
                  fontWeight: active ? 600 : 400,
                }}
              >
                {opt.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm text-muted-foreground cursor-pointer">
          <input
            type="checkbox"
            checked={showSafeOnly}
            onChange={(e) => setShowSafeOnly(e.target.checked)}
            className="rounded"
          />
          Show only foods safe for {formatStageDisplay(patientStage)}
        </label>
      </div>

      <div className="flex flex-col sm:flex-row sm:flex-wrap items-stretch sm:items-center gap-3 sm:gap-4 min-w-0">
        <span style={{ color: theme.textSecondary, fontSize: '0.8rem', whiteSpace: 'nowrap' }}>Potassium (mg)</span>
        <input
          type="range"
          min={0}
          max={1800}
          value={kRange[0]}
          onChange={(e) => setKRange([Math.min(Number(e.target.value), kRange[1]), kRange[1]])}
          className="flex-1 min-w-0"
        />
        <input
          type="range"
          min={0}
          max={1800}
          value={kRange[1]}
          onChange={(e) => setKRange([kRange[0], Math.max(Number(e.target.value), kRange[0])])}
          className="flex-1 min-w-0"
        />
        <span style={{ color: theme.text, fontSize: '0.8rem', minWidth: 90 }}>
          {kRange[0]} – {kRange[1]}
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-10 gap-4 lg:gap-5 items-start min-w-0 lg:items-stretch">
        <div className="w-full min-w-0 lg:col-span-7 flex flex-col min-h-0">
          <div className="rounded-2xl overflow-hidden w-full flex-1 min-h-0" style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}>
            <div className="w-full overflow-x-auto max-h-[min(540px,65vh)] lg:max-h-[calc(100vh-17rem)]" style={{ overflowY: 'auto' }}>
              <table
                className="w-full border-collapse"
                style={{ minWidth: TABLE_MIN_WIDTH, tableLayout: 'fixed' }}
              >
                <colgroup>
                  <col style={{ width: '22%' }} />
                  <col style={{ width: '12%' }} />
                  <col style={{ width: '10%' }} />
                  <col style={{ width: '10%' }} />
                  <col style={{ width: '11%' }} />
                  <col style={{ width: '8%' }} />
                  <col style={{ width: '27%' }} />
                </colgroup>
                <thead
                  className="sticky top-0 z-10"
                  style={{
                    background: theme.cardBg,
                    borderBottom: `1px solid ${theme.cardBorder}`,
                    boxShadow: isDark ? '0 1px 0 rgba(255,255,255,0.06)' : '0 1px 0 rgba(0,0,0,0.06)',
                  }}
                >
                  <tr>
                    <th className="px-3 py-2.5 text-left whitespace-nowrap" style={thStyle}>Food</th>
                    <th className="px-3 py-2.5 text-left whitespace-nowrap" style={thStyle}>Kinyarwanda</th>
                    <th className="px-3 py-2.5 text-left whitespace-nowrap" style={thStyle}>Category</th>
                    <th className="px-3 py-2.5 text-left whitespace-nowrap" style={thStyle}>Potassium</th>
                    <th className="px-3 py-2.5 text-left whitespace-nowrap" style={thStyle}>Phosphorus</th>
                    <th className="px-3 py-2.5 text-left whitespace-nowrap" style={thStyle}>Protein</th>
                    <th className="px-3 py-2.5 text-left whitespace-nowrap" style={{ ...thStyle, fontSize: '0.72rem' }}>
                      {stageColumnHeader}
                    </th>
                  </tr>
                </thead>
                <tbody>
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={7}>
                      <div className="py-14 flex flex-col items-center gap-2">
                        <Search size={26} style={{ color: theme.textTertiary }} />
                        <p style={{ color: theme.textSecondary, fontSize: '0.9rem' }}>No foods match your filters</p>
                      </div>
                    </td>
                  </tr>
                )}
                {filtered.map((food, i) => {
                  const isSelected = selected?.id === food.id;
                  const kColor = potassiumColor(food.potassium_mg);
                  const frenchName = foodTranslation(food.french);
                  const kinyarwandaName = foodTranslation(food.kinyarwanda);
                  const safety = foodStageSafety(food, patientStage);
                  return (
                    <tr
                      key={food.id}
                      onClick={() => setSelected(isSelected ? null : food)}
                      className="cursor-pointer transition-colors"
                      style={{
                        borderBottom: i < filtered.length - 1 ? `1px solid ${theme.cardBorder}` : 'none',
                        background: isSelected ? (isDark ? 'rgba(46,134,171,0.1)' : 'rgba(46,134,171,0.07)') : 'transparent',
                      }}
                    >
                      <td className="px-3 py-3 align-middle">
                        <div className="min-w-0">
                          <div className="capitalize truncate" style={{ color: theme.text, fontSize: '0.85rem', fontWeight: 600 }}>
                            {food.english}
                          </div>
                          {frenchName && (
                            <div className="truncate" style={{ color: theme.text, fontSize: '0.8rem', marginTop: 2 }}>
                              {frenchName}
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="px-3 py-3 align-middle truncate italic" style={{ color: theme.textSecondary, fontSize: '0.75rem' }}>
                        {kinyarwandaName ?? ''}
                      </td>
                      <td className="px-3 py-3 align-middle">
                        <span
                          className="inline-block max-w-full truncate px-2 py-0.5 rounded-full"
                          style={{ background: 'rgba(46,134,171,0.12)', color: '#2E86AB', fontSize: '0.68rem', fontWeight: 500 }}
                        >
                          {food.category}
                        </span>
                      </td>
                      <td className="px-3 py-3 align-middle tabular-nums whitespace-nowrap" style={{ color: kColor, fontSize: '0.85rem', fontWeight: 600 }}>{food.potassium_mg}</td>
                      <td className="px-3 py-3 align-middle tabular-nums whitespace-nowrap" style={{ color: theme.text, fontSize: '0.85rem' }}>{food.phosphorus_mg}</td>
                      <td className="px-3 py-3 align-middle tabular-nums whitespace-nowrap" style={{ color: theme.text, fontSize: '0.85rem' }}>{food.protein_g}</td>
                      <td className="px-3 py-2 align-middle">
                        <span className={`text-xs font-medium block ${safety.color}`}>
                          {safety.label}
                        </span>
                        <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
                          {getTableSafetyReason(food, patientStage)}
                        </p>
                      </td>
                    </tr>
                  );
                })}
                </tbody>
              </table>
            </div>
          </div>
          <p style={{ color: theme.textTertiary, fontSize: '0.75rem', marginTop: 8 }}>
            Showing {filtered.length} of {FOODS.length} foods · Tap a row for details
          </p>
        </div>

        <div className="w-full min-w-0 lg:col-span-3 space-y-3 lg:sticky lg:top-24 lg:self-start lg:max-h-[calc(100vh-6.5rem)] lg:overflow-y-auto">
          {selected ? (() => {
            const selectedSafety = foodStageSafety(selected, patientStage);
            return (
            <>
              <div className="p-5 sm:p-6 rounded-2xl" style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}>
                <div className="flex items-start justify-between gap-3 mb-4">
                  <div>
                    <div className="capitalize" style={{ color: theme.text, fontWeight: 700, fontSize: '1rem' }}>{selected.english}</div>
                    {foodTranslation(selected.french) && (
                      <div style={{ color: theme.text, fontSize: '0.9rem', marginTop: 4 }}>{foodTranslation(selected.french)}</div>
                    )}
                    {foodTranslation(selected.kinyarwanda) && (
                      <div className="italic" style={{ color: theme.textSecondary, fontSize: '0.85rem', marginTop: 4 }}>{foodTranslation(selected.kinyarwanda)}</div>
                    )}
                    <div className="flex gap-2 mt-2 flex-wrap items-center">
                      <span className="px-2.5 py-0.5 rounded-full" style={{ background: 'rgba(46,134,171,0.12)', color: '#2E86AB', fontSize: '0.72rem' }}>
                        {selected.category}
                      </span>
                      <span className={`text-xs font-medium ${selectedSafety.color}`}>
                        {selectedSafety.label}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">{selectedSafety.detail}</p>
                  </div>
                  <button onClick={() => setSelected(null)} className="p-1">
                    <X size={14} style={{ color: theme.textSecondary }} />
                  </button>
                </div>
                <div className="flex items-start gap-2 p-3 rounded-xl mb-4" style={{ background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)' }}>
                  <Info size={12} style={{ color: '#2E86AB', marginTop: 1, flexShrink: 0 }} />
                  <p style={{ color: theme.textSecondary, fontSize: '0.8rem', lineHeight: 1.55 }}>
                    {generateClinicalNote(selected, patientStage)}
                  </p>
                </div>
                <div className="space-y-3">
                  {nutrientBars.map((b) => {
                    const pct = Math.min(100, (b.value / b.limit) * 100);
                    const color = pct > 80 ? '#E74C3C' : pct > 50 ? '#F39C12' : b.color;
                    return (
                      <div key={b.name}>
                        <div className="flex justify-between mb-1">
                          <span style={{ color: theme.text, fontSize: '0.8rem', fontWeight: 500 }}>{b.name}</span>
                          <span style={{ color: theme.textSecondary, fontSize: '0.75rem' }}>{b.value}{b.unit} / {b.limit}{b.unit}</span>
                        </div>
                        <div className="rounded-full overflow-hidden" style={{ height: 6, background: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)' }}>
                          <div style={{ width: `${pct}%`, height: 6, background: color, borderRadius: 9999 }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
              <div className="p-4 sm:p-5 rounded-2xl" style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}>
                <div className="flex items-center gap-2 mb-1">
                  <BarChart2 size={14} style={{ color: '#2E86AB' }} />
                  <span style={{ color: theme.text, fontWeight: 600, fontSize: '0.875rem' }}>Nutrient risk profile</span>
                </div>
                <ResponsiveContainer width="100%" height={175}>
                  <RadarChart data={radarData} outerRadius={60}>
                    <PolarGrid stroke={isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'} />
                    <PolarAngleAxis dataKey="nutrient" tick={{ fill: theme.textSecondary, fontSize: 11 }} />
                    <Radar dataKey="value" stroke="#2E86AB" fill="#2E86AB" fillOpacity={0.2} strokeWidth={2} dot={{ r: 3, fill: '#2E86AB' }} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </>
            );
          })() : (
            <div
              className="rounded-2xl flex flex-col items-center justify-center text-center p-8"
              style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}`, minHeight: 320 }}
            >
              <Search size={22} style={{ color: theme.textTertiary, marginBottom: 12 }} />
              <p style={{ color: theme.textSecondary, fontWeight: 500 }}>Select a food to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
