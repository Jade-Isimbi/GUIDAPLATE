import { useState } from 'react';
import { Search, Filter, ChevronDown, Info, BarChart2, X } from 'lucide-react';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer } from 'recharts';
import { CATEGORIES, FOODS, type Food, isSafeForStage, potassiumColor } from '../../data/foodDatabase';

interface FoodExplorerProps {
  isDark: boolean;
  theme: Record<string, string>;
}

const STAGE_FILTER_OPTIONS = [
  { label: 'All', value: 'all' },
  { label: 'Safe for G2', value: '2' },
  { label: 'Safe for G3', value: '3' },
  { label: 'Safe for G4', value: '4' },
] as const;

export function FoodExplorer({ isDark, theme }: FoodExplorerProps) {
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState('All');
  const [stageFilter, setStageFilter] = useState<(typeof STAGE_FILTER_OPTIONS)[number]['value']>('all');
  const [kRange, setKRange] = useState<[number, number]>([0, 1800]);
  const [selected, setSelected] = useState<Food | null>(null);
  const [showCatMenu, setShowCatMenu] = useState(false);

  const filtered = FOODS.filter((f) => {
    const q = search.trim().toLowerCase();
    const matchSearch =
      !q ||
      f.english.toLowerCase().includes(q) ||
      f.french.toLowerCase().includes(q) ||
      f.kinyarwanda.toLowerCase().includes(q);
    const matchCat = category === 'All' || f.category === category;
    const matchStage =
      stageFilter === 'all' || isSafeForStage(f.ckd_stage_safe, parseInt(stageFilter, 10));
    const matchK = f.potassium_mg >= kRange[0] && f.potassium_mg <= kRange[1];
    return matchSearch && matchCat && matchStage && matchK;
  });

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

  const gridCols = '2fr 1.2fr 0.9fr 0.7fr 0.7fr 0.7fr 0.9fr';

  return (
    <div className="space-y-5 sm:space-y-6">
      <div>
        <div style={{ color: theme.text, fontSize: '1.4rem', fontWeight: 600 }}>Rwanda CKD Food Explorer</div>
        <p style={{ color: theme.textSecondary, marginTop: 4, fontSize: '0.9rem' }}>
          Explore commonly consumed Rwandan foods and their CKD safety ratings — {FOODS.length} foods in database
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

      <div className="flex items-center gap-4">
        <span style={{ color: theme.textSecondary, fontSize: '0.8rem', whiteSpace: 'nowrap' }}>Potassium (mg)</span>
        <input
          type="range"
          min={0}
          max={1800}
          value={kRange[0]}
          onChange={(e) => setKRange([Math.min(Number(e.target.value), kRange[1]), kRange[1]])}
          className="flex-1"
        />
        <input
          type="range"
          min={0}
          max={1800}
          value={kRange[1]}
          onChange={(e) => setKRange([kRange[0], Math.max(Number(e.target.value), kRange[0])])}
          className="flex-1"
        />
        <span style={{ color: theme.text, fontSize: '0.8rem', minWidth: 90 }}>
          {kRange[0]} – {kRange[1]}
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-5 sm:gap-6">
        <div className="lg:col-span-3">
          <div className="rounded-2xl overflow-hidden" style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}>
            <div className="overflow-x-auto">
              <div
                className="grid px-4 sm:px-5 py-3"
                style={{
                  gridTemplateColumns: gridCols,
                  borderBottom: `1px solid ${theme.cardBorder}`,
                  background: isDark ? 'rgba(255,255,255,0.025)' : 'rgba(0,0,0,0.025)',
                  minWidth: 640,
                }}
              >
                {['Food', 'Kinyarwanda', 'Category', 'K (mg)', 'P (mg)', 'Pro (g)', 'CKD Safe'].map((h) => (
                  <span key={h} style={{ color: theme.textSecondary, fontSize: '0.68rem', textTransform: 'uppercase', letterSpacing: '0.07em', fontWeight: 600 }}>
                    {h}
                  </span>
                ))}
              </div>
              <div style={{ maxHeight: 420, overflowY: 'auto' }}>
                {filtered.length === 0 && (
                  <div className="py-14 flex flex-col items-center gap-2">
                    <Search size={26} style={{ color: theme.textTertiary }} />
                    <p style={{ color: theme.textSecondary, fontSize: '0.9rem' }}>No foods match your filters</p>
                  </div>
                )}
                {filtered.map((food, i) => {
                  const isSelected = selected?.id === food.id;
                  const kColor = potassiumColor(food.potassium_mg);
                  return (
                    <button
                      key={food.id}
                      onClick={() => setSelected(isSelected ? null : food)}
                      className="w-full text-left grid px-4 sm:px-5 py-3"
                      style={{
                        gridTemplateColumns: gridCols,
                        borderBottom: i < filtered.length - 1 ? `1px solid ${theme.cardBorder}` : 'none',
                        background: isSelected ? (isDark ? 'rgba(46,134,171,0.1)' : 'rgba(46,134,171,0.07)') : 'transparent',
                        alignItems: 'center',
                        minWidth: 640,
                      }}
                    >
                      <div className="min-w-0 pr-2">
                        <div className="capitalize truncate" style={{ color: theme.text, fontSize: '0.85rem', fontWeight: 600 }}>
                          {food.english}
                        </div>
                        <div className="truncate" style={{ color: theme.text, fontSize: '0.8rem', marginTop: 2 }}>
                          {food.french}
                        </div>
                      </div>
                      <span style={{ color: theme.textSecondary, fontSize: '0.75rem' }} className="truncate italic">
                        {food.kinyarwanda}
                      </span>
                      <span
                        className="inline-block w-fit px-2 py-0.5 rounded-full"
                        style={{ background: 'rgba(46,134,171,0.12)', color: '#2E86AB', fontSize: '0.68rem', fontWeight: 500 }}
                      >
                        {food.category}
                      </span>
                      <span style={{ color: kColor, fontSize: '0.85rem', fontWeight: 600 }}>{food.potassium_mg}</span>
                      <span style={{ color: theme.text, fontSize: '0.85rem' }}>{food.phosphorus_mg}</span>
                      <span style={{ color: theme.text, fontSize: '0.85rem' }}>{food.protein_g}</span>
                      <span
                        className="inline-block w-fit px-2 py-0.5 rounded-full"
                        style={{ background: 'rgba(39,174,96,0.12)', color: '#27AE60', fontSize: '0.68rem', fontWeight: 600 }}
                      >
                        {food.ckd_stage_safe}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
          <p style={{ color: theme.textTertiary, fontSize: '0.75rem', marginTop: 8 }}>
            Showing {filtered.length} of {FOODS.length} foods · Tap a row for details
          </p>
        </div>

        <div className="lg:col-span-2 space-y-4">
          {selected ? (
            <>
              <div className="p-5 sm:p-6 rounded-2xl" style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}>
                <div className="flex items-start justify-between gap-3 mb-4">
                  <div>
                    <div className="capitalize" style={{ color: theme.text, fontWeight: 700, fontSize: '1rem' }}>{selected.english}</div>
                    <div style={{ color: theme.text, fontSize: '0.9rem', marginTop: 4 }}>{selected.french}</div>
                    <div className="italic" style={{ color: theme.textSecondary, fontSize: '0.85rem', marginTop: 4 }}>{selected.kinyarwanda}</div>
                    <div className="flex gap-2 mt-2 flex-wrap">
                      <span className="px-2.5 py-0.5 rounded-full" style={{ background: 'rgba(46,134,171,0.12)', color: '#2E86AB', fontSize: '0.72rem' }}>
                        {selected.category}
                      </span>
                      <span className="px-2.5 py-0.5 rounded-full" style={{ background: 'rgba(39,174,96,0.12)', color: '#27AE60', fontSize: '0.72rem', fontWeight: 600 }}>
                        Safe stages {selected.ckd_stage_safe}
                      </span>
                    </div>
                  </div>
                  <button onClick={() => setSelected(null)} className="p-1">
                    <X size={14} style={{ color: theme.textSecondary }} />
                  </button>
                </div>
                <div className="flex items-start gap-2 p-3 rounded-xl mb-4" style={{ background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)' }}>
                  <Info size={12} style={{ color: '#2E86AB', marginTop: 1, flexShrink: 0 }} />
                  <p style={{ color: theme.textSecondary, fontSize: '0.8rem', lineHeight: 1.55 }}>{selected.notes}</p>
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
          ) : (
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
