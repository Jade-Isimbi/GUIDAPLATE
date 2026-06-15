import { useState } from 'react';
import { Search, Filter, ChevronDown, Info, BarChart2, CheckCircle2, AlertTriangle, XCircle, X } from 'lucide-react';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer } from 'recharts';

interface FoodExplorerProps {
  isDark: boolean;
  theme: Record<string, string>;
}

type Safety = 'Safe' | 'Moderate' | 'Avoid';

interface Food {
  id: number;
  name: string;
  category: string;
  potassium: number;
  phosphorus: number;
  protein: number;
  sodium: number;
  safety: Safety;
  notes: string;
}

const foods: Food[] = [
  { id: 1,  name: 'White Rice (cooked)',     category: 'Grains',     potassium: 55,  phosphorus: 43,  protein: 2.7,  sodium: 1,   safety: 'Safe',     notes: 'Excellent staple for CKD. Very low in potassium and phosphorus — ideal base grain.' },
  { id: 2,  name: 'Cassava (boiled)',         category: 'Tubers',     potassium: 271, phosphorus: 28,  protein: 1.4,  sodium: 14,  safety: 'Safe',     notes: 'Good energy source. Leach by soaking before cooking to further reduce potassium.' },
  { id: 3,  name: 'Plantain (green, cooked)',  category: 'Fruits',    potassium: 499, phosphorus: 34,  protein: 1.3,  sodium: 4,   safety: 'Moderate', notes: 'Moderate potassium — limit to ½ cup per serving. Greener plantains are safer than ripe.' },
  { id: 4,  name: 'Beans (kidney, cooked)',   category: 'Legumes',    potassium: 403, phosphorus: 244, protein: 15.4, sodium: 2,   safety: 'Avoid',    notes: 'High in phosphorus and potassium. Not recommended for CKD stages G3–G5.' },
  { id: 5,  name: 'Sweet Potato (boiled)',    category: 'Tubers',     potassium: 475, phosphorus: 50,  protein: 2.0,  sodium: 36,  safety: 'Moderate', notes: 'Boil in large water and discard cooking liquid — this significantly reduces potassium.' },
  { id: 6,  name: 'Cabbage (raw)',            category: 'Vegetables', potassium: 170, phosphorus: 26,  protein: 1.3,  sodium: 18,  safety: 'Safe',     notes: 'Excellent low-potassium vegetable. Good source of vitamins C and K. Eat freely.' },
  { id: 7,  name: 'Egg (whole, boiled)',      category: 'Protein',    potassium: 126, phosphorus: 172, protein: 12.6, sodium: 142, safety: 'Moderate', notes: 'Good protein quality. Egg whites preferred — they contain less phosphorus than yolks.' },
  { id: 8,  name: 'Tilapia (grilled)',        category: 'Fish',       potassium: 302, phosphorus: 204, protein: 26.2, sodium: 56,  safety: 'Moderate', notes: 'Good lean protein. Limit to 85g per meal. Avoid battered or heavily salted versions.' },
  { id: 9,  name: 'Spinach (cooked)',         category: 'Vegetables', potassium: 839, phosphorus: 101, protein: 5.4,  sodium: 164, safety: 'Avoid',    notes: 'Very high potassium. Avoid in CKD stages G3–G5. Substitute with cabbage or lettuce.' },
  { id: 10, name: 'Sorghum flour',            category: 'Grains',     potassium: 350, phosphorus: 287, protein: 10.6, sodium: 2,   safety: 'Moderate', notes: 'Traditional grain. Use in moderation. White sorghum has lower phosphorus than red.' },
  { id: 11, name: 'Avocado',                  category: 'Fruits',     potassium: 485, phosphorus: 52,  protein: 2.0,  sodium: 7,   safety: 'Avoid',    notes: 'Very high potassium content. Only small portions for stages G1–G2.' },
  { id: 12, name: 'White Bread',              category: 'Grains',     potassium: 96,  phosphorus: 57,  protein: 7.6,  sodium: 477, safety: 'Moderate', notes: 'High sodium from salt in processing. Choose low-sodium versions when available.' },
  { id: 13, name: 'Pineapple (fresh)',         category: 'Fruits',    potassium: 109, phosphorus: 8,   protein: 0.5,  sodium: 1,   safety: 'Safe',     notes: 'One of the safest fruits for CKD. Low in potassium and phosphorus. Enjoy freely.' },
  { id: 14, name: 'Maize meal (ugali)',        category: 'Grains',    potassium: 130, phosphorus: 89,  protein: 3.7,  sodium: 5,   safety: 'Safe',     notes: 'Traditional staple with an acceptable nutrient profile. Good energy source for CKD.' },
  { id: 15, name: 'Milk (whole)',              category: 'Dairy',      potassium: 150, phosphorus: 233, protein: 8.0,  sodium: 107, safety: 'Avoid',    notes: 'High phosphorus and potassium. Limit to 120 ml per day or use phosphorus-free alternatives.' },
  { id: 16, name: 'Carrot (raw)',              category: 'Vegetables', potassium: 320, phosphorus: 35,  protein: 0.9,  sodium: 69,  safety: 'Safe',     notes: 'Good source of beta-carotene. Safe in moderate amounts for most CKD stages.' },
  { id: 17, name: 'Chicken Breast (grilled)',  category: 'Protein',   potassium: 220, phosphorus: 196, protein: 27.3, sodium: 74,  safety: 'Moderate', notes: 'Lean protein source. Remove skin. Limit to 85–100 g per meal.' },
  { id: 18, name: 'Banana (ripe)',             category: 'Fruits',    potassium: 358, phosphorus: 22,  protein: 1.1,  sodium: 1,   safety: 'Avoid',    notes: 'High potassium fruit. One small banana approaches the daily limit — avoid in G3+.' },
];

const categories = ['All', 'Grains', 'Tubers', 'Fruits', 'Vegetables', 'Legumes', 'Protein', 'Fish', 'Dairy'];
const safetyFilters: Array<Safety | 'All'> = ['All', 'Safe', 'Moderate', 'Avoid'];

const safetyConfig: Record<Safety, { color: string; bg: string; icon: typeof CheckCircle2 }> = {
  Safe:     { color: '#27AE60', bg: 'rgba(39,174,96,0.12)',  icon: CheckCircle2 },
  Moderate: { color: '#F39C12', bg: 'rgba(243,156,18,0.12)', icon: AlertTriangle },
  Avoid:    { color: '#E74C3C', bg: 'rgba(231,76,60,0.12)',  icon: XCircle },
};

export function FoodExplorer({ isDark, theme }: FoodExplorerProps) {
  const [search,      setSearch]      = useState('');
  const [category,    setCategory]    = useState('All');
  const [safety,      setSafety]      = useState<Safety | 'All'>('All');
  const [selected,    setSelected]    = useState<Food | null>(null);
  const [showCatMenu, setShowCatMenu] = useState(false);

  const filtered = foods.filter((f) => {
    const matchSearch = f.name.toLowerCase().includes(search.toLowerCase());
    const matchCat    = category === 'All' || f.category === category;
    const matchSafety = safety   === 'All' || f.safety   === safety;
    return matchSearch && matchCat && matchSafety;
  });

  const safetyCount = (s: Safety | 'All') =>
    s === 'All' ? foods.length : foods.filter((f) => f.safety === s).length;

  const radarData = selected
    ? [
        { nutrient: 'Potassium',  value: Math.min(100, (selected.potassium  / 3000) * 100) },
        { nutrient: 'Phosphorus', value: Math.min(100, (selected.phosphorus / 1000) * 100) },
        { nutrient: 'Protein',    value: Math.min(100, (selected.protein    /   75) * 100) },
        { nutrient: 'Sodium',     value: Math.min(100, (selected.sodium     / 2300) * 100) },
      ]
    : [];

  const nutrientBars = selected
    ? [
        { name: 'Potassium',  value: selected.potassium,  limit: 3000, unit: 'mg', color: '#2E86AB' },
        { name: 'Phosphorus', value: selected.phosphorus, limit: 1000, unit: 'mg', color: '#F39C12' },
        { name: 'Protein',    value: selected.protein,    limit: 75,   unit: 'g',  color: '#27AE60' },
        { name: 'Sodium',     value: selected.sodium,     limit: 2300, unit: 'mg', color: '#E74C3C' },
      ]
    : [];

  const gridCols = '2fr 1fr 0.7fr 0.7fr 0.7fr 0.7fr 1fr';

  return (
    <div className="space-y-5 sm:space-y-6">
      {/* Header */}
      <div>
        <div style={{ color: theme.text, fontSize: '1.4rem', fontWeight: 600 }}>Food Explorer</div>
        <p style={{ color: theme.textSecondary, marginTop: 4, fontSize: '0.9rem' }}>
          Browse {foods.length} Rwandan foods with complete CKD suitability ratings and nutrient profiles
        </p>
      </div>

      {/* Controls */}
      <div className="flex gap-2 sm:gap-3 flex-wrap items-center">
        {/* Search */}
        <div
          className="flex items-center gap-2 px-3 sm:px-4 py-2.5 rounded-xl"
          style={{
            flex: '1 1 160px',
            minWidth: 0,
            background: theme.cardBg,
            border: `1px solid ${theme.cardBorder}`,
          }}
        >
          <Search size={14} style={{ color: theme.textSecondary, flexShrink: 0 }} />
          <input
            className="flex-1 bg-transparent outline-none min-w-0"
            style={{ color: theme.text, fontSize: '0.875rem' }}
            placeholder="Search foods..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          {search && (
            <button onClick={() => setSearch('')}>
              <X size={13} style={{ color: theme.textSecondary }} />
            </button>
          )}
        </div>

        {/* Category dropdown */}
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
            <Filter size={13} style={{ color: category !== 'All' ? '#2E86AB' : theme.textSecondary }} />
            <span className="hidden sm:inline">{category}</span>
            <span className="sm:hidden">{category === 'All' ? 'Category' : category}</span>
            <ChevronDown size={12} style={{ color: theme.textSecondary }} />
          </button>
          {showCatMenu && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setShowCatMenu(false)} />
              <div
                className="absolute top-full mt-1.5 left-0 z-20 rounded-xl overflow-hidden shadow-xl"
                style={{
                  background: isDark ? '#111827' : '#fff',
                  border: `1px solid ${theme.cardBorder}`,
                  minWidth: 150,
                }}
              >
                {categories.map((c) => (
                  <button
                    key={c}
                    className="w-full text-left px-4 py-2.5 transition-colors"
                    style={{
                      color:      category === c ? '#2E86AB' : theme.text,
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

        {/* Safety filter pills */}
        <div className="flex gap-1.5 sm:gap-2 flex-wrap">
          {safetyFilters.map((s) => {
            const active = safety === s;
            const accentColor = s === 'All' ? '#2E86AB' : safetyConfig[s as Safety].color;
            return (
              <button
                key={s}
                onClick={() => setSafety(s)}
                className="flex items-center gap-1.5 px-3 sm:px-4 py-2 sm:py-2.5 rounded-xl transition-all duration-150"
                style={{
                  background: active ? (s === 'All' ? 'rgba(46,134,171,0.12)' : safetyConfig[s as Safety].bg) : theme.cardBg,
                  border: `1px solid ${active ? accentColor + '55' : theme.cardBorder}`,
                  color: active ? accentColor : theme.textSecondary,
                  fontSize: '0.82rem',
                  fontWeight: active ? 600 : 400,
                }}
              >
                {s}
                <span
                  className="hidden sm:inline px-1.5 py-0.5 rounded-md"
                  style={{
                    background: active ? accentColor + '20' : isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)',
                    color: active ? accentColor : theme.textTertiary,
                    fontSize: '0.7rem',
                    fontWeight: 600,
                  }}
                >
                  {safetyCount(s)}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main layout — stacked on mobile, side-by-side on lg+ */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-5 sm:gap-6">
        {/* Table */}
        <div className="lg:col-span-3">
          <div
            className="rounded-2xl overflow-hidden"
            style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}
          >
            {/* Scrollable table wrapper */}
            <div className="overflow-x-auto">
              {/* Table header */}
              <div
                className="grid px-4 sm:px-5 py-3"
                style={{
                  gridTemplateColumns: gridCols,
                  borderBottom: `1px solid ${theme.cardBorder}`,
                  background: isDark ? 'rgba(255,255,255,0.025)' : 'rgba(0,0,0,0.025)',
                  minWidth: 520,
                }}
              >
                {['Food', 'Category', 'K (mg)', 'P (mg)', 'Pro (g)', 'Na (mg)', 'CKD Safety'].map((h) => (
                  <span key={h} style={{ color: theme.textSecondary, fontSize: '0.68rem', textTransform: 'uppercase', letterSpacing: '0.07em', fontWeight: 600 }}>
                    {h}
                  </span>
                ))}
              </div>

              {/* Rows */}
              <div style={{ maxHeight: 420, overflowY: 'auto' }}>
                {filtered.length === 0 && (
                  <div className="py-14 flex flex-col items-center gap-2">
                    <Search size={26} style={{ color: theme.textTertiary }} />
                    <p style={{ color: theme.textSecondary, fontSize: '0.9rem' }}>No foods match your search</p>
                  </div>
                )}
                {filtered.map((food, i) => {
                  const cfg        = safetyConfig[food.safety];
                  const SafetyIcon = cfg.icon;
                  const isSelected = selected?.id === food.id;
                  return (
                    <button
                      key={food.id}
                      onClick={() => setSelected(isSelected ? null : food)}
                      className="w-full text-left grid px-4 sm:px-5 py-3 transition-colors duration-100"
                      style={{
                        gridTemplateColumns: gridCols,
                        borderBottom: i < filtered.length - 1 ? `1px solid ${theme.cardBorder}` : 'none',
                        background: isSelected
                          ? isDark ? 'rgba(46,134,171,0.1)' : 'rgba(46,134,171,0.07)'
                          : 'transparent',
                        alignItems: 'center',
                        minWidth: 520,
                      }}
                      onMouseEnter={(e) => {
                        if (!isSelected) (e.currentTarget as HTMLButtonElement).style.background = isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.03)';
                      }}
                      onMouseLeave={(e) => {
                        if (!isSelected) (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
                      }}
                    >
                      <span
                        className="truncate pr-3"
                        style={{ color: theme.text, fontSize: '0.85rem', fontWeight: isSelected ? 600 : 400 }}
                        title={food.name}
                      >
                        {food.name}
                      </span>
                      <span style={{ color: theme.textSecondary, fontSize: '0.78rem' }}>{food.category}</span>
                      <span style={{ color: theme.text, fontSize: '0.85rem' }}>{food.potassium}</span>
                      <span style={{ color: theme.text, fontSize: '0.85rem' }}>{food.phosphorus}</span>
                      <span style={{ color: theme.text, fontSize: '0.85rem' }}>{food.protein}</span>
                      <span style={{ color: theme.text, fontSize: '0.85rem' }}>{food.sodium}</span>
                      <span
                        className="inline-flex items-center gap-1.5 w-fit rounded-full px-2.5 py-1"
                        style={{ background: cfg.bg, color: cfg.color, fontSize: '0.7rem', fontWeight: 600, whiteSpace: 'nowrap' }}
                      >
                        <SafetyIcon size={10} />
                        {food.safety}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
          <p style={{ color: theme.textTertiary, fontSize: '0.75rem', marginTop: 8 }}>
            Showing {filtered.length} of {foods.length} foods · Tap a row to view details
          </p>
        </div>

        {/* Detail panel */}
        <div className="lg:col-span-2 space-y-4">
          {selected ? (
            <>
              {/* Food card */}
              <div
                className="p-5 sm:p-6 rounded-2xl"
                style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}
              >
                <div className="flex items-start justify-between gap-3 mb-4">
                  <div className="min-w-0">
                    <div style={{ color: theme.text, fontWeight: 600, fontSize: '1rem', lineHeight: 1.3 }}>{selected.name}</div>
                    <span
                      className="inline-block mt-1.5 px-2.5 py-0.5 rounded-full"
                      style={{ background: 'rgba(46,134,171,0.12)', color: '#2E86AB', fontSize: '0.72rem', fontWeight: 500 }}
                    >
                      {selected.category}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <span
                      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1"
                      style={{ background: safetyConfig[selected.safety].bg, color: safetyConfig[selected.safety].color, fontSize: '0.72rem', fontWeight: 600 }}
                    >
                      {(() => { const Icon = safetyConfig[selected.safety].icon; return <Icon size={12} />; })()}
                      {selected.safety}
                    </span>
                    <button
                      onClick={() => setSelected(null)}
                      className="p-1 rounded-lg transition-opacity hover:opacity-60"
                    >
                      <X size={14} style={{ color: theme.textSecondary }} />
                    </button>
                  </div>
                </div>

                {/* Notes */}
                <div
                  className="flex items-start gap-2 p-3 rounded-xl mb-4"
                  style={{ background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)' }}
                >
                  <Info size={12} style={{ color: '#2E86AB', marginTop: 1, flexShrink: 0 }} />
                  <p style={{ color: theme.textSecondary, fontSize: '0.8rem', lineHeight: 1.55 }}>{selected.notes}</p>
                </div>

                {/* Nutrient bars */}
                <div className="space-y-3">
                  {nutrientBars.map((b) => {
                    const pct   = Math.min(100, (b.value / b.limit) * 100);
                    const color = pct > 80 ? '#E74C3C' : pct > 50 ? '#F39C12' : b.color;
                    return (
                      <div key={b.name}>
                        <div className="flex justify-between mb-1">
                          <span style={{ color: theme.text, fontSize: '0.8rem', fontWeight: 500 }}>{b.name}</span>
                          <span style={{ color: theme.textSecondary, fontSize: '0.75rem' }}>
                            {b.value}{b.unit} / {b.limit}{b.unit}
                          </span>
                        </div>
                        <div
                          className="rounded-full overflow-hidden"
                          style={{ height: 6, background: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)' }}
                        >
                          <div style={{ width: `${pct}%`, height: 6, background: color, borderRadius: 9999 }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Radar chart */}
              <div
                className="p-4 sm:p-5 rounded-2xl"
                style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}
              >
                <div className="flex items-center gap-2 mb-1">
                  <BarChart2 size={14} style={{ color: '#2E86AB' }} />
                  <span style={{ color: theme.text, fontWeight: 600, fontSize: '0.875rem' }}>Nutrient risk profile</span>
                </div>
                <ResponsiveContainer width="100%" height={175}>
                  <RadarChart data={radarData} outerRadius={60}>
                    <PolarGrid stroke={isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'} />
                    <PolarAngleAxis
                      dataKey="nutrient"
                      tick={{ fill: theme.textSecondary, fontSize: 11 }}
                    />
                    <Radar
                      dataKey="value"
                      stroke="#2E86AB"
                      fill="#2E86AB"
                      fillOpacity={0.2}
                      strokeWidth={2}
                      dot={{ r: 3, fill: '#2E86AB' }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
                <p className="text-center" style={{ color: theme.textTertiary, fontSize: '0.72rem' }}>
                  % of daily CKD limit per serving (G3a)
                </p>
              </div>
            </>
          ) : (
            <div
              className="rounded-2xl flex flex-col items-center justify-center text-center p-8 sm:p-10"
              style={{
                background: theme.cardBg,
                border: `1px solid ${theme.cardBorder}`,
                minHeight: 320,
              }}
            >
              <div
                className="w-12 h-12 sm:w-14 sm:h-14 rounded-2xl flex items-center justify-center mb-4"
                style={{ background: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)' }}
              >
                <Search size={22} style={{ color: theme.textTertiary }} />
              </div>
              <p style={{ color: theme.textSecondary, fontWeight: 500 }}>Select a food to view details</p>
              <p style={{ color: theme.textTertiary, fontSize: '0.825rem', marginTop: 6, maxWidth: 200, lineHeight: 1.5 }}>
                Nutrient breakdown and CKD safety guidance will appear here
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
