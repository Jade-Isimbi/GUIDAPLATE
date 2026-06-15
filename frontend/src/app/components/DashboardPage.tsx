import { Activity, Database, FlaskConical, Shield, ChevronRight, BookOpen, Utensils, Heart, Globe, Award, ArrowRight, Brain, TrendingUp, Eye, ListChecks } from 'lucide-react';

interface DashboardPageProps {
  isDark: boolean;
  theme: Record<string, string>;
  onNavigate: (page: string) => void;
}

const metrics = [
  { label: 'Rwandan Foods', value: '386', sub: 'in our database', color: '#2E86AB', icon: Database },
  { label: 'Nutrients Tracked', value: '4', sub: 'key biomarkers', color: '#27AE60', icon: FlaskConical },
  { label: 'CKD Stages', value: '4 (G2-G4)', sub: 'clinical stages', color: '#F39C12', icon: Shield },
  { label: 'Risk Assessments', value: '1,862 NHANES patients', sub: 'training cohort', color: '#E74C3C', icon: Activity },
];

const steps = [
  {
    n: '01',
    title: 'Explore Foods',
    desc: 'Browse our curated database of commonly consumed Rwandan foods with complete nutrient profiles and CKD suitability ratings.',
    icon: Utensils,
    color: '#2E86AB',
  },
  {
    n: '02',
    title: 'Assess Your Risk',
    desc: 'Enter your dietary intake and CKD stage to receive an instant, personalised risk assessment based on clinical thresholds.',
    icon: FlaskConical,
    color: '#F39C12',
  },
  {
    n: '03',
    title: 'Get Recommendations',
    desc: 'Receive AI-powered dietary guidance tailored to your CKD stage, with safer food alternatives and practical meal suggestions.',
    icon: Heart,
    color: '#27AE60',
  },
];

const mlComponents = [
  {
    title: 'XGBoost Risk Classifier',
    desc: 'Predicts HIGH / MODERATE / LOW dietary risk from 9 features including nutrient intake and CKD stage. Trained on 1,862 NHANES CKD patients.',
    icon: Brain,
    color: '#2E86AB',
    status: 'In Development' as const,
  },
  {
    title: 'LSTM Pattern Analyzer',
    desc: 'Detects dangerous eating patterns across 6 meal occasions over 2 days using sequential nutrient data.',
    icon: TrendingUp,
    color: '#F39C12',
    status: 'In Development' as const,
  },
  {
    title: 'SHAP Explainability',
    desc: 'Explains every prediction by showing which nutrients contributed most to the risk score. Essential for clinical trust.',
    icon: Eye,
    color: '#8E44AD',
    status: 'In Development' as const,
  },
  {
    title: 'Food Recommender',
    desc: 'Suggests safer Rwandan food alternatives when a nutrient limit is exceeded. Grounded in KDOQI 2020 guidelines.',
    icon: ListChecks,
    color: '#27AE60',
    status: 'Active' as const,
  },
];

const insights = [
  { stat: '10–15%', label: 'of adults in sub-Saharan Africa live with CKD' },
  { stat: '60%', label: 'of CKD complications are linked to dietary non-adherence' },
  { stat: '3×', label: 'higher hospitalisation risk with unmanaged phosphorus' },
];

const miniNutrients = [
  { label: 'Potassium', val: '2,400 mg', limit: '3,000 mg', pct: 80, color: '#2E86AB' },
  { label: 'Phosphorus', val: '850 mg', limit: '1,000 mg', pct: 85, color: '#F39C12' },
  { label: 'Protein', val: '52 g', limit: '75 g', pct: 69, color: '#27AE60' },
  { label: 'Sodium', val: '1,900 mg', limit: '2,300 mg', pct: 83, color: '#E74C3C' },
];

export function DashboardPage({ isDark, theme, onNavigate }: DashboardPageProps) {
  return (
    <div className="space-y-8 sm:space-y-10">
      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <div
        className="rounded-2xl sm:rounded-3xl overflow-hidden relative"
        style={{
          background: isDark
            ? 'linear-gradient(135deg, #0d2137 0%, #1a3a52 50%, #0d1f2d 100%)'
            : 'linear-gradient(135deg, #ddeef8 0%, #cce5f5 50%, #ddeef8 100%)',
          border: `1px solid ${isDark ? 'rgba(46,134,171,0.3)' : 'rgba(46,134,171,0.25)'}`,
        }}
      >
        {/* Decorative blobs */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          <div
            className="absolute rounded-full"
            style={{
              width: 340, height: 340,
              top: -80, right: -80,
              background: '#2E86AB',
              opacity: isDark ? 0.06 : 0.08,
            }}
          />
          <div
            className="absolute rounded-full"
            style={{
              width: 200, height: 200,
              bottom: -60, left: -40,
              background: '#27AE60',
              opacity: isDark ? 0.06 : 0.08,
            }}
          />
        </div>

        <div className="relative px-6 sm:px-12 py-8 sm:py-14 flex flex-col lg:flex-row gap-8 lg:gap-12 items-start lg:items-center">
          {/* Left: copy */}
          <div className="flex-1 min-w-0">
            <div
              className="inline-flex items-center gap-2 px-3 sm:px-4 py-1.5 rounded-full mb-5"
              style={{
                background: 'rgba(46,134,171,0.15)',
                border: '1px solid rgba(46,134,171,0.3)',
                color: '#2E86AB',
                fontSize: '0.78rem',
              }}
            >
              <Globe size={12} />
              Designed for Rwanda · CKD Dietary Guidance
            </div>

            <div
              style={{
                color: theme.text,
                fontSize: 'clamp(1.6rem, 5vw, 2.6rem)',
                fontWeight: 700,
                lineHeight: 1.15,
                letterSpacing: '-0.025em',
                marginBottom: '1rem',
              }}
            >
              Smarter eating for
              <br />
              <span style={{ color: '#2E86AB' }}>kidney health</span>
            </div>

            <p
              style={{
                color: theme.textSecondary,
                fontSize: '0.95rem',
                lineHeight: 1.65,
                maxWidth: 460,
                marginBottom: '1.75rem',
              }}
            >
              GuidaPlate is an AI-powered dietary guidance platform that helps CKD patients
              understand nutrient risks in everyday Rwandan foods and make safer, informed meal choices.
            </p>

            <div className="flex gap-3 flex-wrap">
              <button
                onClick={() => onNavigate('assessment')}
                className="flex items-center gap-2 px-5 py-2.5 sm:px-6 sm:py-3 rounded-xl text-white transition-all duration-200 hover:opacity-90"
                style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)', fontSize: '0.875rem', fontWeight: 600 }}
              >
                Start Meal Assessment
                <ArrowRight size={15} />
              </button>
              <button
                onClick={() => onNavigate('explorer')}
                className="flex items-center gap-2 px-5 py-2.5 sm:px-6 sm:py-3 rounded-xl transition-all duration-200 hover:opacity-80"
                style={{
                  background: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(46,134,171,0.1)',
                  border: `1px solid ${isDark ? 'rgba(255,255,255,0.14)' : 'rgba(46,134,171,0.3)'}`,
                  color: isDark ? theme.text : '#1A5F7A',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                }}
              >
                Explore Foods
              </button>
            </div>
          </div>

          {/* Right: mini nutrient cards — hidden on small, shown on lg+ */}
          <div className="grid grid-cols-2 gap-3 w-full lg:w-auto lg:shrink-0" style={{ maxWidth: 300 }}>
            {miniNutrients.map((n) => (
              <div
                key={n.label}
                className="p-3 sm:p-4 rounded-2xl"
                style={{
                  background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.75)',
                  border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.95)'}`,
                  backdropFilter: 'blur(8px)',
                }}
              >
                <p style={{ color: theme.textSecondary, fontSize: '0.72rem', marginBottom: 4 }}>{n.label}</p>
                <p style={{ color: n.color, fontSize: '0.85rem', fontWeight: 700, marginBottom: 6 }}>{n.val}</p>
                <div className="w-full rounded-full overflow-hidden" style={{ height: 5, background: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)' }}>
                  <div style={{ width: `${n.pct}%`, height: 5, background: n.color, borderRadius: 9999 }} />
                </div>
                <p style={{ color: theme.textTertiary, fontSize: '0.68rem', marginTop: 4 }}>of {n.limit}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Metrics ──────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5">
        {metrics.map((m) => {
          const Icon = m.icon;
          return (
            <div
              key={m.label}
              className="p-5 sm:p-6 rounded-2xl"
              style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}
            >
              <div
                className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl flex items-center justify-center mb-3 sm:mb-4"
                style={{ background: `${m.color}18` }}
              >
                <Icon size={18} style={{ color: m.color }} />
              </div>
              <div style={{ color: theme.text, fontSize: 'clamp(1.4rem, 3vw, 1.9rem)', fontWeight: 700, lineHeight: 1 }}>{m.value}</div>
              <p style={{ color: theme.text, fontSize: '0.85rem', fontWeight: 500, marginTop: 4 }}>{m.label}</p>
              <p style={{ color: theme.textTertiary, fontSize: '0.72rem', marginTop: 2 }}>{m.sub}</p>
            </div>
          );
        })}
      </div>

      {/* ── ML Architecture ──────────────────────────────────────────────── */}
      <div>
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ color: theme.text, fontSize: '1.3rem', fontWeight: 600 }}>ML Architecture</div>
          <p style={{ color: theme.textSecondary, marginTop: 4, fontSize: '0.875rem' }}>Four components powering GuidaPlate risk intelligence</p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-5">
          {mlComponents.map((c) => {
            const Icon = c.icon;
            const isActive = c.status === 'Active';
            return (
              <div
                key={c.title}
                className="p-5 sm:p-7 rounded-2xl"
                style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}
              >
                <div className="flex items-start justify-between gap-3 mb-4">
                  <div
                    className="w-10 h-10 sm:w-11 sm:h-11 rounded-xl flex items-center justify-center shrink-0"
                    style={{ background: `${c.color}15` }}
                  >
                    <Icon size={20} style={{ color: c.color }} />
                  </div>
                  <span
                    className="px-2.5 py-0.5 rounded-full shrink-0"
                    style={{
                      background: isActive ? 'rgba(39,174,96,0.12)' : 'rgba(243,156,18,0.12)',
                      color: isActive ? '#27AE60' : '#F39C12',
                      fontSize: '0.68rem',
                      fontWeight: 600,
                    }}
                  >
                    {c.status}
                  </span>
                </div>
                <div style={{ color: theme.text, fontSize: '0.95rem', fontWeight: 600, marginBottom: 8 }}>{c.title}</div>
                <p style={{ color: theme.textSecondary, fontSize: '0.85rem', lineHeight: 1.65 }}>{c.desc}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── How it works ─────────────────────────────────────────────────── */}
      <div>
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ color: theme.text, fontSize: '1.3rem', fontWeight: 600 }}>How GuidaPlate works</div>
          <p style={{ color: theme.textSecondary, marginTop: 4, fontSize: '0.875rem' }}>Three steps from food to safe decision</p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 sm:gap-5">
          {steps.map((s) => {
            const Icon = s.icon;
            return (
              <div
                key={s.n}
                className="p-5 sm:p-7 rounded-2xl"
                style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}
              >
                <div className="flex items-center gap-3 mb-4">
                  <div
                    className="w-10 h-10 sm:w-11 sm:h-11 rounded-xl flex items-center justify-center shrink-0"
                    style={{ background: `${s.color}15` }}
                  >
                    <Icon size={20} style={{ color: s.color }} />
                  </div>
                  <span style={{ color: s.color, fontWeight: 700, fontSize: '0.8rem', letterSpacing: '0.08em', fontFamily: 'monospace' }}>
                    {s.n}
                  </span>
                </div>
                <div style={{ color: theme.text, fontSize: '0.95rem', fontWeight: 600, marginBottom: 8 }}>{s.title}</div>
                <p style={{ color: theme.textSecondary, fontSize: '0.85rem', lineHeight: 1.65 }}>{s.desc}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Clinical insight + Quick nav ──────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-5 sm:gap-6">
        {/* Clinical */}
        <div
          className="lg:col-span-3 p-6 sm:p-8 rounded-2xl"
          style={{
            background: isDark
              ? 'linear-gradient(135deg, #0d1f2d 0%, #1a3a52 100%)'
              : 'linear-gradient(135deg, #ddeef8 0%, #cce5f5 100%)',
            border: `1px solid rgba(46,134,171,0.25)`,
          }}
        >
          <div className="flex items-center gap-2 mb-5">
            <BookOpen size={16} style={{ color: '#2E86AB' }} />
            <span style={{ color: theme.text, fontWeight: 600, fontSize: '0.95rem' }}>Clinical Context</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 sm:gap-5 mb-5">
            {insights.map((i) => (
              <div key={i.stat}>
                <div style={{ color: '#2E86AB', fontSize: 'clamp(1.2rem, 3vw, 1.6rem)', fontWeight: 700 }}>{i.stat}</div>
                <p style={{ color: theme.textSecondary, fontSize: '0.8rem', marginTop: 4, lineHeight: 1.45 }}>{i.label}</p>
              </div>
            ))}
          </div>
          <p style={{ color: theme.textSecondary, fontSize: '0.85rem', lineHeight: 1.65 }}>
            Chronic Kidney Disease is a leading cause of morbidity in Rwanda, yet dietary management remains
            challenging due to limited access to localised, culturally relevant nutritional guidance.
            GuidaPlate bridges this gap by combining validated clinical thresholds with local food data.
          </p>
        </div>

        {/* Quick nav */}
        <div className="lg:col-span-2 flex flex-col gap-4">
          {[
            { label: 'Food Explorer', desc: 'Search and filter 247 Rwandan foods', page: 'explorer', icon: Utensils, color: '#27AE60' },
            { label: 'Meal Assessment', desc: 'Log a meal and get your dietary risk score', page: 'assessment', icon: Shield, color: '#E74C3C' },
          ].map((nav) => {
            const Icon = nav.icon;
            return (
              <button
                key={nav.page}
                onClick={() => onNavigate(nav.page)}
                className="w-full text-left p-4 sm:p-5 rounded-2xl transition-all duration-150 hover:shadow-md"
                style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3 sm:gap-3.5">
                    <div
                      className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
                      style={{ background: `${nav.color}15` }}
                    >
                      <Icon size={19} style={{ color: nav.color }} />
                    </div>
                    <div>
                      <p style={{ color: theme.text, fontWeight: 600, fontSize: '0.9rem' }}>{nav.label}</p>
                      <p style={{ color: theme.textSecondary, fontSize: '0.8rem', marginTop: 2 }}>{nav.desc}</p>
                    </div>
                  </div>
                  <ChevronRight size={16} style={{ color: theme.textTertiary, flexShrink: 0 }} />
                </div>
              </button>
            );
          })}

          {/* About */}
          <div
            className="p-4 sm:p-5 rounded-2xl flex-1"
            style={{ background: isDark ? 'rgba(39,174,96,0.07)' : 'rgba(39,174,96,0.06)', border: '1px solid rgba(39,174,96,0.2)' }}
          >
            <div className="flex items-center gap-2 mb-2">
              <Award size={15} style={{ color: '#27AE60' }} />
              <span style={{ color: theme.text, fontWeight: 600, fontSize: '0.85rem' }}>About this project</span>
            </div>
            <p style={{ color: theme.textSecondary, fontSize: '0.8rem', lineHeight: 1.6 }}>
              GuidaPlate is a research and MVP demonstration project combining clinical nutrition science with machine learning to address a critical gap in CKD
              dietary management across East Africa.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
