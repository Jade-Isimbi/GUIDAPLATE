import { useState } from 'react';
import { Activity, Sun, Moon, LayoutDashboard, Utensils, Shield } from 'lucide-react';
import { DashboardPage } from './components/DashboardPage';
import { FoodExplorer } from './components/FoodExplorer';
import { RiskAssessment } from './components/RiskAssessment';

type Page = 'dashboard' | 'explorer' | 'assessment';

const navItems: Array<{ id: Page; label: string; icon: typeof LayoutDashboard }> = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'explorer', label: 'Food Explorer', icon: Utensils },
  { id: 'assessment', label: 'Meal Assessment', icon: Shield },
];

export default function App() {
  {/* MARKER-MAKE-KIT-INVOKED */}
  const [isDark, setIsDark] = useState(true);
  const [page, setPage] = useState<Page>('dashboard');

  const theme = {
    bg: isDark
      ? 'linear-gradient(160deg, #080c14 0%, #0e1625 50%, #080c14 100%)'
      : 'linear-gradient(160deg, #f0f4f8 0%, #e8eef5 50%, #f0f4f8 100%)',
    navBg: isDark ? 'rgba(8,12,20,0.85)' : 'rgba(255,255,255,0.85)',
    navBorder: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.08)',
    cardBg: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.85)',
    cardBorder: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)',
    text: isDark ? '#f0f4f8' : '#0e1625',
    textSecondary: isDark ? '#8a9ab0' : '#5a6a80',
    textTertiary: isDark ? '#4a5a6a' : '#9aaac0',
    chartGrid: isDark ? '#1e2a38' : '#e0e8f0',
    chartAxis: isDark ? '#8a9ab0' : '#6a7a90',
    progressBg: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)',
  };

  return (
    <div className="min-h-screen" style={{ background: theme.bg }}>
      {/* Navigation */}
      <nav
        className="sticky top-0 z-50 backdrop-blur-xl"
        style={{
          background: theme.navBg,
          borderBottom: `1px solid ${theme.navBorder}`,
        }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-8 h-14 sm:h-16 flex items-center justify-between gap-3">
          {/* Brand */}
          <div className="flex items-center gap-2 sm:gap-3 shrink-0">
            <div
              className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl flex items-center justify-center"
              style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)' }}
            >
              <Activity size={18} className="text-white" />
            </div>
            <div className="flex items-center gap-2">
              <span style={{ color: '#2E86AB', fontWeight: 700, letterSpacing: '-0.02em', fontSize: '1rem' }}>
                GuidaPlate
              </span>
              <span
                className="hidden sm:inline text-xs px-2 py-0.5 rounded-full"
                style={{ background: 'rgba(46,134,171,0.12)', color: '#2E86AB' }}
              >
                CKD Dietary Platform
              </span>
            </div>
          </div>

          {/* Page nav */}
          <div
            className="flex items-center gap-0.5 sm:gap-1 px-1 sm:px-1.5 py-1 sm:py-1.5 rounded-xl"
            style={{
              background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.05)',
              border: `1px solid ${theme.navBorder}`,
            }}
          >
            {navItems.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setPage(id)}
                className="flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-4 py-2 rounded-lg text-sm transition-all duration-150"
                style={{
                  background: page === id
                    ? isDark ? 'rgba(46,134,171,0.2)' : 'rgba(255,255,255,0.9)'
                    : 'transparent',
                  color: page === id ? '#2E86AB' : theme.textSecondary,
                  fontWeight: page === id ? 600 : 400,
                  boxShadow: page === id && !isDark ? '0 1px 4px rgba(0,0,0,0.1)' : 'none',
                }}
              >
                <Icon size={15} />
                <span className="hidden sm:inline">{label}</span>
              </button>
            ))}
          </div>

          {/* Right actions */}
          <div className="flex items-center gap-2 sm:gap-4 shrink-0">
            <button
              onClick={() => setIsDark(!isDark)}
              className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl flex items-center justify-center transition-all duration-200 hover:opacity-70"
              style={{
                background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
                border: `1px solid ${theme.navBorder}`,
                color: theme.textSecondary,
              }}
            >
              {isDark ? <Sun size={15} /> : <Moon size={15} />}
            </button>
            <div
              className="flex items-center gap-2 sm:gap-2.5 px-2 sm:px-3 py-1.5 rounded-xl"
              style={{
                background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)',
                border: `1px solid ${theme.navBorder}`,
              }}
            >
              <div
                className="w-7 h-7 rounded-lg flex items-center justify-center text-xs text-white shrink-0"
                style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #27AE60 100%)', fontWeight: 700 }}
              >
                PD
              </div>
              <div className="hidden sm:block leading-none">
                <p className="text-sm" style={{ color: theme.text, fontWeight: 500 }}>Patient Demo</p>
                <p className="text-xs" style={{ color: theme.textSecondary }}>Stage G3a · ID #12847</p>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Page content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-8 py-6 sm:py-8">
        {page === 'dashboard' && (
          <DashboardPage isDark={isDark} theme={theme} onNavigate={(p) => setPage(p as Page)} />
        )}
        {page === 'explorer' && (
          <FoodExplorer isDark={isDark} theme={theme} />
        )}
        {page === 'assessment' && (
          <RiskAssessment isDark={isDark} theme={theme} />
        )}
      </main>

      {/* Footer */}
      <footer
        className="mt-12 sm:mt-16 py-5 sm:py-6"
        style={{ borderTop: `1px solid ${theme.navBorder}` }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-8 flex items-center justify-center">
          <div className="flex items-center gap-2">
            <Activity size={13} style={{ color: '#2E86AB' }} />
            <span className="text-xs" style={{ color: theme.textSecondary }}>
              GuidaPlate · CKD Dietary Guidance Platform
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
