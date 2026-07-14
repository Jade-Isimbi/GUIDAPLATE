import { useEffect, useMemo, useState } from 'react';
import { Activity, Sun, Moon, LayoutDashboard, Utensils, Shield, LogOut, TrendingUp, Settings, BarChart3, Menu } from 'lucide-react';
import { DashboardPage } from './components/DashboardPage';
import { FoodExplorer } from './components/FoodExplorer';
import { RiskAssessment } from './components/RiskAssessment';
import { WeeklyTrendPage } from './components/WeeklyTrendPage';
import { ModelComparisonPage } from './components/ModelComparisonPage';
import { MealPlannerButton } from './components/MealPlannerButton';
import { MealPlannerPage } from './components/MealPlannerPage';
import { LoginPage } from './components/LoginPage';
import { ResetPasswordPage } from './components/ResetPasswordPage';
import { SettingsPage } from './components/SettingsPage';
import { SignupPage } from './components/SignupPage';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from './components/ui/sheet';
import { clearAuthSession, getAuthToken, onUnauthorized } from '../utils/auth';
import { formatStageDisplay } from '../utils/riskDisplay';

type Page = 'dashboard' | 'explorer' | 'assessment' | 'weekly' | 'model-comparison' | 'meal-planner';
type AuthScreen = 'login' | 'signup' | 'app';

function getPageFromPathname(pathname: string): Page {
  if (pathname === '/food-explorer') return 'explorer';
  if (pathname === '/meal-check') return 'assessment';
  if (pathname === '/weekly-trend') return 'weekly';
  if (pathname === '/model-comparison') return 'model-comparison';
  if (pathname === '/meal-planner') return 'meal-planner';
  return 'dashboard';
}

function pathForPage(page: Page): string {
  if (page === 'explorer') return '/food-explorer';
  if (page === 'assessment') return '/meal-check';
  if (page === 'weekly') return '/weekly-trend';
  if (page === 'model-comparison') return '/model-comparison';
  if (page === 'meal-planner') return '/meal-planner';
  return '/';
}

/** Auth screens follow the Settings/reset-password path pattern (not Page map). */
function getAuthScreenFromPathname(pathname: string): 'login' | 'signup' | null {
  if (pathname === '/signup') return 'signup';
  if (pathname === '/login') return 'login';
  return null;
}

function replacePath(path: string) {
  if (window.location.pathname !== path) {
    window.history.replaceState({}, '', path);
  }
}

function pushPath(path: string) {
  if (window.location.pathname !== path) {
    window.history.pushState({}, '', path);
  }
}

const navItems: Array<{ id: Page; label: string; icon: typeof LayoutDashboard }> = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'explorer', label: 'Food Explorer', icon: Utensils },
  { id: 'assessment', label: 'Meal Check', icon: Shield },
  { id: 'weekly', label: 'Diet Pattern', icon: TrendingUp },
  { id: 'model-comparison', label: 'Model Comparison', icon: BarChart3 },
];

const THEME_STORAGE_KEY = 'guidaplate_theme';

function persistProfile(ckdStage: string | null | undefined, weightKg: number | null | undefined) {
  if (ckdStage) localStorage.setItem('ckd_stage', ckdStage);
  if (weightKg != null && weightKg > 0) {
    localStorage.setItem('weight_kg', weightKg.toString());
  }
}

function readThemePreference(): boolean {
  const stored = localStorage.getItem(THEME_STORAGE_KEY);
  if (stored === 'dark') return true;
  if (stored === 'light') return false;
  return false;
}

export default function App() {
  {/* MARKER-MAKE-KIT-INVOKED */}
  const [isDark, setIsDark] = useState(readThemePreference);
  const [page, setPage] = useState<Page>(() => getPageFromPathname(window.location.pathname));
  const [auth, setAuth] = useState<AuthScreen>(() => {
    if (getAuthToken()) return 'app';
    // Only /login and /signup pin auth screen; other paths (e.g. /meal-check) still show Login.
    return getAuthScreenFromPathname(window.location.pathname) ?? 'login';
  });
  const [userName, setUserName] = useState(() =>
    localStorage.getItem('guidaplate_user_name') || '',
  );
  const [userStage, setUserStage] = useState(() =>
    localStorage.getItem('ckd_stage') || 'G3a',
  );
  const [userWeight, setUserWeight] = useState<number>(() => {
    const stored = localStorage.getItem('weight_kg');
    return stored ? parseFloat(stored) || 65 : 65;
  });
  const [showSettings, setShowSettings] = useState(
    () => window.location.pathname === '/settings',
  );
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

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

  const goToLoginScreen = () => {
    setAuth('login');
    setShowSettings(false);
    setPage('dashboard');
    replacePath('/login');
  };

  const goToSignupScreen = () => {
    setAuth('signup');
    setShowSettings(false);
    pushPath('/signup');
  };

  /** After login/signup, leave /login|/signup|/settings for the intended app path. */
  const leaveAuthPathForApp = (currentPage: Page) => {
    const path = window.location.pathname;
    if (path === '/login' || path === '/signup' || path === '/settings') {
      replacePath(pathForPage(currentPage));
    }
  };

  useEffect(() => {
    getAuthToken();
    return onUnauthorized(() => {
      goToLoginScreen();
    });
  }, []);

  const handleLogin = (data: { name: string; ckdStage: string | null; weightKg: number | null }) => {
    setUserName(data.name);
    setUserStage(data.ckdStage || 'G3a');
    setUserWeight(data.weightKg || 65);
    localStorage.setItem('guidaplate_user_name', data.name);
    persistProfile(data.ckdStage, data.weightKg);
    setAuth('app');
    setShowSettings(false);
    leaveAuthPathForApp(page);
  };
  const handleSignup = (data: { name: string; ckdStage: string; weightKg: number; dob: string; sex: string; email: string; phone: string }) => {
    setUserName(data.name);
    setUserStage(data.ckdStage);
    setUserWeight(data.weightKg);
    localStorage.setItem('guidaplate_user_name', data.name);
    persistProfile(data.ckdStage, data.weightKg);
    setAuth('app');
    setShowSettings(false);
    leaveAuthPathForApp(page);
  };

  const handleProfileUpdated = (ckdStage: string, weightKg: number) => {
    setUserStage(ckdStage);
    setUserWeight(weightKg);
    persistProfile(ckdStage, weightKg);
  };

  const navigateToPage = (next: Page) => {
    setPage(next);
    setMobileNavOpen(false);
    const nextPath = pathForPage(next);
    if (window.location.pathname !== nextPath) {
      window.history.pushState({}, '', nextPath);
    }
  };

  const openSettings = () => {
    setShowSettings(true);
    pushPath('/settings');
  };

  const openMealPlanner = () => {
    setShowSettings(false);
    setPage('meal-planner');
    pushPath('/meal-planner');
  };

  useEffect(() => {
    const onPopState = () => {
      const path = window.location.pathname;
      // Logged-out: only /login and /signup control auth UI; other paths still show Login.
      if (!getAuthToken()) {
        const authFromPath = getAuthScreenFromPathname(path);
        setAuth(authFromPath ?? 'login');
        setShowSettings(false);
        if (path !== '/settings') {
          setPage(getPageFromPathname(path));
        }
        return;
      }
      setShowSettings(path === '/settings');
      if (path !== '/settings') {
        setPage(getPageFromPathname(path));
      }
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  const resetSuccessMessage = useMemo(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get('reset') === 'success') {
      replacePath('/login');
      return 'Password reset successfully. Please log in.';
    }
    return undefined;
  }, []);

  const isResetPasswordPage = window.location.pathname === '/reset-password';

  const initials = userName.split(' ').map((n) => n[0]).join('').slice(0, 2).toUpperCase();
  const activeNavLabel = navItems.find((item) => item.id === page)?.label ?? 'GuidaPlate';

  const renderNavButton = (id: Page, label: string, Icon: typeof LayoutDashboard, fullWidth = false) => (
    <button
      key={id}
      type="button"
      onClick={() => navigateToPage(id)}
      className={`flex items-center gap-3 rounded-xl text-sm transition-all duration-150 ${
        fullWidth ? 'w-full px-4 py-3' : 'px-2 sm:px-4 py-2'
      }`}
      style={{
        background: page === id
          ? isDark ? 'rgba(46,134,171,0.2)' : 'rgba(46,134,171,0.1)'
          : 'transparent',
        color: page === id ? '#2E86AB' : theme.textSecondary,
        fontWeight: page === id ? 600 : 400,
        boxShadow: page === id && !isDark && !fullWidth ? '0 1px 4px rgba(0,0,0,0.1)' : 'none',
      }}
    >
      <Icon size={fullWidth ? 18 : 15} />
      <span>{label}</span>
    </button>
  );

  if (isResetPasswordPage) {
    return <ResetPasswordPage isDark={isDark} theme={theme} />;
  }

  if (auth === 'login') {
    return (
      <LoginPage
        isDark={isDark}
        theme={theme}
        onLogin={handleLogin}
        onGoToSignup={goToSignupScreen}
        initialMessage={resetSuccessMessage}
      />
    );
  }
  if (auth === 'signup') {
    return (
      <SignupPage
        isDark={isDark}
        theme={theme}
        onSignup={handleSignup}
        onGoToLogin={() => {
          setAuth('login');
          pushPath('/login');
        }}
      />
    );
  }

  if (showSettings) {
    return (
      <div className="min-h-screen" style={{ background: theme.bg }}>
        <main className="max-w-[1600px] mx-auto px-4 sm:px-8 py-6 sm:py-8">
          <SettingsPage
            isDark={isDark}
            theme={theme}
            onProfileUpdated={handleProfileUpdated}
          />
        </main>
        <MealPlannerButton onClick={openMealPlanner} />
      </div>
    );
  }

  return (
    <div
      className={
        page === 'meal-planner'
          ? 'h-screen min-w-0 overflow-hidden flex flex-col'
          : 'min-h-screen min-w-0 overflow-x-hidden flex flex-col'
      }
      style={{ background: theme.bg }}
    >
      {/* Navigation */}
      <nav
        className="sticky top-0 z-50 backdrop-blur-xl"
        style={{
          background: theme.navBg,
          borderBottom: `1px solid ${theme.navBorder}`,
        }}
      >
        <div className="max-w-[1600px] mx-auto px-4 sm:px-8 min-w-0 flex items-center gap-2 sm:gap-3 h-14 sm:h-16">
          {/* Mobile menu */}
          <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
            <button
              type="button"
              className="lg:hidden w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
              style={{
                background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
                border: `1px solid ${theme.navBorder}`,
                color: theme.textSecondary,
              }}
              aria-label="Open navigation menu"
              onClick={() => setMobileNavOpen(true)}
            >
              <Menu size={18} />
            </button>
            <SheetContent side="left" className="w-[min(100vw-2rem,320px)] p-0 gap-0 flex flex-col h-full">
              <SheetHeader className="border-b px-5 py-4 text-left">
                <div className="flex items-center gap-3 pr-8">
                  <div
                    className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
                    style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)' }}
                  >
                    <Activity size={18} className="text-white" />
                  </div>
                  <div className="min-w-0">
                    <SheetTitle className="text-base">GuidaPlate</SheetTitle>
                    <p className="text-xs text-muted-foreground truncate">
                      {userName || 'GuidaPlate User'} · {formatStageDisplay(userStage)}
                    </p>
                  </div>
                </div>
              </SheetHeader>
              <nav className="flex flex-col gap-1 p-3">
                {navItems.map(({ id, label, icon: Icon }) => renderNavButton(id, label, Icon, true))}
              </nav>
              <div className="mt-auto border-t p-3 flex flex-col gap-2">
                <button
                  type="button"
                  onClick={() => {
                    setMobileNavOpen(false);
                    openSettings();
                  }}
                  className="flex items-center gap-3 w-full px-4 py-3 rounded-xl text-sm"
                  style={{ color: theme.textSecondary }}
                >
                  <Settings size={18} />
                  Settings
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setIsDark((prev) => {
                      const next = !prev;
                      localStorage.setItem(THEME_STORAGE_KEY, next ? 'dark' : 'light');
                      return next;
                    });
                  }}
                  className="flex items-center gap-3 w-full px-4 py-3 rounded-xl text-sm"
                  style={{ color: theme.textSecondary }}
                >
                  {isDark ? <Sun size={18} /> : <Moon size={18} />}
                  {isDark ? 'Light mode' : 'Dark mode'}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setMobileNavOpen(false);
                    clearAuthSession(false);
                    goToLoginScreen();
                  }}
                  className="flex items-center gap-3 w-full px-4 py-3 rounded-xl text-sm"
                  style={{ color: theme.textSecondary }}
                >
                  <LogOut size={18} />
                  Log out
                </button>
              </div>
            </SheetContent>
          </Sheet>

          {/* Brand — desktop only; mobile uses drawer + page title */}
          <div className="hidden lg:flex items-center gap-2 sm:gap-3 shrink-0 min-w-0">
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
                Kidney Health Platform
              </span>
            </div>
          </div>

          <p
            className="lg:hidden flex-1 min-w-0 truncate text-sm font-medium"
            style={{ color: theme.text }}
          >
            {activeNavLabel}
          </p>

          {/* Desktop page nav */}
          <div className="hidden lg:flex flex-1 min-w-0 justify-center">
          <div
            className="flex items-center gap-1 px-1.5 py-1.5 rounded-xl"
            style={{
              background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.05)',
              border: `1px solid ${theme.navBorder}`,
            }}
          >
            {navItems.map(({ id, label, icon: Icon }) => renderNavButton(id, label, Icon))}
          </div>
          </div>

          {/* Right actions */}
          <div className="hidden lg:flex items-center gap-2 sm:gap-4 shrink-0 ml-auto">
            <button
              onClick={() => {
                setIsDark((prev) => {
                  const next = !prev;
                  localStorage.setItem(THEME_STORAGE_KEY, next ? 'dark' : 'light');
                  return next;
                });
              }}
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
              className="hidden md:flex items-center gap-2 sm:gap-2.5 px-2 sm:px-3 py-1.5 rounded-xl"
              style={{
                background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)',
                border: `1px solid ${theme.navBorder}`,
              }}
            >
              <div
                className="w-7 h-7 rounded-lg flex items-center justify-center text-xs text-white shrink-0"
                style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #27AE60 100%)', fontWeight: 700 }}
              >
                {initials || 'GP'}
              </div>
              <div className="hidden sm:block leading-none">
                <p className="text-sm" style={{ color: theme.text, fontWeight: 500 }}>{userName || 'GuidaPlate User'}</p>
                <p className="text-xs" style={{ color: theme.textSecondary }}>{formatStageDisplay(userStage)}</p>
              </div>
            </div>
            <button
              onClick={() => {
                clearAuthSession(false);
                goToLoginScreen();
              }}
              className="w-8 h-8 rounded-xl flex items-center justify-center transition-all duration-200 hover:opacity-70"
              title="Log out"
              style={{
                background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)',
                border: `1px solid ${theme.navBorder}`,
                color: theme.textSecondary,
              }}
            >
              <LogOut size={14} />
            </button>
          </div>

          {/* Mobile quick actions */}
          <div className="flex lg:hidden items-center gap-2 shrink-0">
            <button
              onClick={() => {
                setIsDark((prev) => {
                  const next = !prev;
                  localStorage.setItem(THEME_STORAGE_KEY, next ? 'dark' : 'light');
                  return next;
                });
              }}
              className="w-8 h-8 rounded-xl flex items-center justify-center transition-all duration-200 hover:opacity-70"
              style={{
                background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
                border: `1px solid ${theme.navBorder}`,
                color: theme.textSecondary,
              }}
              aria-label="Toggle theme"
            >
              {isDark ? <Sun size={15} /> : <Moon size={15} />}
            </button>
          </div>
        </div>
      </nav>

      {/* Page content */}
      <main
        className={
          page === 'meal-planner'
            ? 'flex-1 overflow-hidden min-h-0 w-full'
            : 'flex-1 mx-auto px-4 sm:px-8 py-5 sm:py-6 lg:py-5 max-w-[1600px] w-full min-w-0 overflow-x-hidden'
        }
      >
        {page === 'dashboard' && (
          <DashboardPage isDark={isDark} theme={theme} onNavigate={(p) => navigateToPage(p as Page)} />
        )}
        {page === 'explorer' && (
          <FoodExplorer isDark={isDark} theme={theme} patientStage={userStage} />
        )}
        {page === 'assessment' && (
          <RiskAssessment
            key={`${userStage}-${userWeight}`}
            isDark={isDark}
            theme={theme}
            initialBodyWeight={userWeight}
            initialStage={userStage}
          />
        )}
        {page === 'weekly' && (
          <WeeklyTrendPage isDark={isDark} theme={theme} onNavigate={(p) => navigateToPage(p as Page)} />
        )}
        {page === 'model-comparison' && (
          <ModelComparisonPage isDark={isDark} theme={theme} />
        )}
        {page === 'meal-planner' && (
          <MealPlannerPage isDark={isDark} theme={theme} />
        )}
      </main>

      {/* Footer */}
      {page !== 'meal-planner' && (
      <footer
        className="shrink-0 mt-auto py-5 sm:py-6"
        style={{ borderTop: `1px solid ${theme.navBorder}` }}
      >
        <div className="max-w-[1600px] mx-auto px-4 sm:px-8 flex items-center justify-center">
          <div className="flex items-center gap-2">
            <Activity size={13} style={{ color: '#2E86AB' }} />
            <span className="text-xs" style={{ color: theme.textSecondary }}>
              GuidaPlate · Kidney Health Diet Guide
            </span>
          </div>
        </div>
      </footer>
      )}

      <button
        type="button"
        onClick={openSettings}
        title="Settings"
        className={`fixed right-6 z-50 w-12 h-12 rounded-full shadow-lg flex items-center justify-center transition-all duration-200 hover:scale-105 ${
          // Meal Planner composer Send sits bottom-right; lift Settings so they don't collide.
          page === 'meal-planner' ? 'bottom-28' : 'bottom-6'
        }`}
        style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)' }}
        aria-label="Settings"
      >
        <Settings size={20} className="text-white" />
      </button>

      {page !== 'meal-planner' && (
        <MealPlannerButton
          onClick={openMealPlanner}
          fixedLabel={page === 'weekly' ? 'Build a meal plan' : undefined}
        />
      )}
    </div>
  );
}
