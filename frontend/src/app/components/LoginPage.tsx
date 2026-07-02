import { useState } from 'react';
import { Activity, Eye, EyeOff } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface LoginPageProps {
  isDark: boolean;
  theme: Record<string, string>;
  onLogin: (data: { name: string; ckdStage: string | null; weightKg: number | null }) => void;
  onGoToSignup: () => void;
  initialMessage?: string;
}

export function LoginPage({ isDark, theme, onLogin, onGoToSignup, initialMessage }: LoginPageProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPass, setShowPass] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState(initialMessage ?? '');
  const [isLoading, setIsLoading] = useState(false);
  const [showForgotModal, setShowForgotModal] = useState(false);
  const [forgotEmail, setForgotEmail] = useState('');
  const [forgotSent, setForgotSent] = useState(false);
  const [forgotLoading, setForgotLoading] = useState(false);

  const inputStyle = {
    background: isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.04)',
    border: `1px solid ${isDark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.14)'}`,
    borderRadius: 12,
    color: isDark ? '#f0f4f8' : '#0e1625',
    fontSize: '0.9rem',
    padding: '0.7rem 1rem',
    width: '100%',
    outline: 'none',
    transition: 'border-color 0.15s',
  };

  const labelStyle = {
    color: isDark ? '#c8d6e6' : '#2a3a4a',
    fontSize: '0.82rem',
    fontWeight: 600,
    marginBottom: 6,
    display: 'block' as const,
  };

  const closeForgotModal = () => {
    setShowForgotModal(false);
    setForgotEmail('');
    setForgotSent(false);
    setForgotLoading(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError('Please enter your email and password.');
      return;
    }

    setError(null);
    setSuccessMessage('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        throw new Error('Invalid email or password');
      }

      const data = await response.json();

      localStorage.setItem('guidaplate_token', data.access_token);
      localStorage.setItem('token', data.access_token);
      localStorage.setItem('guidaplate_user_id', data.user_id);

      onLogin({
        name: data.name,
        ckdStage: data.ckd_stage,
        weightKg: data.weight_kg,
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Login failed. Please check your credentials.';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleForgotSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setForgotLoading(true);
    try {
      await fetch(`${API_BASE}/api/auth/forgot-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: forgotEmail }),
      });
      setForgotSent(true);
    } catch {
      setForgotSent(true);
    } finally {
      setForgotLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center px-4 py-12"
      style={{ background: theme.bg }}
    >
      <div className="flex items-center gap-3 mb-10">
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center"
          style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)' }}
        >
          <Activity size={22} className="text-white" />
        </div>
        <span style={{ color: '#2E86AB', fontWeight: 700, fontSize: '1.2rem', letterSpacing: '-0.02em' }}>
          GuidaPlate
        </span>
      </div>

      <div
        className="w-full max-w-md rounded-3xl p-8 sm:p-10"
        style={{
          background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.92)',
          border: `1px solid ${isDark ? 'rgba(255,255,255,0.09)' : 'rgba(0,0,0,0.09)'}`,
          backdropFilter: 'blur(16px)',
        }}
      >
        <h1 style={{ color: theme.text, fontSize: '1.6rem', fontWeight: 700, letterSpacing: '-0.025em', marginBottom: 6 }}>
          Welcome back
        </h1>
        <p style={{ color: isDark ? '#8a9ab0' : '#5a6a80', fontSize: '0.9rem', marginBottom: 32 }}>
          Log in to continue your dietary tracking
        </p>

        {successMessage && (
          <p
            style={{
              color: '#27AE60',
              fontSize: '0.8rem',
              background: 'rgba(39,174,96,0.08)',
              border: '1px solid rgba(39,174,96,0.25)',
              borderRadius: 10,
              padding: '0.6rem 0.9rem',
              marginBottom: 16,
            }}
          >
            {successMessage}
          </p>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label style={labelStyle}>Email address</label>
            <input
              type="email"
              style={inputStyle}
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label style={labelStyle}>Password</label>
              <button
                type="button"
                onClick={() => setShowForgotModal(true)}
                className="text-sm text-teal-600 hover:underline"
                style={{ color: '#2E86AB', fontSize: '0.78rem', fontWeight: 500, background: 'none', border: 'none', cursor: 'pointer' }}
              >
                Forgot password?
              </button>
            </div>
            <div className="relative">
              <input
                type={showPass ? 'text' : 'password'}
                style={{ ...inputStyle, paddingRight: '2.8rem' }}
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
              />
              <button
                type="button"
                onClick={() => setShowPass(!showPass)}
                className="absolute right-3 top-1/2 -translate-y-1/2 transition-opacity hover:opacity-70"
                style={{ color: isDark ? '#8a9ab0' : '#6a7a90', background: 'none', border: 'none', cursor: 'pointer' }}
              >
                {showPass ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          {error && (
            <p style={{ color: '#E74C3C', fontSize: '0.8rem', background: 'rgba(231,76,60,0.08)', border: '1px solid rgba(231,76,60,0.25)', borderRadius: 10, padding: '0.6rem 0.9rem' }}>
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 rounded-xl text-white font-semibold transition-all duration-200 hover:opacity-90 active:scale-[0.98]"
            style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)', fontSize: '0.95rem', marginTop: 8, opacity: isLoading ? 0.7 : 1 }}
          >
            {isLoading ? 'Logging in…' : 'Log In'}
          </button>
        </form>

        <p className="text-center mt-6" style={{ color: isDark ? '#8a9ab0' : '#5a6a80', fontSize: '0.875rem' }}>
          Don&apos;t have an account?{' '}
          <button
            onClick={onGoToSignup}
            style={{ color: '#2E86AB', fontWeight: 600, background: 'none', border: 'none', cursor: 'pointer' }}
          >
            Create one
          </button>
        </p>
      </div>

      <p className="mt-8 text-xs text-center" style={{ color: isDark ? '#4a5a6a' : '#9aaac0', maxWidth: 320 }}>
        GuidaPlate is a decision-support tool. Always consult your healthcare provider.
      </p>

      {showForgotModal && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center px-4">
          <div
            className="rounded-xl shadow-xl max-w-sm w-full p-6 relative"
            style={{
              background: isDark ? 'rgba(14,22,37,0.98)' : 'white',
              border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.08)'}`,
            }}
          >
            <button
              type="button"
              onClick={closeForgotModal}
              className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 text-xl leading-none"
              aria-label="Close"
            >
              ×
            </button>

            <h3 style={{ color: theme.text, fontSize: '1.1rem', fontWeight: 600, marginBottom: 16 }}>
              Reset your password
            </h3>

            {!forgotSent ? (
              <form onSubmit={handleForgotSubmit} className="space-y-4">
                <div>
                  <label style={labelStyle}>Email address</label>
                  <input
                    type="email"
                    style={inputStyle}
                    placeholder="you@example.com"
                    value={forgotEmail}
                    onChange={(e) => setForgotEmail(e.target.value)}
                    required
                  />
                </div>
                <button
                  type="submit"
                  disabled={forgotLoading}
                  className="w-full py-2.5 rounded-xl text-white font-semibold transition-all duration-200 hover:opacity-90"
                  style={{
                    background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)',
                    opacity: forgotLoading ? 0.7 : 1,
                  }}
                >
                  {forgotLoading ? 'Sending...' : 'Send reset link'}
                </button>
              </form>
            ) : (
              <div className="space-y-4">
                <p style={{ color: isDark ? '#8a9ab0' : '#5a6a80', fontSize: '0.875rem' }}>
                  Check your email — a reset link has been sent to {forgotEmail}
                </p>
                <button
                  type="button"
                  onClick={closeForgotModal}
                  className="w-full py-2.5 rounded-xl text-white font-semibold transition-all duration-200 hover:opacity-90"
                  style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)' }}
                >
                  Back to login
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
