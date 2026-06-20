import { useState } from 'react';
import { Activity, Eye, EyeOff } from 'lucide-react';

interface LoginPageProps {
  isDark: boolean;
  theme: Record<string, string>;
  onLogin: (data: { name: string; ckdStage: string | null; weightKg: number | null }) => void;
  onGoToSignup: () => void;
}

export function LoginPage({ isDark, theme, onLogin, onGoToSignup }: LoginPageProps) {
  const [email,       setEmail]       = useState('');
  const [password,    setPassword]    = useState('');
  const [showPass,    setShowPass]    = useState(false);
  const [error,       setError]       = useState<string | null>(null);
  const [isLoading,   setIsLoading]   = useState(false);

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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError('Please enter your email and password.');
      return;
    }

    setError(null);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        throw new Error('Invalid email or password');
      }

      const data = await response.json();

      localStorage.setItem('guidaplate_token', data.access_token);
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

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center px-4 py-12"
      style={{ background: theme.bg }}
    >
      {/* Brand */}
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

      {/* Card */}
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

        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Email */}
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

          {/* Password */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label style={labelStyle}>Password</label>
              <button
                type="button"
                onClick={() => {}}
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

          {/* Error */}
          {error && (
            <p style={{ color: '#E74C3C', fontSize: '0.8rem', background: 'rgba(231,76,60,0.08)', border: '1px solid rgba(231,76,60,0.25)', borderRadius: 10, padding: '0.6rem 0.9rem' }}>
              {error}
            </p>
          )}

          {/* Submit */}
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
          Don't have an account?{' '}
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
    </div>
  );
}
