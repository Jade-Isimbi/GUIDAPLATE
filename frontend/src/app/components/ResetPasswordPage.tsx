import { useMemo, useState } from 'react';
import { Activity, Eye, EyeOff } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

interface ResetPasswordPageProps {
  isDark: boolean;
  theme: Record<string, string>;
}

export function ResetPasswordPage({ isDark, theme }: ResetPasswordPageProps) {
  const token = useMemo(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('token');
  }, []);

  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

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
    setError('');

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (!token) {
      setError('Invalid reset link');
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/auth/reset-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, new_password: password }),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        setError(typeof data.detail === 'string' ? data.detail : 'Password reset failed');
        return;
      }

      window.location.href = '/?reset=success';
    } catch {
      setError('Password reset failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (!token) {
    return (
      <div
        className="min-h-screen flex flex-col items-center justify-center px-4 py-12"
        style={{ background: theme.bg }}
      >
        <div
          className="w-full max-w-md rounded-3xl p-8 sm:p-10 text-center"
          style={{
            background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.92)',
            border: `1px solid ${isDark ? 'rgba(255,255,255,0.09)' : 'rgba(0,0,0,0.09)'}`,
          }}
        >
          <h1 style={{ color: theme.text, fontSize: '1.4rem', fontWeight: 700, marginBottom: 8 }}>
            Invalid reset link
          </h1>
          <p style={{ color: isDark ? '#8a9ab0' : '#5a6a80', fontSize: '0.9rem', marginBottom: 24 }}>
            This password reset link is missing or invalid. Please request a new one.
          </p>
          <a
            href="/"
            className="inline-block py-2.5 px-6 rounded-xl text-white font-semibold"
            style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)' }}
          >
            Back to login
          </a>
        </div>
      </div>
    );
  }

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
          Reset your password
        </h1>
        <p style={{ color: isDark ? '#8a9ab0' : '#5a6a80', fontSize: '0.9rem', marginBottom: 32 }}>
          Enter a new password for your account
        </p>

        {error && (
          <p
            style={{
              color: '#E74C3C',
              fontSize: '0.8rem',
              background: 'rgba(231,76,60,0.08)',
              border: '1px solid rgba(231,76,60,0.25)',
              borderRadius: 10,
              padding: '0.6rem 0.9rem',
              marginBottom: 16,
            }}
          >
            {error}
          </p>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label style={labelStyle}>New password</label>
            <div className="relative">
              <input
                type={showPassword ? 'text' : 'password'}
                style={{ ...inputStyle, paddingRight: '2.8rem' }}
                placeholder="Enter new password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 transition-opacity hover:opacity-70"
                style={{ color: isDark ? '#8a9ab0' : '#6a7a90', background: 'none', border: 'none', cursor: 'pointer' }}
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          <div>
            <label style={labelStyle}>Confirm password</label>
            <input
              type={showPassword ? 'text' : 'password'}
              style={inputStyle}
              placeholder="Confirm new password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 rounded-xl text-white font-semibold transition-all duration-200 hover:opacity-90 active:scale-[0.98]"
            style={{
              background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)',
              fontSize: '0.95rem',
              marginTop: 8,
              opacity: loading ? 0.7 : 1,
            }}
          >
            {loading ? 'Resetting...' : 'Reset Password'}
          </button>
        </form>

        <p className="text-center mt-6" style={{ color: isDark ? '#8a9ab0' : '#5a6a80', fontSize: '0.875rem' }}>
          <a href="/" style={{ color: '#2E86AB', fontWeight: 600 }}>
            Back to login
          </a>
        </p>
      </div>
    </div>
  );
}
