import { useState } from 'react';
import { Activity, Eye, EyeOff, ChevronDown } from 'lucide-react';
import { stageOptionLabel } from '../../utils/riskDisplay';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface SignupPageProps {
  isDark: boolean;
  theme: Record<string, string>;
  onSignup: (data: {
    name: string;
    ckdStage: string;
    weightKg: number;
    dob: string;
    sex: string;
    language: string;
    email: string;
    phone: string;
  }) => void;
  onGoToLogin: () => void;
}

const CKD_STAGES = [
  { value: 'G2',  label: 'Stage 2',  gfr: '60–89',  desc: 'Mildly decreased' },
  { value: 'G3a', label: 'Stage 3a', gfr: '45–59',  desc: 'Mild to moderate decrease' },
  { value: 'G3b', label: 'Stage 3b', gfr: '30–44',  desc: 'Moderate to severe decrease' },
  { value: 'G4',  label: 'Stage 4',  gfr: '15–29',  desc: 'Severe decrease' },
];

const COUNTRY_CODES = [
  { code: '+250', flag: '🇷🇼', name: 'Rwanda' },
  { code: '+254', flag: '🇰🇪', name: 'Kenya' },
  { code: '+255', flag: '🇹🇿', name: 'Tanzania' },
  { code: '+256', flag: '🇺🇬', name: 'Uganda' },
  { code: '+243', flag: '🇨🇩', name: 'DR Congo' },
  { code: '+1',   flag: '🇺🇸', name: 'USA' },
  { code: '+44',  flag: '🇬🇧', name: 'UK' },
  { code: '+33',  flag: '🇫🇷', name: 'France' },
];

const LANGUAGES = ['English', 'French', 'Kinyarwanda'];

export function SignupPage({ isDark, theme, onSignup, onGoToLogin }: SignupPageProps) {
  const [name,        setName]        = useState('');
  const [email,       setEmail]       = useState('');
  const [countryCode, setCountryCode] = useState('+250');
  const [phone,       setPhone]       = useState('');
  const [password,    setPassword]    = useState('');
  const [confirmPass, setConfirmPass] = useState('');
  const [showPass,    setShowPass]    = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [ckdStage,    setCkdStage]    = useState('');
  const [weight,      setWeight]      = useState('');
  const [dob,         setDob]         = useState('');
  const [sex,         setSex]         = useState('');
  const [language,    setLanguage]    = useState('English');
  const [agreed,      setAgreed]      = useState(false);
  const [errors,      setErrors]      = useState<Record<string, string>>({});
  const [showCC,      setShowCC]      = useState(false);
  const [isLoading,   setIsLoading]   = useState(false);
  const [error,       setError]       = useState<string | null>(null);

  const selectedStage = CKD_STAGES.find((s) => s.value === ckdStage);

  const inputBase: React.CSSProperties = {
    background: isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.04)',
    border: `1px solid ${isDark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.14)'}`,
    borderRadius: 12,
    color: isDark ? '#f0f4f8' : '#0e1625',
    fontSize: '0.9rem',
    padding: '0.7rem 1rem',
    width: '100%',
    outline: 'none',
  };

  const labelStyle: React.CSSProperties = {
    color: isDark ? '#c8d6e6' : '#2a3a4a',
    fontSize: '0.82rem',
    fontWeight: 600,
    marginBottom: 6,
    display: 'block',
  };

  const helperStyle: React.CSSProperties = {
    color: isDark ? '#8a9ab0' : '#6a7a90',
    fontSize: '0.75rem',
    marginTop: 5,
  };

  const sectionHeading = (label: string) => (
    <div className="flex items-center gap-3 mb-5">
      <span style={{ color: '#2E86AB', fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase' as const }}>
        {label}
      </span>
      <div className="flex-1 h-px" style={{ background: isDark ? 'rgba(46,134,171,0.25)' : 'rgba(46,134,171,0.2)' }} />
    </div>
  );

  const validate = () => {
    const e: Record<string, string> = {};
    if (!name.trim())         e.name        = 'Full name is required';
    if (!email.trim())        e.email       = 'Email is required';
    if (!password)            e.password    = 'Password is required';
    if (password.length < 8)  e.password    = 'Password must be at least 8 characters';
    if (password !== confirmPass) e.confirmPass = 'Passwords do not match';
    if (!ckdStage)            e.ckdStage    = 'Please select your kidney disease stage';
    if (!weight)              e.weight      = 'Body weight is required';
    if (!dob)                 e.dob         = 'Date of birth is required';
    if (!sex)                 e.sex         = 'Please select your sex';
    if (!agreed)              e.agreed      = 'You must accept the terms to continue';
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleSubmit = async (ev: React.FormEvent) => {
    ev.preventDefault();
    if (!validate()) return;

    setError(null);
    setIsLoading(true);

    const fullPhone = `${countryCode} ${phone.trim()}`;

    try {
      const response = await fetch(`${API_BASE}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          email,
          password,
          phone: fullPhone,
          ckd_stage: ckdStage,
          weight_kg: parseFloat(weight),
          dob,
          sex,
          language,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Registration failed');
      }

      const data = await response.json();

      localStorage.setItem('guidaplate_token', data.access_token);
      localStorage.setItem('token', data.access_token);
      localStorage.setItem('guidaplate_user_id', data.user_id);

      onSignup({
        name: data.name,
        ckdStage: data.ckd_stage,
        weightKg: data.weight_kg,
        dob,
        sex,
        language,
        email,
        phone: fullPhone,
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Something went wrong. Please try again.';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const errText = (key: string) =>
    errors[key] ? <p style={{ color: '#E74C3C', fontSize: '0.75rem', marginTop: 5 }}>{errors[key]}</p> : null;

  const selectedCC = COUNTRY_CODES.find((c) => c.code === countryCode) ?? COUNTRY_CODES[0];

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-start px-4 py-12"
      style={{ background: theme.bg }}
    >
      {/* Brand */}
      <div className="flex items-center gap-3 mb-8">
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
        className="w-full max-w-lg rounded-3xl p-7 sm:p-10"
        style={{
          background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.92)',
          border: `1px solid ${isDark ? 'rgba(255,255,255,0.09)' : 'rgba(0,0,0,0.09)'}`,
          backdropFilter: 'blur(16px)',
        }}
      >
        <h1 style={{ color: theme.text, fontSize: '1.5rem', fontWeight: 700, letterSpacing: '-0.025em', marginBottom: 6 }}>
          Create your account
        </h1>
        <p style={{ color: isDark ? '#8a9ab0' : '#5a6a80', fontSize: '0.875rem', marginBottom: 32 }}>
          Set up your profile for personalised kidney health dietary guidance
        </p>

        <form onSubmit={handleSubmit} className="space-y-5">

          {/* ── Account Details ──────────────────────────── */}
          {sectionHeading('Account Details')}

          {/* Full Name */}
          <div>
            <label style={labelStyle}>Full Name</label>
            <input style={inputBase} placeholder="Jean-Pierre Nkurunziza" value={name} onChange={(e) => setName(e.target.value)} />
            {errText('name')}
          </div>

          {/* Email */}
          <div>
            <label style={labelStyle}>Email Address</label>
            <input type="email" style={inputBase} placeholder="you@example.com" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" />
            {errText('email')}
          </div>

          {/* Phone */}
          <div>
            <label style={labelStyle}>Phone Number</label>
            <div className="flex gap-2">
              {/* Country code */}
              <div className="relative shrink-0">
                <button
                  type="button"
                  onClick={() => setShowCC(!showCC)}
                  className="flex items-center gap-1.5 px-3 rounded-xl h-full"
                  style={{
                    background: isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.04)',
                    border: `1px solid ${isDark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.14)'}`,
                    color: isDark ? '#f0f4f8' : '#0e1625',
                    fontSize: '0.875rem',
                    minHeight: 44,
                    whiteSpace: 'nowrap',
                  }}
                >
                  <span>{selectedCC.flag}</span>
                  <span style={{ fontWeight: 600 }}>{selectedCC.code}</span>
                  <ChevronDown size={13} style={{ color: isDark ? '#8a9ab0' : '#6a7a90' }} />
                </button>
                {showCC && (
                  <>
                    <div className="fixed inset-0 z-10" onClick={() => setShowCC(false)} />
                    <div
                      className="absolute top-full mt-1.5 left-0 z-20 rounded-xl overflow-hidden shadow-xl"
                      style={{ background: isDark ? '#111827' : '#fff', border: `1px solid ${isDark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.1)'}`, minWidth: 180 }}
                    >
                      {COUNTRY_CODES.map((c) => (
                        <button
                          key={c.code}
                          type="button"
                          className="w-full text-left px-4 py-2.5 flex items-center gap-2 transition-colors"
                          style={{
                            color: countryCode === c.code ? '#2E86AB' : (isDark ? '#f0f4f8' : '#0e1625'),
                            background: countryCode === c.code ? 'rgba(46,134,171,0.1)' : 'transparent',
                            fontSize: '0.85rem',
                          }}
                          onMouseDown={() => { setCountryCode(c.code); setShowCC(false); }}
                        >
                          <span>{c.flag}</span>
                          <span style={{ fontWeight: 500 }}>{c.name}</span>
                          <span style={{ color: isDark ? '#8a9ab0' : '#6a7a90', marginLeft: 'auto' }}>{c.code}</span>
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </div>
              <input
                type="tel"
                style={{ ...inputBase, flex: 1 }}
                placeholder="788 000 000"
                value={phone}
                onChange={(e) => setPhone(e.target.value)}
              />
            </div>
          </div>

          {/* Password */}
          <div>
            <label style={labelStyle}>Password</label>
            <div className="relative">
              <input
                type={showPass ? 'text' : 'password'}
                style={{ ...inputBase, paddingRight: '2.8rem' }}
                placeholder="At least 8 characters"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="new-password"
              />
              <button type="button" onClick={() => setShowPass(!showPass)} className="absolute right-3 top-1/2 -translate-y-1/2 transition-opacity hover:opacity-70" style={{ color: isDark ? '#8a9ab0' : '#6a7a90', background: 'none', border: 'none', cursor: 'pointer' }}>
                {showPass ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            {errText('password')}
          </div>

          {/* Confirm Password */}
          <div>
            <label style={labelStyle}>Confirm Password</label>
            <div className="relative">
              <input
                type={showConfirm ? 'text' : 'password'}
                style={{ ...inputBase, paddingRight: '2.8rem' }}
                placeholder="Re-enter your password"
                value={confirmPass}
                onChange={(e) => setConfirmPass(e.target.value)}
                autoComplete="new-password"
              />
              <button type="button" onClick={() => setShowConfirm(!showConfirm)} className="absolute right-3 top-1/2 -translate-y-1/2 transition-opacity hover:opacity-70" style={{ color: isDark ? '#8a9ab0' : '#6a7a90', background: 'none', border: 'none', cursor: 'pointer' }}>
                {showConfirm ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            {errText('confirmPass')}
          </div>

          {/* ── Health Profile ───────────────────────────── */}
          <div className="pt-4">
            {sectionHeading('Health Profile')}
          </div>

          {/* Kidney disease stage */}
          <div>
            <label style={labelStyle}>Kidney Disease Stage</label>
            <div className="grid grid-cols-4 gap-2">
              {CKD_STAGES.map((s) => (
                <button
                  key={s.value}
                  type="button"
                  onClick={() => setCkdStage(s.value)}
                  className="py-2.5 rounded-xl transition-all duration-150"
                  style={{
                    background: ckdStage === s.value ? 'rgba(46,134,171,0.15)' : isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)',
                    border: `1px solid ${ckdStage === s.value ? 'rgba(46,134,171,0.5)' : (isDark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.14)')}`,
                    color: ckdStage === s.value ? '#2E86AB' : (isDark ? '#f0f4f8' : '#0e1625'),
                    fontWeight: ckdStage === s.value ? 700 : 500,
                    fontSize: '0.875rem',
                  }}
                >
                  {stageOptionLabel(s.value)}
                </button>
              ))}
            </div>
            {selectedStage && (
              <p style={{ ...helperStyle, color: '#2E86AB', fontWeight: 500, marginTop: 8 }}>
                {selectedStage.label} — Kidney Function Score {selectedStage.gfr} mL/min/1.73m² · {selectedStage.desc}
              </p>
            )}
            {errText('ckdStage')}
          </div>

          {/* Body Weight */}
          <div>
            <label style={labelStyle}>Body Weight</label>
            <div className="flex gap-2 items-center">
              <input
                type="number"
                style={{ ...inputBase, flex: 1 }}
                placeholder="e.g. 68"
                value={weight}
                onChange={(e) => setWeight(e.target.value)}
                min="20"
                max="250"
              />
              <span style={{ color: isDark ? '#8a9ab0' : '#6a7a90', fontSize: '0.875rem', whiteSpace: 'nowrap', flexShrink: 0 }}>kg</span>
            </div>
            <p style={helperStyle}>Used to calculate your personalised protein limit</p>
            {errText('weight')}
          </div>

          {/* Date of Birth */}
          <div>
            <label style={labelStyle}>Date of Birth</label>
            <input
              type="date"
              style={inputBase}
              value={dob}
              onChange={(e) => setDob(e.target.value)}
              max={new Date().toISOString().split('T')[0]}
            />
            {errText('dob')}
          </div>

          {/* Sex */}
          <div>
            <label style={labelStyle}>Sex</label>
            <div className="grid grid-cols-2 gap-2">
              {['Male', 'Female'].map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setSex(s)}
                  className="py-2.5 rounded-xl transition-all duration-150"
                  style={{
                    background: sex === s ? 'rgba(46,134,171,0.15)' : isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)',
                    border: `1px solid ${sex === s ? 'rgba(46,134,171,0.5)' : (isDark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.14)')}`,
                    color: sex === s ? '#2E86AB' : (isDark ? '#f0f4f8' : '#0e1625'),
                    fontWeight: sex === s ? 700 : 500,
                    fontSize: '0.875rem',
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
            {errText('sex')}
          </div>

          {/* Preferred Language */}
          <div>
            <label style={labelStyle}>Preferred Language</label>
            <div className="grid grid-cols-3 gap-2">
              {LANGUAGES.map((l) => (
                <button
                  key={l}
                  type="button"
                  onClick={() => setLanguage(l)}
                  className="py-2.5 rounded-xl transition-all duration-150"
                  style={{
                    background: language === l ? 'rgba(46,134,171,0.15)' : isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)',
                    border: `1px solid ${language === l ? 'rgba(46,134,171,0.5)' : (isDark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.14)')}`,
                    color: language === l ? '#2E86AB' : (isDark ? '#f0f4f8' : '#0e1625'),
                    fontWeight: language === l ? 700 : 500,
                    fontSize: '0.82rem',
                  }}
                >
                  {l}
                </button>
              ))}
            </div>
          </div>

          {/* Consent checkbox */}
          <div className="pt-2">
            <label
              className="flex items-start gap-3 cursor-pointer"
              style={{ color: isDark ? '#c8d6e6' : '#2a3a4a' }}
            >
              <div className="relative mt-0.5 shrink-0">
                <input
                  type="checkbox"
                  className="sr-only"
                  checked={agreed}
                  onChange={(e) => setAgreed(e.target.checked)}
                />
                <div
                  className="w-5 h-5 rounded-md flex items-center justify-center transition-all duration-150"
                  style={{
                    background: agreed ? '#2E86AB' : (isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.05)'),
                    border: `2px solid ${agreed ? '#2E86AB' : (isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)')}`,
                  }}
                >
                  {agreed && (
                    <svg width="11" height="8" viewBox="0 0 11 8" fill="none">
                      <path d="M1 4L4 7L10 1" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  )}
                </div>
              </div>
              <span style={{ fontSize: '0.82rem', lineHeight: 1.55 }}>
                I understand GuidaPlate is a decision-support tool and not a substitute for professional medical advice
              </span>
            </label>
            {errText('agreed')}
          </div>

          {/* Submit */}
          {error && (
            <p style={{ color: '#E74C3C', fontSize: '0.8rem', background: 'rgba(231,76,60,0.08)', border: '1px solid rgba(231,76,60,0.25)', borderRadius: 10, padding: '0.6rem 0.9rem' }}>
              {error}
            </p>
          )}
          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 rounded-xl text-white font-semibold transition-all duration-200 hover:opacity-90 active:scale-[0.98] mt-2"
            style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)', fontSize: '0.95rem', opacity: isLoading ? 0.7 : 1 }}
          >
            {isLoading ? 'Creating account…' : 'Create Account'}
          </button>
        </form>

        <p className="text-center mt-6" style={{ color: isDark ? '#8a9ab0' : '#5a6a80', fontSize: '0.875rem' }}>
          Already have an account?{' '}
          <button onClick={onGoToLogin} style={{ color: '#2E86AB', fontWeight: 600, background: 'none', border: 'none', cursor: 'pointer' }}>
            Log in
          </button>
        </p>
      </div>

      <p className="mt-8 text-xs text-center" style={{ color: isDark ? '#4a5a6a' : '#9aaac0', maxWidth: 320 }}>
        GuidaPlate is a decision-support tool. Always consult your healthcare provider.
      </p>
    </div>
  );
}
