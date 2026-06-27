import { useEffect, useState } from 'react';

const API_BASE = 'http://localhost:8000/api';
const CKD_STAGES = ['G2', 'G3a', 'G3b', 'G4'] as const;

type CKDStage = (typeof CKD_STAGES)[number];

interface SettingsPageProps {
  isDark: boolean;
  theme: Record<string, string>;
  onProfileUpdated?: (ckdStage: string, weightKg: number) => void;
}

export function SettingsPage({ isDark, theme, onProfileUpdated }: SettingsPageProps) {
  const [ckdStage, setCkdStage] = useState<CKDStage>('G3a');
  const [weightKg, setWeightKg] = useState(65);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    const token = localStorage.getItem('guidaplate_token');
    if (!token) {
      setLoading(false);
      return;
    }

    const loadProfile = async () => {
      try {
        const response = await fetch(`${API_BASE}/patient/profile`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!response.ok) throw new Error('Failed to load profile');
        const data = await response.json();
        if (data.ckd_stage && CKD_STAGES.includes(data.ckd_stage)) {
          setCkdStage(data.ckd_stage);
        }
        if (typeof data.weight_kg === 'number' && data.weight_kg > 0) {
          setWeightKg(data.weight_kg);
        }
      } catch {
        const storedStage = localStorage.getItem('ckd_stage');
        const storedWeight = localStorage.getItem('weight_kg');
        if (storedStage && CKD_STAGES.includes(storedStage as CKDStage)) {
          setCkdStage(storedStage as CKDStage);
        }
        if (storedWeight) {
          setWeightKg(parseFloat(storedWeight) || 65);
        }
      } finally {
        setLoading(false);
      }
    };

    void loadProfile();
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setSuccessMessage('');
    setErrorMessage('');

    const token = localStorage.getItem('guidaplate_token');
    if (!token) {
      setErrorMessage('Failed to update profile. Please try again.');
      setSaving(false);
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/patient/profile`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ ckd_stage: ckdStage, weight_kg: weightKg }),
      });

      if (!response.ok) throw new Error('Update failed');

      const data = await response.json();
      const stage = data.ckd_stage ?? ckdStage;
      const weight = data.weight_kg ?? weightKg;

      localStorage.setItem('ckd_stage', stage);
      localStorage.setItem('weight_kg', weight.toString());

      setSuccessMessage('✓ Profile updated successfully');
      onProfileUpdated?.(stage, weight);
    } catch {
      setErrorMessage('Failed to update profile. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="max-w-lg mx-auto px-4 py-8 sm:py-10">
      <button
        type="button"
        onClick={() => window.history.back()}
        className="mb-6 text-sm font-medium transition-opacity hover:opacity-70"
        style={{ color: '#2E86AB', background: 'none', border: 'none', cursor: 'pointer' }}
      >
        ← Back
      </button>

      <h1 style={{ color: theme.text, fontSize: '1.6rem', fontWeight: 700, marginBottom: 6 }}>
        Settings
      </h1>
      <p style={{ color: theme.textSecondary, fontSize: '0.9rem', marginBottom: 32 }}>
        Update your CKD profile
      </p>

      {loading ? (
        <p style={{ color: theme.textSecondary, fontSize: '0.875rem' }}>Loading profile…</p>
      ) : (
        <div
          className="rounded-2xl p-6 sm:p-8 space-y-6"
          style={{ background: theme.cardBg, border: `1px solid ${theme.cardBorder}` }}
        >
          <div>
            <div style={{ color: theme.text, fontWeight: 600, marginBottom: 14 }}>CKD Stage</div>
            <div className="grid grid-cols-4 gap-2">
              {CKD_STAGES.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setCkdStage(s)}
                  className="py-2.5 rounded-xl transition-all duration-150"
                  style={{
                    background: ckdStage === s ? 'rgba(46,134,171,0.15)' : isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)',
                    border: `1px solid ${ckdStage === s ? 'rgba(46,134,171,0.5)' : theme.cardBorder}`,
                    color: ckdStage === s ? '#2E86AB' : theme.text,
                    fontWeight: ckdStage === s ? 700 : 400,
                    fontSize: '0.875rem',
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label
              htmlFor="settings-weight"
              style={{ color: theme.text, fontWeight: 600, marginBottom: 8, display: 'block' }}
            >
              Body Weight (kg)
            </label>
            <input
              id="settings-weight"
              type="number"
              min={20}
              max={300}
              step={0.5}
              value={weightKg}
              onChange={(e) => setWeightKg(parseFloat(e.target.value) || 65)}
              className="w-full px-4 py-2.5 rounded-xl outline-none"
              style={{
                background: isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.04)',
                border: `1px solid ${isDark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.14)'}`,
                color: theme.text,
                fontSize: '0.9rem',
              }}
            />
            <p style={{ color: theme.textSecondary, fontSize: '0.72rem', marginTop: 8 }}>
              Used to calculate your protein limit
            </p>
          </div>

          {successMessage && (
            <p style={{ color: '#27AE60', fontSize: '0.875rem' }}>{successMessage}</p>
          )}
          {errorMessage && (
            <p style={{ color: '#E74C3C', fontSize: '0.875rem' }}>{errorMessage}</p>
          )}

          <button
            type="button"
            onClick={() => void handleSave()}
            disabled={saving}
            className="w-full py-3 rounded-xl text-white font-semibold transition-all duration-200 hover:opacity-90 disabled:opacity-60"
            style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)' }}
          >
            {saving ? 'Saving…' : 'Save Changes'}
          </button>
        </div>
      )}
    </div>
  );
}
