/** Shared constants for GuidaPlate Playwright E2E. */

export const E2E_PASSWORD = process.env.E2E_PASSWORD ?? 'E2eTestPass123!';

/** Stable shared account for authenticated specs (created/reused in auth.setup). */
export const SHARED_E2E_EMAIL = process.env.E2E_EMAIL ?? 'e2e.shared@example.com';

export const SHARED_PROFILE = {
  name: 'E2E Shared User',
  phone: '+250 788000099',
  ckd_stage: 'G3a',
  weight_kg: 70,
  dob: '1990-06-15',
  sex: 'female',
} as const;

export function apiURLFromBase(baseURL: string | undefined): string {
  if (!baseURL) return 'http://localhost:8000';
  if (baseURL.includes('vercel.app')) {
    return 'https://guidaplate-production.up.railway.app';
  }
  return 'http://localhost:8000';
}

export function storagePathForBase(baseURL: string | undefined): string {
  if (baseURL?.includes('vercel.app')) {
    return 'e2e/.auth/user.production.json';
  }
  return 'e2e/.auth/user.local.json';
}
