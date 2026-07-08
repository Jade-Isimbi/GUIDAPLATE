const TOKEN_KEY = 'guidaplate_token';
const LEGACY_TOKEN_KEY = 'token';
const UNAUTHORIZED_EVENT = 'guidaplate:unauthorized';

const MEAL_RESULTS_PREFIX = 'results_by_occasion';
const MEAL_RESULTS_DATE_PREFIX = 'results_by_occasion_date';
const MEAL_OCCASION_PREFIX = 'guidaplate_meal_occasion';

export function getAuthToken(): string | null {
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) return token;

  const legacy = localStorage.getItem(LEGACY_TOKEN_KEY);
  if (!legacy) return null;

  localStorage.setItem(TOKEN_KEY, legacy);
  localStorage.removeItem(LEGACY_TOKEN_KEY);
  return legacy;
}

export function clearAuthSession(notify = true): void {
  const userId = localStorage.getItem('guidaplate_user_id');

  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(LEGACY_TOKEN_KEY);
  localStorage.removeItem('guidaplate_user_id');

  localStorage.removeItem('ckd_stage');
  localStorage.removeItem('weight_kg');
  localStorage.removeItem('guidaplate_user_name');

  localStorage.removeItem(MEAL_RESULTS_PREFIX);
  localStorage.removeItem(MEAL_RESULTS_DATE_PREFIX);
  localStorage.removeItem(MEAL_OCCASION_PREFIX);

  if (userId) {
    localStorage.removeItem(`${MEAL_RESULTS_PREFIX}:${userId}`);
    localStorage.removeItem(`${MEAL_RESULTS_DATE_PREFIX}:${userId}`);
    localStorage.removeItem(`${MEAL_OCCASION_PREFIX}:${userId}`);
  }

  for (let i = localStorage.length - 1; i >= 0; i -= 1) {
    const key = localStorage.key(i);
    if (!key) continue;
    if (
      key.startsWith(`${MEAL_RESULTS_PREFIX}:`) ||
      key.startsWith(`${MEAL_RESULTS_DATE_PREFIX}:`) ||
      key.startsWith(`${MEAL_OCCASION_PREFIX}:`)
    ) {
      localStorage.removeItem(key);
    }
  }

  if (notify) {
    window.dispatchEvent(new CustomEvent(UNAUTHORIZED_EVENT));
  }
}

export function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const token = getAuthToken();
  if (!token) return { ...extra };
  return { ...extra, Authorization: `Bearer ${token}` };
}

export async function authFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const token = getAuthToken();
  if (!token) {
    clearAuthSession();
    throw new Error('Not authenticated');
  }

  const headers = new Headers(init.headers);
  headers.set('Authorization', `Bearer ${token}`);

  const response = await fetch(input, { ...init, headers });
  if (response.status === 401) {
    clearAuthSession();
  }
  return response;
}

export function onUnauthorized(handler: () => void): () => void {
  window.addEventListener(UNAUTHORIZED_EVENT, handler);
  return () => window.removeEventListener(UNAUTHORIZED_EVENT, handler);
}
