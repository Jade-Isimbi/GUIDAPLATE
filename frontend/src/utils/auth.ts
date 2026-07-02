const TOKEN_KEY = 'guidaplate_token';
const LEGACY_TOKEN_KEY = 'token';
const UNAUTHORIZED_EVENT = 'guidaplate:unauthorized';

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
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(LEGACY_TOKEN_KEY);
  localStorage.removeItem('guidaplate_user_id');
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
