import type { APIRequestContext } from '@playwright/test';
import { E2E_PASSWORD, SHARED_PROFILE, SHARED_E2E_EMAIL } from './constants';

export type AuthPayload = {
  access_token: string;
  user_id: string;
  name: string;
  ckd_stage: string | null;
  weight_kg: number | null;
};

export async function registerOrLogin(
  request: APIRequestContext,
  apiURL: string,
  opts?: { email?: string; password?: string; name?: string },
): Promise<AuthPayload> {
  const email = opts?.email ?? SHARED_E2E_EMAIL;
  const password = opts?.password ?? E2E_PASSWORD;
  const name = opts?.name ?? SHARED_PROFILE.name;

  const registerBody = {
    name,
    email,
    password,
    phone: SHARED_PROFILE.phone,
    ckd_stage: SHARED_PROFILE.ckd_stage,
    weight_kg: SHARED_PROFILE.weight_kg,
    dob: SHARED_PROFILE.dob,
    sex: SHARED_PROFILE.sex,
  };

  const reg = await request.post(`${apiURL}/api/auth/register`, { data: registerBody });
  if (reg.ok()) {
    return (await reg.json()) as AuthPayload;
  }

  const login = await request.post(`${apiURL}/api/auth/login`, {
    data: { email, password },
  });
  if (!login.ok()) {
    throw new Error(
      `Auth failed for ${email}: register=${reg.status()} ${await reg.text()} | login=${login.status()} ${await login.text()}`,
    );
  }
  return (await login.json()) as AuthPayload;
}

export async function postFoodLog(
  request: APIRequestContext,
  apiURL: string,
  token: string,
  payload: Record<string, unknown>,
) {
  const res = await request.post(`${apiURL}/api/patient/food-log`, {
    headers: { Authorization: `Bearer ${token}` },
    data: payload,
  });
  if (!res.ok()) {
    throw new Error(`food-log failed: ${res.status()} ${await res.text()}`);
  }
  return res.json();
}

/** Seed several low-burden meal logs for Diet Pattern (LSTM sequence). */
export async function seedMealHistory(
  request: APIRequestContext,
  apiURL: string,
  token: string,
  count = 6,
) {
  const occasions = ['Breakfast', 'Lunch', 'Dinner', 'Snack', 'Breakfast', 'Lunch'];
  for (let i = 0; i < count; i += 1) {
    await postFoodLog(request, apiURL, token, {
      food_name: 'cabbage',
      category: 'Vegetable',
      stage_safe_range: '1-5',
      portion_grams: 80 + i * 5,
      meal_occasion: occasions[i % occasions.length],
      potassium_mg: 180,
      phosphorus_mg: 40,
      protein_g: 1.5,
      sodium_mg: 20,
    });
  }
}
