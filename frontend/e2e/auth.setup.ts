import { test as setup, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { registerOrLogin } from './helpers/api';
import {
  SHARED_E2E_EMAIL,
  SHARED_PROFILE,
  apiURLFromBase,
  storagePathForBase,
} from './helpers/constants';

/**
 * Creates (or reuses) the shared E2E account via API, writes localStorage
 * keys the SPA expects, and saves Playwright storageState for authenticated specs.
 */
setup('authenticate shared e2e user', async ({ page, request, baseURL }) => {
  const apiURL = apiURLFromBase(baseURL);
  const outPath = storagePathForBase(baseURL);
  fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });

  const auth = await registerOrLogin(request, apiURL, {
    email: SHARED_E2E_EMAIL,
    name: SHARED_PROFILE.name,
  });

  await page.goto('/login');
  await expect(page.getByText('Welcome back')).toBeVisible();

  await page.evaluate(
    ({ token, userId, name, stage, weight }) => {
      localStorage.setItem('guidaplate_token', token);
      localStorage.setItem('token', token);
      localStorage.setItem('guidaplate_user_id', userId);
      localStorage.setItem('guidaplate_user_name', name);
      localStorage.setItem('ckd_stage', stage);
      localStorage.setItem('weight_kg', String(weight));
    },
    {
      token: auth.access_token,
      userId: auth.user_id,
      name: auth.name,
      stage: auth.ckd_stage ?? SHARED_PROFILE.ckd_stage,
      weight: auth.weight_kg ?? SHARED_PROFILE.weight_kg,
    },
  );

  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Dashboard' })).toBeVisible({ timeout: 20_000 });

  await page.context().storageState({ path: outPath });
});
