import { test, expect } from '@playwright/test';
import { seedMealHistory } from '../helpers/api';
import { apiURLFromBase, SHARED_E2E_EMAIL, E2E_PASSWORD } from '../helpers/constants';
import { registerOrLogin } from '../helpers/api';

/** UC6 — View Dietary Patterns (Diet Pattern / weekly-trend) */
test.describe('UC6 View Dietary Patterns', () => {
  test('shows pattern analysis after seeded meal history', async ({ page, request, baseURL }) => {
    const apiURL = apiURLFromBase(baseURL);
    const auth = await registerOrLogin(request, apiURL, {
      email: SHARED_E2E_EMAIL,
      password: E2E_PASSWORD,
    });
    await seedMealHistory(request, apiURL, auth.access_token, 6);

    await page.goto('/weekly-trend');
    await expect(page.getByRole('heading', { name: 'Your Dietary Pattern' })).toBeVisible();

    // Empty state should not win once history exists
    await expect(page.getByText('No meals logged this week')).toHaveCount(0, { timeout: 25_000 });

    await expect(
      page.getByText(/LSTM pattern analysis|Based on your last|What this means for you|risk ·/i).first(),
    ).toBeVisible({ timeout: 30_000 });
  });
});
