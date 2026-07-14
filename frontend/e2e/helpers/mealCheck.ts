import type { Page } from '@playwright/test';
import { expect } from '@playwright/test';
import { apiURLFromBase } from './constants';

type FoodLogRow = { logged_at?: string | null };

function isLoggedToday(loggedAt: string | null | undefined): boolean {
  if (!loggedAt) return false;
  return new Date(loggedAt).toDateString() === new Date().toDateString();
}

function waitForFoodLogHistory(page: Page) {
  return page.waitForResponse(
    (r) =>
      r.url().includes('/api/patient/food-log/history') &&
      r.request().method() === 'GET' &&
      r.ok(),
    { timeout: 45_000 },
  );
}

/** Clear today's logs via API — used when Reset isn't rendered (no Check result yet). */
async function clearTodayViaApi(page: Page) {
  const token = await page.evaluate(
    () => localStorage.getItem('guidaplate_token') || localStorage.getItem('token'),
  );
  if (!token) {
    throw new Error('goMealCheck: missing auth token to clear food-log day');
  }

  const apiBase = apiURLFromBase(page.url());
  const res = await page.request.delete(`${apiBase}/api/patient/food-log/day`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok()) {
    throw new Error(`goMealCheck: DELETE food-log/day failed (${res.status()})`);
  }

  // Drop local occasion assessments so reload doesn't resurrect a results panel.
  await page.evaluate(() => {
    const uid = localStorage.getItem('guidaplate_user_id');
    if (!uid) return;
    localStorage.removeItem(`results_by_occasion:${uid}`);
    localStorage.removeItem(`results_by_occasion_date:${uid}`);
    localStorage.removeItem(`guidaplate_meal_occasion:${uid}`);
  });
}

/**
 * Navigate to Meal Check and clear leftover food-log state for the shared e2e account.
 * Always waits for /food-log/history to hydrate before deciding whether a reset is needed.
 */
export async function goMealCheck(page: Page) {
  const historyPromise = waitForFoodLogHistory(page);
  await page.goto('/meal-check');
  await expect(page.getByText('What did you eat?', { exact: true })).toBeVisible({
    timeout: 20_000,
  });

  const historyRes = await historyPromise;
  const logs = (await historyRes.json()) as FoodLogRow[];
  const todayCount = Array.isArray(logs)
    ? logs.filter((l) => isLoggedToday(l.logged_at)).length
    : 0;

  if (todayCount > 0) {
    const reset = page.getByRole('button', { name: /Reset today's log/i });
    const canUiReset =
      (await reset.isVisible().catch(() => false)) &&
      (await reset.isEnabled().catch(() => false));

    if (canUiReset) {
      page.once('dialog', (d) => d.accept());
      const deleteDone = page.waitForResponse(
        (r) =>
          r.url().includes('/api/patient/food-log/day') &&
          r.request().method() === 'DELETE' &&
          r.ok(),
        { timeout: 30_000 },
      );
      await reset.click();
      await deleteDone;
    } else {
      // Shared-account leftover without an on-screen Check result: Reset is not in the DOM.
      await clearTodayViaApi(page);
      const historyAgain = waitForFoodLogHistory(page);
      await page.reload();
      await expect(page.getByText('What did you eat?', { exact: true })).toBeVisible({
        timeout: 20_000,
      });
      const again = await historyAgain;
      const after = (await again.json()) as FoodLogRow[];
      const remaining = after.filter((l) => isLoggedToday(l.logged_at)).length;
      expect(remaining, 'today food logs should be cleared for shared e2e account').toBe(0);
    }
  }

  // Search only mounts when the current occasion is not in logged view — wait until clear.
  await expect(page.getByTestId('meal-food-search')).toBeVisible({ timeout: 15_000 });
}

export async function selectOccasion(page: Page, occasion: 'Breakfast' | 'Lunch' | 'Dinner' | 'Snack') {
  await page.getByTestId(`occasion-${occasion.toLowerCase()}`).click();
}

/**
 * Meal Check: pick first food from search dropdown and confirm Add.
 * Uses data-testid attrs — does not depend on visible food english text for the click.
 */
export async function addFoodToMeal(page: Page, query: string, _foodName: string, qtyClicks = 0) {
  const search = page.getByTestId('meal-food-search');
  if (!(await search.isVisible().catch(() => false))) {
    await page.getByRole('button', { name: /Add another food/i }).click();
    await expect(search).toBeVisible({ timeout: 10_000 });
  }

  await search.fill(query);
  await expect(page.getByTestId('meal-food-dropdown')).toBeVisible();
  // Prefer exact name match via data-food-name when provided; else first option.
  const byName = page.locator(
    `[data-testid="meal-food-option"][data-food-name="${_foodName}" i]`,
  );
  if (await byName.count()) {
    await byName.first().click();
  } else {
    await page.getByTestId('meal-food-option').first().click();
  }

  for (let i = 0; i < qtyClicks; i += 1) {
    await page.getByRole('button', { name: '+', exact: true }).click();
  }
  await page.getByTestId('meal-food-add').click();
}
