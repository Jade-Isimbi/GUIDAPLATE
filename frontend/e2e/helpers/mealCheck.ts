import type { Page } from '@playwright/test';
import { expect } from '@playwright/test';

/** Ensure Meal Check shows the food search composer (not a prior logged occasion). */
export async function goMealCheck(page: Page) {
  await page.goto('/meal-check');
  await page.getByText('What did you eat?', { exact: true }).waitFor({ state: 'visible' });

  const reset = page.getByRole('button', { name: /Reset today's log/i });
  if (await reset.isVisible().catch(() => false)) {
    page.once('dialog', (d) => d.accept());
    await reset.click();
    const confirm = page.getByRole('button', { name: /confirm|yes|reset/i });
    if (await confirm.isVisible({ timeout: 1500 }).catch(() => false)) {
      await confirm.click();
    }
    await expect(page.getByTestId('meal-food-search')).toBeVisible({ timeout: 15_000 });
  }
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
