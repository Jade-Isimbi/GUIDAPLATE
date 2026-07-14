import { test, expect } from '@playwright/test';

/** UC3 — Lookup Food Nutrients (Food Explorer) */
test.describe('UC3 Lookup Food Nutrients', () => {
  test('searches a food and shows nutrient risk profile', async ({ page }) => {
    await page.goto('/food-explorer');
    await expect(page.getByText('Kidney Health Food Explorer')).toBeVisible();

    await page.getByPlaceholder('Search English, French, or Kinyarwanda...').fill('cabbage');
    await page.getByText('cabbage', { exact: false }).first().click();

    await expect(page.getByText('Nutrient risk profile')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText('Potassium', { exact: false }).first()).toBeVisible();
  });
});
