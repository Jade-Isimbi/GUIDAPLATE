import { test, expect } from '@playwright/test';
import { addFoodToMeal, goMealCheck, selectOccasion } from '../helpers/mealCheck';

/**
 * UC7 — View Food Recommendations
 * Log a HIGH-burden meal first so substitutes / next-meal panels are more likely.
 */
test.describe('UC7 View Food Recommendations', () => {
  test('HIGH meal check surfaces recommendations / alternatives', async ({ page }) => {
    await goMealCheck(page);
    await selectOccasion(page, 'Breakfast');

    // Large avocado portions — high potassium vs breakfast meal caps
    await addFoodToMeal(page, 'avocado', 'avocado', 8);

    await page.getByRole('button', { name: 'Check this meal' }).click();
    await expect(page.getByText('What you should do')).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText('Reduce Intake', { exact: true })).toBeVisible({ timeout: 15_000 });

    await expect(page.getByTestId('alternatives-section')).toBeVisible({ timeout: 25_000 });
  });
});
