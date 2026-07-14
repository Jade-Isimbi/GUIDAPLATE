import { test, expect } from '@playwright/test';
import { addFoodToMeal, goMealCheck, selectOccasion } from '../helpers/mealCheck';

/** UC5 — View SHAP Explanation (Meal Check “What drove this result”) */
test.describe('UC5 View SHAP Explanation', () => {
  test('shows SHAP contribution section after a live risk check', async ({ page }) => {
    await goMealCheck(page);
    await selectOccasion(page, 'Dinner');
    await addFoodToMeal(page, 'beef meat', 'beef meat');

    await page.getByRole('button', { name: 'Check this meal' }).click();
    await expect(page.getByText('What you should do')).toBeVisible({ timeout: 30_000 });

    await expect(page.getByTestId('shap-section')).toBeVisible({ timeout: 20_000 });
  });
});
