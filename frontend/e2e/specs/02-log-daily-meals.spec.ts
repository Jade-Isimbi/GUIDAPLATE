import { test, expect } from '@playwright/test';
import { addFoodToMeal, goMealCheck, selectOccasion } from '../helpers/mealCheck';

/** UC2 — Log Daily Meals (Meal Check food log + assess) */
test.describe('UC2 Log Daily Meals', () => {
  test('adds a food and checks the meal', async ({ page }) => {
    await goMealCheck(page);

    await selectOccasion(page, 'Lunch');
    await addFoodToMeal(page, 'cabbage', 'cabbage');

    await expect(page.getByText(/cabbage/i).first()).toBeVisible();
    await page.getByRole('button', { name: 'Check this meal' }).click();

    await expect(page.getByText('What you should do')).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText(/Safe|Caution|Reduce Intake/).first()).toBeVisible();
  });
});
