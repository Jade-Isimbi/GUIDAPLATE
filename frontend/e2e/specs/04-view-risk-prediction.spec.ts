import { test, expect } from '@playwright/test';
import { addFoodToMeal, goMealCheck, selectOccasion } from '../helpers/mealCheck';

/** UC4 — View Risk Prediction (Meal Check result headline) */
test.describe('UC4 View Risk Prediction', () => {
  test('shows a stage-safe risk label after checking a meal', async ({ page }) => {
    await goMealCheck(page);
    await selectOccasion(page, 'Breakfast');
    await addFoodToMeal(page, 'rice', 'rice');

    await page.getByRole('button', { name: 'Check this meal' }).click();

    await expect(page.getByText('What you should do')).toBeVisible({ timeout: 30_000 });
    await expect(
      page
        .getByText('Safe', { exact: true })
        .or(page.getByText('Caution', { exact: true }))
        .or(page.getByText('Reduce Intake', { exact: true }))
        .first(),
    ).toBeVisible();
    await expect(page.getByText(/Your Stage|daily allowance/i).first()).toBeVisible();
  });
});
