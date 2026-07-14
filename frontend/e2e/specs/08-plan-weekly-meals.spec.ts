import { test, expect } from '@playwright/test';

/**
 * UC8 — Plan Weekly Meals (Meal Planner / Groq)
 *
 * Hits real Groq when GROQ_API_KEY is set on the backend.
 * Run sparingly: one local proof pass is enough; avoid hammering on every iteration.
 * Tag: @groq
 */
test.describe('UC8 Plan Weekly Meals @groq', () => {
  test('opens Meal Suggestions and receives a weekly-plan style reply', async ({ page }) => {
    test.setTimeout(120_000);

    await page.goto('/meal-planner');

    await expect(page.getByText('Meal Suggestions').first()).toBeVisible();
    await expect(page.getByText('Online · Kidney Diet Assistant')).toBeVisible();

    // Discard any persisted session so we don't assert on stale assistant text.
    const clear = page.getByRole('button', { name: /Clear|New conversation/i });
    if (await clear.isVisible().catch(() => false)) {
      await clear.click();
    }
    await expect(page.getByText('Ask me about safe meals for your kidneys')).toBeVisible({
      timeout: 10_000,
    });

    const prompt =
      'Give me a simple 3-day kidney-friendly meal plan using Rwandan foods for my stage.';
    const input = page.getByPlaceholder('Message GuidaPlate AI...');
    await input.fill(prompt);

    // Real click on Send via stable testid (distinct from settings-fab).
    await page.getByTestId('meal-planner-send').click();

    await expect(
      page.getByText(/Looking up your guidelines|Checking safe foods|almost there/i).first(),
    ).toBeVisible({ timeout: 15_000 });

    await expect(page.getByText(prompt)).toBeVisible();

    await expect(
      page.getByText(/Looking up your guidelines|Checking safe foods|almost there/i),
    ).toHaveCount(0, { timeout: 90_000 });

    await expect(
      page
        .locator('main')
        .getByText(/Breakfast|Lunch|Dinner|cassava|rice|cabbage|potassium|kidney|meal plan/i)
        .first(),
    ).toBeVisible({ timeout: 15_000 });
  });
});
