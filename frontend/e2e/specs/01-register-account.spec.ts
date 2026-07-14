import { test, expect } from '@playwright/test';
import { E2E_PASSWORD } from '../helpers/constants';

/**
 * UC1 — Register Account
 * Runs logged-out (clears storageState inherited from the local project).
 */
test.use({ storageState: { cookies: [], origins: [] } });

test.describe('UC1 Register Account', () => {
  test('registers a new account via Signup UI and lands in the app', async ({ page }) => {
    const email = `e2e.register.${Date.now()}@example.com`;

    await page.goto('/signup');
    await expect(page.getByRole('heading', { name: 'Create your account' })).toBeVisible();

    await page.getByPlaceholder('Jean-Pierre Nkurunziza').fill('E2E Register User');
    await page.getByPlaceholder('you@example.com').fill(email);
    await page.getByPlaceholder('788 000 000').fill('788123456');
    await page.getByPlaceholder('At least 8 characters').fill(E2E_PASSWORD);
    await page.getByPlaceholder('Re-enter your password').fill(E2E_PASSWORD);

    await page.getByRole('button', { name: 'Stage 3a', exact: true }).click();
    await page.getByPlaceholder('e.g. 68').fill('70');
    await page.locator('input[type="date"]').fill('1992-03-20');
    await page.getByRole('button', { name: 'Female', exact: true }).click();

    await page.getByTestId('signup-consent').check({ force: true });

    await page.getByRole('button', { name: 'Create Account' }).click();

    await expect(page.getByRole('button', { name: 'Dashboard' })).toBeVisible({ timeout: 25_000 });
    await expect(page).not.toHaveURL(/\/signup/);
  });
});
