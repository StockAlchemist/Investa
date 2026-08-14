import { test, expect } from '@playwright/test';
import { loginAsMockUser } from './helpers/mock-api';

test.describe('Dashboard UI', () => {
    test('has title and loads successfully for authenticated user', async ({ page }) => {
        await loginAsMockUser(page);
        await page.goto('/');

        await expect(page).toHaveTitle(/Investa/i);

        // Check sidebar navigation items
        await expect(page.getByRole('button', { name: 'Dashboard' }).first()).toBeVisible();
        await expect(page.getByRole('button', { name: 'Portfolio' }).first()).toBeVisible();
        await expect(page.getByRole('button', { name: 'Transactions' }).first()).toBeVisible();

        // Check dashboard metrics and holdings table
        await expect(page.getByText('$125,450.00').first()).toBeVisible();
        await expect(page.getByText('AAPL').first()).toBeVisible();
    });
});
