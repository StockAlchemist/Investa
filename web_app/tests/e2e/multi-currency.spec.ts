import { test, expect } from '@playwright/test';
import { loginAsMockUser } from './helpers/mock-api';

test.describe('Multi-Currency Switching Journey', () => {
    test.beforeEach(async ({ page }) => {
        await loginAsMockUser(page);
    });

    test('switches currency from USD to EUR and updates valuations', async ({ page }) => {
        await page.goto('/');

        // Verify initial currency is USD
        const currencyTrigger = page.locator('button[title^="Currency:"]').first();
        await expect(currencyTrigger).toBeVisible();
        await expect(currencyTrigger).toContainText('USD');

        // Initial USD Market Value from mock data ($125,450.00)
        await expect(page.getByText('$125,450.00').first()).toBeVisible();

        // Open currency dropdown
        await currencyTrigger.click();

        // Select EUR
        await page.getByRole('button', { name: /EUR/i }).click();

        // Verify trigger text updates to EUR
        await expect(currencyTrigger).toContainText('EUR');

        // Verify EUR Market Value from mock data (€114,045.50 / 114,045.50)
        await expect(page.getByText('114,045.50').first()).toBeVisible();

        // Verify currency is stored in localStorage
        const storedCurrency = await page.evaluate(() => localStorage.getItem('investa_currency'));
        expect(storedCurrency).toBe('EUR');
    });

    test('switches currency to THB and verifies exchange rate and formatting', async ({ page }) => {
        await page.goto('/');

        const currencyTrigger = page.locator('button[title^="Currency:"]').first();
        await currencyTrigger.click();

        // Select THB
        await page.getByRole('button', { name: /THB/i }).click();

        // Verify trigger text updates to THB
        await expect(currencyTrigger).toContainText('THB');

        // Verify THB Market Value (4,390,750.00)
        await expect(page.getByText('4,390,750').first()).toBeVisible();

        // Verify localStorage persistence
        const storedCurrency = await page.evaluate(() => localStorage.getItem('investa_currency'));
        expect(storedCurrency).toBe('THB');
    });

    test('reloads page with persisted currency selection', async ({ page }) => {
        await page.goto('/');

        const currencyTrigger = page.locator('button[title^="Currency:"]').first();
        await currencyTrigger.click();
        await page.getByRole('button', { name: /EUR/i }).click();
        await expect(currencyTrigger).toContainText('EUR');

        // Reload page
        await page.reload();

        // Verify currency remains EUR after reload
        await expect(page.locator('button[title^="Currency:"]').first()).toContainText('EUR');
        await expect(page.getByText('114,045.50').first()).toBeVisible();
    });
});
