import { test, expect } from '@playwright/test';
import { loginAsMockUser } from './helpers/mock-api';

test.describe('Account Group Switching Journey', () => {
    test.beforeEach(async ({ page }) => {
        await loginAsMockUser(page);
    });

    test('switches account group to Retirement and filters holdings and metrics', async ({ page }) => {
        await page.goto('/');

        // Initial state: All Accounts selected
        const accountTrigger = page.locator('button:has-text("All Accounts"), button:has-text("All")').first();
        await expect(accountTrigger).toBeVisible();

        // Initial all accounts holdings in performance/portfolio
        await expect(page.getByText('$125,450.00').first()).toBeVisible();

        // Open AccountSelector dropdown
        await accountTrigger.click();

        // Select "Retirement" group
        await page.getByRole('button', { name: 'Retirement' }).click();

        // Dropdown closes / trigger updates to Roth IRA
        await expect(page.getByText('Roth IRA').first()).toBeVisible();

        // Summary market value should reflect Roth IRA ($32,450.00)
        await expect(page.getByText('$32,450.00').first()).toBeVisible();

        // Verify localStorage persistence
        const storedAccounts = await page.evaluate(() => localStorage.getItem('investa_selected_accounts'));
        expect(storedAccounts).toContain('Roth IRA');
    });

    test('switches back to All Accounts and restores entire portfolio', async ({ page }) => {
        await page.goto('/');

        // Switch to Retirement group first
        const accountTrigger = page.locator('button:has-text("All Accounts"), button:has-text("All")').first();
        await accountTrigger.click();
        await page.getByRole('button', { name: 'Retirement' }).click();

        // Verify Roth IRA metric
        await expect(page.getByText('$32,450.00').first()).toBeVisible();

        // Open AccountSelector and switch back to All Accounts
        const rothTrigger = page.locator('button:has-text("Roth IRA")').first();
        await rothTrigger.click();
        await page.getByRole('button', { name: 'All Accounts' }).click();

        // Verify full portfolio total is restored
        await expect(page.getByText('$125,450.00').first()).toBeVisible();
    });

    test('selects Trading group (multi-account selection)', async ({ page }) => {
        await page.goto('/');

        const accountTrigger = page.locator('button:has-text("All Accounts"), button:has-text("All")').first();
        await accountTrigger.click();

        // Select "Trading" group (Taxable + Crypto)
        await page.getByRole('button', { name: 'Trading' }).click();

        // Verify multi-account label
        await expect(page.getByText(/2 Accounts Selected|2 Accs|Taxable/i).first()).toBeVisible();

        // Taxable + Crypto market value ($93,000.00)
        await expect(page.getByText('$93,000.00').first()).toBeVisible();
    });
});
