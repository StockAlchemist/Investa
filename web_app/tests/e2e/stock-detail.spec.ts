import { test, expect } from '@playwright/test';
import { loginAsMockUser } from './helpers/mock-api';

test.describe('Stock Detail View Journey', () => {
    test.beforeEach(async ({ page }) => {
        await loginAsMockUser(page);
    });

    test('opens Stock Detail modal from Holdings table and displays Overview', async ({ page }) => {
        await page.goto('/');

        // Navigate to Portfolio tab where HoldingsTable is rendered
        await page.getByRole('button', { name: 'Portfolio' }).click();

        // Click on AAPL holding symbol in the Holdings table
        await page.locator('table').getByText('AAPL', { exact: true }).first().click();

        // Stock Detail Modal should appear with company header
        await expect(page.getByText('Apple Inc.')).toBeVisible();
        await expect(page.getByText('$225.00').first()).toBeVisible();

        // Overview tab content: Your Position and Market Value
        await expect(page.getByText('Your Position').first()).toBeVisible();
        await expect(page.getByText('Market Value').first()).toBeVisible();
    });

    test('interacts with Valuation tab (DCF and Graham models)', async ({ page }) => {
        await page.goto('/');
        await page.getByRole('button', { name: 'Portfolio' }).click();
        await page.locator('table').getByText('AAPL', { exact: true }).first().click();

        // Click Valuation tab
        await page.getByRole('button', { name: 'Valuation' }).click();

        // Verify Valuation header metrics
        await expect(page.getByText('Blended Intrinsic Value')).toBeVisible();
        await expect(page.getByText('$245.80').first()).toBeVisible();
        await expect(page.getByText('Margin of Safety').first()).toBeVisible();

        // Verify DCF Discount Rate / Parameters
        await expect(page.getByText(/Discount Rate|WACC/i).first()).toBeVisible();
    });

    test('interacts with Analysis tab (Buffett quality rank & track record)', async ({ page }) => {
        await page.goto('/');
        await page.getByRole('button', { name: 'Portfolio' }).click();
        await page.locator('table').getByText('AAPL', { exact: true }).first().click();

        // Click Analysis tab
        await page.getByRole('button', { name: 'Analysis', exact: true }).click();

        // Verify AI Scorecard / Moat analysis / Track record
        await expect(page.getByText(/Moat|Financial Strength|Quality Score|Track Record/i).first()).toBeVisible();
    });

    test('interacts with Financials tab (statements and period toggles)', async ({ page }) => {
        await page.goto('/');
        await page.getByRole('button', { name: 'Portfolio' }).click();
        await page.locator('table').getByText('AAPL', { exact: true }).first().click();

        // Click Financials tab
        await page.getByRole('button', { name: 'Financials' }).click();

        // Verify Financial Statement table items (Revenue, Net Income)
        await expect(page.getByText(/Total Revenue|Revenue/i).first()).toBeVisible();

        // Toggle statement type (e.g. Balance Sheet)
        const balanceSheetButton = page.getByRole('button', { name: /Balance Sheet/i });
        if (await balanceSheetButton.isVisible()) {
            await balanceSheetButton.click();
            await expect(page.getByText(/Total Assets|Cash/i).first()).toBeVisible();
        }

        // Toggle Cash Flow Statement
        const cashFlowButton = page.getByRole('button', { name: /Cash Flow/i });
        if (await cashFlowButton.isVisible()) {
            await cashFlowButton.click();
            await expect(page.getByText(/Operating Cash Flow|Free Cash Flow/i).first()).toBeVisible();
        }
    });

    test('interacts with Ratios & Trends tab', async ({ page }) => {
        await page.goto('/');
        await page.getByRole('button', { name: 'Portfolio' }).click();
        await page.locator('table').getByText('AAPL', { exact: true }).first().click();

        // Click Ratios & Trends tab
        await page.getByRole('button', { name: /Ratios & Trends|Ratios/i }).click();

        // Verify ratio metrics (ROE, ROIC, Gross Margin, Net Margin)
        await expect(page.getByText(/ROE|Return on Equity|Profitability/i).first()).toBeVisible();
    });

    test('closes Stock Detail modal via close button or Escape key', async ({ page }) => {
        await page.goto('/');
        await page.getByRole('button', { name: 'Portfolio' }).click();
        await page.locator('table').getByText('AAPL', { exact: true }).first().click();

        await expect(page.getByText('Your Position').first()).toBeVisible();

        // Close modal via close button
        await page.getByLabel('Close modal').click();

        // Modal should disappear
        await expect(page.getByText('Your Position')).not.toBeVisible();
    });
});
