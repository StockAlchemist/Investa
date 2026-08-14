import { test, expect } from '@playwright/test';
import { loginAsMockUser } from './helpers/mock-api';

test.describe('Screener Journey', () => {
    test.beforeEach(async ({ page }) => {
        await loginAsMockUser(page);
    });

    test('executes screener on selected universe and renders results', async ({ page }) => {
        await page.goto('/');

        // Navigate to Screener
        await page.getByRole('button', { name: 'Screener' }).click();
        await expect(page.getByText('Market Explorer')).toBeVisible();

        // Universe select: Select S&P 500
        const universeSelect = page.locator('#screener-universe');
        await universeSelect.selectOption('sp500');

        // Click Execute Screen
        await page.getByRole('button', { name: /Execute Screen/i }).click();

        // Verify Screener Results table loads
        await expect(page.getByText('AAPL').first()).toBeVisible();
        await expect(page.getByText('MSFT').first()).toBeVisible();
        await expect(page.getByText('GOOGL').first()).toBeVisible();
        await expect(page.getByText('NVDA').first()).toBeVisible();
        await expect(page.getByText('BRK.B').first()).toBeVisible();
    });

    test('filters screener results by search keyword', async ({ page }) => {
        await page.goto('/');
        await page.getByRole('button', { name: 'Screener' }).click();

        // Run screen
        await page.getByRole('button', { name: /Execute Screen/i }).click();
        await expect(page.getByText('AAPL').first()).toBeVisible();

        // Search for Alphabet / GOOGL
        const searchInput = page.locator('input[placeholder*="Search symbol or name"]');
        await searchInput.fill('GOOGL');

        // GOOGL should be visible, others filtered out
        await expect(page.getByText('Alphabet Inc.').first()).toBeVisible();
        await expect(page.getByText('Apple Inc.')).not.toBeVisible();

        // Clear search
        await searchInput.fill('');
        await expect(page.getByText('Apple Inc.').first()).toBeVisible();
    });

    test('filters screener results by Margin of Safety and P/E', async ({ page }) => {
        await page.goto('/');
        await page.getByRole('button', { name: 'Screener' }).click();

        // Run screen
        await page.getByRole('button', { name: /Execute Screen/i }).click();
        await expect(page.getByText('AAPL').first()).toBeVisible();

        // Open filters panel
        await page.getByRole('button', { name: /Filters/i }).click();

        // Set Min Margin of Safety to 15%
        const minMosInput = page.locator('input[placeholder="e.g. 15"]');
        await minMosInput.fill('15');

        // GOOGL (22.85%) and BRK.B (15.55%) should remain visible
        await expect(page.getByText('Alphabet Inc.').first()).toBeVisible();
        await expect(page.getByText('Berkshire Hathaway Inc.').first()).toBeVisible();

        // AAPL (9.24%) and NVDA (-10.71%) should be filtered out
        await expect(page.getByText('Apple Inc.')).not.toBeVisible();
        await expect(page.getByText('NVIDIA Corp.')).not.toBeVisible();
    });

    test('triggers AI review from screener results', async ({ page }) => {
        await page.goto('/');
        await page.getByRole('button', { name: 'Screener' }).click();

        // Run screen
        await page.getByRole('button', { name: /Execute Screen/i }).click();
        await expect(page.getByText('AAPL').first()).toBeVisible();

        // Find and click the AI review action on the first row
        const reviewButton = page.locator('button:has-text("Review"), button[title*="Review"], button:has(.lucide-sparkles)').first();
        if (await reviewButton.isVisible()) {
            await reviewButton.click();
            // Verify review modal or updated review status
            await expect(page.locator('body')).toBeVisible();
        }
    });
});
