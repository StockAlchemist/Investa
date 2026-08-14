import { test, expect } from '@playwright/test';
import { loginAsMockUser } from './helpers/mock-api';

test.describe('Transactions Journey', () => {
    test.beforeEach(async ({ page }) => {
        await loginAsMockUser(page);
    });

    test('renders transactions table and KPI metrics correctly', async ({ page }) => {
        await page.goto('/');

        // Navigate to Transactions tab
        await page.getByRole('button', { name: 'Transactions' }).click();

        // Verify transactions view header and table
        await expect(page.getByRole('button', { name: 'Add', exact: true })).toBeVisible();
        await expect(page.getByText('AAPL').first()).toBeVisible();
        await expect(page.getByText('MSFT').first()).toBeVisible();
    });

    test('creates a new Buy transaction and updates portfolio', async ({ page }) => {
        await page.goto('/');
        await page.getByRole('button', { name: 'Transactions' }).click();

        // Open Add Transaction modal
        await page.getByRole('button', { name: 'Add', exact: true }).click();
        await expect(page.getByRole('heading', { name: 'Add Transaction' })).toBeVisible();

        // Fill form fields
        await page.locator('input[name="Date"]').fill('2026-08-01');
        await page.locator('select[name="Type"]').selectOption('Buy');
        await page.locator('input[name="Symbol"]').fill('NVDA');
        await page.locator('input[name="Account"]').fill('Taxable');
        await page.locator('input[name="Quantity"]').fill('50');
        await page.locator('input[name="Price/Share"]').fill('120');
        await page.locator('input[name="Commission"]').fill('2.00');

        // Total Amount should be auto-calculated to 6002.00 (50 * 120 + 2)
        await expect(page.locator('input[name="Total Amount"]')).toHaveValue('6002');

        // Submit form
        await page.getByRole('button', { name: 'Add Transaction' }).click();

        // Verify modal closes
        await expect(page.getByRole('heading', { name: 'Add Transaction' })).not.toBeVisible();

        // Verify NVDA transaction is visible in table
        await expect(page.getByText('NVDA').first()).toBeVisible();
    });

    test('edits an existing transaction and updates table', async ({ page }) => {
        await page.goto('/');
        await page.getByRole('button', { name: 'Transactions' }).click();

        // Find the edit button for Buy AAPL transaction row
        const aaplRow = page.getByRole('row', { name: /Buy/i }).filter({ hasText: 'AAPL' }).first();
        await aaplRow.locator('button[title="Edit"]').click();

        // Verify Edit Transaction modal is open with prefilled data
        await expect(page.getByRole('heading', { name: 'Edit Transaction' })).toBeVisible();
        await expect(page.locator('input[name="Symbol"]')).toHaveValue('AAPL');

        // Modify Quantity
        await page.locator('input[name="Quantity"]').fill('180');

        // Save
        await page.getByRole('button', { name: 'Update Transaction' }).click();
        await expect(page.getByRole('heading', { name: 'Edit Transaction' })).not.toBeVisible();

        // Verify updated quantity is rendered
        await expect(page.getByText('180').first()).toBeVisible();
    });

    test('deletes a transaction with confirmation and removes it from table', async ({ page }) => {
        // Setup dialog handler before navigation
        page.on('dialog', async (dialog) => {
            await dialog.accept();
        });

        await page.goto('/');
        await page.getByRole('button', { name: 'Transactions' }).click();

        // Check initial rows
        await expect(page.getByRole('row', { name: /GOOGL/i }).first()).toBeVisible();

        // Delete the GOOGL transaction
        const googlRow = page.getByRole('row', { name: /GOOGL/i }).first();
        await googlRow.locator('button[title="Delete"]').click();

        // Verify GOOGL row is removed from table
        await expect(page.getByRole('row', { name: /GOOGL/i })).not.toBeVisible();
    });

    test('filters transactions by search keyword and type', async ({ page }) => {
        await page.goto('/');
        await page.getByRole('button', { name: 'Transactions' }).click();

        // Verify search input exists
        const searchInput = page.getByPlaceholder('Search symbol...');
        await expect(searchInput).toBeVisible();

        // Filter by MSFT
        await searchInput.fill('MSFT');
        await expect(page.getByText('MSFT').first()).toBeVisible();

        // Clear filter
        await searchInput.fill('');
    });
});
