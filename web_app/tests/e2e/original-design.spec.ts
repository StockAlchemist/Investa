import { expect, test } from '@playwright/test';
import { loginAsMockUser } from './helpers/mock-api';
import { MOCK_SETTINGS } from './helpers/mock-data';
import { INITIAL_VISIBLE_ITEMS } from '../../lib/dashboard_constants';

const redesign = ['portfolioHero', 'performanceGraph', 'annualTWR', 'totalReturn', 'cashBalance', 'ledgerAllocation', 'ledgerHoldings'];

test('restores the original interface from a saved Ledger layout', async ({ page }) => {
    await loginAsMockUser(page);
    await page.addInitScript(items => localStorage.setItem('investa_dashboard_visible_items', JSON.stringify(items)), redesign);
    await page.goto('/');
    await expect(page.getByText('$125,450.00').first()).toBeVisible();
    await expect.poll(() => page.evaluate(() => JSON.parse(localStorage.getItem('investa_dashboard_visible_items') || '[]'))).toEqual(INITIAL_VISIBLE_ITEMS);
    await expect(page.getByRole('region', { name: 'Ledger overview' })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Add transaction', exact: true })).toHaveCount(0);
    await page.screenshot({ path: test.info().outputPath('original-desktop.png'), fullPage: true });
});

test('retains the saved server layout when there is no local layout', async ({ page }) => {
    await loginAsMockUser(page, { initialSettings: { ...MOCK_SETTINGS, visible_items: ['portfolioHero', 'cashBalance'] } });
    await page.goto('/');
    await expect(page.getByText('$125,450.00').first()).toBeVisible();
    await expect.poll(() => page.evaluate(() => JSON.parse(localStorage.getItem('investa_dashboard_visible_items') || '[]'))).toEqual(['portfolioHero', 'cashBalance']);
});

test('phone keeps the original navigation and transaction entry', async ({ page }) => {
    await loginAsMockUser(page);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/');
    await expect(page.getByText('$125,450.00').first()).toBeVisible();
    await expect(page.getByRole('region', { name: 'Ledger overview' })).toHaveCount(0);
    await page.screenshot({ path: test.info().outputPath('original-phone.png'), fullPage: true });
    await page.evaluate(() => { history.pushState(null, '', '/?section=transactions&currency=USD'); dispatchEvent(new PopStateEvent('popstate')); });
    await page.getByRole('button', { name: /Add Transaction/i }).filter({ visible: true }).click();
    await expect(page.getByRole('dialog')).toBeVisible();
});
