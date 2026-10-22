import { test, expect } from '@playwright/test';
import { loginAsMockUser } from './helpers/mock-api';
import { MOCK_SETTINGS, MOCK_SUMMARY_USD } from './helpers/mock-data';
import { INITIAL_VISIBLE_ITEMS } from '../../lib/dashboard_constants';

for (const width of [390, 1280]) {
    test(`reset restores the default dashboard at ${width}px`, async ({ page }) => {
        await loginAsMockUser(page, { initialSettings: { ...MOCK_SETTINGS, visible_items: ['cashBalance'] } });
        await page.setViewportSize({ width, height: 844 });
        await page.goto('/');
        await expect(page.getByText('Cash Balance', { exact: true })).toBeVisible();
        await page.getByRole('button', { name: 'Dashboard Elements', exact: true }).click();
        await page.getByText('Reset to Default', { exact: true }).click();
        await expect(page.getByText('$125,450.00').first()).toBeVisible();
        await expect.poll(() => page.evaluate(() => JSON.parse(localStorage.getItem('investa_dashboard_visible_items') ?? '[]')))
            .toEqual(INITIAL_VISIBLE_ITEMS);
    });
}

test('the live hero chart renders with one intraday point and a newer valuation', async ({ page }) => {
    await loginAsMockUser(page, { initialSettings: { ...MOCK_SETTINGS, visible_items: ['portfolioHero'] } });
    await page.route(/\/api\/history(\?|$)/, route => route.fulfill({ json: [
        { date: '2026-10-21T13:30:00Z', value: 100000 },
    ] }));
    await page.route(/\/api\/summary(\/headline)?(\?|$)/, route => route.fulfill({ json: {
        ...MOCK_SUMMARY_USD, freshness: { calculated_at: Date.parse('2026-10-21T13:35:00Z') / 1000, refreshing: false },
    } }));
    await page.goto('/');
    await expect(page.getByText('$125,450.00').first()).toBeVisible();
    await expect(page.locator('.recharts-area-curve')).toBeVisible();
});
