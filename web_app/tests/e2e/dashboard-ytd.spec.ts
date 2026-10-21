import { test, expect } from '@playwright/test';
import { loginAsMockUser } from './helpers/mock-api';
import { MOCK_SETTINGS, MOCK_SUMMARY_USD } from './helpers/mock-data';

for (const width of [390, 1280]) {
    for (const fraction of [0.182, 0, -0.051]) {
        test(`YTD card loads independently of Risk Analytics at ${width}px (${fraction})`, async ({ page }) => {
            await loginAsMockUser(page, {
                initialSettings: { ...MOCK_SETTINGS, visible_items: ['ytdReturn'] },
            });
            await page.setViewportSize({ width, height: 844 });
            // Real summaries need not include YTD; it is supplied by /risk_metrics.
            const metrics = { ...MOCK_SUMMARY_USD.metrics };
            delete metrics.ytd_return;
            await page.route(/\/api\/summary(\/headline)?(\?|$)/, route =>
                route.fulfill({ json: { ...MOCK_SUMMARY_USD, metrics } }));
            const requests: URL[] = [];
            await page.route(/\/api\/risk_metrics(\?|$)/, route => {
                requests.push(new URL(route.request().url()));
                return route.fulfill({ json: { 'YTD Return': fraction } });
            });
            await page.goto('/');
            const card = page.locator('.card-standard').filter({ has: page.getByText('YTD Return', { exact: true }) });
            await expect(card.getByText(`${(fraction * 100).toFixed(2)}%`, { exact: true })).toBeVisible();
            expect(requests.length).toBeGreaterThan(0);
            expect(requests[0].searchParams.get('currency')).toBe('USD');
            await expect(page.getByRole('button', { name: 'Show detailed analytics', exact: true }))
                .toHaveAttribute('aria-expanded', 'false');
            if (fraction > 0) await card.screenshot({ path: test.info().outputPath(`ytd-${width}.png`) });
        });
    }
}
