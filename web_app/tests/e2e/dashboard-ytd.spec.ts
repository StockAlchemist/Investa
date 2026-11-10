import { test, expect } from '@playwright/test';
import { loginAsMockUser } from './helpers/mock-api';
import { MOCK_SETTINGS, MOCK_SUMMARY_USD } from './helpers/mock-data';

// The YTD card reads the summary's `ytd_return` (a percent). It used to fetch
// /risk_metrics for it even with Risk Analytics hidden — the summary field was
// declared and never set — so with Risk Analytics hidden it must not ask now.
for (const width of [390, 1280]) {
    for (const percent of [18.2, 0, -5.1]) {
        test(`YTD card reads the summary without Risk Analytics at ${width}px (${percent})`, async ({ page }) => {
            await loginAsMockUser(page, {
                initialSettings: { ...MOCK_SETTINGS, visible_items: ['ytdReturn'] },
            });
            await page.setViewportSize({ width, height: 844 });
            const metrics = { ...MOCK_SUMMARY_USD.metrics, ytd_return: percent };
            await page.route(/\/api\/summary(\/headline)?(\?|$)/, route =>
                route.fulfill({ json: { ...MOCK_SUMMARY_USD, metrics } }));
            const riskRequests: URL[] = [];
            await page.route(/\/api\/risk_metrics(\?|$)/, route => {
                riskRequests.push(new URL(route.request().url()));
                // A different figure: if the card ever showed it, it read the wrong source.
                return route.fulfill({ json: { 'YTD Return': 0.99 } });
            });
            await page.goto('/');
            const card = page.locator('.card-standard').filter({ has: page.getByText('YTD Return', { exact: true }) });
            await expect(card.getByText(`${percent.toFixed(2)}%`, { exact: true })).toBeVisible();
            expect(riskRequests).toHaveLength(0);
            await expect(page.getByRole('button', { name: 'Show detailed analytics', exact: true }))
                .toHaveAttribute('aria-expanded', 'false');
            if (percent > 0) await card.screenshot({ path: test.info().outputPath(`ytd-${width}.png`) });
        });
    }
}
