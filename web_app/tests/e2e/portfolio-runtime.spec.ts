import { test, expect } from '@playwright/test';
import { loginAsMockUser } from './helpers/mock-api';
import { MOCK_SUMMARY_USD } from './helpers/mock-data';

test('default dashboard does not fetch hidden analytics or unrelated screens', async ({ page }) => {
    await loginAsMockUser(page);
    const requests: string[] = [];
    page.on('request', request => { if (request.url().includes('/api/')) requests.push(request.url()); });
    await page.goto('/');
    await expect(page.getByText('$125,450.00').first()).toBeVisible();
    // The old eight-second timer enabled every unrelated dataset. Allow that
    // window to elapse to guard against its reintroduction.
    await page.waitForTimeout(8500);
    for (const endpoint of ['transactions', 'capital_gains', 'dividends', 'asset_change', 'risk_metrics', 'projection', 'screener']) {
        expect(requests.some(path => new URL(path).pathname.endsWith(`/${endpoint}`))).toBe(false);
    }
    const histories = [...new Set(requests.filter(url => new URL(url).pathname.endsWith('/history')))];
    expect(histories.length).toBeLessThanOrEqual(3);
    await test.info().attach('dashboard-request-budget', { body: JSON.stringify({ requests: requests.length, historyRequests: histories.length }), contentType: 'application/json' });
});

test('failed holdings request displays Retry rather than an empty portfolio', async ({ page }) => {
    await loginAsMockUser(page);
    await page.route(/\/api\/holdings(\?|$)/, route => route.fulfill({ status: 500, json: { detail: 'Unavailable' } }));
    await page.goto('/?section=allocation&currency=USD');
    await expect(page.getByRole('alert').filter({ hasText: 'Could not load your holdings.' })).toContainText('Could not load your holdings.', { timeout: 15000 });
    await expect(page.getByRole('button', { name: 'Retry', exact: true })).toBeVisible();
    await expect(page.getByText('No portfolio data yet')).toHaveCount(0);
});

test('currency switch does not retain the old headline under the new currency', async ({ page }) => {
    await loginAsMockUser(page);
    await page.goto('/');
    await expect(page.getByText('$125,450.00').first()).toBeVisible();
    // Delay the new scope long enough to inspect the pending state.
    await page.route(/\/api\/summary(\/headline)?\?.*currency=THB/, async route => {
        await new Promise(resolve => setTimeout(resolve, 1500));
        await route.fulfill({ json: { metrics: { market_value: 4500000, day_change_display: 100, day_change_percent: 0.1 }, account_metrics: {} } });
    });
    await page.evaluate(() => {
        history.pushState(null, '', '/?section=performance&currency=THB');
        dispatchEvent(new PopStateEvent('popstate'));
    });
    await expect(page.getByText('$125,450.00', { exact: true })).toHaveCount(0);
});

test('browser Back restores the selected screen', async ({ page }) => {
    await loginAsMockUser(page);
    await page.goto('/?section=performance&currency=USD');
    await page.getByRole('button', { name: 'Portfolio', exact: true }).first().click();
    await expect(page).toHaveURL(/section=allocation/);
    await page.goBack();
    await expect(page).toHaveURL(/section=performance/);
    await expect(page.getByRole('button', { name: 'Dashboard', exact: true }).first()).toHaveAttribute('aria-current', 'page');
});


test('holdings presets and keyboard sorting work on the desktop', async ({ page }) => {
    await loginAsMockUser(page);
    await page.goto('/?section=allocation&currency=USD');
    await page.getByRole('combobox', { name: 'Holdings view preset' }).selectOption('Income');
    const symbol = page.getByRole('button', { name: 'Sort by Symbol', exact: true });
    await symbol.focus();
    await page.keyboard.press('Enter');
    await expect(symbol.locator('..')).toHaveAttribute('aria-sort', 'descending');
    await page.keyboard.press('Enter');
    await expect(symbol.locator('..')).toHaveAttribute('aria-sort', 'ascending');
    await page.keyboard.press('Alt+ArrowRight');
    await expect(page.getByRole('columnheader').nth(1)).toContainText('Symbol');
    await expect(page.getByRole('button', { name: 'Sort by Divs', exact: true })).toBeVisible();
    await page.screenshot({ path: test.info().outputPath('holdings-desktop.png'), fullPage: true });
});

test('phone dashboard keeps detailed analytics behind an explicit control', async ({ page }) => {
    await loginAsMockUser(page);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/');
    await expect(page.getByRole('button', { name: 'Show detailed analytics', exact: true })).toHaveAttribute('aria-expanded', 'false');
    await page.screenshot({ path: test.info().outputPath('dashboard-phone.png'), fullPage: true });
});


test('dashboard shows current figures while historical returns finish and polls for completion', async ({ page }) => {
    await loginAsMockUser(page);
    let summaryRequests = 0;
    const pending = {
        ...MOCK_SUMMARY_USD,
        metrics: { ...MOCK_SUMMARY_USD.metrics, cumulative_twr: null, annualized_twr: null },
        freshness: { calculated_at: 1791792000, refreshing: true },
    };
    await page.route(/\/api\/summary(\/headline)?(\?|$)/, route => {
        if (new URL(route.request().url()).pathname.endsWith('/headline')) {
            return route.fulfill({ json: { ...pending, freshness: { ...pending.freshness, refreshing: false } } });
        }
        summaryRequests += 1;
        return route.fulfill({ json: summaryRequests < 3 ? pending : {
            ...MOCK_SUMMARY_USD, freshness: { calculated_at: 1791792000, refreshing: false },
        } });
    });
    await page.goto('/');
    await expect(page.getByText('$125,450.00').first()).toBeVisible();
    await expect(page.getByText('No portfolio data yet')).toHaveCount(0);
    await expect(page.getByText('32.10%', { exact: true }).first()).toBeVisible({ timeout: 15000 });
    expect(summaryRequests).toBeGreaterThanOrEqual(3);
    await expect(page.getByRole('alert').filter({ hasText: 'Could not load your portfolio.' })).toHaveCount(0);
});
