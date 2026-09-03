import { test, expect, Page } from '@playwright/test';
import { loginAsMockUser } from './helpers/mock-api';

/** A forecast with enough horizons for the card's chart and milestone table. */
function projectionPayload() {
    const horizons = Array.from({ length: 20 }, (_, i) => {
        const y = i + 1;
        const median = 1_000_000 * Math.pow(1.1, y);
        return {
            years: y,
            median_value: median,
            median_return_pct: (Math.pow(1.1, y) - 1) * 100,
            expected_value: median * 1.02,
            p10: median * 0.6,
            p25: median * 0.8,
            p75: median * 1.25,
            p90: median * 1.6,
        };
    });
    return {
        available: true,
        current_value: 1_000_000,
        annual_return_pct: 10,
        annual_volatility_pct: 18,
        currency: 'USD',
        horizons,
    };
}

/** A 5-year replay plus a two-row calibration table. */
function backtestPayload() {
    const points = Array.from({ length: 13 }, (_, i) => {
        const years = (i * 5) / 12;
        const median = 500_000 * Math.pow(1.1, years);
        return {
            date: `20${20 + Math.floor(i / 3)}-${String(((i * 5) % 12) + 1).padStart(2, '0')}-01`,
            years,
            actual: 500_000 * Math.pow(1.12, years),
            median,
            p10: median * 0.7,
            p25: median * 0.85,
            p75: median * 1.2,
            p90: median * 1.5,
        };
    });
    return {
        available: true,
        reason: null,
        currency: 'USD',
        market_timezone: 'America/New_York',
        history_years: 20.5,
        history_start: '2005-06-30',
        history_end: '2026-01-01',
        min_history_years: 5,
        horizons: [
            {
                years: 1, samples: 180, std_z: 1.05, in_band_pct: 78.5, below_p10_pct: 11.2,
                above_p90_pct: 10.3, mean_u: 0.49, median_actual_return_pct: 12.4,
                median_projected_return_pct: 10.1, verdict: 'calibrated',
            },
            {
                years: 5, samples: 120, std_z: 1.4, in_band_pct: 61.0, below_p10_pct: 22.0,
                above_p90_pct: 17.0, mean_u: 0.44, median_actual_return_pct: 41.0,
                median_projected_return_pct: 61.0, verdict: 'narrow',
            },
        ],
        replay: {
            anchor_date: points[0].date,
            years: 5,
            start_value: 500_000,
            indexed: false,
            fit_years: 15.5,
            annual_return_pct: 10,
            annual_volatility_pct: 18,
            final_actual: points[points.length - 1].actual,
            final_median: points[points.length - 1].median,
            final_p10: points[points.length - 1].p10,
            final_p90: points[points.length - 1].p90,
            outcome: 'inside',
            points,
        },
    };
}

async function mockProjection(page: Page) {
    const json = (body: unknown) => ({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
    // Registered after the shared mocks (and backtest after projection) so the
    // more specific route wins — Playwright matches the most recent first.
    await page.route(/\/api\/projection(\?|$)/, route => route.fulfill(json(projectionPayload())));
    await page.route(/\/api\/projection\/backtest/, route => route.fulfill(json(backtestPayload())));
}

test.describe('Projection backtest', () => {
    test('the Backtest tab replays the model against real history', async ({ page }) => {
        await loginAsMockUser(page);
        await mockProjection(page);
        await page.goto('/');

        const card = page.locator('div').filter({ hasText: /^Projected Value/ }).first();
        await expect(page.getByText('Assumed return')).toBeVisible();

        await page.getByRole('button', { name: 'Backtest' }).click();

        // The replay's verdict line, the calibration table and the per-horizon
        // verdicts all come from the backtest payload.
        await expect(page.getByText(/finished inside the 10–90% band/)).toBeVisible();
        await expect(page.getByRole('columnheader', { name: 'Inside 10–90%' })).toBeVisible();
        await expect(page.getByText('Well calibrated')).toBeVisible();
        await expect(page.getByText('Bands too narrow')).toBeVisible();
        // The forecast's assumptions row gives way to the backtest.
        await expect(page.getByText('Assumed return')).toBeHidden();
        await expect(card).toBeVisible();

        await page.getByRole('button', { name: 'Forecast' }).click();
        await expect(page.getByText('Assumed return')).toBeVisible();
    });
});
