import { test, expect } from '@playwright/test';
import { loginAsMockUser } from './helpers/mock-api';

/**
 * `fetchStrategyAllocation` casts its response through `as unknown as`, so the
 * non-optional `warnings`/`sleeves` on `StrategyAllocation` are a claim about
 * the backend, not a guarantee about the bytes that arrive. A payload missing
 * them used to throw inside the render and take the whole screen into the app
 * error boundary; the allocation section should simply not draw instead.
 */
test.describe('Strategies resilience', () => {
    test.beforeEach(async ({ page }) => {
        await loginAsMockUser(page);
    });

    for (const [label, body] of [
        ['an empty object', '{}'],
        ['a payload with no sleeves', '{"warnings":[]}'],
        ['a payload with no warnings', '{"sleeves":[]}'],
    ] as const) {
        test(`survives ${label} from the allocation endpoint`, async ({ page }) => {
            await page.route(/\/api\/strategies\/[^/]+\/allocation/, (route) =>
                route.fulfill({ status: 200, contentType: 'application/json', body })
            );

            await page.goto('/');
            await page.getByRole('button', { name: 'Strategies', exact: true }).first().click();

            // The screen still renders its own content...
            await expect(
                page.getByRole('heading', { name: 'Strategies', exact: true }).first()
            ).toBeVisible();
            await expect(page.getByText('The rule', { exact: true })).toBeVisible();
            // ...rather than the app error boundary.
            await expect(page.getByText('Something went wrong')).toHaveCount(0);
        });
    }
});
