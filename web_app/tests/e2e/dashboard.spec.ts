import { test, expect } from '@playwright/test';

test.describe('Dashboard UI', () => {
    test('has title and loads successfully', async ({ page }) => {
        await page.goto('/');

        // Expect a title "to contain" a substring. Next.js default is often Investa or Create Next App
        // We will check for some generic element on your dashboard to verify it loads.
        await expect(page).toHaveTitle(/Investa|React/i);

        // Check if the overall structure rendered (assuming basic page wrapper or typical div)
        const body = page.locator('body');
        await expect(body).toBeVisible();
    });
});
