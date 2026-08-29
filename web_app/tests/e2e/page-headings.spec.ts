import { test, expect } from '@playwright/test';
import { loginAsMockUser } from './helpers/mock-api';

/**
 * Every screen must name itself at every width.
 *
 * PageHeader's <h1> is `hidden md:block`, so below md it contributes nothing —
 * the in-page headings are what name the screen there. They were once removed
 * as "duplicates", which they only are at md and up, which left five screens
 * anonymous on a phone.
 */
const PHONE = { width: 390, height: 844 };
const DESKTOP = { width: 1280, height: 800 };

/** Screens reached from the nav list, which the hamburger hides below md. */
const NAV_SCREENS = ['Screener', 'Rankings', 'Strategies', 'AI Insights'] as const;

test.describe('Page headings', () => {
    test.beforeEach(async ({ page }) => {
        await loginAsMockUser(page);
    });

    for (const name of NAV_SCREENS) {
        test(`names "${name}" on a phone`, async ({ page }) => {
            await page.setViewportSize(PHONE);
            await page.goto('/');
            await page.getByLabel('Open Navigation Menu').click();
            await page.getByRole('button', { name, exact: true }).first().click();
            await expect(
                page.getByRole('heading', { name, exact: true }).first()
            ).toBeVisible();
        });

        test(`names "${name}" on a desktop`, async ({ page }) => {
            await page.setViewportSize(DESKTOP);
            await page.goto('/');
            await page.getByRole('button', { name, exact: true }).first().click();
            await expect(
                page.getByRole('heading', { name, exact: true }).first()
            ).toBeVisible();
        });
    }

    // Settings has no nav row of its own: on a phone it is reached by tapping
    // the user at the foot of the drawer, on a desktop from the sidebar.
    test('names "Settings" on a phone', async ({ page }) => {
        await page.setViewportSize(PHONE);
        await page.goto('/');
        await page.getByLabel('Open Navigation Menu').click();
        await page.getByText('E2E Test User', { exact: true })
            .filter({ visible: true })
            .click();
        await expect(
            page.getByRole('heading', { name: 'Settings', exact: true }).first()
        ).toBeVisible();
    });

    test('names "Settings" on a desktop', async ({ page }) => {
        await page.setViewportSize(DESKTOP);
        await page.goto('/');
        await page.getByRole('button', { name: 'Settings', exact: true }).first().click();
        await expect(
            page.getByRole('heading', { name: 'Settings', exact: true }).first()
        ).toBeVisible();
    });

    // This route renders no PageHeader at any width, so it must carry its own.
    for (const [label, size] of [['phone', PHONE], ['desktop', DESKTOP]] as const) {
        test(`the standalone /screener route has a heading on a ${label}`, async ({ page }) => {
            await page.setViewportSize(size);
            await page.goto('/screener');
            await expect(
                page.getByRole('heading', { name: 'Screener', exact: true }).first()
            ).toBeVisible();
        });
    }
});
