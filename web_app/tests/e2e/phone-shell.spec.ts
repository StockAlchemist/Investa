import { test, expect } from '@playwright/test';
import { loginAsMockUser } from './helpers/mock-api';

/**
 * The phone shell is the web twin of the native iPhone app's
 * `MainView.phoneShell`: a navigation bar, a control bar, and a tab bar holding
 * the first four sections with the rest behind More.
 *
 * These assertions are about that anatomy, not about pixels. They exist because
 * the shell they replaced — a hamburger drawer over a three-item bar whose
 * "Home" stood for nine tabs at once — was swapped wholesale, and the drawer's
 * absence is easy to reintroduce by accident.
 */
const PHONE = { width: 390, height: 844 };
const DESKTOP = { width: 1280, height: 800 };

/** The four the native tab bar carries, in `AppSection.allCases.prefix(4)` order. */
const TAB_BAR = ['Dashboard', 'Portfolio', 'Performance', 'Transactions'] as const;

test.describe('Phone shell', () => {
    test.beforeEach(async ({ page }) => {
        await loginAsMockUser(page);
        await page.setViewportSize(PHONE);
    });

    test('the tab bar carries the four main sections plus More', async ({ page }) => {
        await page.goto('/');
        const tabBar = page.locator('nav.ios-bar');
        await expect(tabBar).toBeVisible();
        for (const label of TAB_BAR) {
            await expect(tabBar.getByRole('button', { name: label, exact: true })).toBeVisible();
        }
        await expect(tabBar.getByRole('button', { name: 'More', exact: true })).toBeVisible();
    });

    test('More lists every section the tab bar has no room for', async ({ page }) => {
        await page.goto('/');
        await page.getByLabel('More').click();
        for (const label of ['Income', 'Capital Gains', 'Screener', 'Rankings',
                             'Strategies', 'Watchlist', 'Markets', 'AI Insights', 'Settings']) {
            await expect(page.getByRole('button', { name: label, exact: true }).first()).toBeVisible();
        }
        // Sections already in the tab bar are not repeated on More.
        for (const label of TAB_BAR) {
            await expect(page.locator('main').getByRole('button', { name: label, exact: true })).toHaveCount(0);
        }
    });

    test('More stays the selected tab while you are inside a section it holds', async ({ page }) => {
        await page.goto('/');
        await page.getByLabel('More').click();
        await page.getByRole('button', { name: 'Watchlist', exact: true }).first().click();
        const tabBar = page.locator('nav.ios-bar');
        await expect(tabBar.getByRole('button', { name: 'More', exact: true })).toHaveAttribute('aria-current', 'page');
        await expect(tabBar.getByRole('button', { name: 'Dashboard', exact: true })).not.toHaveAttribute('aria-current', 'page');
    });

    test('the control bar carries the account, layout, closed, refresh, currency and settings controls', async ({ page }) => {
        await page.goto('/');
        const bar = page.getByRole('toolbar', { name: 'Portfolio controls' });
        for (const label of ['Accounts', 'Dashboard Elements', 'Refresh', 'Display currency', 'Settings']) {
            await expect(bar.getByLabel(label)).toBeVisible();
        }
        // The show-closed toggle names the action it performs, so it renames itself.
        await expect(bar.getByLabel(/closed positions/i)).toBeAttached();
    });

    test('a focused search takes over the control bar', async ({ page }) => {
        await page.goto('/');
        const bar = page.getByRole('toolbar', { name: 'Portfolio controls' });
        await expect(bar.getByLabel('Accounts')).toBeVisible();
        await bar.locator('input[placeholder="Search symbol…"]').click();
        // Every sibling control hides while the field is active — the iOS pattern,
        // and what stops the expanded field shoving them off-screen.
        await expect(bar.getByLabel('Accounts')).toHaveCount(0);
        await expect(bar.getByLabel('Display currency')).toHaveCount(0);
    });

    test('the desktop keeps its sidebar and header, not the phone shell', async ({ page }) => {
        await page.setViewportSize(DESKTOP);
        await page.goto('/');
        await expect(page.locator('nav.ios-bar')).toBeHidden();
        await expect(page.getByRole('heading', { name: 'Dashboard', exact: true }).first()).toBeVisible();
    });
});
