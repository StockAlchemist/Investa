import { test, expect } from '@playwright/test';
import { loginAsMockUser } from './helpers/mock-api';
import { MOCK_HOLDINGS } from './helpers/mock-data';

/**
 * The phone's holdings card — the web twin of `iosHoldingRow` in
 * macos_app HoldingsTableView.swift.
 *
 * Its header row used to draw both the symbol and the amount at text-xl bold,
 * which made it the loudest type on the phone and left the two ends of the row
 * no room to coexist: `SCBRMS&P500` ran under its own figure. These assertions
 * hold the row to the native's own rule — the identifier is shown in full and
 * the amount is what gets shorter.
 */
const PHONE = { width: 390, height: 844 };

/**
 * A fund ticker long enough to crowd its amount, with a big value beside it.
 *
 * The value is in the billions so the row overflows by ~26px rather than by a
 * hair. The amount is drawn in the platform's own UI face (`.tabular-nums` in
 * globals.css asks for SF Pro and takes whatever the machine has), so its width
 * is not the same on a Mac as on the Linux box CI runs on. At $61.7M the two
 * ends of this row missed each other by 1.5px here and met on CI, where the
 * fallback face is a shade narrower — the row stopped overflowing and the rule
 * under test correctly declined to fire. A margin wider than any face
 * substitution is what keeps this about the rule.
 */
const LONG_SYMBOL = {
    ...MOCK_HOLDINGS[0],
    Symbol: 'SCBRMS&P500',
    Quantity: 12345,
    'Market Value': 6170535542.18,
};

/** Opens Portfolio on a phone, where cards are what the tab opens in. */
async function openPortfolio(page: import('@playwright/test').Page) {
    await page.addInitScript(() => localStorage.setItem('investa_active_tab', 'allocation'));
    await page.setViewportSize(PHONE);
    await page.goto('/');
}

test.describe('Holdings cards on a phone', () => {
    test('cards are what the tab opens in, with the table a tap away', async ({ page }) => {
        await loginAsMockUser(page);
        await openPortfolio(page);

        // No toggle first: a phone table shows two of its columns and hides the
        // rest behind a sideways scroll, so it is not where the tab should land.
        await expect(page.locator('h3.text-\\[17px\\]').first()).toBeVisible();

        // The table is still there for anyone who wants it. Both views stay in
        // the DOM — the toggle switches which one is shown, so this is about
        // visibility, not presence.
        await page.locator('[title="Switch to Table View"]').first().click();
        await expect(page.locator('h3.text-\\[17px\\]').first()).toBeHidden();
        await page.locator('[title="Switch to Card View"]').first().click();
        await expect(page.locator('h3.text-\\[17px\\]').first()).toBeVisible();
    });

    test('the market value reads as money, not a bare number', async ({ page }) => {
        await loginAsMockUser(page);
        await openPortfolio(page);
        const amount = page.locator('h3.text-\\[17px\\]').first()
            .locator('xpath=ancestor::div[contains(@class,"justify-between")][1]')
            .locator('div.text-\\[15px\\]');
        // "Mkt Val" does not contain "Value", so the shared formatter used to
        // miss it and print `44,000` — no currency, and three decimals on a
        // figure that has none.
        await expect(amount).toHaveText(/^\$[\d,]+$/);
    });

    test('a long symbol is shown in full, and the amount shortens instead', async ({ page }) => {
        await loginAsMockUser(page, { initialHoldings: [LONG_SYMBOL, ...MOCK_HOLDINGS] });
        await openPortfolio(page);

        const symbol = page.locator('h3.text-\\[17px\\]').first();
        await expect(symbol).toHaveText('SCBRMS&P500');

        // Shown in full means measured in full: a clipped symbol still reports
        // its own text, so the assertion above alone would pass on `SCBRMS&P5…`.
        await expect.poll(async () => symbol.evaluate((el) => {
            const range = document.createRange();
            range.selectNodeContents(el);
            const natural = range.getBoundingClientRect().width;
            return natural - el.getBoundingClientRect().width;
        })).toBeLessThanOrEqual(0.5);

        const amount = symbol.locator('xpath=ancestor::div[contains(@class,"justify-between")][1]')
            .locator('div.text-\\[15px\\]');
        await expect(amount).toHaveText(/^\$[\d.]+[KMB]$/);

        // And it stays there: stepping down frees the width that forced the step,
        // so a rule that re-measured freely would flip back and oscillate.
        await page.waitForTimeout(1200);
        await expect(amount).toHaveText(/^\$[\d.]+[KMB]$/);
    });
});
