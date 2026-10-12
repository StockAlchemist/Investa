import { expect, test } from '@playwright/test';
import { loginAsMockUser } from './helpers/mock-api';
import { MOCK_TRANSACTIONS } from './helpers/mock-data';

for (const mode of ['all', 'selected', 'row'] as const) {
    test(`IBKR ${mode} approval uses the chosen row accounts`, async ({ page }) => {
        await loginAsMockUser(page);
        let pending = [
            { ...MOCK_TRANSACTIONS[0], id: 701, Symbol: 'ADBE', Account: 'IBKR' },
            { ...MOCK_TRANSACTIONS[0], id: 702, Symbol: 'MA', Account: 'IBKR' },
        ];
        let approved: unknown;
        await page.route(/\/api\/sync\/ibkr\/pending(\?|$)/, route => route.fulfill({ json: pending }));
        await page.route(/\/api\/sync\/ibkr\/approve(\?|$)/, route => {
            const body = route.request().postDataJSON() as { ids: number[]; accounts: Record<string, string> };
            approved = body;
            pending = pending.filter(tx => !body.ids.includes(tx.id));
            return route.fulfill({ json: { status: 'success', count: body.ids.length } });
        });
        await page.goto('/?section=transactions&currency=USD&accounts=Taxable');
        const selector = page.getByRole('combobox', { name: 'Account for ADBE pending transaction 701' });
        await expect(selector).toHaveValue('IBKR');
        await selector.selectOption('Roth IRA');
        await page.getByRole('combobox', { name: 'Account for MA pending transaction 702' }).selectOption('Taxable');
        if (mode === 'all') {
            await page.screenshot({ path: test.info().outputPath('ibkr-account-selectors.png'), fullPage: true });
            await page.getByRole('button', { name: 'Approve All', exact: true }).click();
        } else if (mode === 'selected') {
            await page.getByRole('checkbox', { name: 'Select pending transaction 701', exact: true }).check();
            await page.getByRole('button', { name: 'Approve Selected (1)', exact: true }).click();
        } else {
            await selector.locator('xpath=ancestor::tr').getByRole('button', { name: 'Approve', exact: true }).click();
        }
        await expect.poll(() => approved).toEqual(mode === 'all'
            ? { ids: [701, 702], accounts: { '701': 'Roth IRA', '702': 'Taxable' } }
            : { ids: [701], accounts: { '701': 'Roth IRA' } });
        await expect(selector).toHaveCount(0);
        if (mode !== 'all') await expect(page.getByRole('combobox', { name: 'Account for MA pending transaction 702' })).toHaveValue('Taxable');
    });
}

test('failed IBKR approval retains the account choice for retry', async ({ page }) => {
    await loginAsMockUser(page);
    const pending = [{ ...MOCK_TRANSACTIONS[0], id: 701, Symbol: 'ADBE', Account: 'IBKR' }];
    await page.route(/\/api\/sync\/ibkr\/pending(\?|$)/, route => route.fulfill({ json: pending }));
    await page.route(/\/api\/sync\/ibkr\/approve(\?|$)/, route => route.fulfill({ status: 500, json: { detail: 'Unavailable' } }));
    page.on('dialog', dialog => dialog.accept());
    await page.goto('/?section=transactions&currency=USD');
    const selector = page.getByRole('combobox', { name: 'Account for ADBE pending transaction 701' });
    await selector.selectOption('Roth IRA');
    await page.getByRole('button', { name: 'Approve All', exact: true }).click();
    await expect(selector).toBeEnabled();
    await expect(selector).toHaveValue('Roth IRA');
});
