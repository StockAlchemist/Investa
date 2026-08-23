import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ValuationTab } from '@/components/stock-detail/tabs/ValuationTab';

/**
 * Twelve model cards stacked under one another buried the answer the tab exists
 * to give. The selector opens on the backend's best-fit method and keeps every
 * other model one choice away.
 */
const intrinsicValue = {
    current_price: 400,
    average_intrinsic_value: 420,
    margin_of_safety_pct: 5,
    recommended_method: { method_key: 'dcf', name: 'Discounted Free Cash Flow', rationale: 'Stable free cash flow.' },
    models: {
        dcf: { model: 'Discounted Free Cash Flow', intrinsic_value: 392.89, parameters: {} },
        graham: { model: "Graham's Revised Formula", intrinsic_value: 671.56, parameters: {} },
        ddm: { model: 'Dividend Discount', intrinsic_value: 14.48, parameters: {} },
    },
};

const renderTab = (iv: Record<string, unknown> = intrinsicValue) =>
    render(
        <ValuationTab
            symbol="TEST"
            intrinsicValue={iv}
            fundamentals={{ currency: 'USD', sector: 'Technology' }}
            currency="USD"
            fxRate={1}
        />,
    );

const selector = () => screen.getByLabelText('Valuation method') as HTMLSelectElement;

const MODEL_HEADINGS = ['Discounted Free Cash Flow', "Graham's Revised Formula", 'Dividend Discount'];

/** Model cards title themselves with an h3, sharing that level with the spectrum
 *  chart and appending badges ("Primary") to the text — so match by model name. */
const visibleModelTitles = () =>
    screen
        .queryAllByRole('heading', { level: 3 })
        .map((h) => MODEL_HEADINGS.find((m) => (h.textContent ?? '').startsWith(m)))
        .filter((m): m is string => m !== undefined);

describe('ValuationTab method selector', () => {
    it('shows only the best-fit model by default', () => {
        renderTab();

        expect(selector().value).toBe('__best_fit__');
        const titles = visibleModelTitles();
        expect(titles).toContain('Discounted Free Cash Flow');
        expect(titles).not.toContain("Graham's Revised Formula");
    });

    it('offers only the models the backend returned', () => {
        renderTab();

        const values = Array.from(selector().querySelectorAll('option')).map((o) => o.value);
        expect(values).toEqual(['__best_fit__', '__all__', 'dcf', 'ddm', 'graham']);
    });

    it('reveals every model when "All Methods" is chosen', async () => {
        renderTab();

        await userEvent.selectOptions(selector(), '__all__');

        const titles = visibleModelTitles();
        expect(titles).toContain('Discounted Free Cash Flow');
        expect(titles).toContain("Graham's Revised Formula");
        expect(titles).toContain('Dividend Discount');
    });

    it('switches to a single non-recommended model on request', async () => {
        renderTab();

        await userEvent.selectOptions(selector(), 'graham');

        const titles = visibleModelTitles();
        expect(titles).toEqual(["Graham's Revised Formula"]);
    });

    it('falls back to showing every model when there is no best fit', () => {
        renderTab({ ...intrinsicValue, recommended_method: undefined });

        const titles = visibleModelTitles();
        expect(titles).toContain('Discounted Free Cash Flow');
        expect(titles).toContain("Graham's Revised Formula");
    });
});
