import { describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { ValuationTab } from '@/components/stock-detail/tabs/ValuationTab';

/**
 * The backend deliberately reports doubt about a blended intrinsic value rather
 * than resolving it by fiat, and the native clients show it. The web app used to
 * render `valuation_note` only for outright refusals, so a headline built from
 * models three times apart — or a value that was clamped before display — looked
 * exactly as trustworthy as one the models agreed on.
 */
const baseIntrinsic = {
    current_price: 400,
    average_intrinsic_value: 400.8,
    margin_of_safety_pct: 0.2,
    models: {
        dcf: { model: 'Discounted Free Cash Flow', intrinsic_value: 392.89, parameters: {} },
        graham: { model: "Graham's Revised Formula", intrinsic_value: 671.56, parameters: {} },
        ddm: { model: 'Dividend Discount', intrinsic_value: 14.48, parameters: {} },
    },
    model_weights: { dcf: 0.5, graham: 0.3, ddm: 0.2 },
};

/** The caveat text and its per-model chips share one container; the same figures
 *  also appear in the model cards further down, so scope assertions to the banner. */
const banner = (title: string) => within(screen.getByText(title).closest('div')!);

const renderTab = (intrinsicValue: Record<string, unknown>, fxRate = 1, currency = 'USD') =>
    render(
        <ValuationTab
            symbol="TEST"
            intrinsicValue={intrinsicValue}
            fundamentals={{ currency: 'USD', sector: 'Technology' }}
            currency={currency}
            fxRate={fxRate}
        />,
    );

describe('ValuationTab blend caveats', () => {
    it('warns when the contributing models disagree', () => {
        renderTab({
            ...baseIntrinsic,
            valuation_status: 'low_confidence',
            model_spread_pct: 164,
            valuation_note:
                'Models disagree by 164% of the blended value (dcf=392.89, graham=671.56, ddm=14.48).',
        });

        const b = banner('Models disagree');
        // The raw parenthetical is native-currency and is replaced by chips below.
        expect(b.getByText('Models disagree by 164% of the blended value.')).toBeInTheDocument();
        expect(b.getByText(/\$671\.56/)).toBeInTheDocument();
        expect(b.getByText(/\$14\.48/)).toBeInTheDocument();
    });

    it('converts the per-model figures into the displayed currency', () => {
        renderTab(
            {
                ...baseIntrinsic,
                valuation_status: 'low_confidence',
                valuation_note:
                    'Models disagree by 164% of the blended value (dcf=392.89, graham=671.56, ddm=14.48).',
            },
            32,
            'THB',
        );

        const b = banner('Models disagree');
        expect(b.getByText(/฿21,489/)).toBeInTheDocument();
        expect(b.queryByText(/671\.56/)).not.toBeInTheDocument();
    });

    it('says so when the displayed value was clamped, not merely blended', () => {
        renderTab({
            ...baseIntrinsic,
            valuation_status: 'clamped',
            valuation_note:
                'Model output 6.3x price is outside the credible band; clamped to 5.0x. Treat as low confidence.',
        });

        expect(
            banner('Output outside credible range').getByText(/clamped to 5\.0x/),
        ).toBeInTheDocument();
    });

    it('stays quiet when the models agree', () => {
        renderTab({ ...baseIntrinsic, valuation_status: 'ok' });

        expect(screen.queryByText('Models disagree')).not.toBeInTheDocument();
        expect(screen.queryByText('Output outside credible range')).not.toBeInTheDocument();
    });
});
