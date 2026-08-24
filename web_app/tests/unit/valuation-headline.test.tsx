import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import ValuationHeadlineCard from '@/components/stock-detail/ValuationHeadlineCard';

/**
 * The headline replaced three stat cards with one that draws the price and the
 * blended value on a shared scale. What matters is that every figure the old
 * cards carried still reaches the reader — the price in particular, which no
 * longer has a card of its own.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any -- partial IntrinsicValueResponse
const base: any = {
    current_price: 341.83,
    average_intrinsic_value: 502.4,
    margin_of_safety_pct: 46.97,
    valuation_status: 'ok',
    valuation_confidence: 0.47,
    earnings_power_floor: 154.96,
    blend_profile: 'operating',
    range: { bear: 374.0, bull: 671.57 },
    model_weights: { dcf: 0.45, dcfo: 0.3, graham: 0.25 },
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- partial IntrinsicValueResponse
const renderCard = (iv: any, extra: Record<string, unknown> = {}) =>
    render(
        <ValuationHeadlineCard
            intrinsicValue={iv}
            displayAverage={iv.average_intrinsic_value}
            displayMos={iv.margin_of_safety_pct}
            hasAnyCustom={false}
            currency="USD"
            fxRate={1}
            {...extra}
        />,
    );

describe('ValuationHeadlineCard', () => {
    it('shows the blended value, the margin of safety, and the price it is measured against', () => {
        renderCard(base);
        expect(screen.getByText('Blended Intrinsic Value')).toBeTruthy();
        expect(screen.getByText('$502.40')).toBeTruthy();
        expect(screen.getByText('+47.0%')).toBeTruthy();
        expect(screen.getByText('Undervalued vs market')).toBeTruthy();
        expect(screen.getByText('$341.83')).toBeTruthy();
    });

    it('labels the model range beside the band it draws', () => {
        renderCard(base);
        expect(screen.getByText('Model range')).toBeTruthy();
        expect(screen.getByText('$374.00 – $671.57')).toBeTruthy();
    });

    it('carries the qualifiers that used to sit in the third card', () => {
        renderCard(base);
        expect(screen.getByText('47%')).toBeTruthy();
        expect(screen.getByText('No-growth floor')).toBeTruthy();
        expect(screen.getByText('$154.96')).toBeTruthy();
        expect(screen.getByText('3 models')).toBeTruthy();
    });

    it('marks an overvalued stock in the negative direction', () => {
        renderCard({ ...base, average_intrinsic_value: 250, margin_of_safety_pct: -26.9 });
        expect(screen.getByText('-26.9%')).toBeTruthy();
        expect(screen.getByText('Overvalued vs market')).toBeTruthy();
    });

    it('shows the default value beside an edited one, and drops the backend confidence', () => {
        render(
            <ValuationHeadlineCard
                intrinsicValue={base}
                displayAverage={430}
                displayMos={25.8}
                hasAnyCustom
                currency="USD"
                fxRate={1}
            />,
        );
        expect(screen.getByText('Custom Blended Value')).toBeTruthy();
        expect(screen.getByText('$430.00')).toBeTruthy();
        expect(screen.getByText(/Default \$502\.40/)).toBeTruthy();
        expect(screen.queryByText('Confidence')).toBeNull();
    });

    it('reports a refusal without inventing a scale for it', () => {
        render(
            <ValuationHeadlineCard
                intrinsicValue={{ ...base, average_intrinsic_value: null, margin_of_safety_pct: null }}
                displayAverage={null}
                displayMos={null}
                hasAnyCustom={false}
                currency="USD"
                fxRate={1}
            />,
        );
        expect(screen.getByText('Not valued')).toBeTruthy();
        expect(screen.getByText('No estimate available')).toBeTruthy();
        expect(screen.queryByText('Model range')).toBeNull();
    });

    it('converts every figure at the display rate', () => {
        renderCard(base, { fxRate: 32, currency: 'THB' });
        expect(screen.getByText('฿16,076.80')).toBeTruthy();
        expect(screen.getByText('฿10,938.56')).toBeTruthy();
    });
});
