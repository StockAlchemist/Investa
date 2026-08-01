import { describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import StockKeyMetrics from '@/components/StockKeyMetrics';
// From the shared catalogue, not a copy: the panel's verdicts have to be the
// same ones the heatmap paints.
import { metricTone } from '@/lib/metrics';

/** A company that reads well on every measure the panel groups. */
const metrics = {
    pe_ratio: 12,
    forward_pe: 10,
    peg_ratio: 0.9,
    ps_ratio: 1.2,
    pb_ratio: 1.5,
    p_fcf: 11,
    ev_ebitda: 7,
    ev_sales: 1.1,
    dividend_yield: 0.035,       // fraction on the wire, "3.50%" on screen
    eps_ttm: 8.71,
    eps_qoq: 0.27,
    eps_growth_3y: 0.068,
    eps_growth_5y: 0.178,
    eps_surprise: 0.0674,
    sales_ttm: 466_822_987_776,
    sales_qoq: 0.164,
    sales_growth_3y: 0.018,
    sales_growth_5y: 0.086,
    roa: 0.27,
    roe: 0.42,
    roic: 0.31,
    gross_margin: 0.486,
    operating_margin: 0.326,
    net_margin: 0.276,
    quick_ratio: 1.4,
    current_ratio: 2.1,
    debt_equity: 30,             // percent points, like the filed figure
    lt_debt_equity: 25,
    relative_volume: 2.24,
    float_short: 0.01,
    analyst_recom: 2.04,
    earnings_days: 12,
};

const panel = (group: string) =>
    screen.getByText(group).closest('div') as HTMLElement;

describe('StockKeyMetrics', () => {
    it('renders the four groups and none of the performance metrics', () => {
        render(<StockKeyMetrics metrics={metrics} />);
        for (const group of ['Valuation', 'Earnings & Sales', 'Profitability', 'Market']) {
            expect(screen.getByText(group)).toBeInTheDocument();
        }
        // Performance belongs to the Chart tab; eleven more rows here would be
        // the clutter this panel exists to avoid.
        expect(screen.queryByText('Year to Date')).not.toBeInTheDocument();
        expect(screen.queryByText('1-Year Performance')).not.toBeInTheDocument();
    });

    it('scales fractions to percentages and leaves ratios alone', () => {
        render(<StockKeyMetrics metrics={metrics} />);
        expect(within(panel('Profitability')).getByText('+27.00%')).toBeInTheDocument();   // ROA
        expect(within(panel('Valuation')).getByText('12.00')).toBeInTheDocument();          // P/E
        expect(within(panel('Earnings & Sales')).getByText('$8.71')).toBeInTheDocument();   // EPS TTM
        expect(within(panel('Earnings & Sales')).getByText('$466.8B')).toBeInTheDocument(); // Sales TTM
    });

    it('drops the sign from magnitudes, which have no direction to report', () => {
        render(<StockKeyMetrics metrics={metrics} />);
        // Dividend yield is dividends over price: "+3.50%" would read as a
        // change in the yield rather than as the yield.
        expect(within(panel('Valuation')).getByText('3.50%')).toBeInTheDocument();
        expect(within(panel('Valuation')).queryByText('+3.50%')).not.toBeInTheDocument();
    });

    it('counts days to the next report, and says so once it is past', () => {
        render(<StockKeyMetrics metrics={{ ...metrics, earnings_days: -3 }} />);
        expect(within(panel('Market')).getByText('3d ago')).toBeInTheDocument();
    });

    it('shows an absent reading as absent rather than dropping the row', () => {
        render(<StockKeyMetrics metrics={{ ...metrics, peg_ratio: null }} />);
        expect(screen.getByText('PEG')).toBeInTheDocument();
        expect(within(panel('Valuation')).getAllByText('–').length).toBe(1);
    });

    it('hides a group with nothing in it, and the whole block when empty', () => {
        // An ETF carries none of the company metrics.
        const { container } = render(<StockKeyMetrics metrics={{}} />);
        expect(container.firstChild).toBeNull();

        render(<StockKeyMetrics metrics={{ pe_ratio: 12, pb_ratio: 1.5 }} />);
        expect(screen.getByText('Valuation')).toBeInTheDocument();
        expect(screen.queryByText('Profitability')).not.toBeInTheDocument();
    });

    it('shows beta and average volume beside the other market readings', () => {
        render(<StockKeyMetrics metrics={metrics} beta={1.24} averageVolume={56_842_919} />);
        const market = panel('Market');
        expect(within(market).getByText('1.24')).toBeInTheDocument();
        // A share count, not money — it must not wear a currency sign.
        expect(within(market).getByText('56.8M')).toBeInTheDocument();
    });

    it('renders nothing at all when the backend sent no metric block', () => {
        const { container } = render(<StockKeyMetrics metrics={undefined} />);
        expect(container.firstChild).toBeNull();
    });
});

describe('metric tone', () => {
    const pe = { key: 'pe', mid: 25, clamp: 15, inverted: true };

    it('reads a cheap multiple as better and an expensive one as worse', () => {
        expect(metricTone(10, pe)).toContain('emerald');
        expect(metricTone(40, pe)).toContain('rose');
    });

    it('leaves a near-typical reading plain inside the panel dead zone', () => {
        // 23 is 2 points from the index median, well inside 15% of the clamp.
        expect(metricTone(23, pe, 0.15)).toBe('text-foreground');
        // The heatmap tooltip passes no dead zone and still takes a side.
        expect(metricTone(23, pe)).toContain('emerald');
    });

    it('never colours a missing reading as a verdict', () => {
        expect(metricTone(null, pe, 0.15)).toBe('text-muted-foreground');
    });
});
