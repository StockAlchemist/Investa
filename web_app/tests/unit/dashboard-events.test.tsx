import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import DashboardEvents from '@/components/dashboard/DashboardEvents';
import type { DividendEvent, EarningsEvent } from '@/lib/api';

/**
 * The dashboard Events panel.
 *
 * The case worth pinning down is the earnings report that has already happened.
 * A scheduled report leaves the panel the moment it is in the past, which used
 * to mean a company reported and the dashboard simply went quiet about it. A
 * `reported` row is the resolution of that date: it looks *backwards*, carries
 * what was actually printed, and says so plainly in the gap before the figures
 * are published.
 */

vi.mock('@/context/StockModalContext', () => ({
    useStockModal: () => ({ openStockDetail: vi.fn() }),
}));

// Renders no text of its own — the ticker must stay a unique match in the row.
vi.mock('@/components/StockIcon', () => ({
    default: ({ symbol }: { symbol: string }) => <span data-testid={`icon-${symbol}`} />,
}));

// A Tuesday afternoon in New York; every date below is relative to it.
const NOW = new Date('2026-07-21T18:00:00Z');

const earnings = (overrides: Partial<EarningsEvent> = {}): EarningsEvent => ({
    symbol: 'NFLX',
    name: 'Netflix, Inc.',
    earnings_date: '2026-07-20',
    status: 'reported',
    market_timezone: 'America/New_York',
    ...overrides,
});

const dividend = (): DividendEvent =>
    ({
        symbol: 'KO',
        name: 'Coca-Cola',
        dividend_date: '2026-07-24',
        ex_dividend_date: '2026-07-22',
        amount: 42,
        status: 'confirmed',
        market_timezone: 'America/New_York',
    } as DividendEvent);

beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(NOW);
});

afterEach(() => {
    vi.useRealTimers();
});

describe('DashboardEvents — reported quarters', () => {
    it('keeps a report on the timeline after the date has passed', () => {
        render(
            <DashboardEvents
                events={[]}
                earnings={[earnings({ eps_actual: 2.1, eps_estimate: 1.95, surprise_pct: 7.69 })]}
                currency="USD"
            />,
        );
        expect(screen.getByText('NFLX')).toBeInTheDocument();
        expect(screen.getByText('reported')).toBeInTheDocument();
        expect(screen.getByText('1d ago')).toBeInTheDocument();
    });

    it('shows what was printed and the beat against consensus', () => {
        render(
            <DashboardEvents
                events={[]}
                earnings={[earnings({ eps_actual: 2.1, eps_estimate: 1.95, surprise_pct: 7.69 })]}
                currency="USD"
            />,
        );
        expect(screen.getByText('2.10 EPS')).toBeInTheDocument();
        expect(screen.getByText('+7.7%')).toBeInTheDocument();
        expect(screen.getByTitle('Reported EPS 2.10 vs 1.95 expected')).toBeInTheDocument();
    });

    it('marks a miss as negative', () => {
        render(
            <DashboardEvents
                events={[]}
                earnings={[earnings({ eps_actual: 0.8, eps_estimate: 1.0, surprise_pct: -20 })]}
                currency="USD"
            />,
        );
        expect(screen.getByText('-20.0%')).toBeInTheDocument();
    });

    it('says the report happened even before the figures are published', () => {
        render(<DashboardEvents events={[]} earnings={[earnings({ eps_actual: null })]} currency="USD" />, );
        expect(screen.getByText('Reported')).toBeInTheDocument();
        expect(screen.getByTitle('Reported — figures not published yet')).toBeInTheDocument();
    });

    it('drops a reported row older than the lookback the backend filters to', () => {
        render(
            <DashboardEvents
                events={[]}
                earnings={[earnings({ earnings_date: '2026-07-10', eps_actual: 2.1 })]}
                currency="USD"
            />,
        );
        expect(screen.queryByText('NFLX')).not.toBeInTheDocument();
    });

    it('still drops a *scheduled* earnings date that has gone by', () => {
        render(
            <DashboardEvents
                events={[]}
                earnings={[earnings({ status: 'confirmed', eps_estimate: 1.95 })]}
                currency="USD"
            />,
        );
        expect(screen.queryByText('NFLX')).not.toBeInTheDocument();
    });

    it('interleaves a reported quarter with upcoming dividends', () => {
        render(
            <DashboardEvents
                events={[dividend()]}
                earnings={[
                    earnings({ eps_actual: 2.1, eps_estimate: 1.95, surprise_pct: 7.69 }),
                    earnings({ symbol: 'MSFT', name: 'Microsoft', status: 'confirmed', earnings_date: '2026-07-29', eps_estimate: 3.4 }),
                ]}
                currency="USD"
            />,
        );
        expect(screen.getByText('NFLX')).toBeInTheDocument();
        expect(screen.getByText('KO')).toBeInTheDocument();
        expect(screen.getByText('MSFT')).toBeInTheDocument();
        expect(screen.getByText('3.40 EPS')).toBeInTheDocument();
    });
});
