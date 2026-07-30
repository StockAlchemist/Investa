import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import MarketTrendPanel from '@/components/dashboard/MarketTrendPanel';
import { fetchTrendSignal, type TrendSignal } from '@/lib/api';

/**
 * The market-trend panel.
 *
 * What is worth pinning down is not the layout but the honesty of the reading.
 * The panel shows several markets at once precisely because they disagree, so
 * the cases here are: each market is named and read independently; a mid-month
 * price is presented as a preview and never as the current state; a market whose
 * prices are missing is named as missing rather than dropped (an absent row
 * would look like a market that agrees); and a half-populated payload cannot
 * take the dashboard down with it.
 */

vi.mock('@/lib/api', async (importOriginal) => {
    const actual = await importOriginal<typeof import('@/lib/api')>();
    return { ...actual, fetchTrendSignal: vi.fn() };
});

const mockFetch = vi.mocked(fetchTrendSignal);

/** Twelve month-ends rising to `end`, the last three with an average to compare. */
const history = (end: number): TrendSignal['history'] =>
    Array.from({ length: 12 }, (_, i) => {
        const close = end - (11 - i) * 5;
        return { date: `2026-${String(i + 1).padStart(2, '0')}-28`, close, sma: i >= 9 ? close - 8 : null };
    });

/** A reading shaped like the backend's payload. */
const signal = (overrides: Partial<TrendSignal> = {}): TrendSignal => ({
    advisory_only: true,
    signal_symbol: 'SPY',
    signal_name: 'S&P 500',
    market_timezone: 'America/New_York',
    state: 'in',
    sma_months: 10,
    decision_date: '2026-06-30',
    decision_close: 620,
    sma: 588,
    governs_month: '2026-07',
    provisional_state: 'in',
    provisional_sma: 592,
    latest_close: 625,
    latest_date: '2026-07-29',
    flip_close: 612,
    distance_pct: 2.1,
    would_flip: false,
    next_decision_date: '2026-07-31',
    history: history(620),
    ...overrides,
});

const nasdaq = (overrides: Partial<TrendSignal> = {}): TrendSignal =>
    signal({ signal_symbol: 'QQQ', signal_name: 'NASDAQ 100', history: history(500), ...overrides });

/**
 * A text matcher that spans nested elements, resolving to the innermost one.
 *
 * The panel emphasises market names and prices inline, so a sentence like
 * "NASDAQ 100 unavailable" is split across a nested span and the default
 * matcher — which only sees an element's own text nodes — never sees the whole
 * sentence.
 */
const spanning = (pattern: RegExp) => (_: string, element: Element | null) =>
    !!element
    && pattern.test(element.textContent ?? '')
    && !Array.from(element.children).some(child => pattern.test(child.textContent ?? ''));

/** The panel retries a failed fetch once, so failure paths need room to settle. */
const SETTLE = { timeout: 5000 };

function renderPanel() {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return render(
        <QueryClientProvider client={client}>
            <MarketTrendPanel />
        </QueryClientProvider>,
    );
}

beforeEach(() => {
    mockFetch.mockReset();
});

describe('MarketTrendPanel', () => {
    it('reads every market independently, and shows where they disagree', async () => {
        mockFetch.mockImplementation(async (symbol) =>
            symbol === 'QQQ'
                ? nasdaq({ state: 'out', decision_close: 500, sma: 530, flip_close: 524, distance_pct: -1.4 })
                : signal(),
        );

        renderPanel();

        expect(await screen.findByText('S&P 500')).toBeInTheDocument();
        expect(screen.getByText('NASDAQ 100')).toBeInTheDocument();
        // One index being up says nothing about the other — which is the reason
        // both are shown rather than one standing in for "the market".
        expect(screen.getByText('Uptrend')).toBeInTheDocument();
        expect(screen.getByText('Downtrend')).toBeInTheDocument();
        // The margin beside each state comes from the same comparison as the
        // state itself: the deciding close against the average it was read
        // against, never the mid-month price.
        expect(screen.getByText('+5.4%')).toBeInTheDocument();      // 620 vs 588
        expect(screen.getByText('-5.7%')).toBeInTheDocument();      // 500 vs 530
        // And the panel never tells the reader to act on any of it.
        expect(screen.getByText(/no strategy acts on these/i)).toBeInTheDocument();
    });

    it('states the shared timing once, not per market', async () => {
        mockFetch.mockImplementation(async (symbol) => (symbol === 'QQQ' ? nasdaq() : signal()));

        renderPanel();

        await screen.findByText('S&P 500');
        // Both readings were set at the same month-end and are next checked on
        // the same date, so the panel says so once in the footer — the date
        // itself is formatted for the viewer's locale, hence the loose match.
        const footers = screen.getAllByText(spanning(/Set at the .+ close, governing July 2026/));
        expect(footers).toHaveLength(1);
    });

    it('presents a diverging mid-month price as a preview, not the current state', async () => {
        mockFetch.mockImplementation(async (symbol) =>
            symbol === 'SPY'
                ? signal({ provisional_state: 'out', would_flip: true, latest_close: 580, distance_pct: -5.2 })
                : nasdaq(),
        );

        renderPanel();

        // The active reading is unchanged by the running month...
        expect(await screen.findByText('S&P 500')).toBeInTheDocument();
        expect(screen.getAllByText('Uptrend')).toHaveLength(2);
        // ...and the provisional one is phrased as what a future close would do.
        expect(screen.getByText(/On track to turn down/)).toBeInTheDocument();
        // The unflipped market states its threshold instead.
        expect(screen.getByText(/Turns down below/)).toBeInTheDocument();
    });

    it('names a market whose prices are unavailable rather than dropping it', async () => {
        mockFetch.mockImplementation(async (symbol) => {
            if (symbol === 'QQQ') throw new Error('Failed to fetch the trend signal');
            return signal();
        });

        renderPanel();

        expect(await screen.findByText('S&P 500', {}, SETTLE)).toBeInTheDocument();
        expect(screen.getByText(spanning(/NASDAQ 100 unavailable/))).toBeInTheDocument();
        // The surviving market still carries the shared footer.
        expect(screen.getByText(spanning(/no strategy acts on these/i))).toBeInTheDocument();
    });

    it('treats a half-populated reading as unavailable instead of throwing', async () => {
        // The panel shares the dashboard's top section: a missing field that
        // threw during render would take the whole page down with it.
        mockFetch.mockImplementation(async (symbol) =>
            symbol === 'QQQ' ? ({ signal_symbol: 'QQQ' } as unknown as TrendSignal) : signal(),
        );

        renderPanel();

        expect(await screen.findByText('S&P 500')).toBeInTheDocument();
        expect(screen.getByText(spanning(/NASDAQ 100 unavailable/))).toBeInTheDocument();
    });

    it('reports an unreadable panel rather than an empty one', async () => {
        mockFetch.mockRejectedValue(new Error('Failed to fetch the trend signal'));

        renderPanel();

        expect(
            await screen.findByText(spanning(/Market trend unavailable/), {}, SETTLE),
        ).toBeInTheDocument();
    });
});
