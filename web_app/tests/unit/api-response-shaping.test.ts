import { describe, it, expect, vi, beforeEach } from 'vitest';

const GET = vi.fn();
const POST = vi.fn();
vi.mock('@/src/api/client', () => ({
    apiClient: {
        GET: (...a: unknown[]) => GET(...a),
        POST: (...a: unknown[]) => POST(...a),
    },
}));

import {
    fetchHoldings,
    fetchTransactions,
    fetchCapitalGains,
    fetchDividends,
    fetchHistory,
    fetchWatchlist,
    getWatchlists,
    fetchSymbolSearch,
    fetchMarketNews,
    fetchSettings,
    fetchRatios,
    fetchBuffettRankings,
    fetchTrendSignal,
    fetchStrategies,
    fetchStrategyAllocation,
} from '@/lib/api';

/** A response in the shape openapi-fetch returns. */
const ok = (data: unknown) => ({ data, error: undefined, response: { status: 200 } });

/**
 * Every fetcher reaches its return type through `as unknown as`, so the types
 * in lib/api.ts describe what the backend promises, not what arrived. A
 * component that maps over an array the type marks as *required* has been told
 * it is safe not to guard — so a payload missing it used to throw mid-render
 * and put the app error boundary over the whole screen.
 */
describe('whole-response lists', () => {
    beforeEach(() => { GET.mockReset(); POST.mockReset(); });

    // Rendering "no holdings" for a malformed response would read as a true
    // answer about someone's money, so these refuse rather than empty out.
    const listFetchers: [string, () => Promise<unknown>][] = [
        ['fetchHoldings', () => fetchHoldings()],
        ['fetchTransactions', () => fetchTransactions()],
        ['fetchCapitalGains', () => fetchCapitalGains()],
        ['fetchDividends', () => fetchDividends()],
        ['fetchHistory', () => fetchHistory()],
        ['fetchWatchlist', () => fetchWatchlist()],
        ['getWatchlists', () => getWatchlists()],
    ];

    for (const [name, call] of listFetchers) {
        it(`${name} throws rather than returning a false empty list`, async () => {
            for (const body of [null, undefined, {}, 'text', 42]) {
                GET.mockResolvedValue(ok(body));
                await expect(call()).rejects.toThrow();
            }
        });

        it(`${name} passes a real list straight through`, async () => {
            GET.mockResolvedValue(ok([{ a: 1 }, { a: 2 }]));
            await expect(call()).resolves.toHaveLength(2);
        });
    }

    // These three already swallow transport errors and answer []; a malformed
    // payload has to follow the same contract rather than start throwing.
    it('best-effort lookups stay best-effort', async () => {
        for (const body of [null, {}, 'text']) {
            GET.mockResolvedValue(ok(body));
            await expect(fetchSymbolSearch('aapl')).resolves.toEqual([]);
            await expect(fetchMarketNews()).resolves.toEqual([]);
        }
    });
});

describe('required list and map fields on object responses', () => {
    beforeEach(() => { GET.mockReset(); });

    it('fetchSettings fills every required list and map', async () => {
        GET.mockResolvedValue(ok({ display_currency: 'USD' }));

        const settings = await fetchSettings();

        expect(settings.user_excluded_symbols).toEqual([]);
        expect(settings.available_currencies).toEqual([]);
        // Object.entries() on these throws just as .map() would.
        expect(settings.account_groups).toEqual({});
        expect(settings.manual_overrides).toEqual({});
        expect(settings.user_symbol_map).toEqual({});
        expect(settings.account_currency_map).toEqual({});
        // Whatever did arrive is preserved.
        expect(settings.display_currency).toBe('USD');
    });

    it('fetchRatios guarantees historical', async () => {
        GET.mockResolvedValue(ok({ symbol: 'AAPL' }));
        await expect(fetchRatios('AAPL')).resolves.toMatchObject({ historical: [] });
    });

    it('fetchBuffettRankings guarantees rows', async () => {
        GET.mockResolvedValue(ok({ total: 0 }));
        await expect(fetchBuffettRankings({})).resolves.toMatchObject({ rows: [] });
    });

    it('fetchTrendSignal guarantees history', async () => {
        GET.mockResolvedValue(ok({ symbol: 'SPY' }));
        await expect(fetchTrendSignal()).resolves.toMatchObject({ history: [] });
    });

    it('object responses throw when the payload is not an object', async () => {
        for (const body of [null, 'text', 42, ['a', 'list']]) {
            GET.mockResolvedValue(ok(body));
            await expect(fetchSettings()).rejects.toThrow(/Failed to fetch settings/);
        }
    });
});

describe('the strategies path', () => {
    beforeEach(() => { GET.mockReset(); });

    it('guarantees the allocation arrays, one level deep', async () => {
        GET.mockResolvedValue(ok({ sleeves: [{ key: 'a' }, { key: 'b', positions: null }] }));

        const allocation = await fetchStrategyAllocation('x', 1000);

        expect(allocation.warnings).toEqual([]);
        expect(allocation.sleeves).toHaveLength(2);
        for (const sleeve of allocation.sleeves) {
            expect(Array.isArray(sleeve.positions)).toBe(true);
        }
    });

    it('drops list entries that are not objects, and non-string warnings', async () => {
        GET.mockResolvedValue(ok({ sleeves: [null, 'nope', { key: 'a' }], warnings: [1, 'real'] }));

        const allocation = await fetchStrategyAllocation('x', 1000);

        expect(allocation.sleeves).toHaveLength(1);
        expect(allocation.warnings).toEqual(['real']);
    });

    it('guarantees each strategy has risks and a backtest object', async () => {
        GET.mockResolvedValue(ok({ strategies: [{ id: 'a', name: 'A' }] }));

        const catalogue = await fetchStrategies();

        expect(catalogue.strategies[0].risks).toEqual([]);
        // StrategyCard reads strategy.backtest.cagr without guarding.
        expect(catalogue.strategies[0].backtest).toEqual({});
        expect(catalogue.default).toBe('');
    });
});
