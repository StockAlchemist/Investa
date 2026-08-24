import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import RebalanceHelper from '@/components/portfolio/RebalanceHelper';
import type { Holding } from '@/lib/api';

/**
 * The Rebalance Helper must not price trades against a portfolio it cannot classify.
 *
 * "Unknown" is not an asset class — it is the bucket a holding falls into when its
 * metadata fetch failed. When a failed fetch parked GOOG, AMZN, ASML, NVDA and PLTR
 * there, the card read "Sell $1,103,057 of Unknown / Buy $1,105,034 of EQUITY": a
 * ~$1.1M round trip into the same shares, plus the capital gains. Every other
 * bucket's *current* weight is understated by exactly the unclassified share, so no
 * row on the card can be trusted while it is material.
 *
 * The line worth holding: the block is on the unclassified *share*, not on the
 * presence of an Unknown row. A trace of unclassified value is noise and must not
 * suppress a card the reader relies on.
 */

vi.mock('@/context/AuthContext', () => ({
    useAuth: () => ({ user: { username: 'tester' } }),
}));

const targets = { quoteType: { EQUITY: 95, ETF: 5 } };

vi.mock('@/lib/api', async (importOriginal) => ({
    ...(await importOriginal<typeof import('@/lib/api')>()),
    fetchSettings: () => Promise.resolve({ target_allocation: targets }),
}));

const holding = (
    Symbol: string,
    marketValue: number,
    quoteType: string | null,
): Holding =>
    ({
        Symbol,
        Account: 'IBKR Atcha',
        'Market Value (USD)': marketValue,
        quoteType,
        Sector: quoteType ? 'Technology' : null,
        Country: quoteType ? 'United States' : null,
    }) as unknown as Holding;

async function renderCard(holdings: Holding[]) {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
        <QueryClientProvider client={client}>
            <RebalanceHelper holdings={holdings} currency="USD" />
        </QueryClientProvider>,
    );
    // The card needs the settings query to land before targets exist.
    await screen.findByText(/Asset Type/);
    await new Promise(r => setTimeout(r, 0));
}

describe('RebalanceHelper with unclassified holdings', () => {
    it('suppresses trades when the Unknown bucket is material', async () => {
        await renderCard([
            holding('MSFT', 166_050, 'EQUITY'),
            holding('GOOG', 455_097, null),
            holding('AMZN', 299_500, null),
        ]);

        expect(await screen.findByText(/No trades suggested/)).toBeTruthy();
        expect(screen.queryByText(/^Buy /)).toBeNull();
        expect(screen.queryByText(/^Sell /)).toBeNull();
    });

    it('names the affected symbols so the reader can chase the data', async () => {
        await renderCard([
            holding('MSFT', 166_050, 'EQUITY'),
            holding('GOOG', 455_097, null),
            holding('AMZN', 299_500, null),
        ]);

        const affected = await screen.findByText(/AMZN, GOOG/);
        expect(affected).toBeTruthy();
    });

    it('reports the unclassified share of the portfolio', async () => {
        await renderCard([
            holding('MSFT', 500_000, 'EQUITY'),
            holding('GOOG', 500_000, null),
        ]);

        expect(await screen.findByText(/50\.0% of the portfolio is unclassified/)).toBeTruthy();
    });

    it('still suggests trades when the unclassified share is a trace', async () => {
        await renderCard([
            holding('MSFT', 999_000, 'EQUITY'),   // 99.9%
            holding('GOOG', 1_000, null),         // 0.1% — below the 0.5% tolerance
        ]);

        expect(await screen.findByText(/Trades to align each bucket/)).toBeTruthy();
        expect(screen.queryByText(/No trades suggested/)).toBeNull();
    });

    it('leaves the no-targets case alone', async () => {
        targets.quoteType = {} as typeof targets.quoteType;
        await renderCard([holding('GOOG', 455_097, null)]);

        expect(await screen.findByText(/No targets set for asset type/)).toBeTruthy();
        expect(screen.queryByText(/No trades suggested/)).toBeNull();
        targets.quoteType = { EQUITY: 95, ETF: 5 };
    });
});
