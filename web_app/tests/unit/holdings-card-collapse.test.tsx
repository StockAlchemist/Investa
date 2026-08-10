import { describe, expect, it, vi } from 'vitest';
import { render, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import HoldingsTable from '@/components/HoldingsTable';
import type { Holding } from '@/lib/api';

/**
 * The mobile holding card.
 *
 * A card collapses to a single summary line, because the full metric grid is
 * tall enough that one expanded holding fills a phone screen. What's worth
 * pinning down is that *everything* detailed lives inside that expansion —
 * including the account/tax-lots strip. When the lots control sat outside it, a
 * "collapsed" holding was still two rows tall and the collapse bought nothing.
 */

vi.mock('@/context/StockModalContext', () => ({
    useStockModal: () => ({ openStockDetail: vi.fn() }),
}));

vi.mock('@/components/WatchlistStar', () => ({
    default: ({ symbol }: { symbol: string }) => <span data-testid={`star-${symbol}`} />,
}));

// Keys mirror the backend payload, which is what `getValue` resolves against:
// display headers map to "Market Value"/"Day Change" and carry a currency suffix.
const holding = (): Holding => ({
    Symbol: 'GOOG',
    Account: 'IBKR Atcha',
    Quantity: 1314,
    'Price (USD)': 353.39,
    'Market Value (USD)': 464354.48,
    'Day Change (USD)': -4244.19,
    'Day Change %': -0.91,
    'Avg Cost (USD)': 151.65,
    lots: Array.from({ length: 15 }, (_, i) => ({
        Date: `2024-01-${String(i + 1).padStart(2, '0')}`,
        Quantity: 10,
        'Cost Basis (USD)': 1000,
    })),
} as unknown as Holding);

function renderCards() {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return render(
        <QueryClientProvider client={client}>
            <HoldingsTable holdings={[holding()]} currency="USD" />
        </QueryClientProvider>,
    );
}

/** The mobile card column — the desktop table renders lots of its own. */
function cardView(container: HTMLElement): HTMLElement {
    const el = container.querySelector('.md\\:hidden.space-y-4');
    if (!el) throw new Error('mobile card container not found');
    return el as HTMLElement;
}

describe('mobile holding card', () => {
    it('collapses to a summary line, with the tax-lots strip hidden', () => {
        const { container } = renderCards();
        const cards = within(cardView(container));

        // The summary stays: symbol, market value, and the day change.
        expect(cards.getByText('GOOG')).toBeInTheDocument();
        expect(cards.getByText(/-0\.91/)).toBeInTheDocument();

        // The detail does not — neither the lots strip nor the metric grid.
        expect(cards.queryByText('15 Lots')).not.toBeInTheDocument();
        expect(cards.queryByText('IBKR Atcha')).not.toBeInTheDocument();
        expect(cards.queryByText('Avg Cost:')).not.toBeInTheDocument();
    });

    it('reveals the tax-lots strip only once the card is expanded', async () => {
        const user = userEvent.setup();
        const { container } = renderCards();
        const cards = within(cardView(container));

        await user.click(cards.getByTitle('Show GOOG details'));

        expect(cards.getByText('15 Lots')).toBeInTheDocument();
        expect(cards.getByText('IBKR Atcha')).toBeInTheDocument();
        expect(cards.getByText('Avg Cost:')).toBeInTheDocument();

        // ...and collapsing puts it away again.
        await user.click(cards.getByTitle('Hide GOOG details'));
        expect(cards.queryByText('15 Lots')).not.toBeInTheDocument();
    });
});
