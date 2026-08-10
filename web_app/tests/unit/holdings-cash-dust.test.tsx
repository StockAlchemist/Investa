import { describe, expect, it, vi } from 'vitest';
import { render, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import HoldingsTable from '@/components/HoldingsTable';
import type { Holding } from '@/lib/api';

/**
 * Cash rows in the holdings table.
 *
 * Brokers leave sub-cent cash residue behind after FX conversion and fee
 * postings, which showed up as a run of rows reading 0.0021 and -0.0002 —
 * one per account, crowding out real positions. Those are filtered out.
 *
 * The line worth holding: the test is on *magnitude*, not sign. A negative cash
 * balance is margin debt, and hiding real money owed because it is below zero
 * would be a considerably worse bug than the one being fixed.
 */

vi.mock('@/context/StockModalContext', () => ({
    useStockModal: () => ({ openStockDetail: vi.fn() }),
}));

vi.mock('@/components/WatchlistStar', () => ({
    default: ({ symbol }: { symbol: string }) => <span data-testid={`star-${symbol}`} />,
}));

const cash = (account: string, marketValue: number, symbol = 'Cash ($)'): Holding => ({
    Symbol: symbol,
    Account: account,
    Quantity: marketValue,
    'Price (USD)': 1,
    'Market Value (USD)': marketValue,
} as unknown as Holding);

const stock = (): Holding => ({
    Symbol: 'GOOG',
    Account: 'IBKR Atcha',
    Quantity: 1314,
    'Price (USD)': 353.39,
    'Market Value (USD)': 464354.48,
} as unknown as Holding);

function renderTable(holdings: Holding[]) {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const utils = render(
        <QueryClientProvider client={client}>
            <HoldingsTable holdings={holdings} currency="USD" />
        </QueryClientProvider>,
    );
    // Scope to the desktop table; the mobile card column renders the same rows.
    const table = utils.container.querySelector('table');
    if (!table) throw new Error('holdings table not found');
    return within(table as HTMLElement);
}

// The Account column is hidden by default, so rows aggregate by symbol and only
// the symbol renders — that is what these assert on.
describe('cash rows in the holdings table', () => {
    it('drops sub-cent cash residue, positive or negative', () => {
        for (const residue of [0.0021, -0.0002, 0]) {
            const table = renderTable([stock(), cash('Kim Eng', residue)]);
            expect(table.getByText('GOOG')).toBeInTheDocument();
            expect(table.queryByText('Cash ($)')).not.toBeInTheDocument();
        }
    });

    it('drops residue in a non-USD cash line too', () => {
        const table = renderTable([stock(), cash('Eastspring', -0.0092, 'Cash (฿)')]);

        expect(table.queryByText('Cash (฿)')).not.toBeInTheDocument();
    });

    it('keeps cash balances above a cent', () => {
        const table = renderTable([stock(), cash('IBKR Atcha', 733.35)]);

        expect(table.getByText('Cash ($)')).toBeInTheDocument();
    });

    it('keeps a real negative cash balance — margin debt is not noise', () => {
        const table = renderTable([stock(), cash('Morgan Stanley', -5000)]);

        expect(table.getByText('Cash ($)')).toBeInTheDocument();
    });

    it('treats exactly one cent as residue and just above it as a balance', () => {
        expect(renderTable([stock(), cash('Dime!', 0.01)]).queryByText('Cash ($)')).not.toBeInTheDocument();
        expect(renderTable([stock(), cash('Dime!', 0.02)]).getByText('Cash ($)')).toBeInTheDocument();
    });

    it('leaves non-cash holdings alone however small', () => {
        const tiny = { ...stock(), Symbol: 'PENNY', 'Market Value (USD)': 0.001 } as unknown as Holding;
        const table = renderTable([stock(), tiny]);

        expect(table.getByText('PENNY')).toBeInTheDocument();
    });
});
