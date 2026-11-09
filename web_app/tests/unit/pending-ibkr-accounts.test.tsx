import { describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as api from '@/lib/api';
import { PendingIbkrBanner } from '@/components/transactions/PendingIbkrBanner';

vi.mock('@/components/StockTicker', () => ({
    default: ({ symbol }: { symbol: string }) => <span>{symbol}</span>,
}));

vi.mock('@/context/AuthContext', () => ({
    useAuth: () => ({ user: { username: 'dheematan' } }),
}));

vi.mock('@/lib/api', async (importOriginal) => {
    const actual = await importOriginal<typeof api>();
    return {
        ...actual,
        fetchPendingIbkr: vi.fn().mockResolvedValue([
            { id: 1, Date: '2026-11-06', Type: 'Buy', Symbol: 'AAPL', Quantity: 1, 'Price/Share': 200, 'Total Amount': 200, Account: 'IBKR Atcha' },
        ]),
        // account_metrics is keyed by the engine's upper-cased ids.
        fetchSummary: vi.fn().mockResolvedValue({
            metrics: { _available_accounts: ['E*Trade', 'IBKR Atcha', 'Dime!'] },
            account_metrics: { 'E*TRADE': {}, 'IBKR ATCHA': {}, 'DIME!': {}, 'NEW BROKER': {} },
        }),
    };
});

describe('IBKR approval window accounts', () => {
    it('lists each account once, as typed, whatever case the engine keys it by', async () => {
        const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
        render(
            <QueryClientProvider client={client}>
                <PendingIbkrBanner availableAccounts={['E*Trade', 'Dime!']} />
            </QueryClientProvider>,
        );
        const select = await screen.findByRole('combobox', { name: /Account for AAPL/ });
        await vi.waitFor(() => expect(within(select).getAllByRole('option').length).toBe(4));
        const options = within(select).getAllByRole('option').map(o => o.textContent);
        expect(options).toEqual(['Dime!', 'E*Trade', 'IBKR Atcha', 'NEW BROKER']);
        expect((select as HTMLSelectElement).value).toBe('IBKR Atcha');
    });
});
