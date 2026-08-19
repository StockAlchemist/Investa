import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TransactionsToolbar } from '@/components/transactions/TransactionsToolbar';
import TransactionsTable from '@/components/TransactionsTable';
import * as api from '@/lib/api';

vi.mock('@/lib/api', async (importOriginal) => {
    const actual = await importOriginal<typeof api>();
    return {
        ...actual,
        fetchSettings: vi.fn().mockResolvedValue({}),
        fetchPendingIbkr: vi.fn().mockResolvedValue([]),
        syncIbkr: vi.fn().mockResolvedValue({ status: 'success', message: 'IBKR sync complete.' }),
    };
});

describe('Transactions IBKR Sync', () => {
    describe('TransactionsToolbar', () => {
        const defaultProps = {
            symbolFilter: '',
            setSymbolFilter: vi.fn(),
            accountFilter: '',
            setAccountFilter: vi.fn(),
            uniqueAccounts: ['Account 1'],
            filterTypes: [],
            toggleFilterType: vi.fn(),
            availableTypes: ['Buy', 'Sell'],
            datePreset: 'all' as const,
            setDatePreset: vi.fn(),
            customFrom: '',
            setCustomFrom: vi.fn(),
            customTo: '',
            setCustomTo: vi.fn(),
            resetFilters: vi.fn(),
            hasActiveFilters: false,
            viewMode: 'table' as const,
            setViewMode: vi.fn(),
            onOpenAddModal: vi.fn(),
            onOpenImportModal: vi.fn(),
            filteredTransactions: [],
        };

        it('renders the Sync with IBKR button when onSyncIbkr is provided', () => {
            const onSyncIbkr = vi.fn();
            render(
                <TransactionsToolbar
                    {...defaultProps}
                    onSyncIbkr={onSyncIbkr}
                />
            );

            const syncButton = screen.getByRole('button', { name: /Sync with IBKR/i });
            expect(syncButton).toBeInTheDocument();
            fireEvent.click(syncButton);
            expect(onSyncIbkr).toHaveBeenCalledTimes(1);
        });

        it('disables the button and shows Syncing... when isSyncingIbkr is true', () => {
            const onSyncIbkr = vi.fn();
            render(
                <TransactionsToolbar
                    {...defaultProps}
                    onSyncIbkr={onSyncIbkr}
                    isSyncingIbkr={true}
                />
            );

            const syncButton = screen.getByRole('button', { name: /Syncing.../i });
            expect(syncButton).toBeInTheDocument();
            expect(syncButton).toBeDisabled();
        });
    });

    describe('TransactionsTable integration', () => {
        let queryClient: QueryClient;

        beforeEach(() => {
            queryClient = new QueryClient({
                defaultOptions: {
                    queries: { retry: false },
                },
            });
            vi.clearAllMocks();
        });

        it('calls syncIbkr and displays success message upon sync completion', async () => {
            vi.mocked(api.syncIbkr).mockResolvedValueOnce({
                status: 'success',
                message: 'Synced 3 transactions from IBKR',
            });

            render(
                <QueryClientProvider client={queryClient}>
                    <TransactionsTable transactions={[]} currency="USD" />
                </QueryClientProvider>
            );

            const syncButton = screen.getByRole('button', { name: /Sync with IBKR/i });
            fireEvent.click(syncButton);

            expect(api.syncIbkr).toHaveBeenCalledTimes(1);

            await waitFor(() => {
                expect(screen.getByText('Synced 3 transactions from IBKR')).toBeInTheDocument();
            });
        });

        it('displays error message when syncIbkr fails', async () => {
            vi.mocked(api.syncIbkr).mockRejectedValueOnce(
                new Error('IBKR API not configured. Please set IBKR Token and Query ID in your settings.')
            );

            render(
                <QueryClientProvider client={queryClient}>
                    <TransactionsTable transactions={[]} currency="USD" />
                </QueryClientProvider>
            );

            const syncButton = screen.getByRole('button', { name: /Sync with IBKR/i });
            fireEvent.click(syncButton);

            await waitFor(() => {
                expect(screen.getByText('IBKR API not configured. Please set IBKR Token and Query ID in your settings.')).toBeInTheDocument();
            });
        });
    });
});
