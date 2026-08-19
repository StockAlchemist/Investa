import { describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TransactionsToolbar } from '@/components/transactions/TransactionsToolbar';
import { ImportReviewModal } from '@/components/transactions/ImportReviewModal';
import * as api from '@/lib/api';
import type { Transaction } from '@/lib/api';

vi.mock('@/lib/api', async (importOriginal) => {
    const actual = await importOriginal<typeof api>();
    return {
        ...actual,
        fetchTransactions: vi.fn().mockResolvedValue([]),
        addTransactionsBatch: vi.fn().mockResolvedValue({ status: 'success', count: 1 }),
    };
});

describe('Transactions Import Account Selection', () => {
    const defaultProps = {
        symbolFilter: '',
        setSymbolFilter: vi.fn(),
        accountFilter: '',
        setAccountFilter: vi.fn(),
        uniqueAccounts: ['IBKR Atcha', 'Manual Account'],
        availableAccounts: ['IBKR Atcha', 'Manual Account'],
        accountCashModeMap: {
            'IBKR Atcha': 'Auto',
            'Manual Account': 'Manual',
        },
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

    it('renders Auto-add cash when manual account is selected', () => {
        render(
            <TransactionsToolbar
                {...defaultProps}
                importAccount="Manual Account"
                onToggleAutoAddCash={vi.fn()}
            />
        );

        const importBtn = screen.getByTitle('Import PDF or CSV statements to an account');
        fireEvent.click(importBtn);

        expect(screen.getByText('Auto-add cash')).toBeInTheDocument();
    });

    it('hides Auto-add cash when auto cash account is selected', () => {
        render(
            <TransactionsToolbar
                {...defaultProps}
                importAccount="IBKR Atcha"
                onToggleAutoAddCash={vi.fn()}
            />
        );

        const importBtn = screen.getByTitle('Import PDF or CSV statements to an account');
        fireEvent.click(importBtn);

        expect(screen.queryByText('Auto-add cash')).not.toBeInTheDocument();
    });

    it('hides Auto-add cash in ImportReviewModal for auto cash accounts and passes false to batch import', async () => {
        const queryClient = new QueryClient({
            defaultOptions: { queries: { retry: false } },
        });

        const sampleTx: Transaction = {
            id: 1,
            Date: '2026-08-15',
            Type: 'Buy',
            Symbol: 'AAPL',
            Quantity: 10,
            'Price/Share': 200,
            'Total Amount': -2000,
            Account: 'IBKR Atcha',
        };

        const setReviewTransactions = vi.fn();
        const setImportAccount = vi.fn();
        const setIsReviewing = vi.fn();

        render(
            <QueryClientProvider client={queryClient}>
                <ImportReviewModal
                    isReviewing={true}
                    setIsReviewing={setIsReviewing}
                    reviewTransactions={[sampleTx]}
                    setReviewTransactions={setReviewTransactions}
                    importAccount="IBKR Atcha"
                    setImportAccount={setImportAccount}
                    autoAddCash={true}
                    accountCashModeMap={{ 'IBKR Atcha': 'Auto' }}
                    availableAccounts={['IBKR Atcha', 'Manual Account']}
                    transactions={[]}
                />
            </QueryClientProvider>
        );

        // Auto-add cash option must not be rendered
        expect(screen.queryByText('Auto-add cash')).not.toBeInTheDocument();

        // Click confirm
        const confirmBtn = screen.getByRole('button', { name: /Confirm & Import All/i });
        fireEvent.click(confirmBtn);

        await waitFor(() => {
            expect(api.addTransactionsBatch).toHaveBeenCalledWith([sampleTx], false);
        });
    });
});
