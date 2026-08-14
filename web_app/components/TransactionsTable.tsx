"use client";

import React, { useRef, useState, useMemo } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Transaction, deleteTransaction, addTransaction, updateTransaction, parseDocument, fetchSettings } from '../lib/api';
import { TransactionsTableProps } from './transactions/types';
import { useTransactionsFilter } from './transactions/hooks/useTransactionsFilter';
import { TransactionsToolbar } from './transactions/TransactionsToolbar';
import { TransactionsDesktopTable } from './transactions/TransactionsDesktopTable';
import { TransactionsMobileCards } from './transactions/TransactionsMobileCards';
import { TransactionsPagination } from './transactions/TransactionsPagination';
import { PendingIbkrBanner } from './transactions/PendingIbkrBanner';
import { ImportReviewModal } from './transactions/ImportReviewModal';
import TxKpiStrip from './transactions/TxKpiStrip';
import TransactionModal from './TransactionModal';
import TableSkeleton from './skeletons/TableSkeleton';

export type { TransactionsTableProps };

export default function TransactionsTable({ transactions = [], currency = 'USD', isLoading = false }: TransactionsTableProps) {
    const queryClient = useQueryClient();
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Modal state for Add/Edit
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalMode, setModalMode] = useState<'add' | 'edit'>('add');
    const [currentTransaction, setCurrentTransaction] = useState<Transaction | null>(null);

    // Import review state
    const [isReviewing, setIsReviewing] = useState(false);
    const [reviewTransactions, setReviewTransactions] = useState<Transaction[]>([]);
    const [importAccount] = useState('');

    const { data: settings } = useQuery({
        queryKey: ['settings'],
        queryFn: fetchSettings,
        staleTime: 5 * 60 * 1000,
    });

    const {
        symbolFilter,
        setSymbolFilter,
        accountFilter,
        setAccountFilter,
        filterTypes,
        toggleFilterType,
        datePreset,
        setDatePreset,
        customFrom,
        setCustomFrom,
        customTo,
        setCustomTo,
        sortBy,
        sortDirection,
        handleSort,
        currentPage,
        setCurrentPage,
        pageSize,
        setPageSize,
        viewMode,
        setViewMode,
        resetFilters,
        hasActiveFilters,
        uniqueAccounts,
        availableTypes,
        duplicateKeys,
        filteredTransactions,
        paginatedTransactions,
        totalPages,
    } = useTransactionsFilter({ transactions });

    const existingSymbols = useMemo(() => {
        const set = new Set<string>();
        transactions.forEach(t => { if (t.Symbol) set.add(t.Symbol); });
        return Array.from(set).sort();
    }, [transactions]);

    const accountCurrencyMap = useMemo(() => {
        const map: Record<string, string> = { ...(settings?.account_currency_map || {}) };
        transactions.forEach(t => {
            if (t.Account && t['Local Currency'] && !map[t.Account]) {
                map[t.Account] = t['Local Currency'];
            }
        });
        return map;
    }, [settings, transactions]);

    const handleAdd = () => {
        setModalMode('add');
        setCurrentTransaction(null);
        setIsModalOpen(true);
    };

    const handleEdit = (tx: Transaction) => {
        setModalMode('edit');
        setCurrentTransaction(tx);
        setIsModalOpen(true);
    };

    const handleDelete = async (tx: Transaction) => {
        if (!tx.id) {
            alert("Cannot delete transaction without ID");
            return;
        }
        if (window.confirm(`Are you sure you want to delete transaction ${tx.Symbol} on ${tx.Date}?`)) {
            try {
                await deleteTransaction(tx.id);
                queryClient.invalidateQueries({ queryKey: ['transactions'] });
                queryClient.invalidateQueries({ queryKey: ['summary'] });
                queryClient.invalidateQueries({ queryKey: ['holdings'] });
            } catch (error) {
                console.error("Failed to delete transaction:", error);
                alert("Failed to delete transaction");
            }
        }
    };

    const handleModalSubmit = async (transaction: Transaction) => {
        try {
            if (modalMode === 'add') {
                await addTransaction(transaction);
            } else {
                if (!transaction.id) throw new Error("Transaction ID missing for update");
                await updateTransaction(transaction.id, transaction);
            }
            queryClient.invalidateQueries({ queryKey: ['transactions'] });
            queryClient.invalidateQueries({ queryKey: ['summary'] });
            queryClient.invalidateQueries({ queryKey: ['holdings'] });
        } catch (error) {
            console.error("Failed to save transaction:", error);
            throw error;
        }
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        try {
            const result = await parseDocument(file);
            if (result.transactions && result.transactions.length > 0) {
                const outflowTypes = new Set(['buy', 'withdrawal', 'fees', 'fee', 'tax', 'withholding tax', 'split', 'stock split', 'buy to cover']);
                const enrichedTransactions = result.transactions.map(tx => {
                    const total = Number(tx['Total Amount'] ?? 0);
                    const txType = (tx.Type || '').toLowerCase().trim();
                    const signed = (outflowTypes.has(txType) && isFinite(total) && Math.abs(total) > 1e-9) ? -Math.abs(total) : Math.abs(total);
                    return {
                        ...tx,
                        Account: importAccount || tx.Account || 'Default',
                        'Total Amount': isFinite(total) ? signed : tx['Total Amount'],
                    };
                });
                setReviewTransactions(enrichedTransactions);
                setIsReviewing(true);
            } else {
                alert(result.message || "No transactions found in document.");
            }
        } catch (error) {
            console.error("Failed to parse document:", error);
            alert("Failed to parse document. Check console for details.");
        } finally {
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    if (isLoading) {
        return <TableSkeleton />;
    }

    return (
        <div className="space-y-6">
            {/* Hidden file input for statement uploads */}
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept=".pdf,.csv"
                className="hidden"
            />

            {/* IBKR Pending sync banner */}
            <PendingIbkrBanner />

            {/* Document parser AI review section */}
            <ImportReviewModal
                isReviewing={isReviewing}
                setIsReviewing={setIsReviewing}
                reviewTransactions={reviewTransactions}
                setReviewTransactions={setReviewTransactions}
                importAccount={importAccount}
                transactions={transactions}
            />

            {/* KPI Overview Strip */}
            <TxKpiStrip transactions={filteredTransactions} preferredCurrency={currency} />

            {/* Filter toolbar */}
            <TransactionsToolbar
                symbolFilter={symbolFilter}
                setSymbolFilter={setSymbolFilter}
                accountFilter={accountFilter}
                setAccountFilter={setAccountFilter}
                uniqueAccounts={uniqueAccounts}
                filterTypes={filterTypes}
                toggleFilterType={toggleFilterType}
                availableTypes={availableTypes}
                datePreset={datePreset}
                setDatePreset={setDatePreset}
                customFrom={customFrom}
                setCustomFrom={setCustomFrom}
                customTo={customTo}
                setCustomTo={setCustomTo}
                resetFilters={resetFilters}
                hasActiveFilters={hasActiveFilters}
                viewMode={viewMode}
                setViewMode={setViewMode}
                onOpenAddModal={handleAdd}
                onOpenImportModal={() => fileInputRef.current?.click()}
                filteredTransactions={filteredTransactions}
            />

            {/* Transactions View: Table or Cards */}
            {viewMode === 'table' ? (
                <TransactionsDesktopTable
                    transactions={paginatedTransactions}
                    sortBy={sortBy}
                    sortDirection={sortDirection}
                    handleSort={handleSort}
                    duplicateKeys={duplicateKeys}
                    onEdit={handleEdit}
                    onDelete={handleDelete}
                />
            ) : (
                <TransactionsMobileCards
                    transactions={paginatedTransactions}
                    duplicateKeys={duplicateKeys}
                    onEdit={handleEdit}
                    onDelete={handleDelete}
                />
            )}

            {/* Pagination controls */}
            <TransactionsPagination
                currentPage={currentPage}
                totalPages={totalPages}
                pageSize={pageSize}
                setPageSize={setPageSize}
                setCurrentPage={setCurrentPage}
                totalCount={filteredTransactions.length}
            />

            {/* Add / Edit Transaction Modal */}
            <TransactionModal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                onSubmit={handleModalSubmit}
                initialData={currentTransaction}
                mode={modalMode}
                existingAccounts={uniqueAccounts}
                existingSymbols={existingSymbols}
                accountCurrencyMap={accountCurrencyMap}
                accountCashModeMap={settings?.account_cash_mode_map}
            />
        </div>
    );
}
