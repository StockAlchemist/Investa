import React, { useState } from 'react';
import { useQueryClient, useQuery } from '@tanstack/react-query';
import { exportToCSV } from '../lib/export';
import { Transaction, addTransaction, updateTransaction, deleteTransaction, addToWatchlist, fetchPendingIbkr, approveIbkr, rejectIbkr } from '../lib/api';
import { Trash2, Star, Pencil, Plus, Filter, ChevronUp, ChevronDown, Download, Eye, EyeOff, LayoutGrid, Table as TableIcon, CheckCircle, XCircle, AlertCircle, Clock } from 'lucide-react';
import TransactionModal from './TransactionModal';
import StockTicker from './StockTicker';
import TableSkeleton from './skeletons/TableSkeleton';

interface TransactionsTableProps {
    transactions: Transaction[];
    isLoading?: boolean;
}

export default function TransactionsTable({ transactions, isLoading }: TransactionsTableProps) {
    const [symbolFilter, setSymbolFilter] = useState('');
    const [accountFilter, setAccountFilter] = useState('');
    const [filterType, setFilterType] = useState('');
    const [showFilters, setShowFilters] = useState(false);
    const [visibleRows, setVisibleRows] = useState(10);
    const [showInternalCash, setShowInternalCash] = useState(false);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalMode, setModalMode] = useState<'add' | 'edit'>('add');
    const [currentTransaction, setCurrentTransaction] = useState<Transaction | null>(null);
    const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
    const [mobileViewMode, setMobileViewMode] = useState<'card' | 'table'>('table');

    // React Query for pending items
    const { data: pendingTransactions = [], isLoading: isPendingLoading } = useQuery<Transaction[]>({
        queryKey: ['pendingIbkr'],
        queryFn: fetchPendingIbkr,
    });
    const [selectedPendingIds, setSelectedPendingIds] = useState<Set<number>>(new Set());
    const [isApproving, setIsApproving] = useState(false);

    const queryClient = useQueryClient();

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
                // Invalidate queries to refresh data
                queryClient.invalidateQueries({ queryKey: ['transactions'] });
                queryClient.invalidateQueries({ queryKey: ['summary'] });
                queryClient.invalidateQueries({ queryKey: ['holdings'] });
            } catch (error) {
                console.error("Failed to delete transaction:", error);
                alert("Failed to delete transaction");
            }
        }
    };

    const handleToggleSelect = (id: number) => {
        const next = new Set(selectedIds);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        setSelectedIds(next);
    };

    const handleSelectAll = () => {
        if (selectedIds.size === visibleTransactions.length) {
            setSelectedIds(new Set());
        } else {
            const allIds = visibleTransactions
                .map(tx => tx.id)
                .filter((id): id is number => id !== undefined);
            setSelectedIds(new Set(allIds));
        }
    };

    const handleBulkDelete = async () => {
        if (selectedIds.size === 0) return;
        if (window.confirm(`Are you sure you want to delete ${selectedIds.size} selected transactions?`)) {
            try {
                // Delete one by one for now (or implement bulk API if needed)
                for (const id of Array.from(selectedIds)) {
                    await deleteTransaction(id);
                }
                setSelectedIds(new Set());
                queryClient.invalidateQueries({ queryKey: ['transactions'] });
                queryClient.invalidateQueries({ queryKey: ['summary'] });
                queryClient.invalidateQueries({ queryKey: ['holdings'] });
                alert(`Successfully deleted ${selectedIds.size} transactions.`);
            } catch (error) {
                console.error("Failed bulk delete:", error);
                alert("Failed to delete some transactions.");
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
            // Invalidate queries to refresh data
            queryClient.invalidateQueries({ queryKey: ['transactions'] });
            queryClient.invalidateQueries({ queryKey: ['summary'] });
            queryClient.invalidateQueries({ queryKey: ['holdings'] });
        } catch (error) {
            console.error("Failed to save transaction:", error);
            throw error; // Re-throw to be handled by modal
        }
    };

    const resetFilters = () => {
        setSymbolFilter('');
        setAccountFilter('');
        setFilterType('');
    };

    if (isLoading) {
        return <TableSkeleton />;
    }

    const uniqueTypes = new Set<string>();
    (transactions || []).forEach(tx => {
        if (tx.Type) uniqueTypes.add(tx.Type);
    });
    const existingTypes = Array.from(uniqueTypes).sort();

    const filteredTransactions = (transactions || []).filter(tx => {
        const symbolMatch = tx.Symbol.toLowerCase().includes(symbolFilter.toLowerCase());
        const accountMatch = tx.Account.toLowerCase().includes(accountFilter.toLowerCase());
        const typeMatch = filterType ? tx.Type === filterType : true;
        const internalCashMatch = showInternalCash
            ? true
            : (tx.Symbol !== '$CASH' || ['deposit', 'withdrawal', 'interest', 'dividend'].includes(tx.Type.toLowerCase()));
        return symbolMatch && accountMatch && typeMatch && internalCashMatch;
    });

    const visibleTransactions = filteredTransactions.slice(0, visibleRows);

    const handleShowMore = () => {
        setVisibleRows(prev => prev + 20);
    };

    const handleShowAll = () => {
        setVisibleRows(filteredTransactions.length);
    };

    // --- Pending Syncs Logic ---
    const handlePendingAction = async (action: 'approve' | 'reject', ids?: number[]) => {
        const idsToProcess = ids || Array.from(selectedPendingIds);
        if (idsToProcess.length === 0) return;

        // Optimistic update: Remove from UI immediately
        queryClient.setQueryData(['pendingIbkr'], (old: Transaction[] | undefined) =>
            old ? old.filter(tx => !idsToProcess.includes(tx.id!)) : []
        );
        setSelectedPendingIds(new Set());

        setIsApproving(true);
        try {
            if (action === 'approve') {
                await approveIbkr(idsToProcess);
            } else {
                await rejectIbkr(idsToProcess);
            }
            // Refetch to ensure we're in sync with server
            await queryClient.invalidateQueries({ queryKey: ['pendingIbkr'] });
            if (action === 'approve') {
                await queryClient.invalidateQueries({ queryKey: ['transactions'] });
            }
            queryClient.invalidateQueries({ queryKey: ['summary'] });
            queryClient.invalidateQueries({ queryKey: ['holdings'] });
        } catch (error) {
            console.error(`Failed to ${action} transactions:`, error);
            alert(`Error ${action}ing transactions`);
            // Roll back on error? For now, just refetch
            queryClient.invalidateQueries({ queryKey: ['pendingIbkr'] });
        } finally {
            setIsApproving(false);
        }
    };

    // Calculate Account -> Currency Map from existing transactions
    // Also extract unique accounts and symbols for suggestions
    const accountCurrencyMap: Record<string, string> = {};
    const uniqueAccounts = new Set<string>();
    const uniqueSymbols = new Set<string>();

    if (transactions) {
        transactions.forEach(tx => {
            if (tx.Account) {
                uniqueAccounts.add(tx.Account);
                if (tx["Local Currency"]) {
                    accountCurrencyMap[tx.Account] = tx["Local Currency"];
                }
            }
            if (tx.Symbol) {
                uniqueSymbols.add(tx.Symbol);
            }
        });
    }

    const existingAccounts = Array.from(uniqueAccounts).sort();
    const existingSymbols = Array.from(uniqueSymbols).sort();

    const getTransactionTypeStyle = (type: string) => {
        const t = type.toUpperCase();
        if (['BUY', 'DEPOSIT'].includes(t)) {
            return 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20';
        }
        if (['SELL', 'WITHDRAWAL'].includes(t)) {
            return 'bg-red-500/10 text-red-600 dark:text-red-500 border border-red-500/20';
        }
        if (['DIVIDEND', 'INTEREST'].includes(t)) {
            return 'bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border border-cyan-500/20';
        }
        return 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 border border-gray-200 dark:border-gray-700';
    };

    const formatTransactionType = (type: string) => {
        return type.replace(/\w\S*/g, (txt) => txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase());
    };

    return (
        <div className="space-y-6">
            {/* Pending Transactions Staging Area */}
            {pendingTransactions.length > 0 && (
                <div className="bg-cyan-500/5 border border-cyan-500/20 rounded-xl overflow-hidden animate-in fade-in slide-in-from-top-4 duration-500">
                    <div className="px-4 py-3 bg-cyan-500/10 border-b border-cyan-500/20 flex justify-between items-center">
                        <div className="flex items-center gap-2">
                            <Clock className="h-4 w-4 text-cyan-500" />
                            <h3 className="text-sm font-semibold text-cyan-700 dark:text-cyan-400 uppercase tracking-wider">
                                Pending Syncs ({pendingTransactions.length})
                            </h3>
                        </div>
                        <div className="flex gap-2">
                            <button
                                onClick={() => handlePendingAction('approve')}
                                disabled={selectedPendingIds.size === 0 || isApproving}
                                className="px-3 py-1 bg-cyan-600 text-white rounded text-xs font-bold hover:bg-cyan-700 transition-colors disabled:opacity-50 flex items-center gap-1.5"
                            >
                                <CheckCircle className="h-3 w-3" />
                                Approve Selected ({selectedPendingIds.size})
                            </button>
                            <button
                                onClick={() => handlePendingAction('reject')}
                                disabled={selectedPendingIds.size === 0 || isApproving}
                                className="px-3 py-1 bg-red-600/10 text-red-600 border border-red-600/20 rounded text-xs font-bold hover:bg-red-600/20 transition-colors disabled:opacity-50 flex items-center gap-1.5"
                            >
                                <XCircle className="h-3 w-3" />
                                Discard
                            </button>
                        </div>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-black/5 dark:divide-white/10">
                            <thead className="bg-cyan-500/5 text-xs font-semibold text-cyan-700 dark:text-cyan-400">
                                <tr>
                                    <th className="px-4 py-2 text-left w-10">
                                        <input
                                            type="checkbox"
                                            checked={selectedPendingIds.size === pendingTransactions.length}
                                            onChange={() => {
                                                if (selectedPendingIds.size === pendingTransactions.length) setSelectedPendingIds(new Set());
                                                else setSelectedPendingIds(new Set(pendingTransactions.map(t => t.id!).filter(Boolean)));
                                            }}
                                            className="rounded border-cyan-500/30 text-cyan-500 focus:ring-cyan-500"
                                        />
                                    </th>
                                    <th className="px-4 py-2">Date</th>
                                    <th className="px-4 py-2">Type</th>
                                    <th className="px-4 py-2">Symbol</th>
                                    <th className="px-4 py-2 text-right">Qty</th>
                                    <th className="px-4 py-2 text-right">Price</th>
                                    <th className="px-4 py-2 text-right">Comm.</th>
                                    <th className="px-4 py-2 text-right">Total</th>
                                    <th className="px-4 py-2 text-right">Action</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-black/5 dark:divide-white/10 text-sm">
                                {pendingTransactions.map((tx) => (
                                    <tr key={tx.id} className="hover:bg-cyan-500/10 transition-colors">
                                        <td className="px-4 py-2">
                                            <input
                                                type="checkbox"
                                                checked={selectedPendingIds.has(tx.id!)}
                                                onChange={() => {
                                                    const next = new Set(selectedPendingIds);
                                                    if (next.has(tx.id!)) next.delete(tx.id!);
                                                    else next.add(tx.id!);
                                                    setSelectedPendingIds(next);
                                                }}
                                                className="rounded border-cyan-500/30 text-cyan-500 focus:ring-cyan-500"
                                            />
                                        </td>
                                        <td className="px-4 py-2 whitespace-nowrap text-muted-foreground">{tx.Date}</td>
                                        <td className="px-4 py-2 font-bold text-cyan-500 uppercase text-[10px] tracking-tighter">{tx.Type}</td>
                                        <td className="px-4 py-2 font-medium">{tx.Symbol}</td>
                                        <td className="px-4 py-2 text-right tabular-nums">{tx.Quantity}</td>
                                        <td className="px-4 py-2 text-right tabular-nums">{tx["Price/Share"]?.toFixed(2)}</td>
                                        <td className="px-4 py-2 text-right text-muted-foreground tabular-nums">{tx.Commission?.toFixed(2) || '0.00'}</td>
                                        <td className="px-4 py-2 text-right font-bold tabular-nums">
                                            {tx["Total Amount"]?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                        </td>
                                        <td className="px-4 py-2 text-right whitespace-nowrap">
                                            <div className="flex justify-end gap-1">
                                                <button onClick={() => handlePendingAction('approve', [tx.id!])} className="p-1 hover:bg-cyan-500/20 text-cyan-500 rounded transition-colors" title="Approve">
                                                    <CheckCircle className="h-4 w-4" />
                                                </button>
                                                <button onClick={() => handlePendingAction('reject', [tx.id!])} className="p-1 hover:bg-red-500/20 text-red-500 rounded transition-colors" title="Discard">
                                                    <XCircle className="h-4 w-4" />
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}


            <TransactionModal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                onSubmit={handleModalSubmit}
                initialData={currentTransaction}
                mode={modalMode}
                accountCurrencyMap={accountCurrencyMap}
                existingAccounts={existingAccounts}
                existingSymbols={existingSymbols}
            />

            <div className="flex flex-col gap-4">
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                    <div className="flex flex-nowrap gap-2 w-full md:w-auto">
                        <button
                            onClick={handleAdd}
                            className="flex-1 md:flex-none px-2 md:px-4 py-2 bg-[#0097b2] text-white rounded-md hover:bg-[#0086a0] transition-colors text-sm font-medium shadow-sm flex items-center justify-center gap-2"
                        >
                            <Plus className="h-4 w-4" />
                            <span className="hidden md:inline">Add Transaction</span>
                        </button>
                        {selectedIds.size > 0 && (
                            <button
                                onClick={handleBulkDelete}
                                className="flex-1 md:flex-none px-2 md:px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors text-sm font-medium shadow-sm flex items-center justify-center gap-2"
                            >
                                <Trash2 className="h-4 w-4" />
                                <span className="hidden md:inline">Delete Selected ({selectedIds.size})</span>
                            </button>
                        )}
                        <button
                            onClick={() => { setShowFilters(!showFilters); if (showFilters) resetFilters(); }}
                            className="flex-1 md:flex-none flex justify-center md:justify-between w-full md:w-auto px-2 md:px-4 py-2 gap-3 text-sm font-medium text-foreground bg-secondary border border-border rounded-lg shadow-sm hover:bg-accent/10 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 transition-all items-center"
                        >
                            <div className="flex items-center gap-2">
                                <Filter className="h-4 w-4" />
                                <span className="hidden md:inline">{showFilters ? 'Hide Filters' : 'Show Filters'}</span>
                            </div>
                            <span className="text-xs hidden md:inline">{showFilters ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}</span>
                        </button>
                        <button
                            onClick={() => exportToCSV(filteredTransactions, 'transactions.csv')}
                            className="flex-1 md:flex-none px-2 md:px-4 py-2 bg-secondary border border-border text-foreground rounded-md hover:bg-accent/10 transition-colors text-sm font-medium text-center flex items-center justify-center gap-2"
                        >
                            <Download className="h-4 w-4" />
                            <span className="hidden md:inline">Export CSV</span>
                        </button>
                        <button
                            onClick={() => setShowInternalCash(!showInternalCash)}
                            className={`flex-1 md:flex-none flex justify-center md:justify-between w-full md:w-auto px-2 md:px-4 py-2 gap-3 text-sm font-medium border rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 transition-all items-center ${showInternalCash
                                ? 'bg-cyan-500/10 text-cyan-500 border-cyan-500/20'
                                : 'bg-secondary text-foreground border-border hover:bg-accent/10'
                                }`}
                        >
                            <div className="flex items-center gap-2">
                                {showInternalCash ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                                <span className="hidden md:inline">{showInternalCash ? 'Hide Internal Cash' : 'Show Internal Cash'}</span>
                            </div>
                        </button>
                        <button
                            onClick={() => setMobileViewMode(current => current === 'card' ? 'table' : 'card')}
                            className="md:hidden flex-1 md:flex-none flex items-center justify-center gap-1.5 px-2 md:px-4 py-2 text-sm font-medium border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 text-center transition-colors text-foreground bg-secondary border-border hover:bg-accent/10"
                            title={mobileViewMode === 'card' ? 'Switch to Table View' : 'Switch to Card View'}
                        >
                            {mobileViewMode === 'card' ? <TableIcon className="w-4 h-4" /> : <LayoutGrid className="w-4 h-4" />}
                        </button>
                    </div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                        Showing {visibleTransactions.length} of {filteredTransactions.length} transactions
                    </div>
                </div>

                {showFilters && (
                    <div className="flex flex-col md:flex-row gap-2">
                        <div className="relative flex-1">
                            <input
                                type="text"
                                placeholder="Filter Symbol..."
                                value={symbolFilter}
                                onChange={(e) => setSymbolFilter(e.target.value)}
                                className="bg-card border border-border text-foreground rounded-md px-3 py-2 text-sm w-full focus:ring-cyan-500 focus:border-cyan-500"
                            />
                            {symbolFilter && (
                                <button
                                    onClick={() => setSymbolFilter('')}
                                    className="absolute right-2 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
                                >
                                    ✕
                                </button>
                            )}
                        </div>
                        <div className="relative flex-1">
                            <input
                                type="text"
                                placeholder="Filter Account..."
                                value={accountFilter}
                                onChange={(e) => setAccountFilter(e.target.value)}
                                className="bg-card border border-border text-foreground rounded-md px-3 py-2 text-sm w-full focus:ring-cyan-500 focus:border-cyan-500"
                            />
                            {accountFilter && (
                                <button
                                    onClick={() => setAccountFilter('')}
                                    className="absolute right-2 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
                                >
                                    ✕
                                </button>
                            )}
                        </div>
                        <div className="relative flex-1">
                            <select
                                value={filterType}
                                onChange={(e) => setFilterType(e.target.value)}
                                className="bg-card border border-border text-foreground rounded-md px-3 py-2 text-sm w-full focus:ring-cyan-500 focus:border-cyan-500 appearance-none pr-8"
                            >
                                <option value="">All Types</option>
                                {existingTypes.map(type => (
                                    <option key={type} value={type}>{formatTransactionType(type)}</option>
                                ))}
                            </select>
                            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-muted-foreground">
                                <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" /></svg>
                            </div>
                        </div>
                        <button
                            onClick={resetFilters}
                            className="flex-1 md:flex-none px-4 py-2 bg-secondary border border-border text-foreground rounded-md hover:bg-accent/10 transition-colors text-sm font-medium text-center"
                        >
                            Reset Filters
                        </button>
                    </div>
                )}
            </div>

            {/* Desktop Table View */}
            <div className={`bg-card rounded-xl shadow-sm border border-border overflow-hidden ${mobileViewMode === 'table' ? 'block' : 'hidden'} md:block`}>
                <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-black/5 dark:divide-white/10">
                        <thead className="bg-secondary/50 font-semibold border-b border-border">
                            <tr>
                                <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground w-10">
                                    <input
                                        type="checkbox"
                                        checked={selectedIds.size === visibleTransactions.length && visibleTransactions.length > 0}
                                        onChange={handleSelectAll}
                                        className="rounded border-gray-300 text-cyan-500 focus:ring-cyan-500"
                                    />
                                </th>
                                <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Date</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Type</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Symbol</th>
                                <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground">Qty</th>
                                <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground">Price/Share</th>
                                <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground">Total Amount</th>
                                <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground">Commission</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Account</th>
                                <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground">Split Ratio</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Note</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Currency</th>
                                <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-black/5 dark:divide-white/10">
                            {visibleTransactions.length === 0 ? (
                                <tr>
                                    <td colSpan={13} className="px-6 py-12 text-center text-muted-foreground">
                                        {(!transactions || transactions.length === 0)
                                            ? "No transactions added yet. Click 'Add Transaction' to get started."
                                            : "No transactions match your filters."}
                                    </td>
                                </tr>
                            ) : (
                                visibleTransactions.map((tx, index) => (
                                    <tr key={index} className={`hover:bg-accent/5 transition-colors group ${tx.id !== undefined && selectedIds.has(tx.id) ? 'bg-cyan-500/5' : ''}`}>
                                        <td className="px-4 py-3 whitespace-nowrap text-sm">
                                            <input
                                                type="checkbox"
                                                checked={tx.id !== undefined && selectedIds.has(tx.id)}
                                                onChange={() => tx.id !== undefined && handleToggleSelect(tx.id)}
                                                className="rounded border-gray-300 text-cyan-500 focus:ring-cyan-500 cursor-pointer"
                                            />
                                        </td>
                                        <td className="px-4 py-3 text-sm text-foreground whitespace-nowrap">{tx.Date ? tx.Date.split('T')[0].split(' ')[0] : '-'}</td>
                                        <td className="px-4 py-3 text-sm text-muted-foreground">
                                            <span className={`px-2 py-0.5 rounded text-xs font-medium whitespace-nowrap ${getTransactionTypeStyle(tx.Type)}`}>
                                                {formatTransactionType(tx.Type)}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3 whitespace-nowrap">
                                            <StockTicker symbol={tx.Symbol} currency={tx["Local Currency"]} />
                                        </td>
                                        <td className="px-4 py-3 text-sm text-right text-muted-foreground tabular-nums">{tx.Quantity}</td>
                                        <td className="px-4 py-3 text-sm text-right text-muted-foreground tabular-nums">{tx["Price/Share"]?.toFixed(2)}</td>
                                        <td className="px-4 py-3 text-sm text-right font-medium text-foreground tabular-nums">
                                            {tx["Total Amount"] ? Math.abs(tx["Total Amount"]).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '-'}
                                        </td>
                                        <td className="px-4 py-3 text-sm text-right text-muted-foreground tabular-nums">
                                            {tx.Commission ? tx.Commission.toFixed(2) : '-'}
                                        </td>
                                        <td className="px-4 py-3 text-sm text-muted-foreground whitespace-nowrap">{tx.Account}</td>
                                        <td className="px-4 py-3 text-sm text-right text-muted-foreground tabular-nums">{tx["Split Ratio"] ? tx["Split Ratio"] : ''}</td>
                                        <td className="px-4 py-3 text-sm text-muted-foreground truncate max-w-xs" title={tx.Note}>{tx.Note || '-'}</td>
                                        <td className="px-4 py-3 text-sm text-muted-foreground">{tx["Local Currency"]}</td>
                                        <td className="px-4 py-3 text-sm text-right text-foreground whitespace-nowrap">
                                            <button
                                                onClick={() => handleEdit(tx)}
                                                className="text-cyan-500 hover:text-cyan-400 hover:bg-cyan-500/10 p-2 rounded transition-colors mr-1"
                                                title="Edit"
                                            >
                                                <Pencil className="w-4 h-4" />
                                            </button>
                                            <button
                                                onClick={() => handleDelete(tx)}
                                                className="text-red-500 hover:text-red-400 hover:bg-red-500/10 p-2 rounded transition-colors"
                                                title="Delete"
                                            >
                                                <Trash2 className="w-4 h-4" />
                                            </button>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Mobile Card View */}
            <div className={`md:hidden space-y-4 p-4 ${mobileViewMode === 'card' ? 'block' : 'hidden'}`}>
                {visibleTransactions.map((tx, index) => (
                    <div key={`mobile-tx-${index}`} className="bg-card rounded-lg border border-border shadow-sm p-4">
                        <div className="flex justify-between items-start mb-2">
                            <div>
                                <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase tracking-wider ${getTransactionTypeStyle(tx.Type)}`}>
                                    {formatTransactionType(tx.Type)}
                                </span>
                                <h3 className="text-lg font-bold text-foreground mt-1">
                                    <StockTicker symbol={tx.Symbol} currency={tx["Local Currency"]} />
                                </h3>
                            </div>
                            <div className="text-right">
                                <div className="text-sm font-medium text-foreground">
                                    {tx.Date ? tx.Date.split('T')[0] : '-'}
                                </div>
                                <div className="text-xs text-muted-foreground">{tx.Account}</div>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-y-2 text-sm mt-3 pt-3 border-t border-black/5 dark:border-white/10">
                            <div className="text-muted-foreground">Quantity</div>
                            <div className="text-right font-medium text-foreground">{tx.Quantity}</div>

                            <div className="text-muted-foreground">Price</div>
                            <div className="text-right font-medium text-foreground">{tx["Price/Share"]?.toFixed(2)}</div>

                            <div className="text-muted-foreground">Amount</div>
                            <div className="text-right font-bold text-foreground">
                                {tx["Total Amount"] ? Math.abs(tx["Total Amount"]).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '-'} {tx["Local Currency"]}
                            </div>

                            {tx.Commission > 0 && (
                                <>
                                    <div className="text-muted-foreground">Commission</div>
                                    <div className="text-right text-muted-foreground">{tx.Commission.toFixed(2)}</div>
                                </>
                            )}
                        </div>
                        {
                            tx.Note && (
                                <div className="mt-2 pt-2 border-t border-black/5 dark:border-white/10 text-xs text-muted-foreground italic">
                                    {tx.Note}
                                </div>
                            )
                        }
                        < div className="mt-3 pt-2 border-t border-black/5 dark:border-white/10 flex justify-end gap-3" >
                            <button
                                onClick={() => handleEdit(tx)}
                                className="text-cyan-500 hover:text-cyan-400 hover:bg-cyan-500/10 p-2 rounded transition-colors"
                                title="Edit"
                            >
                                <Pencil className="w-5 h-5" />
                            </button>
                            <button
                                onClick={() => handleDelete(tx)}
                                className="text-red-500 hover:text-red-400 hover:bg-red-500/10 p-2 rounded transition-colors"
                                title="Delete"
                            >
                                <Trash2 className="w-5 h-5" />
                            </button>
                        </div>
                    </div>
                ))
                }
            </div >

            {
                visibleRows < filteredTransactions.length && (
                    <div className="flex justify-center gap-4 p-4 border-t border-black/5 dark:border-white/10">
                        <button
                            onClick={handleShowMore}
                            className="px-4 py-2 bg-[#0097b2] text-white rounded-md hover:bg-[#0086a0] transition-colors text-sm font-medium shadow-sm"
                        >
                            Show More
                        </button>
                        <button
                            onClick={handleShowAll}
                            className="px-4 py-2 bg-card text-foreground border border-border rounded-md hover:bg-secondary transition-colors text-sm font-medium shadow-sm"
                        >
                            Show All
                        </button>
                    </div>
                )
            }
        </div >
    );
}
