import React, { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { exportToCSV } from '../lib/export';
import { Transaction, addTransaction, updateTransaction, deleteTransaction, addToWatchlist } from '../lib/api';
import { Trash2, Star } from 'lucide-react';
import TransactionModal from './TransactionModal';

interface TransactionsTableProps {
    transactions: Transaction[];
}

export default function TransactionsTable({ transactions }: TransactionsTableProps) {
    const [symbolFilter, setSymbolFilter] = useState('');
    const [accountFilter, setAccountFilter] = useState('');
    const [filterType, setFilterType] = useState('');
    const [showFilters, setShowFilters] = useState(false);
    const [visibleRows, setVisibleRows] = useState(10);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalMode, setModalMode] = useState<'add' | 'edit'>('add');
    const [currentTransaction, setCurrentTransaction] = useState<Transaction | null>(null);
    const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());

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
                queryClient.invalidateQueries();
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
                queryClient.invalidateQueries();
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
            queryClient.invalidateQueries();
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

    if (!transactions || transactions.length === 0) {
        return <div className="p-4 text-center text-gray-500">No transactions found.</div>;
    }

    const uniqueTypes = new Set<string>();
    transactions.forEach(tx => {
        if (tx.Type) uniqueTypes.add(tx.Type);
    });
    const existingTypes = Array.from(uniqueTypes).sort();

    const filteredTransactions = transactions.filter(tx => {
        const symbolMatch = tx.Symbol.toLowerCase().includes(symbolFilter.toLowerCase());
        const accountMatch = tx.Account.toLowerCase().includes(accountFilter.toLowerCase());
        const typeMatch = filterType ? tx.Type === filterType : true;
        return symbolMatch && accountMatch && typeMatch;
    });

    const visibleTransactions = filteredTransactions.slice(0, visibleRows);

    const handleShowMore = () => {
        setVisibleRows(prev => prev + 20);
    };

    const handleShowAll = () => {
        setVisibleRows(filteredTransactions.length);
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

    return (
        <div className="space-y-4">


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
                    <div className="flex flex-wrap gap-2 w-full md:w-auto">
                        <button
                            onClick={handleAdd}
                            className="flex-1 md:flex-none px-4 py-2 bg-[#0097b2] text-white rounded-md hover:bg-[#0086a0] transition-colors text-sm font-medium shadow-sm flex items-center justify-center gap-2"
                        >
                            <span>+</span> Add Transaction
                        </button>
                        {selectedIds.size > 0 && (
                            <button
                                onClick={handleBulkDelete}
                                className="flex-1 md:flex-none px-4 py-2 bg-rose-600 text-white rounded-md hover:bg-rose-700 transition-colors text-sm font-medium shadow-sm flex items-center justify-center gap-2"
                            >
                                <Trash2 className="h-4 w-4" />
                                Delete Selected ({selectedIds.size})
                            </button>
                        )}
                        <button
                            onClick={() => { setShowFilters(!showFilters); if (showFilters) resetFilters(); }}
                            className="flex justify-between w-full md:w-auto px-4 py-2 gap-3 text-sm font-medium text-foreground bg-secondary border border-border rounded-lg shadow-sm hover:bg-accent/10 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 backdrop-blur-md transition-all items-center"
                        >
                            <span>{showFilters ? 'Hide Filters' : 'Show Filters'}</span>
                            <span className="text-xs">{showFilters ? '▲' : '▼'}</span>
                        </button>
                        <button
                            onClick={() => exportToCSV(filteredTransactions, 'transactions.csv')}
                            className="flex-1 md:flex-none px-4 py-2 bg-secondary border border-border text-foreground rounded-md hover:bg-accent/10 transition-colors text-sm font-medium text-center backdrop-blur-md"
                        >
                            Export CSV
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
                                className="bg-card border border-border text-foreground rounded-md px-3 py-2 text-sm w-full focus:ring-cyan-500 focus:border-cyan-500 backdrop-blur-md"
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
                                className="bg-card border border-border text-foreground rounded-md px-3 py-2 text-sm w-full focus:ring-cyan-500 focus:border-cyan-500 backdrop-blur-md"
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
                                className="bg-card border border-border text-foreground rounded-md px-3 py-2 text-sm w-full focus:ring-cyan-500 focus:border-cyan-500 backdrop-blur-md appearance-none pr-8"
                            >
                                <option value="">All Types</option>
                                {existingTypes.map(type => (
                                    <option key={type} value={type}>{type}</option>
                                ))}
                            </select>
                            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-muted-foreground">
                                <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" /></svg>
                            </div>
                        </div>
                        <button
                            onClick={resetFilters}
                            className="flex-1 md:flex-none px-4 py-2 bg-secondary border border-border text-foreground rounded-md hover:bg-accent/10 transition-colors text-sm font-medium text-center backdrop-blur-md"
                        >
                            Reset Filters
                        </button>
                    </div>
                )}
            </div>

            {/* Desktop Table View */}
            <div className="bg-card backdrop-blur-md rounded-xl shadow-sm border border-border overflow-hidden">
                <div className="hidden md:block overflow-x-auto">
                    <table className="min-w-full divide-y divide-black/5 dark:divide-white/10">
                        <thead className="bg-secondary">
                            <tr>
                                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground w-10">
                                    <input
                                        type="checkbox"
                                        checked={selectedIds.size === visibleTransactions.length && visibleTransactions.length > 0}
                                        onChange={handleSelectAll}
                                        className="rounded border-gray-300 text-cyan-500 focus:ring-cyan-500"
                                    />
                                </th>
                                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">Date</th>
                                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">Type</th>
                                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">Symbol</th>
                                <th className="px-4 py-3 text-right text-xs font-medium uppercase tracking-wider text-muted-foreground">Qty</th>
                                <th className="px-4 py-3 text-right text-xs font-medium uppercase tracking-wider text-muted-foreground">Price/Share</th>
                                <th className="px-4 py-3 text-right text-xs font-medium uppercase tracking-wider text-muted-foreground">Total Amount</th>
                                <th className="px-4 py-3 text-right text-xs font-medium uppercase tracking-wider text-muted-foreground">Commission</th>
                                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">Account</th>
                                <th className="px-4 py-3 text-right text-xs font-medium uppercase tracking-wider text-muted-foreground">Split Ratio</th>
                                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">Note</th>
                                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">Currency</th>
                                <th className="px-4 py-3 text-right text-xs font-medium uppercase tracking-wider text-muted-foreground">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-black/5 dark:divide-white/10">
                            {visibleTransactions.map((tx, index) => (
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
                                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${tx.Type.toUpperCase() === 'BUY' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' :
                                            tx.Type.toUpperCase() === 'SELL' ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' :
                                                'bg-white/10 text-muted-foreground border border-white/10'
                                            }`}>
                                            {tx.Type}
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 text-sm font-medium text-foreground flex items-center gap-2">
                                        <button
                                            onClick={async (e) => {
                                                e.stopPropagation();
                                                try {
                                                    await addToWatchlist(tx.Symbol);
                                                } catch (err) {
                                                    console.error("Failed to add to watchlist", err);
                                                }
                                            }}
                                            className="text-muted-foreground/30 hover:text-yellow-500 transition-colors"
                                            title="Add to Watchlist"
                                        >
                                            <Star className="h-3 w-3" />
                                        </button>
                                        {tx.Symbol}
                                    </td>
                                    <td className="px-4 py-3 text-sm text-right text-muted-foreground">{tx.Quantity}</td>
                                    <td className="px-4 py-3 text-sm text-right text-muted-foreground">{tx["Price/Share"]?.toFixed(2)}</td>
                                    <td className="px-4 py-3 text-sm text-right font-medium text-foreground">
                                        {tx["Total Amount"] ? Math.abs(tx["Total Amount"]).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '-'}
                                    </td>
                                    <td className="px-4 py-3 text-sm text-right text-muted-foreground">
                                        {tx.Commission ? tx.Commission.toFixed(2) : '-'}
                                    </td>
                                    <td className="px-4 py-3 text-sm text-muted-foreground whitespace-nowrap">{tx.Account}</td>
                                    <td className="px-4 py-3 text-sm text-right text-muted-foreground">{tx["Split Ratio"] ? tx["Split Ratio"] : ''}</td>
                                    <td className="px-4 py-3 text-sm text-muted-foreground truncate max-w-xs" title={tx.Note}>{tx.Note || '-'}</td>
                                    <td className="px-4 py-3 text-sm text-muted-foreground">{tx["Local Currency"]}</td>
                                    <td className="px-4 py-3 text-sm text-right text-foreground whitespace-nowrap">
                                        <button
                                            onClick={() => handleEdit(tx)}
                                            className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 mr-2"
                                        >
                                            Edit
                                        </button>
                                        <button
                                            onClick={() => handleDelete(tx)}
                                            className="text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-300"
                                        >
                                            Delete
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Mobile Card View */}
            <div className="block md:hidden space-y-4 p-4">
                {visibleTransactions.map((tx, index) => (
                    <div key={`mobile-tx-${index}`} className="bg-card rounded-lg border border-border shadow-sm p-4 backdrop-blur-md">
                        <div className="flex justify-between items-start mb-2">
                            <div>
                                <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase tracking-wider ${tx.Type.toUpperCase() === 'BUY' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' :
                                    tx.Type.toUpperCase() === 'SELL' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400' :
                                        'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
                                    }`}>
                                    {tx.Type}
                                </span>
                                <h3 className="text-lg font-bold text-foreground mt-1">{tx.Symbol}</h3>
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
                                className="text-sm font-medium text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
                            >
                                Edit
                            </button>
                            <button
                                onClick={() => handleDelete(tx)}
                                className="text-sm font-medium text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-300"
                            >
                                Delete
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
