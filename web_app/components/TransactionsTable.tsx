import React, { useState } from 'react';
import { useQueryClient, useQuery } from '@tanstack/react-query';
import { exportToCSV } from '../lib/export';
import { Transaction, addTransaction, updateTransaction, deleteTransaction, addToWatchlist, fetchPendingIbkr, approveIbkr, rejectIbkr, parseDocument, addTransactionsBatch } from '../lib/api';
import { Trash2, Star, Pencil, Plus, Filter, ChevronUp, ChevronDown, Download, Eye, EyeOff, LayoutGrid, Table as TableIcon, CheckCircle, XCircle, AlertCircle, Clock, FileText } from 'lucide-react';
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
    const fileInputRef = React.useRef<HTMLInputElement>(null);
    const [isImporting, setIsImporting] = useState(false);
    const [autoAddCashOnImport, setAutoAddCashOnImport] = useState(true);
    const [importAccount, setImportAccount] = useState('');
    const [reviewTransactions, setReviewTransactions] = useState<Transaction[]>([]);
    const [isReviewing, setIsReviewing] = useState(false);

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

    const handleImportClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        try {
            setIsImporting(true);
            const result = await parseDocument(file);
            
            if (result.transactions && result.transactions.length > 0) {
                // Apply the selected import account to all extracted transactions if they don't have one
                const enrichedTransactions = result.transactions.map(tx => ({
                    ...tx,
                    Account: importAccount || tx.Account || 'Default'
                }));
                setReviewTransactions(enrichedTransactions);
                setIsReviewing(true);
            } else {
                alert(result.message || "No transactions found in document.");
            }
        } catch (error) {
            console.error("Failed to parse document:", error);
            alert("Failed to parse document. Check console for details.");
        } finally {
            setIsImporting(false);
            if (fileInputRef.current) {
                fileInputRef.current.value = ''; // reset input
            }
        }
    };

    const handleReviewConfirm = async () => {
        if (reviewTransactions.length === 0) return;
        
        setIsImporting(true);
        try {
            const result = await addTransactionsBatch(reviewTransactions, autoAddCashOnImport);
            alert(`Successfully imported ${result.count} transactions!`);
            
            setReviewTransactions([]);
            setIsReviewing(false);
            
            queryClient.invalidateQueries({ queryKey: ['transactions'] });
            queryClient.invalidateQueries({ queryKey: ['summary'] });
            queryClient.invalidateQueries({ queryKey: ['holdings'] });
        } catch (error) {
            console.error("Failed to add batch transactions:", error);
            alert("Failed to add transactions to database.");
        } finally {
            setIsImporting(false);
        }
    };

    const handleRemoveFromReview = (index: number) => {
        setReviewTransactions(prev => prev.filter((_, i) => i !== index));
    };

    const handleUpdateReviewTransaction = (index: number, updated: Transaction) => {
        setReviewTransactions(prev => prev.map((tx, i) => i === index ? updated : tx));
    };

    const resetFilters = () => {
        setSymbolFilter('');
        setAccountFilter('');
        setFilterType('');
    };

    if (isLoading) {
        return <TableSkeleton />;
    }

    const formatTransactionType = (type: string) => {
        return type.replace(/\w\S*/g, (txt) => txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase());
    };

    const uniqueTypes = new Set<string>([
        'Buy', 'Sell', 'Dividend', 'Transfer', 'Interest', 'Fees', 'Deposit', 'Withdrawal', 'Spin-off', 'Split'
    ]);
    (transactions || []).forEach(tx => {
        if (tx.Type) uniqueTypes.add(formatTransactionType(tx.Type));
    });
    const existingTypes = Array.from(uniqueTypes).sort();

    const filteredTransactions = (transactions || []).filter(tx => {
        const symbolMatch = tx.Symbol.toLowerCase().includes(symbolFilter.toLowerCase());
        const accountMatch = tx.Account.toLowerCase().includes(accountFilter.toLowerCase());
        const typeMatch = filterType ? tx.Type.toLowerCase() === filterType.toLowerCase() : true;
        const internalCashMatch = showInternalCash
            ? true
            : (tx.Symbol !== '$CASH' || ['deposit', 'withdrawal', 'interest', 'dividend', 'transfer'].includes(tx.Type.toLowerCase()));
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
            return 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400';
        }
        if (['SELL', 'WITHDRAWAL'].includes(t)) {
            return 'bg-red-500/10 text-red-600 dark:text-red-500';
        }
        if (['DIVIDEND', 'INTEREST'].includes(t)) {
            return 'bg-indigo-500/10 text-indigo-600 dark:text-indigo-400';
        }
        return 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400';
    };

    return (
        <div className="space-y-6">
            {/* AI Review & Confirm Section */}
            {isReviewing && reviewTransactions.length > 0 && (
                <div className="metric-card card-shine overflow-hidden animate-in fade-in zoom-in duration-500 relative border-2 border-indigo-500/20">
                    <div className="absolute top-0 left-0 right-0 h-[2px] bg-indigo-500" />
                    <div className="px-4 py-4 bg-indigo-500/10 flex justify-between items-center border-b border-indigo-500/10">
                        <div className="flex items-center gap-3">
                            <div className="p-2 bg-indigo-500/20 rounded-full">
                                <FileText className="h-5 w-5 text-indigo-500" />
                            </div>
                            <div>
                                <h3 className="text-sm font-bold text-indigo-700 dark:text-indigo-400 uppercase tracking-widest">
                                    Review Extracted Transactions ({reviewTransactions.length})
                                </h3>
                                <p className="text-[10px] text-muted-foreground uppercase font-semibold">AI identified these from your document. Please verify before saving.</p>
                            </div>
                        </div>
                        <div className="flex gap-2">
                            <button
                                onClick={handleReviewConfirm}
                                disabled={isImporting}
                                className="px-5 py-2 bg-indigo-600 text-white rounded-lg text-xs font-black uppercase tracking-wider hover:bg-indigo-700 transition-all shadow-lg hover:shadow-indigo-500/40 disabled:opacity-50 flex items-center gap-2 border-none"
                            >
                                <CheckCircle className="h-4 w-4" />
                                Confirm & Import All
                            </button>
                            <button
                                onClick={() => { setReviewTransactions([]); setIsReviewing(false); }}
                                className="px-4 py-2 bg-secondary text-foreground rounded-lg text-xs font-bold uppercase tracking-wider hover:bg-accent/10 transition-all border-none"
                            >
                                <Trash2 className="h-4 w-4" />
                            </button>
                        </div>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="min-w-full">
                            <thead className="bg-indigo-500/5 text-[10px] font-black text-indigo-700 dark:text-indigo-400 uppercase tracking-tighter">
                                <tr>
                                    <th className="px-4 py-2 text-left">Date</th>
                                    <th className="px-4 py-2 text-left">Type</th>
                                    <th className="px-4 py-2 text-left">Symbol</th>
                                    <th className="px-4 py-2 text-right">Qty</th>
                                    <th className="px-4 py-2 text-right">Price</th>
                                    <th className="px-4 py-2 text-right">Total</th>
                                    <th className="px-4 py-2 text-left">Account</th>
                                    <th className="px-4 py-2 text-right whitespace-nowrap"></th>
                                </tr>
                            </thead>
                            <tbody className="text-sm divide-y divide-indigo-500/5">
                                {reviewTransactions.map((tx, idx) => (
                                    <tr key={`review-${idx}`} className="hover:bg-indigo-500/5 transition-colors group">
                                        <td className="px-4 py-3">
                                            <input 
                                                type="text" 
                                                value={tx.Date} 
                                                onChange={(e) => handleUpdateReviewTransaction(idx, { ...tx, Date: e.target.value })}
                                                className="bg-transparent border-none text-[12px] p-0 w-full focus:ring-0 text-muted-foreground"
                                            />
                                        </td>
                                        <td className="px-4 py-3">
                                            <select 
                                                value={tx.Type} 
                                                onChange={(e) => handleUpdateReviewTransaction(idx, { ...tx, Type: e.target.value })}
                                                className="bg-transparent border-none text-[10px] p-0 font-bold uppercase tracking-widest focus:ring-0 text-indigo-500 appearance-none"
                                            >
                                                <option value="Buy">BUY</option>
                                                <option value="Sell">SELL</option>
                                                <option value="Dividend">DIVIDEND</option>
                                            </select>
                                        </td>
                                        <td className="px-4 py-3">
                                            <input 
                                                type="text" 
                                                value={tx.Symbol} 
                                                onChange={(e) => handleUpdateReviewTransaction(idx, { ...tx, Symbol: e.target.value.toUpperCase() })}
                                                className="bg-transparent border-none text-sm p-0 w-full font-bold focus:ring-0"
                                            />
                                        </td>
                                        <td className="px-4 py-3">
                                            <input 
                                                type="number" 
                                                value={tx.Quantity} 
                                                onChange={(e) => handleUpdateReviewTransaction(idx, { ...tx, Quantity: parseFloat(e.target.value) })}
                                                className="bg-transparent border-none text-right text-sm p-0 w-full focus:ring-0 tabular-nums"
                                            />
                                        </td>
                                        <td className="px-4 py-3">
                                            <input 
                                                type="number" 
                                                value={tx["Price/Share"]} 
                                                onChange={(e) => handleUpdateReviewTransaction(idx, { ...tx, "Price/Share": parseFloat(e.target.value) })}
                                                className="bg-transparent border-none text-right text-sm p-0 w-full focus:ring-0 tabular-nums"
                                            />
                                        </td>
                                        <td className="px-4 py-3">
                                            <input 
                                                type="number" 
                                                value={tx["Total Amount"]} 
                                                onChange={(e) => handleUpdateReviewTransaction(idx, { ...tx, "Total Amount": parseFloat(e.target.value) })}
                                                className="bg-transparent border-none text-right text-sm p-0 w-full focus:ring-0 font-bold tabular-nums"
                                            />
                                        </td>
                                        <td className="px-4 py-3">
                                            <input 
                                                type="text" 
                                                value={tx.Account} 
                                                placeholder="Account"
                                                onChange={(e) => handleUpdateReviewTransaction(idx, { ...tx, Account: e.target.value })}
                                                className="bg-transparent border-none text-xs p-0 w-full focus:ring-0 text-muted-foreground"
                                            />
                                        </td>
                                        <td className="px-4 py-3 text-right">
                                            <button 
                                                onClick={() => handleRemoveFromReview(idx)}
                                                className="p-1.5 text-muted-foreground hover:text-red-500 hover:bg-red-500/10 rounded transition-all"
                                            >
                                                <Trash2 className="h-4 w-4" />
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* IBKR Pending Sync Section */}
            {pendingTransactions.length > 0 && (
                <div className="metric-card card-shine overflow-hidden animate-in slide-in-from-top duration-500 relative border-2 border-cyan-500/20">
                    <div className="absolute top-0 left-0 right-0 h-[2px] bg-cyan-500" />
                    <div className="px-4 py-4 bg-cyan-500/10 flex justify-between items-center border-b border-cyan-500/10">
                        <div className="flex items-center gap-3">
                            <div className="p-2 bg-cyan-500/20 rounded-full">
                                <Clock className="h-5 w-5 text-cyan-500" />
                            </div>
                            <div>
                                <h3 className="text-sm font-bold text-cyan-700 dark:text-cyan-400 uppercase tracking-widest">
                                    Pending IBKR Transactions ({pendingTransactions.length})
                                </h3>
                                <p className="text-[10px] text-muted-foreground uppercase font-semibold">Synced from IBKR. Review and approve to add to your main portfolio.</p>
                            </div>
                        </div>
                        <div className="flex gap-2">
                            {selectedPendingIds.size > 0 && (
                                <>
                                    <button
                                        onClick={() => handlePendingAction('approve')}
                                        disabled={isApproving}
                                        className="px-5 py-2 bg-emerald-600 text-white rounded-lg text-xs font-black uppercase tracking-wider hover:bg-emerald-700 transition-all shadow-lg hover:shadow-emerald-500/40 disabled:opacity-50 flex items-center gap-2 border-none"
                                    >
                                        <CheckCircle className="h-4 w-4" />
                                        Approve Selected ({selectedPendingIds.size})
                                    </button>
                                    <button
                                        onClick={() => handlePendingAction('reject')}
                                        disabled={isApproving}
                                        className="px-5 py-2 bg-red-600 text-white rounded-lg text-xs font-black uppercase tracking-wider hover:bg-red-700 transition-all shadow-lg hover:shadow-red-500/40 disabled:opacity-50 flex items-center gap-2 border-none"
                                    >
                                        <XCircle className="h-4 w-4" />
                                        Reject
                                    </button>
                                </>
                            )}
                            {selectedPendingIds.size === 0 && (
                                <button
                                    onClick={() => handlePendingAction('approve', pendingTransactions.map(tx => tx.id!))}
                                    disabled={isApproving}
                                    className="px-5 py-2 bg-cyan-600 text-white rounded-lg text-xs font-black uppercase tracking-wider hover:bg-cyan-700 transition-all shadow-lg hover:shadow-cyan-500/40 disabled:opacity-50 flex items-center gap-2 border-none"
                                >
                                    <CheckCircle className="h-4 w-4" />
                                    Approve All
                                </button>
                            )}
                        </div>
                    </div>
                    <div className="overflow-x-auto max-h-[300px] overflow-y-auto">
                        <table className="min-w-full">
                            <thead className="bg-cyan-500/5 text-[10px] font-black text-cyan-700 dark:text-cyan-400 uppercase tracking-tighter sticky top-0 bg-card/95 backdrop-blur-sm z-10">
                                <tr>
                                    <th className="px-4 py-2 text-left w-8">
                                        <input 
                                            type="checkbox" 
                                            checked={selectedPendingIds.size === pendingTransactions.length}
                                            onChange={() => {
                                                if (selectedPendingIds.size === pendingTransactions.length) setSelectedPendingIds(new Set());
                                                else setSelectedPendingIds(new Set(pendingTransactions.map(tx => tx.id!)));
                                            }}
                                            className="rounded text-cyan-500"
                                        />
                                    </th>
                                    <th className="px-4 py-2 text-left">Date</th>
                                    <th className="px-4 py-2 text-left">Type</th>
                                    <th className="px-4 py-2 text-left">Symbol</th>
                                    <th className="px-4 py-2 text-right">Qty</th>
                                    <th className="px-4 py-2 text-right">Price</th>
                                    <th className="px-4 py-2 text-right">Total</th>
                                    <th className="px-4 py-2 text-left">Account</th>
                                    <th className="px-4 py-2 text-right">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="text-sm divide-y divide-cyan-500/5">
                                {pendingTransactions.map((tx) => (
                                    <tr key={`pending-${tx.id}`} className={`hover:bg-cyan-500/5 transition-colors group ${selectedPendingIds.has(tx.id!) ? 'bg-cyan-500/10' : ''}`}>
                                        <td className="px-4 py-3">
                                            <input 
                                                type="checkbox" 
                                                checked={selectedPendingIds.has(tx.id!)}
                                                onChange={() => {
                                                    const next = new Set(selectedPendingIds);
                                                    if (next.has(tx.id!)) next.delete(tx.id!);
                                                    else next.add(tx.id!);
                                                    setSelectedPendingIds(next);
                                                }}
                                                className="rounded text-cyan-500"
                                            />
                                        </td>
                                        <td className="px-4 py-3 text-[12px] text-muted-foreground whitespace-nowrap">{tx.Date}</td>
                                        <td className="px-4 py-3">
                                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-widest ${getTransactionTypeStyle(tx.Type)}`}>
                                                {tx.Type}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3">
                                            <StockTicker symbol={tx.Symbol} currency={tx["Local Currency"]} />
                                        </td>
                                        <td className="px-4 py-3 text-right tabular-nums">{tx.Quantity || '-'}</td>
                                        <td className="px-4 py-3 text-right tabular-nums">{tx["Price/Share"]?.toFixed(2) || '-'}</td>
                                        <td className="px-4 py-3 text-right font-bold tabular-nums">
                                            {tx["Total Amount"] ? Math.abs(tx["Total Amount"]).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '-'}
                                        </td>
                                        <td className="px-4 py-3 text-xs text-muted-foreground whitespace-nowrap">{tx.Account}</td>
                                        <td className="px-4 py-3 text-right">
                                            <div className="flex justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                                <button 
                                                    onClick={() => handlePendingAction('approve', [tx.id!])}
                                                    className="p-1.5 text-emerald-500 hover:bg-emerald-500/10 rounded transition-all"
                                                    title="Approve"
                                                >
                                                    <CheckCircle className="h-4 w-4" />
                                                </button>
                                                <button 
                                                    onClick={() => handlePendingAction('reject', [tx.id!])}
                                                    className="p-1.5 text-red-500 hover:bg-red-500/10 rounded transition-all"
                                                    title="Reject"
                                                >
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
                <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                    {/* Primary Actions Group (Left) */}
                    <div className="flex flex-row items-center gap-2 overflow-x-auto pb-1 no-scrollbar">
                        <button
                            onClick={handleAdd}
                            className="flex-shrink-0 px-3 py-2 bg-[#0097b2] text-white rounded-md hover:bg-[#0086a0] transition-colors text-xs font-bold flex items-center gap-1.5"
                        >
                            <Plus className="h-3.5 w-3.5" />
                            <span>Add</span>
                        </button>
                        
                        <div className="flex-shrink-0 flex items-center bg-purple-600 rounded-md overflow-hidden h-9">
                            <button
                                onClick={handleImportClick}
                                disabled={isImporting}
                                className="px-2 h-full flex items-center gap-1.5 text-white hover:bg-purple-700 transition-colors text-xs font-bold disabled:opacity-50 border-none"
                                title="Only used to import from an IBKR trade confirmation PDF file"
                            >
                                <FileText className="h-3.5 w-3.5 shrink-0" />
                                <span className="whitespace-nowrap">{isImporting ? '...' : 'Import'}</span>
                            </button>
                            <div className="w-[1px] h-4 bg-white/20" />
                            <select
                                value={importAccount}
                                onChange={(e) => setImportAccount(e.target.value)}
                                className="bg-purple-700 text-white text-[9px] font-bold h-full border-none focus:ring-0 cursor-pointer appearance-none pl-1.5 pr-6 text-right uppercase tracking-tighter"
                                style={{
                                    backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='white'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7' /%3E%3C/svg%3E")`,
                                    backgroundRepeat: 'no-repeat',
                                    backgroundPosition: 'right 0.3rem center',
                                    backgroundSize: '0.8em'
                                }}
                            >
                                <option value="" className="bg-purple-800 text-[9px]">DEFAULT</option>
                                {existingAccounts.map(acc => (
                                    <option key={acc} value={acc} className="bg-purple-800 text-[9px]">{acc}</option>
                                ))}
                            </select>
                            <div className="w-[1px] h-4 bg-white/20" />
                            <label 
                                className="flex items-center gap-1 px-2 h-full hover:bg-purple-700 transition-colors cursor-pointer select-none"
                                htmlFor="auto-add-cash-import"
                            >
                                <input
                                    type="checkbox"
                                    id="auto-add-cash-import"
                                    checked={autoAddCashOnImport}
                                    onChange={(e) => setAutoAddCashOnImport(e.target.checked)}
                                    className="h-3 w-3 rounded border-none bg-white/10 text-white focus:ring-offset-0 focus:ring-0 cursor-pointer"
                                />
                                <span className="text-[9px] font-bold text-white/90 uppercase tracking-tighter">Auto</span>
                            </label>
                        </div>

                        <input
                            type="file"
                            accept=".pdf,image/*"
                            ref={fileInputRef}
                            style={{ display: 'none' }}
                            onChange={handleFileUpload}
                        />

                        {selectedIds.size > 0 && (
                            <button
                                onClick={handleBulkDelete}
                                className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors text-sm font-medium flex items-center gap-2"
                            >
                                <Trash2 className="h-4 w-4" />
                                <span>Delete ({selectedIds.size})</span>
                            </button>
                        )}
                    </div>

                    {/* Secondary Actions & Info Group (Right) */}
                    <div className="flex flex-wrap items-center gap-2 lg:justify-end flex-1">
                        <div className="text-xs font-medium text-muted-foreground bg-secondary/50 px-3 py-2 rounded-lg border border-border/50 hidden md:block whitespace-nowrap">
                            Showing <span className="text-foreground font-bold">{visibleTransactions.length}</span> of <span className="text-foreground font-bold">{filteredTransactions.length}</span>
                        </div>

                        <div className="flex items-center gap-1 bg-secondary/30 p-1 rounded-lg">
                            <button
                                onClick={() => { setShowFilters(!showFilters); if (showFilters) resetFilters(); }}
                                className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all flex items-center gap-2 ${showFilters 
                                    ? 'bg-[#0097b2] text-white shadow-sm' 
                                    : 'text-foreground hover:bg-accent/10'}`}
                            >
                                <Filter className="h-3.5 w-3.5" />
                                <span>Filters</span>
                            </button>
                            
                            <button
                                onClick={() => setShowInternalCash(!showInternalCash)}
                                className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all flex items-center gap-2 ${showInternalCash
                                    ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
                                    : 'text-foreground hover:bg-accent/10'}`}
                            >
                                {showInternalCash ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                                <span>Internal Cash</span>
                            </button>
                        </div>

                        <button
                            onClick={() => exportToCSV(filteredTransactions, 'transactions.csv')}
                            className="p-2 text-foreground bg-secondary rounded-lg hover:bg-accent/10 transition-all"
                            title="Export CSV"
                        >
                            <Download className="h-4 w-4" />
                        </button>

                        <button
                            onClick={() => setMobileViewMode(current => current === 'card' ? 'table' : 'card')}
                            className="md:hidden p-2 text-foreground bg-secondary rounded-lg hover:bg-accent/10 transition-all"
                            title={mobileViewMode === 'card' ? 'Switch to Table View' : 'Switch to Card View'}
                        >
                            {mobileViewMode === 'card' ? <TableIcon className="w-4 h-4" /> : <LayoutGrid className="w-4 h-4" />}
                        </button>
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
                                className="bg-card border-none text-foreground rounded-md px-3 py-2 text-sm w-full focus:ring-indigo-500 focus:border-indigo-500"
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
                                className="bg-card border-none text-foreground rounded-md px-3 py-2 text-sm w-full focus:ring-indigo-500 focus:border-indigo-500"
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
                                className="bg-card border-none text-foreground rounded-md px-3 py-2 text-sm w-full focus:ring-cyan-500 focus:border-cyan-500 appearance-none pr-8"
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
                            className="flex-1 md:flex-none px-4 py-2 bg-secondary text-foreground rounded-md hover:bg-accent/10 transition-colors text-sm font-medium text-center"
                        >
                            Reset Filters
                        </button>
                    </div>
                )}
            </div>

            {/* Desktop Table View */}
            <div className={`metric-card card-shine overflow-hidden relative ${mobileViewMode === 'table' ? 'block' : 'hidden'} md:block transition-all`}>
                <div className="absolute top-0 left-0 right-0 h-[2px] bg-slate-500 opacity-80" />
                <div className="overflow-x-auto border-none">
                    <table className="min-w-full border-none">
                        <thead className="bg-secondary sticky top-0 z-10 font-semibold border-none">
                            <tr>
                                <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground w-12 sticky left-0 z-20 bg-secondary/95 backdrop-blur-md">
                                    <input
                                        type="checkbox"
                                        checked={selectedIds.size === visibleTransactions.length && visibleTransactions.length > 0}
                                        onChange={handleSelectAll}
                                        className="rounded text-cyan-500"
                                    />
                                </th>
                                <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground min-w-[100px] sticky left-12 z-20 bg-secondary/95 backdrop-blur-md border-none">Date</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground min-w-[80px] sticky left-[148px] z-20 bg-secondary/95 backdrop-blur-md border-none">Type</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground min-w-[100px] sticky left-[228px] z-20 bg-secondary/95 backdrop-blur-md border-none shadow-[2px_0_5px_-2px_rgba(0,0,0,0.3)]">Symbol</th>
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
                        <tbody>
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
                                    <tr key={index} className={`hover:bg-accent/5 transition-colors group border-none ${tx.id !== undefined && selectedIds.has(tx.id) ? 'bg-indigo-500/5' : ''}`}>
                                        <td className="px-4 py-3 whitespace-nowrap text-sm w-12 sticky left-0 z-10 bg-background/95 backdrop-blur-md">
                                            <input
                                                type="checkbox"
                                                checked={tx.id !== undefined && selectedIds.has(tx.id)}
                                                onChange={() => tx.id !== undefined && handleToggleSelect(tx.id)}
                                                className="rounded text-cyan-500"
                                            />
                                        </td>
                                        <td className="px-4 py-3 text-sm text-foreground whitespace-nowrap min-w-[100px] sticky left-12 z-10 bg-background/95 backdrop-blur-md border-none">{tx.Date ? tx.Date.split('T')[0].split(' ')[0] : '-'}</td>
                                        <td className="px-4 py-3 text-sm text-muted-foreground min-w-[80px] sticky left-[148px] z-10 bg-background/95 backdrop-blur-md border-none">
                                            <span className={`px-2 py-0.5 rounded text-xs font-medium whitespace-nowrap ${getTransactionTypeStyle(tx.Type)}`}>
                                                {formatTransactionType(tx.Type)}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3 whitespace-nowrap min-w-[100px] sticky left-[228px] z-10 bg-background/95 backdrop-blur-md border-none shadow-[2px_0_5px_-2px_rgba(0,0,0,0.3)]">
                                            <StockTicker symbol={tx.Symbol} currency={tx["Local Currency"]} />
                                        </td>
                                        <td className="px-4 py-3 text-sm text-right text-muted-foreground tabular-nums">
                                            {tx.Type.toLowerCase() === 'dividend' && tx.Quantity === 0 ? <span className="text-muted-foreground/30">-</span> : tx.Quantity}
                                        </td>
                                        <td className="px-4 py-3 text-sm text-right text-muted-foreground tabular-nums">
                                            {tx.Type.toLowerCase() === 'dividend' && (tx["Price/Share"] === 0 || !tx["Price/Share"]) ? <span className="text-muted-foreground/30">-</span> : tx["Price/Share"]?.toFixed(2)}
                                        </td>
                                        <td className="px-4 py-3 text-sm text-right font-medium text-foreground tabular-nums">
                                            {tx["Total Amount"] ? Math.abs(tx["Total Amount"]).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : <span className="text-muted-foreground/30">-</span>}
                                        </td>
                                        <td className="px-4 py-3 text-sm text-right text-muted-foreground tabular-nums">
                                            {tx.Commission ? tx.Commission.toFixed(2) : <span className="text-muted-foreground/30">-</span>}
                                        </td>
                                        <td className="px-4 py-3 text-sm text-muted-foreground whitespace-nowrap">{tx.Account}</td>
                                        <td className="px-4 py-3 text-sm text-right text-muted-foreground tabular-nums">{tx["Split Ratio"] ? tx["Split Ratio"] : <span className="text-muted-foreground/30">-</span>}</td>
                                        <td className="px-4 py-3 text-sm text-muted-foreground truncate max-w-xs" title={tx.Note}>{tx.Note || <span className="text-muted-foreground/30">-</span>}</td>
                                        <td className="px-4 py-3 text-sm text-muted-foreground">{tx["Local Currency"]}</td>
                                        <td className="px-4 py-3 text-sm text-right text-foreground whitespace-nowrap">
                                            <button
                                                onClick={() => handleEdit(tx)}
                                                className="text-indigo-500 hover:text-indigo-400 hover:bg-indigo-500/10 p-2 rounded transition-colors mr-1"
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
                    <div key={`mobile-tx-${index}`} className="bg-card rounded-lg p-4 border-none shadow-none">
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

                        <div className="grid grid-cols-2 gap-y-2 text-sm mt-3 pt-3">
                            <div className="text-muted-foreground">Quantity</div>
                            <div className="text-right font-medium text-foreground">
                                {tx.Type.toLowerCase() === 'dividend' && tx.Quantity === 0 ? '' : tx.Quantity}
                            </div>

                            <div className="text-muted-foreground">Price</div>
                            <div className="text-right font-medium text-foreground">
                                {tx.Type.toLowerCase() === 'dividend' && (tx["Price/Share"] === 0 || !tx["Price/Share"]) ? '' : tx["Price/Share"]?.toFixed(2)}
                            </div>

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
                                <div className="mt-2 pt-2 text-xs text-muted-foreground italic">
                                    {tx.Note}
                                </div>
                            )
                        }
                        <div className="mt-3 pt-2 flex justify-end gap-3">
                            <button
                                onClick={() => handleEdit(tx)}
                                className="text-indigo-500 hover:text-indigo-400 hover:bg-indigo-500/10 p-2 rounded transition-colors"
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
                ))}
            </div>

            {visibleRows < filteredTransactions.length && (
                <div className="flex justify-center gap-4 p-4">
                    <button
                        onClick={handleShowMore}
                        className="px-4 py-2 bg-[#0097b2] text-white rounded-md hover:bg-[#0086a0] transition-colors text-sm font-medium"
                    >
                        Show More
                    </button>
                    <button
                        onClick={handleShowAll}
                        className="px-4 py-2 bg-card text-foreground rounded-md hover:bg-secondary transition-colors text-sm font-medium"
                    >
                        Show All
                    </button>
                </div>
            )}
        </div>
    );
}
