import React, { useState, useMemo } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { FileText, AlertCircle, CheckCircle, Trash2 } from 'lucide-react';
import { Transaction, addTransactionsBatch, fetchTransactions } from '../../lib/api';
import { importMatchKey } from './transactionsUtils';
import { cn } from '../../lib/utils';

interface ImportReviewModalProps {
    isReviewing: boolean;
    setIsReviewing: (rev: boolean) => void;
    reviewTransactions: Transaction[];
    setReviewTransactions: React.Dispatch<React.SetStateAction<Transaction[]>>;
    importAccount: string;
    setImportAccount?: (acc: string) => void;
    autoAddCash?: boolean;
    setAutoAddCash?: (auto: boolean) => void;
    availableAccounts?: string[];
    accountCashModeMap?: Record<string, string>;
    transactions: Transaction[];
}

export const ImportReviewModal: React.FC<ImportReviewModalProps> = ({
    isReviewing,
    setIsReviewing,
    reviewTransactions,
    setReviewTransactions,
    importAccount,
    setImportAccount,
    autoAddCash = true,
    setAutoAddCash,
    availableAccounts = [],
    accountCashModeMap = {},
    transactions,
}) => {
    const queryClient = useQueryClient();
    const [isImporting, setIsImporting] = useState(false);

    // Fetch all transactions for cross-account duplicate checking
    const { data: allTransactions = [] } = useQuery<Transaction[]>({
        queryKey: ['transactions', 'all-for-dup-check'],
        queryFn: () => fetchTransactions(),
        enabled: isReviewing,
        staleTime: 5 * 60 * 1000,
    });

    const existingTxKeys = useMemo(() => {
        const keys = new Set<string>();
        for (const tx of transactions) keys.add(importMatchKey(tx));
        for (const tx of allTransactions) keys.add(importMatchKey(tx));
        return keys;
    }, [transactions, allTransactions]);

    const reviewDuplicateCount = useMemo(
        () => reviewTransactions.reduce((n, tx) => existingTxKeys.has(importMatchKey(tx)) ? n + 1 : n, 0),
        [reviewTransactions, existingTxKeys],
    );

    const accountList = useMemo(() => {
        const set = new Set<string>(availableAccounts);
        transactions.forEach(t => { if (t.Account) set.add(t.Account); });
        return Array.from(set).filter(acc => acc && acc.trim().toLowerCase() !== 'all accounts' && acc.trim().toLowerCase() !== 'all').sort();
    }, [availableAccounts, transactions]);

    const isSelectedAccountManual = (() => {
        const targetAcc = importAccount || 'Default';
        const mode = (accountCashModeMap[targetAcc] || (importAccount ? 'Manual' : (accountCashModeMap['Default'] || 'Manual'))).toLowerCase();
        return mode === 'manual';
    })();

    if (!isReviewing || reviewTransactions.length === 0) return null;

    const handleUpdateReviewTransaction = (index: number, updated: Transaction) => {
        setReviewTransactions(prev => prev.map((tx, i) => i === index ? updated : tx));
    };

    const handleBulkAccountChange = (newAcc: string) => {
        setImportAccount?.(newAcc);
        const targetAcc = newAcc || 'Default';
        const mode = (accountCashModeMap[targetAcc] || (newAcc ? 'Manual' : (accountCashModeMap['Default'] || 'Manual'))).toLowerCase();
        if (mode !== 'manual') {
            setAutoAddCash?.(false);
        }
        setReviewTransactions(prev => prev.map(tx => ({
            ...tx,
            Account: newAcc || 'Default'
        })));
    };

    const handleRemoveFromReview = (index: number) => {
        setReviewTransactions(prev => prev.filter((_, i) => i !== index));
    };

    const handleReviewConfirm = async () => {
        if (reviewTransactions.length === 0) return;

        setIsImporting(true);
        try {
            const shouldAutoAddCash = isSelectedAccountManual && autoAddCash;
            const result = await addTransactionsBatch(reviewTransactions, shouldAutoAddCash);
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

    return (
        <div className="metric-card card-shine overflow-hidden animate-in fade-in zoom-in duration-500 relative border-2 border-indigo-500/20">
            <datalist id="import-review-accounts">
                <option value="Default" />
                {accountList.map(acc => (
                    <option key={acc} value={acc} />
                ))}
            </datalist>
            <div className="absolute top-0 left-0 right-0 h-[2px] bg-indigo-500" />
            <div className="px-4 py-4 bg-indigo-500/10 flex flex-wrap justify-between items-center gap-3 border-b border-indigo-500/10">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-indigo-500/20 rounded-full">
                        <FileText className="h-5 w-5 text-indigo-500" />
                    </div>
                    <div>
                        <h3 className="text-sm font-bold text-indigo-700 dark:text-indigo-400 uppercase tracking-widest">
                            Review Extracted Transactions ({reviewTransactions.length})
                        </h3>
                        <p className="text-[10px] text-muted-foreground uppercase font-semibold">
                            AI identified these from your document. Please verify before saving.
                        </p>
                        {reviewDuplicateCount > 0 && (
                            <p className="text-[10px] text-amber-600 dark:text-amber-400 uppercase font-bold flex items-center gap-1 mt-0.5">
                                <AlertCircle className="h-3 w-3" />
                                {reviewDuplicateCount} already in your table (highlighted)
                            </p>
                        )}
                    </div>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                    {/* Bulk account selector */}
                    <div className="flex items-center gap-1.5 text-xs">
                        <span className="text-muted-foreground text-[11px] font-medium">Account:</span>
                        <select
                            aria-label="Bulk Target Account"
                            value={importAccount}
                            onChange={e => handleBulkAccountChange(e.target.value)}
                            className="px-2.5 py-1 bg-background border border-indigo-500/30 rounded-lg text-xs font-semibold text-foreground focus:outline-none focus:ring-1 focus:ring-indigo-500"
                        >
                            <option value="">Default (Auto-detect)</option>
                            {accountList.map(acc => (
                                <option key={acc} value={acc}>{acc}</option>
                            ))}
                        </select>
                    </div>

                    {/* Auto-add cash toggle - only shown if selected account is manual */}
                    {setAutoAddCash && isSelectedAccountManual && (
                        <label className="flex items-center gap-1.5 text-xs text-foreground cursor-pointer select-none">
                            <input
                                type="checkbox"
                                checked={autoAddCash}
                                onChange={e => setAutoAddCash(e.target.checked)}
                                className="rounded text-indigo-500 focus:ring-indigo-500"
                            />
                            <span className="text-[11px] font-medium text-muted-foreground">Auto-add cash</span>
                        </label>
                    )}

                    <div className="flex gap-2">
                        <button
                            onClick={handleReviewConfirm}
                            disabled={isImporting}
                            className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-xs font-black uppercase tracking-wider hover:bg-indigo-700 transition-all shadow-lg hover:shadow-indigo-500/40 disabled:opacity-50 flex items-center gap-2 border-none cursor-pointer"
                        >
                            <CheckCircle className="h-4 w-4" />
                            Confirm & Import All
                        </button>
                        <button
                            onClick={() => { setReviewTransactions([]); setIsReviewing(false); }}
                            className="px-3 py-2 bg-secondary text-foreground rounded-lg text-xs font-bold uppercase tracking-wider hover:bg-accent/10 transition-all border-none cursor-pointer"
                            title="Discard"
                        >
                            <Trash2 className="h-4 w-4" />
                        </button>
                    </div>
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
                        {reviewTransactions.map((tx, idx) => {
                            const isDuplicate = existingTxKeys.has(importMatchKey(tx));
                            return (
                                <tr
                                    key={`review-${idx}`}
                                    className={cn(
                                        'transition-colors group',
                                        isDuplicate
                                            ? 'bg-amber-500/10 hover:bg-amber-500/20'
                                            : 'hover:bg-indigo-500/5',
                                    )}
                                    title={isDuplicate ? 'This transaction already exists in your table' : undefined}
                                >
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
                                            aria-label="Transaction Type"
                                            value={tx.Type}
                                            onChange={(e) => handleUpdateReviewTransaction(idx, { ...tx, Type: e.target.value })}
                                            className="bg-transparent border-none text-[10px] p-0 font-bold uppercase tracking-widest focus:ring-0 text-indigo-500 appearance-none cursor-pointer"
                                        >
                                            <option value="Buy">BUY</option>
                                            <option value="Sell">SELL</option>
                                            <option value="Dividend">DIVIDEND</option>
                                            <option value="Transfer">TRANSFER</option>
                                            <option value="Interest">INTEREST</option>
                                            <option value="Fees">FEES</option>
                                            <option value="Tax">TAX</option>
                                            <option value="Deposit">DEPOSIT</option>
                                            <option value="Withdrawal">WITHDRAWAL</option>
                                            <option value="Split">SPLIT</option>
                                            <option value="Spin-off">SPIN-OFF</option>
                                            <option value="Short Sell">SHORT SELL</option>
                                            <option value="Buy To Cover">BUY TO COVER</option>
                                        </select>
                                    </td>
                                    <td className="px-4 py-3">
                                        <div className="flex items-center gap-2">
                                            <input
                                                type="text"
                                                value={tx.Symbol}
                                                onChange={(e) => handleUpdateReviewTransaction(idx, { ...tx, Symbol: e.target.value.toUpperCase() })}
                                                className="bg-transparent border-none text-sm p-0 w-full font-bold focus:ring-0"
                                            />
                                            {isDuplicate && (
                                                <span className="shrink-0 inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-700 dark:text-amber-400 text-[9px] font-black uppercase tracking-wider whitespace-nowrap">
                                                    <AlertCircle className="h-2.5 w-2.5" />
                                                    Duplicate
                                                </span>
                                            )}
                                        </div>
                                    </td>
                                    <td className="px-4 py-3">
                                        <input
                                            type="number"
                                            value={tx.Quantity}
                                            onChange={(e) => handleUpdateReviewTransaction(idx, { ...tx, Quantity: parseFloat(e.target.value) || 0 })}
                                            className="bg-transparent border-none text-right text-sm p-0 w-full focus:ring-0 tabular-nums"
                                        />
                                    </td>
                                    <td className="px-4 py-3">
                                        <input
                                            type="number"
                                            value={tx["Price/Share"]}
                                            onChange={(e) => handleUpdateReviewTransaction(idx, { ...tx, "Price/Share": parseFloat(e.target.value) || 0 })}
                                            className="bg-transparent border-none text-right text-sm p-0 w-full focus:ring-0 tabular-nums"
                                        />
                                    </td>
                                    <td className="px-4 py-3">
                                        <input
                                            type="number"
                                            value={tx["Total Amount"]}
                                            onChange={(e) => handleUpdateReviewTransaction(idx, { ...tx, "Total Amount": parseFloat(e.target.value) || 0 })}
                                            className="bg-transparent border-none text-right text-sm p-0 w-full focus:ring-0 font-bold tabular-nums"
                                        />
                                    </td>
                                    <td className="px-4 py-3">
                                        <input
                                            type="text"
                                            value={tx.Account}
                                            placeholder="Account"
                                            list="import-review-accounts"
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
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
};
