import React, { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Clock, CheckCircle, XCircle } from 'lucide-react';
import { Transaction, fetchPendingIbkr, approveIbkr, rejectIbkr } from '../../lib/api';
import StockTicker from '../StockTicker';
import { formatCalendarDate } from '@/lib/market_time';

function getPendingTypeStyle(type: string): string {
    const t = (type || '').toUpperCase();
    if (['BUY', 'DEPOSIT', 'BUY TO COVER'].includes(t)) {
        return 'bg-up/12 text-up';
    }
    if (['SELL', 'WITHDRAWAL', 'SHORT SELL'].includes(t)) {
        return 'bg-down/12 text-down';
    }
    if (['DIVIDEND', 'INTEREST'].includes(t)) {
        return 'bg-indigo-500/10 text-indigo-600 dark:text-indigo-400';
    }
    if (['FEES', 'FEE', 'TAX', 'WITHHOLDING TAX'].includes(t)) {
        return 'bg-orange-500/10 text-orange-600 dark:text-orange-400';
    }
    return 'bg-violet-500/10 text-violet-600 dark:text-violet-400';
}

export const PendingIbkrBanner: React.FC = () => {
    const queryClient = useQueryClient();
    const [selectedPendingIds, setSelectedPendingIds] = useState<Set<number>>(new Set());
    const [isApproving, setIsApproving] = useState(false);

    const { data: pendingTransactions = [] } = useQuery<Transaction[]>({
        queryKey: ['pendingIbkr'],
        queryFn: fetchPendingIbkr,
    });

    if (pendingTransactions.length === 0) return null;

    const handlePendingAction = async (action: 'approve' | 'reject', explicitIds?: number[]) => {
        const idsToProcess = explicitIds || Array.from(selectedPendingIds);
        if (idsToProcess.length === 0) return;

        setIsApproving(true);
        try {
            if (action === 'approve') {
                await approveIbkr(idsToProcess);
            } else {
                await rejectIbkr(idsToProcess);
            }

            setSelectedPendingIds(prev => {
                const next = new Set(prev);
                idsToProcess.forEach(id => next.delete(id));
                return next;
            });

            queryClient.invalidateQueries({ queryKey: ['pendingIbkr'] });
            queryClient.invalidateQueries({ queryKey: ['transactions'] });
            queryClient.invalidateQueries({ queryKey: ['summary'] });
            queryClient.invalidateQueries({ queryKey: ['holdings'] });
        } catch (error) {
            console.error(`Failed to ${action} pending transactions:`, error);
            alert(`Failed to ${action} transactions`);
        } finally {
            setIsApproving(false);
        }
    };

    return (
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
                                <td className="px-4 py-3 text-[12px] text-muted-foreground whitespace-nowrap">{formatCalendarDate(tx.Date)}</td>
                                <td className="px-4 py-3">
                                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-widest ${getPendingTypeStyle(tx.Type)}`}>
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
                                            className="p-1.5 text-up hover:bg-up/12 rounded transition-all"
                                            title="Approve"
                                        >
                                            <CheckCircle className="h-4 w-4" />
                                        </button>
                                        <button
                                            onClick={() => handlePendingAction('reject', [tx.id!])}
                                            className="p-1.5 text-down hover:bg-down/12 rounded transition-all"
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
    );
};
