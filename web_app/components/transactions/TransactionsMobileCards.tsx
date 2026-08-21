import React from 'react';
import { Pencil, Trash2, AlertCircle } from 'lucide-react';
import { Transaction } from '../../lib/api';
import { formatTransactionType, getTotalAmountStyle, dupKey } from './transactionsUtils';
import StockTicker from '../StockTicker';

interface TransactionsMobileCardsProps {
    transactions: Transaction[];
    duplicateKeys: Set<string>;
    onEdit: (tx: Transaction) => void;
    onDelete: (tx: Transaction) => void;
    currency?: string;
}

export const TransactionsMobileCards: React.FC<TransactionsMobileCardsProps> = ({
    transactions,
    duplicateKeys,
    onEdit,
    onDelete,
    currency = 'USD',
}) => {
    if (transactions.length === 0) {
        return (
            <div className="py-8 text-center text-muted-foreground text-xs bg-card rounded-2xl border border-border/60">
                No transactions match your filters.
            </div>
        );
    }

    return (
        <div className="space-y-3">
            {transactions.map((tx, idx) => {
                const isDup = duplicateKeys.has(dupKey(tx));
                const { className: totalColor, display: totalText } = getTotalAmountStyle(tx);

                return (
                    <div
                        key={tx.id ?? `${tx.Symbol}-${tx.Date}-${idx}`}
                        className="bg-card border border-border/60 rounded-2xl p-4 space-y-3 shadow-sm hover:border-border transition-colors"
                    >
                        {/* Header: Date, Dup badge, Actions */}
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <span className="text-xs text-muted-foreground font-medium tabular-nums">
                                    {(tx.Date || '').split('T')[0].split(' ')[0]}
                                </span>
                                {isDup && (
                                    <span className="inline-flex items-center gap-0.5 px-1.5 py-0.2 text-[9px] font-bold bg-amber-500/15 text-amber-600 dark:text-amber-400 rounded border border-amber-500/30">
                                        <AlertCircle className="w-2.5 h-2.5" />
                                        Dup
                                    </span>
                                )}
                            </div>
                            <div className="flex items-center gap-1">
                                <button
                                    onClick={() => onEdit(tx)}
                                    className="p-1.5 text-muted-foreground hover:text-cyan-500 hover:bg-secondary rounded-lg transition-colors"
                                >
                                    <Pencil className="w-3.5 h-3.5" />
                                </button>
                                <button
                                    onClick={() => onDelete(tx)}
                                    className="p-1.5 text-muted-foreground hover:text-red-500 hover:bg-secondary rounded-lg transition-colors"
                                >
                                    <Trash2 className="w-3.5 h-3.5" />
                                </button>
                            </div>
                        </div>

                        {/* Middle: Symbol & Total */}
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <span className="font-bold text-base text-foreground">
                                    {tx.Symbol ? <StockTicker symbol={tx.Symbol} currency={tx['Local Currency'] || currency} /> : '-'}
                                </span>
                                <span className="px-2 py-0.5 text-[10px] font-bold rounded-full bg-secondary text-muted-foreground">
                                    {formatTransactionType(tx.Type || '')}
                                </span>
                            </div>
                            <div className={`text-base font-bold tabular-nums ${totalColor}`}>
                                {totalText}
                            </div>
                        </div>

                        {/* Details grid */}
                        <div className="grid grid-cols-3 gap-2 pt-2 border-t border-border/40 text-xs">
                            <div>
                                <span className="block text-[10px] text-muted-foreground uppercase tracking-wider">Qty</span>
                                <span className="font-medium text-foreground tabular-nums">
                                    {Number(tx.Quantity || 0).toLocaleString(undefined, { maximumFractionDigits: 4 })}
                                </span>
                            </div>
                            <div>
                                <span className="block text-[10px] text-muted-foreground uppercase tracking-wider">Price</span>
                                <span className="font-medium text-foreground tabular-nums">
                                    {Number(tx['Price/Share'] || 0) > 0 ? Number(tx['Price/Share']).toFixed(2) : '-'}
                                </span>
                            </div>
                            <div>
                                <span className="block text-[10px] text-muted-foreground uppercase tracking-wider">Account</span>
                                <span className="font-medium text-foreground truncate block">
                                    {tx.Account || 'Default'}
                                </span>
                            </div>
                        </div>

                        {/* Note if present */}
                        {tx.Note && (
                            <div className="text-[11px] text-muted-foreground/80 italic pt-1 border-t border-border/20">
                                {tx.Note}
                            </div>
                        )}
                    </div>
                );
            })}
        </div>
    );
};
