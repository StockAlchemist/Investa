import React from 'react';
import { ChevronUp, ChevronDown, Pencil, Trash2, AlertCircle } from 'lucide-react';
import { Transaction } from '../../lib/api';
import { SortableKey } from './types';
import { formatTransactionType, getTotalAmountStyle, dupKey } from './transactionsUtils';
import StockTicker from '../StockTicker';

interface TransactionsDesktopTableProps {
    transactions: Transaction[];
    sortBy: SortableKey;
    sortDirection: 'asc' | 'desc';
    handleSort: (key: SortableKey) => void;
    duplicateKeys: Set<string>;
    onEdit: (tx: Transaction) => void;
    onDelete: (tx: Transaction) => void;
}

function SortIndicator({ active, direction }: { active: boolean; direction: 'asc' | 'desc' }) {
    if (!active) {
        return <ChevronDown className="w-3 h-3 opacity-30 group-hover:opacity-60 transition-opacity" />;
    }
    return direction === 'asc'
        ? <ChevronUp className="w-3 h-3 text-foreground" />
        : <ChevronDown className="w-3 h-3 text-foreground" />;
}

function getTypeBadgeStyle(type: string): string {
    const t = type.toLowerCase();
    if (t === 'buy') return 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20';
    if (t === 'sell') return 'bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20';
    if (t === 'dividend') return 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20';
    if (t === 'deposit') return 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20';
    if (t === 'withdrawal') return 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20';
    if (t === 'transfer') return 'bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20';
    return 'bg-secondary text-muted-foreground border-border/60';
}

export const TransactionsDesktopTable: React.FC<TransactionsDesktopTableProps> = ({
    transactions,
    sortBy,
    sortDirection,
    handleSort,
    duplicateKeys,
    onEdit,
    onDelete,
}) => {
    return (
        <div className="overflow-x-auto rounded-2xl border border-border/60 bg-card shadow-sm">
            <table className="w-full text-left border-collapse">
                <thead>
                    <tr className="border-b border-border/60 bg-muted/30 text-[11px] font-bold uppercase tracking-wider text-muted-foreground select-none">
                        <th
                            onClick={() => handleSort('Date')}
                            className="py-3 px-4 cursor-pointer hover:text-foreground transition-colors group"
                        >
                            <div className="flex items-center gap-1">
                                Date
                                <SortIndicator active={sortBy === 'Date'} direction={sortDirection} />
                            </div>
                        </th>
                        <th
                            onClick={() => handleSort('Type')}
                            className="py-3 px-3 cursor-pointer hover:text-foreground transition-colors group"
                        >
                            <div className="flex items-center gap-1">
                                Type
                                <SortIndicator active={sortBy === 'Type'} direction={sortDirection} />
                            </div>
                        </th>
                        <th
                            onClick={() => handleSort('Symbol')}
                            className="py-3 px-3 cursor-pointer hover:text-foreground transition-colors group"
                        >
                            <div className="flex items-center gap-1">
                                Symbol
                                <SortIndicator active={sortBy === 'Symbol'} direction={sortDirection} />
                            </div>
                        </th>
                        <th
                            onClick={() => handleSort('Quantity')}
                            className="py-3 px-3 text-right cursor-pointer hover:text-foreground transition-colors group"
                        >
                            <div className="flex items-center justify-end gap-1">
                                Quantity
                                <SortIndicator active={sortBy === 'Quantity'} direction={sortDirection} />
                            </div>
                        </th>
                        <th
                            onClick={() => handleSort('Price/Share')}
                            className="py-3 px-3 text-right cursor-pointer hover:text-foreground transition-colors group"
                        >
                            <div className="flex items-center justify-end gap-1">
                                Price/Share
                                <SortIndicator active={sortBy === 'Price/Share'} direction={sortDirection} />
                            </div>
                        </th>
                        <th
                            onClick={() => handleSort('Commission')}
                            className="py-3 px-3 text-right cursor-pointer hover:text-foreground transition-colors group"
                        >
                            <div className="flex items-center justify-end gap-1">
                                Fee
                                <SortIndicator active={sortBy === 'Commission'} direction={sortDirection} />
                            </div>
                        </th>
                        <th
                            onClick={() => handleSort('Total Amount')}
                            className="py-3 px-4 text-right cursor-pointer hover:text-foreground transition-colors group"
                        >
                            <div className="flex items-center justify-end gap-1">
                                Total Amount
                                <SortIndicator active={sortBy === 'Total Amount'} direction={sortDirection} />
                            </div>
                        </th>
                        <th
                            onClick={() => handleSort('Account')}
                            className="py-3 px-3 cursor-pointer hover:text-foreground transition-colors group"
                        >
                            <div className="flex items-center gap-1">
                                Account
                                <SortIndicator active={sortBy === 'Account'} direction={sortDirection} />
                            </div>
                        </th>
                        <th className="py-3 px-4 text-right">Actions</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-border/40 text-xs">
                    {transactions.length === 0 ? (
                        <tr>
                            <td colSpan={9} className="py-8 text-center text-muted-foreground">
                                No transactions match your filters.
                            </td>
                        </tr>
                    ) : (
                        transactions.map((tx, idx) => {
                            const isDup = duplicateKeys.has(dupKey(tx));
                            const { className: totalColor, display: totalText } = getTotalAmountStyle(tx);

                            return (
                                <tr
                                    key={tx.id ?? `${tx.Symbol}-${tx.Date}-${idx}`}
                                    className="hover:bg-muted/20 transition-colors group"
                                >
                                    {/* Date */}
                                    <td className="py-3 px-4 whitespace-nowrap text-muted-foreground tabular-nums">
                                        <div className="flex items-center gap-1.5">
                                            <span>{(tx.Date || '').split('T')[0].split(' ')[0]}</span>
                                            {isDup && (
                                                <span
                                                    className="inline-flex items-center gap-0.5 px-1.5 py-0.2 text-[9px] font-bold bg-amber-500/15 text-amber-600 dark:text-amber-400 rounded border border-amber-500/30"
                                                    title="Possible duplicate transaction"
                                                >
                                                    <AlertCircle className="w-2.5 h-2.5" />
                                                    Dup
                                                </span>
                                            )}
                                        </div>
                                    </td>

                                    {/* Type */}
                                    <td className="py-3 px-3 whitespace-nowrap">
                                        <span className={`inline-block px-2 py-0.5 text-[10px] font-bold rounded-full border ${getTypeBadgeStyle(tx.Type || '')}`}>
                                            {formatTransactionType(tx.Type || '')}
                                        </span>
                                    </td>

                                    {/* Symbol */}
                                    <td className="py-3 px-3 whitespace-nowrap font-bold text-foreground">
                                        {tx.Symbol ? (
                                            <StockTicker symbol={tx.Symbol} />
                                        ) : (
                                            <span className="text-muted-foreground/40">-</span>
                                        )}
                                    </td>

                                    {/* Quantity */}
                                    <td className="py-3 px-3 text-right whitespace-nowrap tabular-nums text-foreground">
                                        {Number(tx.Quantity || 0).toLocaleString(undefined, { maximumFractionDigits: 4 })}
                                    </td>

                                    {/* Price/Share */}
                                    <td className="py-3 px-3 text-right whitespace-nowrap tabular-nums text-muted-foreground">
                                        {Number(tx['Price/Share'] || 0) > 0
                                            ? Number(tx['Price/Share']).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })
                                            : '-'}
                                    </td>

                                    {/* Commission */}
                                    <td className="py-3 px-3 text-right whitespace-nowrap tabular-nums text-muted-foreground">
                                        {Number(tx.Commission || 0) > 0
                                            ? Number(tx.Commission).toFixed(2)
                                            : '-'}
                                    </td>

                                    {/* Total Amount */}
                                    <td className={`py-3 px-4 text-right whitespace-nowrap tabular-nums font-semibold ${totalColor}`}>
                                        {totalText}
                                    </td>

                                    {/* Account */}
                                    <td className="py-3 px-3 whitespace-nowrap text-muted-foreground">
                                        <span className="text-[11px] bg-secondary/80 px-2 py-0.5 rounded-md border border-border/40">
                                            {tx.Account || 'Default'}
                                        </span>
                                    </td>

                                    {/* Actions */}
                                    <td className="py-3 px-4 text-right whitespace-nowrap">
                                        <div className="flex items-center justify-end gap-1 opacity-80 group-hover:opacity-100 transition-opacity">
                                            <button
                                                onClick={() => onEdit(tx)}
                                                className="p-1 text-muted-foreground hover:text-cyan-500 hover:bg-secondary rounded-lg transition-colors"
                                                title="Edit Transaction"
                                            >
                                                <Pencil className="w-3.5 h-3.5" />
                                            </button>
                                            <button
                                                onClick={() => onDelete(tx)}
                                                className="p-1 text-muted-foreground hover:text-red-500 hover:bg-secondary rounded-lg transition-colors"
                                                title="Delete Transaction"
                                            >
                                                <Trash2 className="w-3.5 h-3.5" />
                                            </button>
                                        </div>
                                    </td>
                                </tr>
                            );
                        })
                    )}
                </tbody>
            </table>
        </div>
    );
};
