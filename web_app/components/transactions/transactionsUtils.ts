import { Transaction } from '../../lib/api';
import { CANONICAL_TYPES, DatePreset } from './types';

// Identity key for duplicate detection: same symbol + date + type + |qty| + amount + account + note.
export function dupKey(tx: Transaction): string {
    return `${tx.Symbol}|${(tx.Date || '').split('T')[0].split(' ')[0]}|${tx.Type}|${Math.abs(tx.Quantity || 0)}|${tx['Total Amount'] ?? ''}|${tx.Account}|${tx.Note ?? ''}`;
}

// Looser key used to flag import-review rows that already exist in the table.
export function importMatchKey(tx: Transaction): string {
    const date = (tx.Date || '').split('T')[0].split(' ')[0];
    const sym = (tx.Symbol || '').toUpperCase();
    const type = String(tx.Type || '').toLowerCase();
    if (type === 'dividend' || type === 'tax' || type === 'withholding tax') {
        return `${sym}|${date}|${type}`;
    }
    const qty = Math.abs(Number(tx.Quantity) || 0);
    return `${sym}|${date}|${type}|${qty}`;
}

export function formatTransactionType(type: string): string {
    return type.replace(/\w\S*/g, (txt) => txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase());
}

export function canonicalType(raw: string): string {
    const key = raw.toLowerCase().replace(/[\s-]+/g, '');
    return CANONICAL_TYPES.find(t => t.toLowerCase().replace(/[\s-]+/g, '') === key)
        ?? formatTransactionType(raw);
}

// Cash-impact styling for the Total Amount column:
export const OUTFLOW_TYPES = new Set(['buy', 'withdrawal', 'fees', 'fee', 'tax', 'withholding tax', 'buy to cover']);
export const INFLOW_TYPES = new Set(['sell', 'deposit', 'dividend', 'interest', 'short sell']);

export function isCashSymbol(symbol: string | undefined): boolean {
    const s = (symbol || '').toUpperCase();
    return s === '$CASH' || s === 'CASH' || s.startsWith('CASH (');
}

export function displayAmount(tx: Transaction): number {
    const total = Math.abs(Number(tx['Total Amount']) || 0);
    if (total > 1e-9) return total;
    if (isCashSymbol(tx.Symbol)) return Math.abs(Number(tx.Quantity) || 0);
    return Math.abs((Number(tx.Quantity) || 0) * (Number(tx['Price/Share']) || 0));
}

export function getTotalAmountStyle(tx: Transaction): { className: string; display: string } {
    const mag = displayAmount(tx);
    if (mag < 1e-9) return { className: 'text-muted-foreground/30', display: '-' };
    const formatted = mag.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    const txType = (tx.Type || '').toLowerCase().trim();
    if (OUTFLOW_TYPES.has(txType)) {
        return { className: 'text-red-600 dark:text-red-500', display: `-${formatted}` };
    }
    if (INFLOW_TYPES.has(txType)) {
        return { className: 'text-emerald-600 dark:text-emerald-400', display: formatted };
    }
    // neutral: Transfer, Split, Spin-off, etc.
    return { className: 'text-muted-foreground', display: formatted };
}

// Returns inclusive [from, to] as YYYY-MM-DD strings (or null bounds for open ends).
export function computeDateRange(preset: DatePreset, customFrom: string, customTo: string): { from: string | null; to: string | null } {
    if (preset === 'all') return { from: null, to: null };
    if (preset === 'custom') return { from: customFrom || null, to: customTo || null };
    const now = new Date();
    const iso = (d: Date) => d.toISOString().slice(0, 10);
    const to = iso(now);
    let from: Date;
    if (preset === 'mtd') from = new Date(now.getFullYear(), now.getMonth(), 1);
    else if (preset === 'ytd') from = new Date(now.getFullYear(), 0, 1);
    else if (preset === '30d') from = new Date(now.getTime() - 30 * 86400000);
    else if (preset === '90d') from = new Date(now.getTime() - 90 * 86400000);
    else from = new Date(now.getFullYear() - 1, now.getMonth(), now.getDate()); // 1y
    return { from: iso(from), to };
}
