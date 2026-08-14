import { formatCurrency } from '../../lib/utils';

export function isCashSymbol(symbol: string | undefined): boolean {
    const s = (symbol || '').toUpperCase();
    return s === '$CASH' || s === 'CASH' || s.startsWith('CASH (');
}

export const normalizeMarketName = (market: string): string => {
    if (!market) return 'Unknown';
    const m = market.toUpperCase();
    if (m.includes('NASDAQ') || m === 'NMS' || m === 'NGM' || m === 'NCM') return 'NASDAQ';
    if (m === 'NYQ' || m === 'NYSE' || m.includes('NEW YORK')) return 'NYSE';
    if (m === 'ASE' || m === 'AMEX') return 'AMEX';
    if (m === 'PCX' || m === 'ARCA' || m.includes('ARCA')) return 'NYSE Arca';
    return market; // Return original if no match
};

export const getCellClass = (val: unknown, header: string): string => {
    if (val === null || val === undefined || val === '-' || val === '' || (typeof val === 'number' && Math.abs(val) < 0.0001)) {
        return 'text-muted-foreground/40 font-light';
    }
    if (typeof val !== 'number') return '';
    if (['Day Chg', 'Day Chg %', 'Unreal. G/L', 'Unreal. G/L %', 'Real. G/L', 'Total G/L', 'Total Ret %', 'FX G/L', 'FX G/L %', 'IRR (%)'].includes(header)) {
        if (Math.abs(val) < 0.001) return 'text-muted-foreground/40 font-light';
        return val > 0 ? 'text-emerald-600 dark:text-emerald-400 font-medium' : 'text-red-600 dark:text-red-500 font-medium';
    }
    return '';
};

export const formatHoldingValue = (val: unknown, field: string, currency: string): string => {
    if (val === null || val === undefined) return '-';
    if (typeof val === 'string') return val;
    if (typeof val !== 'number') return String(val);

    const num = val;
    if (field.includes('%') || field.includes('pct_') || field === 'IRR (%)') {
        if (num === Infinity || num === -Infinity) return 'N/A';
        return `${num >= 0 ? '+' : ''}${num.toFixed(2)}%`;
    }
    if (field.includes('Price') || field.includes('Value') || field.includes('Cost') || field.includes('Gain') || field.includes('Div') || field.includes('Balance')) {
        return formatCurrency(num, currency);
    }
    if (field === 'Quantity') {
        return num.toLocaleString(undefined, { maximumFractionDigits: 4 });
    }
    return num.toLocaleString();
};
