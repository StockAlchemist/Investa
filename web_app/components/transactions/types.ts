import { Transaction } from '../../lib/api';

export interface TransactionsTableProps {
    transactions: Transaction[];
    currency?: string;
    isLoading?: boolean;
}

export type SortableKey =
    | 'Date' | 'Type' | 'Symbol' | 'Quantity' | 'Price/Share'
    | 'Total Amount' | 'Commission' | 'Account' | 'Local Currency';

export type DatePreset = 'all' | 'mtd' | 'ytd' | '30d' | '90d' | '1y' | 'custom';

export const DATE_PRESETS: { key: DatePreset; label: string }[] = [
    { key: 'all', label: 'All' },
    { key: 'mtd', label: 'This month' },
    { key: 'ytd', label: 'YTD' },
    { key: '30d', label: '30D' },
    { key: '90d', label: '90D' },
    { key: '1y', label: '1Y' },
    { key: 'custom', label: 'Custom' },
];

export const CANONICAL_TYPES = [
    'Buy', 'Sell', 'Dividend', 'Transfer', 'Interest', 'Fees', 'Tax',
    'Deposit', 'Withdrawal', 'Spin-off', 'Split', 'Short Sell', 'Buy To Cover',
];
