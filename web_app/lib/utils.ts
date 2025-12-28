import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs))
}

export const CURRENCY_SYMBOLS: Record<string, string> = {
    USD: '$',
    EUR: '€',
    GBP: '£',
    JPY: '¥',
    CNY: '¥',
    THB: '฿',
    SGD: 'S$',
};

export function formatCurrency(value: number, currency: string = 'USD'): string {
    const formatted = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: currency,
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(value);

    // Manual override for THB symbol
    if (currency === 'THB') {
        return formatted.replace('THB', '฿');
    }

    return formatted;
}
