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
        // Check if there's a space that needs removing too (Intl sometimes adds NBSP or space)
        return formatted.replace('THB', '฿').replace(/\s/g, '');
    }

    return formatted;
}

export function formatPercent(value: number): string {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(value);
}

export function formatCompactNumber(number: number, currency?: string): string {
    const formatter = new Intl.NumberFormat('en-US', {
        notation: 'compact',
        maximumFractionDigits: 2,
        style: currency ? 'currency' : 'decimal',
        currency: currency,
    });

    let formatted = formatter.format(number);

    if (currency === 'THB') {
        formatted = formatted.replace('THB', '฿').replace(/\s/g, '');
    }

    return formatted;
}

/**
 * Returns a semi-transparent Tailwind background color based on the percentage value provided.
 * Useful for heatmap effects (e.g., green for positive, red for negative).
 */
export function getHeatmapClass(percentage: number | null | undefined): string {
    if (percentage === null || percentage === undefined || isNaN(percentage)) return 'bg-transparent';
    if (Math.abs(percentage) < 0.01) return 'bg-transparent'; // Neutral

    // Cap intensity at 20%
    const intensity = Math.min(Math.abs(percentage) / 20, 1);

    // Choose base color depending on positive or negative, adjusting opacity
    if (percentage > 0) {
        // Emerald scales
        if (intensity < 0.2) return 'bg-emerald-500/5 hover:bg-emerald-500/10 dark:bg-emerald-500/10';
        if (intensity < 0.4) return 'bg-emerald-500/10 hover:bg-emerald-500/20 dark:bg-emerald-500/20';
        if (intensity < 0.7) return 'bg-emerald-500/20 hover:bg-emerald-500/30 dark:bg-emerald-500/30';
        return 'bg-emerald-500/30 hover:bg-emerald-500/40 dark:bg-emerald-500/40';
    } else {
        // Rose scales
        if (intensity < 0.2) return 'bg-rose-500/5 hover:bg-rose-500/10 dark:bg-rose-500/10';
        if (intensity < 0.4) return 'bg-rose-500/10 hover:bg-rose-500/20 dark:bg-rose-500/20';
        if (intensity < 0.7) return 'bg-rose-500/20 hover:bg-rose-500/30 dark:bg-rose-500/30';
        return 'bg-rose-500/30 hover:bg-rose-500/40 dark:bg-rose-500/40';
    }
}

/**
 * Returns soft categorical tailwind color classes (bg and text) based on a string hash.
 * Optimized for consistent, pleasing colors for tags, sectors, etc.
 */
export function getColorForString(str: string | null | undefined): string {
    if (!str) return 'bg-zinc-100 text-zinc-700 dark:bg-zinc-800/60 dark:text-zinc-300';

    // Simple hash function
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }

    const colors = [
        'bg-blue-100 text-blue-700 dark:bg-blue-500/10 dark:text-blue-400',
        'bg-indigo-100 text-indigo-700 dark:bg-indigo-500/10 dark:text-indigo-400',
        'bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-400',
        'bg-fuchsia-100 text-fuchsia-700 dark:bg-fuchsia-500/10 dark:text-fuchsia-400',
        'bg-pink-100 text-pink-700 dark:bg-pink-500/10 dark:text-pink-400',
        'bg-rose-100 text-rose-700 dark:bg-rose-500/10 dark:text-rose-400',
        'bg-orange-100 text-orange-700 dark:bg-orange-500/10 dark:text-orange-400',
        'bg-amber-100 text-amber-700 dark:bg-amber-500/10 dark:text-amber-400',
        'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400',
        'bg-teal-100 text-teal-700 dark:bg-teal-500/10 dark:text-teal-400',
        'bg-cyan-100 text-cyan-700 dark:bg-cyan-500/10 dark:text-cyan-400',
        'bg-sky-100 text-sky-700 dark:bg-sky-500/10 dark:text-sky-400'
    ];

    const index = Math.abs(hash) % colors.length;
    return colors[index];
}
