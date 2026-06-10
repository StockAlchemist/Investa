import { describe, expect, it } from 'vitest';
import {
    cn,
    formatCurrency,
    formatPercent,
    formatCompactNumber,
    getHeatmapClass,
    getColorForString,
} from '@/lib/utils';

describe('cn', () => {
    it('merges conflicting tailwind classes (last wins)', () => {
        expect(cn('p-2', 'p-4')).toBe('p-4');
    });

    it('drops falsy values', () => {
        expect(cn('a', false && 'b', undefined, 'c')).toBe('a c');
    });
});

describe('formatCurrency', () => {
    it('formats USD with two decimals', () => {
        expect(formatCurrency(1234.5)).toBe('$1,234.50');
    });

    it('formats THB with the ฿ symbol and no stray spaces', () => {
        const result = formatCurrency(50000, 'THB');
        expect(result).toContain('฿');
        expect(result).not.toMatch(/\s/);
        expect(result).toContain('50,000');
    });

    it('renders zero-decimal currencies without decimals', () => {
        expect(formatCurrency(1500, 'JPY')).toBe('¥1,500');
        expect(formatCurrency(1500.75, 'KRW')).not.toContain('.');
    });

    it('handles negative values', () => {
        expect(formatCurrency(-42.1)).toBe('-$42.10');
    });
});

describe('formatPercent', () => {
    it('formats a ratio as a percentage with two decimals', () => {
        expect(formatPercent(0.59079)).toBe('59.08%');
    });

    it('handles the special values the API can produce', () => {
        expect(formatPercent(Infinity)).toBe('∞');
        expect(formatPercent(-Infinity)).toBe('-∞');
        expect(formatPercent(NaN)).toBe('-');
    });

    it('formats negatives', () => {
        expect(formatPercent(-0.0512)).toBe('-5.12%');
    });
});

describe('formatCompactNumber', () => {
    it('returns "0" for zero', () => {
        expect(formatCompactNumber(0)).toBe('0');
    });

    it('compacts thousands and millions', () => {
        expect(formatCompactNumber(1500)).toBe('1.5K');
        expect(formatCompactNumber(2_340_000)).toBe('2.34M');
    });

    it('prefixes a currency symbol when given', () => {
        expect(formatCompactNumber(1500, 'USD')).toBe('$1.5K');
    });

    it('replaces THB with ฿', () => {
        expect(formatCompactNumber(50000, 'THB')).toContain('฿');
    });
});

describe('getHeatmapClass', () => {
    it('is transparent for null/undefined/NaN/near-zero', () => {
        expect(getHeatmapClass(null)).toBe('bg-transparent');
        expect(getHeatmapClass(undefined)).toBe('bg-transparent');
        expect(getHeatmapClass(NaN)).toBe('bg-transparent');
        expect(getHeatmapClass(0.005)).toBe('bg-transparent');
    });

    it('uses emerald scales for gains, rose for losses', () => {
        expect(getHeatmapClass(5)).toContain('emerald');
        expect(getHeatmapClass(-5)).toContain('rose');
    });

    it('intensity increases with magnitude and caps at 20%', () => {
        const weak = getHeatmapClass(1);
        const strong = getHeatmapClass(19);
        const capped = getHeatmapClass(100);
        expect(weak).toContain('emerald-500/5');
        expect(strong).toContain('emerald-500/30');
        expect(capped).toBe(strong);
    });
});

describe('getColorForString', () => {
    it('is deterministic for the same input', () => {
        expect(getColorForString('Technology')).toBe(getColorForString('Technology'));
    });

    it('returns the neutral fallback for empty input', () => {
        expect(getColorForString(null)).toContain('zinc');
        expect(getColorForString(undefined)).toContain('zinc');
        expect(getColorForString('')).toContain('zinc');
    });

    it('returns a bg+text class pair', () => {
        const cls = getColorForString('Energy');
        expect(cls).toMatch(/bg-\w+-100/);
        expect(cls).toMatch(/text-\w+-700/);
    });
});
