import { describe, expect, it } from 'vitest';
import { daysUntilLongTerm, longTermDate } from '../../lib/tax_lots';

describe('longTermDate', () => {
    it('is the day after the first anniversary', () => {
        expect(longTermDate('2025-03-10')).toBe('2026-03-11');
        expect(longTermDate('2025-12-31')).toBe('2027-01-01');
    });

    it('does not lose a day to a 29 February in the holding year', () => {
        // 365 days after 1 Feb 2024 is 31 Jan 2025 — two days early.
        expect(longTermDate('2024-02-01')).toBe('2025-02-02');
    });

    it('anchors a 29 February purchase on 28 February', () => {
        expect(longTermDate('2024-02-29')).toBe('2025-03-01');
    });

    it('accepts a datetime and rejects garbage', () => {
        expect(longTermDate('2025-03-10T00:00:00')).toBe('2026-03-11');
        expect(longTermDate('')).toBeNull();
        expect(longTermDate(undefined)).toBeNull();
    });
});

describe('daysUntilLongTerm', () => {
    it('is still short-term on the anniversary itself', () => {
        expect(daysUntilLongTerm('2025-03-10', '2026-03-10')).toBe(1);
        expect(daysUntilLongTerm('2025-03-10', '2026-03-11')).toBe(0);
        expect(daysUntilLongTerm('2025-03-10', '2027-03-11')).toBe(-365);
    });
});
