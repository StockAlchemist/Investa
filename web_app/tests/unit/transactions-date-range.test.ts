import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { computeDateRange } from '@/components/transactions/transactionsUtils';

// Investa's own clock. Node re-reads TZ when it is assigned at runtime.
const originalTz = process.env.TZ;
beforeAll(() => { process.env.TZ = 'Asia/Bangkok'; });
afterAll(() => { process.env.TZ = originalTz; });

// 11:00 in Bangkok, 00:00 in New York: both on 24 Sep.
const MIDDAY = new Date('2026-09-24T04:00:00Z');
// 08:00 in Bangkok on 1 Oct, still 30 Sep in New York.
const BANGKOK_MORNING = new Date('2026-10-01T01:00:00Z');

describe('computeDateRange', () => {
    it('runs on a Bangkok clock (else the cases below prove nothing)', () => {
        expect(MIDDAY.getHours()).toBe(11);
    });

    it('starts "This month" on the 1st, not the last day of the month before', () => {
        expect(computeDateRange('mtd', '', '', MIDDAY)).toEqual({ from: '2026-09-01', to: null });
    });

    it('starts YTD on 1 January, not 31 December', () => {
        expect(computeDateRange('ytd', '', '', MIDDAY).from).toBe('2026-01-01');
    });

    it('counts rolling windows in calendar days', () => {
        expect(computeDateRange('30d', '', '', MIDDAY).from).toBe('2026-08-25');
        expect(computeDateRange('90d', '', '', MIDDAY).from).toBe('2026-06-26');
        expect(computeDateRange('1y', '', '', MIDDAY).from).toBe('2025-09-24');
    });

    it("reckons the month on the market's day, open-ended so Bangkok's today stays in", () => {
        // New York is still in September, so the month is September's; with no
        // upper bound a trade dated 1 Oct in Bangkok still lists.
        expect(computeDateRange('mtd', '', '', BANGKOK_MORNING)).toEqual({ from: '2026-09-01', to: null });
    });

    it('passes custom and open ranges through', () => {
        expect(computeDateRange('all', '', '', MIDDAY)).toEqual({ from: null, to: null });
        expect(computeDateRange('custom', '2026-01-02', '', MIDDAY)).toEqual({ from: '2026-01-02', to: null });
    });
});
