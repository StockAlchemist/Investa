import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { chartDay, chartInstant, formatCalendarDate, formatMarketTime } from '../../lib/market_time';

// Investa's own clock, where every shape below used to go wrong.
// Node re-reads TZ when it is assigned at runtime.
const originalTz = process.env.TZ;
beforeAll(() => { process.env.TZ = 'Asia/Bangkok'; });
afterAll(() => { process.env.TZ = originalTz; });

// The shapes the API actually sends (captured from /history, /stock_history
// and /market_history on 24 Sep 2026).
const DAILY_UTC_MIDNIGHT = '2026-08-05T00:00:00+00:00';
const DAILY_BARE = '2026-08-05';
const INTRADAY_WITH_OFFSET = '2026-09-23T09:30:00-04:00';
const INTRADAY_ZONELESS_UTC = '2026-09-23 13:30:00';

describe('chartInstant', () => {
    it('reads a zoneless /market_history bar as UTC, not on the device clock', () => {
        expect(chartInstant(INTRADAY_ZONELESS_UTC)).toBe(Date.parse('2026-09-23T13:30:00Z'));
    });

    it('passes offsets and numbers through', () => {
        expect(chartInstant(INTRADAY_WITH_OFFSET)).toBe(Date.parse('2026-09-23T13:30:00Z'));
        expect(chartInstant(42)).toBe(42);
    });
});

describe('chartDay', () => {
    it('keeps a daily bar on its own day rather than New York\'s evening before', () => {
        expect(chartDay(DAILY_UTC_MIDNIGHT)).toBe('2026-08-05');
        expect(chartDay(DAILY_BARE)).toBe('2026-08-05');
        expect(chartDay(Date.parse(DAILY_UTC_MIDNIGHT))).toBe('2026-08-05');
        expect(formatCalendarDate(chartDay(DAILY_UTC_MIDNIGHT))).toMatch(/^05 \S+ 2026$/);
    });
});

describe('formatMarketTime', () => {
    it('puts both intraday shapes at the New York open, day first', () => {
        for (const v of [INTRADAY_WITH_OFFSET, INTRADAY_ZONELESS_UTC]) {
            const out = formatMarketTime(v, { weekday: true });
            expect(out).toMatch(/^\S+, 23 \S+ /);
            expect(out).toMatch(/9:30/);
        }
    });

    it('adds the year only when asked, in the Gregorian era', () => {
        expect(formatMarketTime(INTRADAY_WITH_OFFSET, { year: true })).toMatch(/^23 \S+ 2026 /);
        expect(formatMarketTime(INTRADAY_WITH_OFFSET)).not.toMatch(/2026/);
    });
});
