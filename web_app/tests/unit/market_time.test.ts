import { describe, expect, it, afterEach, vi } from 'vitest';
import {
    DEFAULT_MARKET_TIMEZONE,
    formatCalendarDate,
    isWithinMarketMonths,
    marketDayDiff,
    marketToday,
} from '../../lib/market_time';

/** 02:00 UTC on Jul 30 2026 — already Jul 30 in Bangkok, still Jul 29 in New York. */
const BANGKOK_MORNING = new Date('2026-07-30T02:00:00Z');

afterEach(() => {
    vi.useRealTimers();
});

function at(moment: Date) {
    vi.useFakeTimers();
    vi.setSystemTime(moment);
}

describe('marketToday', () => {
    it('reads the date on the market clock, not the device clock', () => {
        at(BANGKOK_MORNING);
        expect(marketToday('Asia/Bangkok')).toBe('2026-07-30');
        expect(marketToday('America/New_York')).toBe('2026-07-29');
    });

    it('defaults to the US market zone when no zone is given', () => {
        at(BANGKOK_MORNING);
        expect(marketToday()).toBe(marketToday(DEFAULT_MARKET_TIMEZONE));
        expect(marketToday(null)).toBe('2026-07-29');
    });

    it('falls back to the default zone for an unusable one', () => {
        at(BANGKOK_MORNING);
        expect(marketToday('Mars/Olympus')).toBe('2026-07-29');
    });
});

describe('marketDayDiff', () => {
    it('counts days against the exchange date', () => {
        at(BANGKOK_MORNING);
        // The regression: a US event on Jul 29 is *today* in New York, even though
        // the viewer's own calendar already says Jul 30.
        expect(marketDayDiff('2026-07-29', 'America/New_York')).toBe(0);
        expect(marketDayDiff('2026-07-29', 'Asia/Bangkok')).toBe(-1);
        expect(marketDayDiff('2026-07-30', 'America/New_York')).toBe(1);
        expect(marketDayDiff('2026-08-05', 'America/New_York')).toBe(7);
    });

    it('is unskewed by a DST transition inside the span', () => {
        at(new Date('2026-10-20T14:00:00Z')); // Oct 20 in New York, pre-transition
        // US DST ends Nov 1 2026; the count must still be a whole number of days.
        expect(marketDayDiff('2026-11-10', 'America/New_York')).toBe(21);
    });

    it('accepts datetimes and rejects junk', () => {
        at(BANGKOK_MORNING);
        expect(marketDayDiff('2026-07-29T20:00:00Z', 'America/New_York')).toBe(0);
        expect(marketDayDiff('', 'America/New_York')).toBeNull();
        expect(marketDayDiff('soon', 'America/New_York')).toBeNull();
        expect(marketDayDiff(null)).toBeNull();
        expect(marketDayDiff(undefined)).toBeNull();
    });
});

describe('isWithinMarketMonths', () => {
    it('measures the horizon from the exchange date', () => {
        at(BANGKOK_MORNING); // Jul 29 in New York, Jul 30 in Bangkok
        expect(isWithinMarketMonths('2026-10-29', 3, 'America/New_York')).toBe(true);
        expect(isWithinMarketMonths('2026-10-30', 3, 'America/New_York')).toBe(false);
        // A Bangkok payment gets its horizon from the Bangkok date, one day later.
        expect(isWithinMarketMonths('2026-10-30', 3, 'Asia/Bangkok')).toBe(true);
    });

    it('spans a full year for the 12-month horizon', () => {
        at(BANGKOK_MORNING);
        expect(isWithinMarketMonths('2027-07-29', 12, 'America/New_York')).toBe(true);
        expect(isWithinMarketMonths('2027-07-30', 12, 'America/New_York')).toBe(false);
    });

    it('keeps past payments in view and drops junk', () => {
        at(BANGKOK_MORNING);
        // The backend already filters out the past; the horizon is an upper bound only.
        expect(isWithinMarketMonths('2026-01-05', 3, 'America/New_York')).toBe(true);
        expect(isWithinMarketMonths('soon', 3, 'America/New_York')).toBe(false);
        expect(isWithinMarketMonths(null, 3)).toBe(false);
    });
});

describe('formatCalendarDate', () => {
    it('formats the calendar day itself rather than a local instant', () => {
        // A date-only value localized in a negative-offset zone would slide back
        // to Aug 4; pinning to UTC keeps the day the market named.
        expect(formatCalendarDate('2026-08-05')).toMatch(/Aug 5, 2026/);
        expect(formatCalendarDate('2026-08-05', { month: 'short', day: 'numeric' })).toMatch(/Aug 5/);
    });

    it('passes unparseable input straight through', () => {
        expect(formatCalendarDate('')).toBe('');
        expect(formatCalendarDate('soon')).toBe('soon');
        expect(formatCalendarDate(null)).toBe('');
    });

    // A date belongs to the market, not to the reader's device — the same rule
    // that pins the zone. A Thai-locale browser defaults to the Buddhist era and
    // would label a quarter ending March 2026 as 2569, which is not a year any
    // filing mentions.
    //
    // The formatting request has to be read back off the call rather than
    // rebuilt here: `formatCalendarDate` renders in the *runtime's* locale, so a
    // test that formats with an explicit 'th-TH' proves only that the platform
    // can do the right thing when asked. What needs guarding is that Investa
    // asks.
    it('names the year in the market’s calendar, not the reader’s', () => {
        const spy = vi.spyOn(Date.prototype, 'toLocaleDateString');
        formatCalendarDate('2026-03-31', { month: 'short', year: 'numeric' });
        expect(spy).toHaveBeenCalledTimes(1);
        const requested = spy.mock.calls[0][1] as Intl.DateTimeFormatOptions;
        spy.mockRestore();

        const rendered = new Date('2026-03-31T00:00:00Z').toLocaleDateString('th-TH', requested);
        expect(rendered).toContain('2026');
        expect(rendered).not.toContain('2569');
    });

    it('renders the day in the market’s zone, not the reader’s', () => {
        const spy = vi.spyOn(Date.prototype, 'toLocaleDateString');
        formatCalendarDate('2026-01-01');
        const requested = spy.mock.calls[0][1] as Intl.DateTimeFormatOptions;
        spy.mockRestore();
        expect(requested.timeZone).toBe('UTC');
    });
});
