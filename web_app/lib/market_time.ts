/**
 * Market-local date reckoning — the client-side twin of `src/server/calendar_events.py`.
 *
 * Every calendar date Investa shows belongs to a market, not to the viewer: an
 * earnings report is "today" while it is still today on that exchange, whether
 * the browser is in Bangkok or Los Angeles. Counting days against the device
 * clock instead is what made a US report a day out for a UTC+7 viewer.
 */

/** Fallback zone when an event does not name its exchange's (Investa is US-first). */
export const DEFAULT_MARKET_TIMEZONE = 'America/New_York';

/** `YYYY-MM-DD` for an ISO date/datetime string, or null if unparseable. */
function calendarDay(iso: string | null | undefined): string | null {
    if (typeof iso !== 'string') return null;
    const day = iso.slice(0, 10);
    return /^\d{4}-\d{2}-\d{2}$/.test(day) ? day : null;
}

/** Today's `YYYY-MM-DD` in a market's own timezone. */
export function marketToday(timeZone?: string | null): string {
    const now = new Date();
    for (const tz of [timeZone, DEFAULT_MARKET_TIMEZONE]) {
        if (!tz) continue;
        try {
            const parts = new Intl.DateTimeFormat('en-US', {
                timeZone: tz,
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
            }).formatToParts(now);
            const at = (type: string) => parts.find(p => p.type === type)?.value ?? '';
            const day = `${at('year')}-${at('month')}-${at('day')}`;
            if (calendarDay(day)) return day;
        } catch {
            // Unknown zone (or an engine without full ICU) — fall through to the default.
        }
    }
    // Last resort: the device's own date, which is what we were trying to avoid.
    return now.toISOString().slice(0, 10);
}

/**
 * Whole days from today-in-`timeZone` to a calendar date. Negative for the past,
 * null if the date can't be read.
 *
 * Both sides are pinned to UTC midnight so the subtraction can't be skewed by a
 * DST transition falling between them.
 */
export function marketDayDiff(iso: string | null | undefined, timeZone?: string | null): number | null {
    const day = calendarDay(iso);
    if (!day) return null;
    const target = Date.parse(`${day}T00:00:00Z`);
    const today = Date.parse(`${marketToday(timeZone)}T00:00:00Z`);
    if (isNaN(target) || isNaN(today)) return null;
    return Math.round((target - today) / 86400000);
}

/**
 * Whether a calendar date falls no later than `months` months past today in a
 * market's own timezone — the horizon filter behind the "3 months / 1 year"
 * calendar toggles.
 *
 * A month here is a calendar month, so the cutoff lands on the same day-of-month
 * (overflowing as JS does: Nov 30 + 3 months reaches into early March).
 */
export function isWithinMarketMonths(
    iso: string | null | undefined,
    months: number,
    timeZone?: string | null,
): boolean {
    const day = calendarDay(iso);
    if (!day) return false;
    const target = Date.parse(`${day}T00:00:00Z`);
    if (isNaN(target)) return false;
    const [year, month, date] = marketToday(timeZone).split('-').map(Number);
    return target <= Date.UTC(year, month - 1 + months, date);
}

/**
 * Format a date-only string ("2026-08-05") for display.
 *
 * Rendered in UTC on purpose: the value is a calendar day, not an instant, so
 * letting the browser localize it would slide it to the previous day for anyone
 * west of UTC.
 */
export function formatCalendarDate(
    iso: string | null | undefined,
    options: Intl.DateTimeFormatOptions = { year: 'numeric', month: 'short', day: 'numeric' },
): string {
    const day = calendarDay(iso);
    if (!day) return typeof iso === 'string' ? iso : '';
    const d = new Date(`${day}T00:00:00Z`);
    if (isNaN(d.getTime())) return day;
    return d.toLocaleDateString(undefined, { ...options, timeZone: 'UTC' });
}
