/**
 * When a tax lot turns long-term — the twin of `MarketTime.daysUntilLongTerm`
 * on the native clients.
 *
 * Long-term means held *more than* one year: the holding period starts the day
 * after purchase, so a lot bought on 10 Mar 2025 is still short-term on 10 Mar
 * 2026 and long-term from 11 Mar. Counting 365 days instead flips it a day
 * early, and two days early across a 29 February — so the app would call a
 * sale long-term on the one day it is not.
 */
import { marketToday } from './market_time';

const DAY_MS = 24 * 60 * 60 * 1000;

/** `YYYY-MM-DD` on which a lot bought on `acquired` becomes long-term, or null. */
export function longTermDate(acquired: string | null | undefined): string | null {
    const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(acquired ?? '');
    if (!m) return null;
    const [year, month, day] = [Number(m[1]), Number(m[2]), Number(m[3])];
    // The anniversary, clamped to the month's end: a 29 Feb purchase has its
    // anniversary on 28 Feb, not on a 1 Mar that Date.UTC would roll over to.
    const lastDay = new Date(Date.UTC(year + 1, month, 0)).getUTCDate();
    const anniversary = Date.UTC(year + 1, month - 1, Math.min(day, lastDay));
    return new Date(anniversary + DAY_MS).toISOString().slice(0, 10);
}

/**
 * Whole days from today on the market's clock until the lot is long-term:
 * positive while short-term, zero or negative once long-term. Null if the
 * purchase date can't be read.
 */
export function daysUntilLongTerm(
    acquired: string | null | undefined,
    today: string = marketToday(),
): number | null {
    const lt = longTermDate(acquired);
    if (!lt) return null;
    return Math.round((Date.parse(`${lt}T00:00:00Z`) - Date.parse(`${today}T00:00:00Z`)) / DAY_MS);
}
