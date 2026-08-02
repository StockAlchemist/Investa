/**
 * The pure parts of the financial-statement chart: which line items open, which
 * colour each one wears, which of them can share a y-axis, and how a period end
 * is named. Kept out of the modal so they can be tested without rendering it.
 */
import { formatCalendarDate } from './market_time';

export type StatementPeriod = 'quarterly' | 'annual';

/** What the chart opens on for each statement, in preference order. */
export const DEFAULT_CHART_METRICS: Record<string, string[]> = {
    income: ['Total Revenue', 'Gross Profit', 'Net Income'],
    balance: ['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity'],
    cash: ['Operating Cash Flow', 'Free Cash Flow', 'Capital Expenditure'],
    equity: ['Stockholders Equity', 'Retained Earnings'],
};

/** Four is where a validated categorical palette stops being separable. */
export const MAX_CHART_SERIES = 4;

/** A hard ceiling on plotted periods, whatever range is asked for. */
export const MAX_CHART_PERIODS = 80;

/**
 * Past this many periods, grouped bars become hairlines and the chart is read
 * as a shape rather than a set of magnitudes — which is a line's job.
 */
export const BAR_TO_LINE_THRESHOLD = 24;

export type StatementRange = '5y' | '10y' | 'max';

/**
 * How many periods a range covers. Quarterly and annual mean the same span in
 * years, which is the unit an investor thinks in — not the same column count.
 */
export function periodsInRange(range: StatementRange, periodType: StatementPeriod): number {
    if (range === 'max') return MAX_CHART_PERIODS;
    const years = range === '5y' ? 5 : 10;
    return periodType === 'quarterly' ? years * 4 : years;
}

/** Quarters get five years by default; a year per bar reads well over ten. */
export function defaultRange(periodType: StatementPeriod): StatementRange {
    return periodType === 'quarterly' ? '5y' : '10y';
}

/**
 * The first four slots of the validated categorical palette, stepped for each
 * surface. The light steps sit below 3:1 on the light surface, so the statement
 * table under the chart carries the relief.
 */
export const SERIES_COLORS_LIGHT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'];
export const SERIES_COLORS_DARK = ['#3987e5', '#d95926', '#199e70', '#c98500'];

/**
 * Two measures whose magnitudes differ by more than this get their own chart
 * rather than a second y-axis — revenue and EPS on one scale is a flat line
 * next to a mountain, and on two scales it is a lie about their relationship.
 *
 * Ten means the smaller series still reaches a tenth of the frame, which is
 * about where a bar stops being a readable magnitude and becomes a mark that
 * something was there. S&P Global's capex is 29x its operating cash flow and
 * splits either way; the cases this setting decides are the ones in between.
 */
export const SAME_SCALE_RATIO = 10;

export function isFiniteNumber(v: unknown): v is number {
    return typeof v === 'number' && Number.isFinite(v);
}

/**
 * How a period end is named on an axis or a column head. A quarter needs its
 * month — four columns a year all say "2026" otherwise — and a year does not,
 * because filed period ends are the company's own 52/53-week dates and two of
 * them can land in one calendar year.
 */
export function periodAxisLabel(iso: string, periodType: StatementPeriod): string {
    return periodType === 'quarterly'
        ? formatCalendarDate(iso, { month: 'short', year: 'numeric' })
        : iso.slice(0, 4);
}

/** Compact for statement magnitudes, plain for per-share figures. */
export function formatStatementValue(val: number): string {
    if (Math.abs(val) >= 1000) {
        return new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 2 }).format(val);
    }
    return new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(val);
}

/**
 * A line item has to be reported across the history to be worth opening on.
 * Gross profit is the case that forces this: Meta never tags it, so it exists
 * for the handful of periods Yahoo derives it in and nowhere else — charted by
 * default it is an empty series next to two full ones, which reads as a broken
 * chart rather than as an unreported number.
 */
export const MIN_DEFAULT_COVERAGE = 0.5;

/**
 * The line items to open on: the statement's preferred ones, less any too
 * sparsely reported to plot. Falls back to the preferred list when that would
 * leave nothing — a company that tags almost nothing still gets a chart.
 */
export function pickDefaultMetrics(
    preferred: string[],
    rows: { label: string; values: (number | null)[] }[],
    limit: number = MAX_CHART_SERIES,
): string[] {
    const byLabel = new Map(rows.map(r => [r.label, r.values]));
    const available = preferred.filter(m => byLabel.has(m));
    const covered = available.filter(m => {
        const values = byLabel.get(m) ?? [];
        if (!values.length) return false;
        return values.filter(isFiniteNumber).length / values.length >= MIN_DEFAULT_COVERAGE;
    });
    return (covered.length ? covered : available).slice(0, limit);
}

/**
 * Add or remove a line item, holding every other one in the colour slot it
 * already had. Removing series 1 of 3 must not repaint series 2 and 3, so a
 * dropped item leaves a hole rather than closing the array up.
 */
export function toggleSlot(
    slots: (string | null)[],
    label: string,
    max: number = MAX_CHART_SERIES,
): (string | null)[] {
    const next = [...slots];
    const at = next.indexOf(label);
    if (at !== -1) {
        next[at] = null;
        return next;
    }
    const free = next.indexOf(null);
    if (free !== -1) {
        next[free] = label;
        return next;
    }
    if (next.length < max) return [...next, label];
    return next;
}

/**
 * Split series into sets that can honestly share one y-axis. Never a second
 * axis: two scales on one frame let the author decide where the lines cross.
 */
export function groupBySharedScale<T extends { maxAbs: number }>(series: T[]): T[][] {
    const groups: T[][] = [];
    [...series]
        .sort((a, b) => b.maxAbs - a.maxAbs)
        .forEach(s => {
            const home = groups.find(g => {
                const lead = g[0].maxAbs;
                // Two all-zero rows share a frame; a zero row never joins a real one.
                if (!lead || !s.maxAbs) return !lead && !s.maxAbs;
                return lead <= s.maxAbs * SAME_SCALE_RATIO;
            });
            if (home) home.push(s);
            else groups.push([s]);
        });
    return groups;
}
