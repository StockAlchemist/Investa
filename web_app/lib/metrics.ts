/**
 * The one catalogue of company metrics, and the scale each is read against.
 *
 * Shared by the S&P 500 heatmap and the stock detail window: a P/E of 31 has to
 * mean the same thing — and paint the same colour — whether it is read off a
 * tile or off a row in the detail window. The backend agrees on the field names
 * (`_fundamental_metrics` in server/routes/market.py builds both payloads).
 *
 * `mid` values are the index's own median for that metric, measured across all
 * 500 constituents, so roughly half the map falls either side and the scale
 * actually discriminates.
 */
import type { SP500HeatmapItem } from './api';

// ---------------------------------------------------------------------------
// Metric definitions
// ---------------------------------------------------------------------------
export type MetricGroup = 'Performance' | 'Valuation' | 'Earnings & Sales' | 'Profitability' | 'Market';

export interface MetricDef {
    key: string;
    label: string;
    group: MetricGroup;
    field: keyof SP500HeatmapItem;
    /** Value that colours neutral (black). Zero for returns; a typical reading
     *  for ratios, which are never meaningfully "zero-centred". */
    mid: number;
    /** Distance from `mid` at which the colour saturates. */
    clamp: number;
    format?: 'pct' | 'ratio' | 'dollar' | 'cap' | 'days';
    inverted?: boolean; // If true, lower is greener (e.g. P/E, Debt/Eq)
    /** How the value maps to colour.
     *
     *  'diverging' (default) encodes *polarity* — which side of `mid` a value
     *  falls on — as red↔green through a dark midpoint. It is only meaningful
     *  when both sides can actually occur.
     *
     *  'sequential' encodes *magnitude* on a one-hue ramp over [mid, mid+clamp].
     *  Used for quantities with a hard floor and no "bad" direction: dividend
     *  yield is dividends over price, so it cannot go below zero, and half a
     *  diverging scale would describe values that do not exist. */
    scale?: 'diverging' | 'sequential';
}

// `mid` values are set from the index's own median for that metric, measured
// across all 500 constituents, so roughly half the map falls either side and
// the scale actually discriminates.
export const METRICS: MetricDef[] = [
    // --- Performance. A return of zero is genuinely neutral, so these stay
    // zero-centred; the clamp widens with the horizon.
    { key: 'day',   label: '1-Day Performance',        group: 'Performance', field: 'change_pct',        mid: 0, clamp: 3,   format: 'pct' },
    { key: '1w',    label: '1-Week Performance',       group: 'Performance', field: 'week_change_pct',   mid: 0, clamp: 5,   format: 'pct' },
    { key: '1m',    label: '1-Month Performance',      group: 'Performance', field: 'month_change_pct',  mid: 0, clamp: 10,  format: 'pct' },
    { key: 'mtd',   label: 'Month to Date',            group: 'Performance', field: 'mtd_change_pct',    mid: 0, clamp: 10,  format: 'pct' },
    { key: '3m',    label: '3-Month Performance',      group: 'Performance', field: '3m_change_pct',     mid: 0, clamp: 15,  format: 'pct' },
    { key: '6m',    label: '6-Month Performance',      group: 'Performance', field: '6m_change_pct',     mid: 0, clamp: 25,  format: 'pct' },
    { key: 'ytd',   label: 'Year to Date',             group: 'Performance', field: 'ytd_change_pct',    mid: 0, clamp: 30,  format: 'pct' },
    { key: '1y',    label: '1-Year Performance',       group: 'Performance', field: '1y_change_pct',     mid: 0, clamp: 40,  format: 'pct' },
    { key: '3y',    label: '3-Year Performance',       group: 'Performance', field: '3y_change_pct',     mid: 0, clamp: 60,  format: 'pct' },
    { key: '5y',    label: '5-Year Performance',       group: 'Performance', field: '5y_change_pct',     mid: 0, clamp: 100, format: 'pct' },
    { key: '10y',   label: '10-Year Performance',      group: 'Performance', field: '10y_change_pct',    mid: 0, clamp: 200, format: 'pct' },
    // Both are one-sided by construction — a price cannot exceed its own
    // 52-week high or fall below its own low — so they take magnitude ramps
    // over their real range instead of a half-empty diverging scale.
    { key: 'dd52',  label: 'Drawdown from 52-Week High', group: 'Performance', field: 'drawdown_52w',      mid: -50, clamp: 50, format: 'pct', scale: 'sequential' },
    { key: 'up52',  label: 'Gain from 52-Week Low',      group: 'Performance', field: 'gain_from_52w_low', mid: 0,   clamp: 100, format: 'pct', scale: 'sequential' },

    // --- Valuation. Centred on a typical large-cap reading so "cheap" reads
    // green and "expensive" reads red; centring on zero would paint every
    // profitable company red.
    { key: 'pe',        label: 'P/E',            group: 'Valuation', field: 'pe_ratio',       mid: 25, clamp: 15,  format: 'ratio', inverted: true },
    { key: 'fwd_pe',    label: 'Forward P/E',    group: 'Valuation', field: 'forward_pe',     mid: 17, clamp: 10,  format: 'ratio', inverted: true },
    { key: 'peg',       label: 'PEG',            group: 'Valuation', field: 'peg_ratio',      mid: 2,  clamp: 1.5, format: 'ratio', inverted: true },
    { key: 'ps',        label: 'P/S',            group: 'Valuation', field: 'ps_ratio',       mid: 3,  clamp: 2.5, format: 'ratio', inverted: true },
    { key: 'pb',        label: 'P/B',            group: 'Valuation', field: 'pb_ratio',       mid: 4,  clamp: 3,   format: 'ratio', inverted: true },
    { key: 'p_fcf',     label: 'P/FCF',          group: 'Valuation', field: 'p_fcf',          mid: 24, clamp: 15,  format: 'ratio', inverted: true },
    { key: 'ev_ebitda', label: 'EV/EBITDA',      group: 'Valuation', field: 'ev_ebitda',      mid: 15, clamp: 10,  format: 'ratio', inverted: true },
    { key: 'ev_sales',  label: 'EV/Sales',       group: 'Valuation', field: 'ev_sales',       mid: 4,  clamp: 3,   format: 'ratio', inverted: true },
    // Dividends over price: floored at zero and one-directional, so it gets a
    // magnitude ramp. Top of scale is 5% — above the 90th percentile of the
    // index (3.9%), clipping only ~3% of constituents.
    { key: 'yield',     label: 'Dividend Yield', group: 'Valuation', field: 'dividend_yield', mid: 0, clamp: 5, format: 'pct', scale: 'sequential' },

    // --- Earnings & sales. Zero is meaningful here (a loss, or a decline), so
    // these stay zero-centred.
    { key: 'eps_ttm',      label: 'EPS TTM',                group: 'Earnings & Sales', field: 'eps_ttm',         mid: 0, clamp: 15, format: 'dollar' },
    { key: 'eps_qoq',      label: 'EPS Q/Q',                group: 'Earnings & Sales', field: 'eps_qoq',         mid: 0, clamp: 50, format: 'pct' },
    { key: 'eps_3y',       label: 'EPS Growth Past 3 Years', group: 'Earnings & Sales', field: 'eps_growth_3y',  mid: 0, clamp: 30, format: 'pct' },
    { key: 'eps_5y',       label: 'EPS Growth Past 5 Years', group: 'Earnings & Sales', field: 'eps_growth_5y',  mid: 0, clamp: 30, format: 'pct' },
    { key: 'eps_surprise', label: 'EPS Surprise',           group: 'Earnings & Sales', field: 'eps_surprise',    mid: 0, clamp: 10, format: 'pct' },
    // Revenue is a magnitude with a hard floor, not a polarity.
    { key: 'sales_ttm',    label: 'Sales TTM',              group: 'Earnings & Sales', field: 'sales_ttm',       mid: 0, clamp: 100e9, format: 'cap', scale: 'sequential' },
    { key: 'sales_qoq',    label: 'Sales Q/Q',              group: 'Earnings & Sales', field: 'sales_qoq',       mid: 0, clamp: 30, format: 'pct' },
    { key: 'sales_3y',     label: 'Sales Growth Past 3 Years', group: 'Earnings & Sales', field: 'sales_growth_3y', mid: 0, clamp: 25, format: 'pct' },
    { key: 'sales_5y',     label: 'Sales Growth Past 5 Years', group: 'Earnings & Sales', field: 'sales_growth_5y', mid: 0, clamp: 25, format: 'pct' },

    // --- Profitability & balance sheet.
    { key: 'roa',          label: 'ROA',            group: 'Profitability', field: 'roa',             mid: 0,  clamp: 20, format: 'pct' },
    { key: 'roe',          label: 'ROE',            group: 'Profitability', field: 'roe',             mid: 0,  clamp: 50, format: 'pct' },
    { key: 'roic',         label: 'ROIC',           group: 'Profitability', field: 'roic',            mid: 0,  clamp: 30, format: 'pct' },
    { key: 'gross_margin', label: 'Gross Margin',   group: 'Profitability', field: 'gross_margin',    mid: 0,  clamp: 60, format: 'pct' },
    { key: 'op_margin',    label: 'Operating Margin', group: 'Profitability', field: 'operating_margin', mid: 0, clamp: 30, format: 'pct' },
    { key: 'net_margin',   label: 'Net Margin',     group: 'Profitability', field: 'net_margin',      mid: 0,  clamp: 25, format: 'pct' },
    // Liquidity: more cover is better, and 1.0 is the classic solvency line.
    { key: 'quick_ratio',  label: 'Quick Ratio',    group: 'Profitability', field: 'quick_ratio',     mid: 1,   clamp: 1,  format: 'ratio' },
    { key: 'current_ratio', label: 'Current Ratio', group: 'Profitability', field: 'current_ratio',   mid: 1.5, clamp: 1,  format: 'ratio' },
    { key: 'lt_debt_eq',   label: 'LT Debt/Equity', group: 'Profitability', field: 'lt_debt_equity',  mid: 50, clamp: 50, format: 'ratio', inverted: true },
    { key: 'debt_equity',  label: 'Debt/Equity',    group: 'Profitability', field: 'debt_equity',     mid: 80, clamp: 80, format: 'ratio', inverted: true },

    // --- Market & sentiment.
    // 1.0 is exactly average volume, so the deviation either side is the signal.
    { key: 'rel_volume',   label: 'Relative Volume', group: 'Market', field: 'relative_volume', mid: 1,   clamp: 1,   format: 'ratio' },
    // Heavily shorted reads red; floored at zero, and the domain lands on 0-10%.
    { key: 'float_short',  label: 'Float Short',     group: 'Market', field: 'float_short',     mid: 5,   clamp: 5,   format: 'pct',   inverted: true },
    // Yahoo's consensus runs 1 (strong buy) to 5 (sell), so lower is greener.
    { key: 'analyst',      label: 'Analysts Recom.', group: 'Market', field: 'analyst_recom',   mid: 2.5, clamp: 1.5, format: 'ratio', inverted: true },
    // Days until the next report; imminent earnings read green.
    { key: 'earnings_days', label: 'Earnings Date',  group: 'Market', field: 'earnings_days',   mid: 45,  clamp: 45,  format: 'days',  inverted: true },
];

// ---------------------------------------------------------------------------
// Color and Metric formatting helpers
// ---------------------------------------------------------------------------

/** Metrics the backend sends as a fraction and we display as a percentage.
 *  `day` is excluded: `change_pct` already arrives in percent points, and so
 *  do the ratio-style fields (P/E, Debt/Equity, ...) which are not scaled. */
const DECIMAL_METRICS = new Set([
    '1w', '1m', 'mtd', '3m', '6m', 'ytd', '1y', '3y', '5y', '10y', 'dd52', 'up52',
    'yield',
    'eps_qoq', 'eps_3y', 'eps_5y', 'eps_surprise', 'sales_qoq', 'sales_3y', 'sales_5y',
    'roa', 'roe', 'roic', 'gross_margin', 'op_margin', 'net_margin',
    'float_short',
]);

export function getScaledValue(v: number | null | undefined, metricKey: string): number | null {
    if (v == null || !isFinite(v)) return null;
    return DECIMAL_METRICS.has(metricKey) ? v * 100 : v;
}

/** Signed distance from the metric's neutral point, positive meaning "good".
 *  Takes an already-scaled value. */
export function signedDeviation(scaled: number, metric: Pick<MetricDef, 'mid' | 'inverted'>): number {
    return metric.inverted ? metric.mid - scaled : scaled - metric.mid;
}

type ScaleSpec = Pick<MetricDef, 'mid' | 'clamp' | 'inverted' | 'scale'>;

/** The [low, high] range the colour scale covers, in display units. */
export function metricDomain(metric: ScaleSpec): [number, number] {
    return metric.scale === 'sequential'
        ? [metric.mid, metric.mid + metric.clamp]
        : [metric.mid - metric.clamp, metric.mid + metric.clamp];
}

export type HeatTheme = 'dark' | 'light';

/** Every colour the map paints itself with. The two themes are the same scale
 *  read against opposite surfaces: on black, "more" means brighter; on white it
 *  means more saturated and darker. Only the endpoints and the neutral point
 *  change — the geometry of the scale is identical, so a tile means the same
 *  thing in either mode. */
export interface HeatPalette {
    /** Colour at the metric's neutral point on the diverging scale. */
    mid: string;
    /** Saturated ends of the diverging scale. */
    pos: string;
    neg: string;
    /** Ends of the one-hue magnitude ramp. */
    seqLow: string;
    seqHigh: string;
    /** A tile with no reading — deliberately achromatic, so it can never be
     *  mistaken for a weak red or a weak green. */
    noData: string;
    /** Surface behind the tiles; it shows through the 1px gaps as the grid. */
    surface: string;
    sectorHeader: string;
    industryHeader: string;
    headerText: string;
    headerRule: string;
    /** Tile luminance above which a label switches from white to dark ink. */
    labelFlip: number;
}

export const HEAT_PALETTES: Record<HeatTheme, HeatPalette> = {
    // Black surface: the neutral point is the surface itself, and colour is
    // added as light. The sequential ramp is validated as an ordinal scale
    // against that surface (monotone lightness, ΔL >= 0.06 per step, 2.35:1 at
    // the low end — comfortably clear of the grey used for "no data").
    dark: {
        mid: '#000000',
        pos: '#30cc5a',
        neg: '#f63538',
        seqLow: '#165433',
        seqHigh: '#30cc5a',
        noData: '#1a1a1a',
        surface: '#000000',
        sectorHeader: '#2a2a2a',
        industryHeader: '#3a3a3a',
        headerText: '#ffffff',
        headerRule: '#000000',
        // Above pure white, i.e. never reached: colour is only ever added to
        // black here, so the dark map is white-labelled throughout — the look
        // it has always had.
        labelFlip: 1.01,
    },
    // Light surface: the neutral point is a near-white the page can carry, and
    // colour is added as ink. The endpoints are darkened relative to the dark
    // theme's so white tile labels keep their contrast, and the sequential ramp
    // runs pale -> deep, since on paper "more" reads as darker.
    light: {
        mid: '#eff1f3',
        pos: '#158f47',
        neg: '#d02b2f',
        seqLow: '#d8efdf',
        seqHigh: '#106b38',
        noData: '#bcc0c7',
        surface: '#b4b9c0',
        sectorHeader: '#c3c8ce',
        industryHeader: '#dbdfe4',
        headerText: '#1f2328',
        headerRule: '#b4b9c0',
        // The crossover that maximises the worst case: white and dark ink reach
        // equal contrast here, so no tile on the light scale falls below 4.2:1.
        labelFlip: 0.2,
    },
};

function hexToRgb(hex: string): [number, number, number] {
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
}

function mix(from: string, to: string, t: number): string {
    const a = hexToRgb(from);
    const b = hexToRgb(to);
    const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/** Colour for an already-scaled value (used directly by the legend, which works
 *  in display units). */
export function heatColorScaled(scaled: number | null, metric: ScaleSpec, theme: HeatTheme = 'dark'): string {
    const p = HEAT_PALETTES[theme];
    if (scaled == null || !isFinite(scaled)) return p.noData;

    if (metric.scale === 'sequential') {
        const [low, high] = metricDomain(metric);
        const t = Math.max(0, Math.min(1, (scaled - low) / (high - low)));
        return mix(p.seqLow, p.seqHigh, t);
    }

    const pct = (signedDeviation(scaled, metric) / metric.clamp) * 3.0; // scale roughly to -3 / +3

    if (pct >= 3.0) return p.pos;
    if (pct <= -3.0) return p.neg;

    if (pct > 0) return mix(p.mid, p.pos, pct / 3.0);
    if (pct < 0) return mix(p.mid, p.neg, -pct / 3.0);
    return p.mid;
}

export function heatColor(v: number | null | undefined, metricKey: string, theme: HeatTheme = 'dark'): string {
    const metric = METRICS.find(m => m.key === metricKey) || METRICS[0];
    return heatColorScaled(getScaledValue(v, metricKey), metric, theme);
}

function parseColor(color: string): [number, number, number] {
    if (color.startsWith('#')) return hexToRgb(color);
    const m = /rgb\((\d+),\s*(\d+),\s*(\d+)\)/.exec(color);
    return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : [0, 0, 0];
}

/** WCAG relative luminance of a CSS colour the map produced. */
export function tileLuminance(color: string): number {
    const [r, g, b] = parseColor(color);
    const lin = (c: number) => {
        const s = c / 255;
        return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

/** Ink for a tile label, picked against the tile's own fill rather than against
 *  the theme: the light map runs from near-white at the neutral point to deep
 *  red/green at the ends, so a single fixed label colour would be unreadable
 *  over half of it. */
export function tileTextColor(color: string, theme: HeatTheme = 'dark'): string {
    return tileLuminance(color) > HEAT_PALETTES[theme].labelFlip ? '#15181d' : '#ffffff';
}

/** Text colour for a metric read as *text* rather than as a tile — the heatmap
 *  tooltip and the detail window's metric rows. Keyed off the same neutral point
 *  as the tile colour, so a reading does not change verdict between the two.
 *
 *  `deadZone` is the share of `clamp` around the neutral point that stays plain
 *  ink, as a fraction. The tooltip shows one value at a time and passes 0, but a
 *  panel of thirty-odd rows needs the near-typical ones to recede or the page
 *  reads as noise; muting them is a display choice, not a different verdict. */
export function metricTone(
    v: number | null | undefined,
    metric: Pick<MetricDef, 'key' | 'mid' | 'clamp' | 'inverted' | 'scale'>,
    deadZone = 0,
): string {
    const scaled = getScaledValue(v, metric.key);
    if (scaled == null) return 'text-muted-foreground';
    // A magnitude has no good or bad side, so it takes plain ink rather than a
    // verdict colour.
    if (metric.scale === 'sequential') return 'text-foreground';
    const deviation = signedDeviation(scaled, metric);
    if (Math.abs(deviation) < metric.clamp * deadZone) return 'text-foreground';
    return deviation >= 0
        ? 'text-emerald-600 dark:text-emerald-400'
        : 'text-rose-600 dark:text-rose-400';
}

/** "$1.2T" / "$840.0B" / "$120M" — the compact money form both the map and
 *  the detail window use for caps and revenue. */
export function formatCompactCap(v: number): string {
    if (v >= 1e12) return `$${(v / 1e12).toFixed(1)}T`;
    if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
    if (v >= 1e6) return `$${(v / 1e6).toFixed(0)}M`;
    return `$${v.toFixed(0)}`;
}

export function formatMetric(v: number | null | undefined, format?: string, metricKey?: string): string {
    return formatScaled(getScaledValue(v, metricKey || ''), format);
}

/** Formats a value that is already in display units. */
export function formatScaled(val: number | null, format?: string): string {
    if (val == null) return 'n/a';
    switch (format) {
        case 'pct':    return `${val >= 0 ? '+' : ''}${val.toFixed(2)}%`;
        case 'ratio':  return val.toFixed(2);
        case 'cap':    return formatCompactCap(val);
        case 'dollar': return `$${val.toFixed(2)}`;
        // Negative means the report has already happened.
        case 'days':   return val >= 0 ? `${Math.round(val)}d` : `${Math.round(-val)}d ago`;
        default:       return val.toFixed(2);
    }
}
