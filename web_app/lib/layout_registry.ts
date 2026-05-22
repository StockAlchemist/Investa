import { DEFAULT_ITEMS, INITIAL_VISIBLE_ITEMS } from './dashboard_constants';

// ── Types ─────────────────────────────────────────────────────────────────────
export interface LayoutItem {
    id: string;
    title: string;
    /** Optional CSS col-span (only used by the Dashboard grid) */
    colSpan?: string;
}

// ── Per-tab layout definitions ────────────────────────────────────────────────
// Each entry lists the toggleable sections a tab can show/hide via the
// Layout configurator.  Tabs not listed here won't show the button.

const ALLOCATION_ITEMS: LayoutItem[] = [
    { id: 'holdingsTable',    title: 'Holdings Table' },
    { id: 'concentrationKpis',title: 'Concentration KPIs' },
    { id: 'categoryDrift',    title: 'Category Drift (Asset, Sector, Country)' },
    { id: 'stockDrift',       title: 'Stock Drift' },
    { id: 'rebalanceHelper',  title: 'Rebalance Helper' },
    { id: 'treemap',          title: 'Treemap' },
    { id: 'donutCharts',      title: 'Donut Charts' },
];

const ASSET_CHANGE_ITEMS: LayoutItem[] = [
    { id: 'kpiStrip',            title: 'KPI Strip' },
    { id: 'returnsChart',       title: 'Returns Chart' },
    { id: 'monthlyHeatmap',     title: 'Monthly Heatmap' },
    { id: 'drawdownTimeline',   title: 'Drawdown Timeline' },
    { id: 'benchmarkScoreboard', title: 'Benchmark Scoreboard' },
    { id: 'sectorAttribution',  title: 'Sector Contribution' },
    { id: 'topContributors',    title: 'Top Contributors' },
];

const CAPITAL_GAINS_ITEMS: LayoutItem[] = [
    { id: 'unrealizedTax',            title: 'Unrealized Tax Lots' },
    { id: 'capitalGainsKpis',         title: 'Realized Gains KPIs' },
    { id: 'annualCapitalGains',       title: 'Annual Realized Gains' },
    { id: 'capitalGainsTransactions', title: 'Realized Gains Transactions' },
];

const DIVIDEND_ITEMS: LayoutItem[] = [
    { id: 'incomeKpis',           title: 'Income KPIs' },
    { id: 'incomeProjector',      title: 'Income Projector' },
    { id: 'dividendCalendar',     title: 'Dividend Calendar' },
    { id: 'topPayers',            title: 'Top Payers' },
    { id: 'byAccount',            title: 'By Account' },
    { id: 'annualDividends',      title: 'Annual Dividends' },
    { id: 'dividendTransactions', title: 'Dividend Transactions' },
];

// ── Registry ──────────────────────────────────────────────────────────────────
// Maps tab id → ordered list of toggleable layout items.
export const TAB_LAYOUT_ITEMS: Record<string, LayoutItem[]> = {
    performance:   DEFAULT_ITEMS as LayoutItem[],
    allocation:    ALLOCATION_ITEMS,
    asset_change:  ASSET_CHANGE_ITEMS,
    capital_gains: CAPITAL_GAINS_ITEMS,
    dividend:      DIVIDEND_ITEMS,
};

// Default visibility per tab (everything ON by default for new tabs;
// dashboard keeps its curated subset).
export const TAB_INITIAL_VISIBLE: Record<string, string[]> = {
    performance:   INITIAL_VISIBLE_ITEMS,
    allocation:    ALLOCATION_ITEMS.map(i => i.id),
    asset_change:  ASSET_CHANGE_ITEMS.map(i => i.id),
    capital_gains: CAPITAL_GAINS_ITEMS.map(i => i.id),
    dividend:      DIVIDEND_ITEMS.map(i => i.id),
};

// ── Section labels for the dropdown header ────────────────────────────────────
export const TAB_SECTION_LABELS: Record<string, string> = {
    performance:   'Dashboard Elements',
    allocation:    'Portfolio Sections',
    asset_change:  'Performance Sections',
    capital_gains: 'Capital Gains Sections',
    dividend:      'Income Sections',
};
