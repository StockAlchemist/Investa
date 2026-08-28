// Items rendered as their own full-width section by the dashboard tab in
// page.tsx (donut, performance graph, risk analytics, attribution widgets).
// Dashboard's compact metric grid skips these; page.tsx renders them one by
// one in the order they appear in DEFAULT_ITEMS below.
export const COMPLEX_METRIC_IDS = [
    'portfolioDonut',
    'performanceGraph',
    'projection',
    'riskMetrics',
    'sectorContribution',
    'topContributors',
];

// IDs that render outside the compact metric grid AND outside the full-width
// analytics grid (hero, today strip, events, insights). Dashboard.tsx renders
// these explicitly at the top of the page.
export const TOP_SECTION_IDS = [
    'portfolioHero',
    'todayStrip',
    'trendSignal',
    'dashboardEvents',
    'dashboardInsights',
];

// Order here drives the dashboard's compact-grid visual order (and the array
// order LayoutConfigurator uses within each group). The `group` field drives
// the dropdown's grouping headers — it does NOT change rendering order.
export const DEFAULT_ITEMS = [
    // Overview (top of page)
    { id: 'portfolioHero', title: 'Portfolio Hero', group: 'Overview' },
    { id: 'todayStrip', title: 'Market Today', group: 'Overview' },
    { id: 'trendSignal', title: 'Market Trend', group: 'Overview' },
    // Insights & Events
    { id: 'dashboardEvents', title: 'Upcoming Events', group: 'Insights & Events' },
    { id: 'dashboardInsights', title: 'Actionable Insights', group: 'Insights & Events' },
    // Returns
    { id: 'totalReturn', title: 'Total Return', group: 'Returns', colSpan: 'col-span-2 md:col-span-1' },
    { id: 'unrealizedGL', title: 'Unrealized G/L', group: 'Returns', colSpan: 'col-span-2 md:col-span-1' },
    { id: 'realizedGain', title: 'Realized Gain', group: 'Returns', colSpan: 'col-span-2 md:col-span-1' },
    { id: 'annualTWR', title: 'Annual TWR', group: 'Returns', colSpan: 'col-span-2 lg:col-span-1' },
    { id: 'mwr', title: 'IRR (MWR)', group: 'Returns', colSpan: 'col-span-1' },
    // Income & Cash
    { id: 'ytdDividends', title: 'Total Dividends', group: 'Income & Cash', colSpan: 'col-span-1' },
    { id: 'dividendYield', title: 'Dividend Yield %', group: 'Income & Cash', colSpan: 'col-span-1' },
    { id: 'ytdReturn', title: 'YTD Return', group: 'Income & Cash', colSpan: 'col-span-1' },
    { id: 'cashBalance', title: 'Cash Balance', group: 'Income & Cash', colSpan: 'col-span-1' },
    // Costs & FX
    { id: 'fxGL', title: 'FX Gain/Loss', group: 'Costs & FX', colSpan: 'col-span-2 md:col-span-1' },
    { id: 'fees', title: 'Fees', group: 'Costs & FX', colSpan: 'col-span-2 md:col-span-1' },
    { id: 'taxes', title: 'Taxes', group: 'Costs & FX', colSpan: 'col-span-2 md:col-span-1' },
    // Charts (full-width sections rendered in this order by page.tsx)
    { id: 'portfolioDonut', title: 'Portfolio Composition', group: 'Charts', colSpan: 'col-span-2 md:col-span-2 lg:col-span-4' },
    { id: 'performanceGraph', title: 'Performance Graph', group: 'Charts', colSpan: 'col-span-2 md:col-span-2 lg:col-span-4' },
    { id: 'projection', title: 'Projected Value', group: 'Charts', colSpan: 'col-span-2 md:col-span-2 lg:col-span-4' },
    // Risk & Attribution
    { id: 'riskMetrics', title: 'Risk Analytics', group: 'Risk & Attribution', colSpan: 'col-span-2 md:col-span-2 lg:col-span-4' },
    { id: 'sectorContribution', title: 'Sector Contribution', group: 'Risk & Attribution', colSpan: 'col-span-2 md:col-span-2 lg:col-span-2' },
    { id: 'topContributors', title: 'Top Contributors', group: 'Risk & Attribution', colSpan: 'col-span-2 md:col-span-2 lg:col-span-2' },
];

export const INITIAL_VISIBLE_ITEMS = [
    'portfolioHero',
    'todayStrip',
    'trendSignal',
    'dashboardEvents',
    'dashboardInsights',
    'totalReturn',
    'unrealizedGL',
    'realizedGain',
    'annualTWR',
    'mwr',
    'ytdDividends',
    'dividendYield',
    'ytdReturn',
    'cashBalance',
    'fxGL',
    'fees',
    'taxes',
    'portfolioDonut',
    'performanceGraph',
    'projection',
];

/**
 * Retired: a twelve-hue map that recoloured the app's chrome by destination —
 * indigo, sky, teal, slate, purple, violet, amber, emerald, lime, rose, cyan,
 * zinc. The chrome changed colour as you navigated, and lime and rose carried
 * no meaning on screens where green and red mean gain and loss.
 *
 * One accent now. The export and its shape are kept so existing call sites
 * type-check; every destination resolves to the same theme. `color` stays
 * 'indigo-500' because MetricCard looks the accent up by that key.
 */
export interface TabTheme {
    color: string;
    glow: string;
    bgGlow: string;
    bgSolid: string;
    textSolid: string;
    shadowSolid: string;
}

export const ACCENT_THEME: TabTheme = {
    color: 'indigo-500',
    glow: 'from-primary/20',
    bgGlow: 'bg-primary/20',
    bgSolid: 'bg-primary',
    textSolid: 'text-primary',
    shadowSolid: 'shadow-primary/30',
};

export const TAB_THEMES: Record<string, TabTheme> = new Proxy(
    { performance: ACCENT_THEME },
    { get: () => ACCENT_THEME },
);
