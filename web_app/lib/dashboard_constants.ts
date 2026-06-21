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

export const TAB_THEMES: Record<string, { color: string; glow: string; bgGlow: string; bgSolid: string; textSolid: string; shadowSolid: string }> = {
    performance: { color: 'indigo-500', glow: 'from-indigo-500/20', bgGlow: 'bg-indigo-500/20', bgSolid: 'bg-indigo-500', textSolid: 'text-indigo-500', shadowSolid: 'shadow-indigo-500/30' },
    watchlist: { color: 'sky-500', glow: 'from-sky-500/20', bgGlow: 'bg-sky-500/20', bgSolid: 'bg-sky-500', textSolid: 'text-sky-500', shadowSolid: 'shadow-sky-500/30' },
    screener: { color: 'teal-500', glow: 'from-teal-500/20', bgGlow: 'bg-teal-500/20', bgSolid: 'bg-teal-500', textSolid: 'text-teal-500', shadowSolid: 'shadow-teal-500/30' },
    transactions: { color: 'slate-500', glow: 'from-slate-500/20', bgGlow: 'bg-slate-500/20', bgSolid: 'bg-slate-500', textSolid: 'text-slate-500', shadowSolid: 'shadow-slate-500/30' },
    allocation: { color: 'purple-500', glow: 'from-purple-500/20', bgGlow: 'bg-purple-500/20', bgSolid: 'bg-purple-500', textSolid: 'text-purple-500', shadowSolid: 'shadow-purple-500/30' },
    asset_change: { color: 'violet-500', glow: 'from-violet-500/20', bgGlow: 'bg-violet-500/20', bgSolid: 'bg-violet-500', textSolid: 'text-violet-500', shadowSolid: 'shadow-violet-500/30' },
    capital_gains: { color: 'amber-500', glow: 'from-amber-500/20', bgGlow: 'bg-amber-500/20', bgSolid: 'bg-amber-500', textSolid: 'text-amber-500', shadowSolid: 'shadow-amber-500/30' },
    dividend: { color: 'emerald-500', glow: 'from-emerald-500/20', bgGlow: 'bg-emerald-500/20', bgSolid: 'bg-emerald-500', textSolid: 'text-emerald-500', shadowSolid: 'shadow-emerald-500/30' },
    ai_review: { color: 'rose-500', glow: 'from-rose-500/20', bgGlow: 'bg-rose-500/20', bgSolid: 'bg-rose-500', textSolid: 'text-rose-500', shadowSolid: 'shadow-rose-500/30' },
    markets: { color: 'cyan-500', glow: 'from-cyan-500/20', bgGlow: 'bg-cyan-500/20', bgSolid: 'bg-cyan-500', textSolid: 'text-cyan-500', shadowSolid: 'shadow-cyan-500/30' },
    settings: { color: 'zinc-500', glow: 'from-zinc-500/20', bgGlow: 'bg-zinc-500/20', bgSolid: 'bg-zinc-500', textSolid: 'text-zinc-500', shadowSolid: 'shadow-zinc-500/30' },
};
