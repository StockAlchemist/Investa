
export const COMPLEX_METRIC_IDS = ['riskMetrics', 'sectorContribution', 'topContributors', 'portfolioDonut'];

export const DEFAULT_ITEMS = [
    { id: 'portfolioValue', title: 'Total Portfolio Value', colSpan: 'col-span-2 md:col-span-2 lg:col-span-2' },
    { id: 'dayGL', title: "Day's Gain/Loss", colSpan: 'col-span-2 md:col-span-2 lg:col-span-2' },
    { id: 'totalReturn', title: 'Total Return', colSpan: 'col-span-2 md:col-span-1' },
    { id: 'unrealizedGL', title: 'Unrealized G/L', colSpan: 'col-span-2 md:col-span-1' },
    { id: 'realizedGain', title: 'Realized Gain', colSpan: 'col-span-2 md:col-span-1' },
    { id: 'annualTWR', title: 'Annual TWR', colSpan: 'col-span-2 lg:col-span-1' },
    { id: 'mwr', title: 'IRR (MWR)', colSpan: 'col-span-1' },
    { id: 'ytdDividends', title: 'Total Dividends', colSpan: 'col-span-1' },
    { id: 'cashBalance', title: 'Cash Balance', colSpan: 'col-span-1' },
    { id: 'fxGL', title: 'FX Gain/Loss', colSpan: 'col-span-2 md:col-span-1' },
    { id: 'fees', title: 'Fees', colSpan: 'col-span-2 md:col-span-1' },
    { id: 'riskMetrics', title: 'Risk Analytics', colSpan: 'col-span-2 md:col-span-2 lg:col-span-4' },
    { id: 'portfolioDonut', title: 'Portfolio Composition', colSpan: 'col-span-2 md:col-span-2 lg:col-span-4' },
    { id: 'sectorContribution', title: 'Sector Contribution', colSpan: 'col-span-2 md:col-span-2 lg:col-span-2' },
    { id: 'topContributors', title: 'Top Contributors', colSpan: 'col-span-2 md:col-span-2 lg:col-span-2' },
];

export const INITIAL_VISIBLE_ITEMS = [
    'portfolioValue',
    'dayGL',
    'realizedGain',
    'annualTWR',
    'mwr',
    'cashBalance',
    'portfolioDonut'
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
