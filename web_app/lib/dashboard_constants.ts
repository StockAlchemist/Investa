
export const COMPLEX_METRIC_IDS = ['riskMetrics', 'sectorContribution', 'topContributors', 'portfolioDonut'];

export const DEFAULT_ITEMS = [
    { id: 'portfolioValue', title: 'Total Portfolio Value', colSpan: 'col-span-2 md:col-span-2 lg:col-span-2' },
    { id: 'dayGL', title: "Day's Gain/Loss", colSpan: 'col-span-2 md:col-span-2 lg:col-span-2' },
    { id: 'totalReturn', title: 'Total Return', colSpan: 'col-span-2 md:col-span-2 lg:col-span-2' },
    { id: 'unrealizedGL', title: 'Unrealized G/L', colSpan: 'col-span-2 md:col-span-2 lg:col-span-2' },
    { id: 'realizedGain', title: 'Realized Gain', colSpan: 'col-span-2 md:col-span-2 lg:col-span-2' },
    { id: 'ytdDividends', title: 'Total Dividends', colSpan: 'col-span-2 md:col-span-2 lg:col-span-2' },
    { id: 'annualTWR', title: 'Annual TWR', colSpan: 'col-span-1' },
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
    'unrealizedGL',
    'annualTWR',
    'cashBalance',
    'portfolioDonut'
];
