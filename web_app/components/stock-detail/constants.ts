import {
    LayoutDashboard,
    LineChart as LineChartIcon,
    Receipt,
    Scale,
    Target,
    PieChart as PieChartIcon,
    Sparkles,
    Newspaper,
} from 'lucide-react';
import { TabType } from './types';

export const RANKING_CONFIG: Record<string, string[]> = {
    income: [
        'Total Revenue',
        'Cost Of Revenue',
        'Gross Profit',
        'Operating Expense',
        'Operating Income',
        'EBITDA',
        'EBIT',
        'Pretax Income',
        'Tax Provision',
        'Net Income Common Stockholders',
        'Net Income',
        'Normalized Income',
        'Basic EPS',
        'Diluted EPS'
    ],
    balance: [
        'Total Assets',
        'Current Assets',
        'Cash And Cash Equivalents',
        'Receivables',
        'Inventory',
        'Total Liabilities Net Minority Interest',
        'Current Liabilities',
        'Total Debt',
        'Net Debt',
        'Total Equity Gross Minority Interest',
        'Stockholders Equity',
        'Common Stock Equity',
        'Retained Earnings',
        'Working Capital',
        'Invested Capital',
        'Tangible Book Value'
    ],
    cash: [
        'Operating Cash Flow',
        'Investing Cash Flow',
        'Financing Cash Flow',
        'Capital Expenditure',
        'Free Cash Flow',
        'End Cash Position',
        'Net Income'
    ],
    equity: [
        'Total Equity Gross Minority Interest',
        'Stockholders Equity',
        'Common Stock Equity',
        'Retained Earnings',
        'Capital Stock',
        'Common Stock'
    ]
};

export const TABS_CONFIG: { id: TabType; label: string; icon: React.ElementType }[] = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'chart', label: 'Chart', icon: LineChartIcon },
    { id: 'financials', label: 'Financials', icon: Receipt },
    { id: 'ratios', label: 'Ratios', icon: Scale },
    { id: 'valuation', label: 'Valuation', icon: Target },
    { id: 'holdings', label: 'Holdings', icon: PieChartIcon },
    { id: 'analysis', label: 'Analysis', icon: Sparkles },
    { id: 'news', label: 'News', icon: Newspaper },
];
