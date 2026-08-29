import { Users, ArrowLeftRight, SlidersHorizontal, Settings2, UserCircle } from 'lucide-react';
import { TabDefinition } from './types';

export const ASSET_TYPES = [
    "",
    "STOCK",
    "ETF",
    "MUTUALFUND",
    "CURRENCY",
    "INDEX",
    "FUTURE",
    "OPTION",
    "CRYPTOCURRENCY",
    "Other",
];

export const SECTORS = [
    "",
    "Other",
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
    "Exchange-Traded Fund",
];

export const PRESET_BENCHMARKS = [
    "S&P 500",
    "NASDAQ",
    "Dow Jones",
    "Russell 2000",
    "Total US Market (VTI)",
    "All-World (VT)",
    "Total International (VXUS)",
    "Emerging Markets (VWO)",
    "Europe (VGK)",
    "Japan (EWJ)",
    "US Total Bond (BND)",
    "20+ Year Treasury (TLT)",
    "Gold (GLD)",
    "Bitcoin (BTC-USD)",
    "US Growth (VUG)",
    "US Value (VTV)",
    "US Dividend (SCHD)",
];

export const TABS: TabDefinition[] = [
    {
        id: 'accounts',
        label: 'Accounts',
        description: 'Account groups, per-account currency, cash automation, and yield assumptions.',
        icon: Users,
        color: 'text-indigo-500 dark:text-indigo-400',
        badgeBg: 'bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border border-indigo-500/20',
        accentDot: 'bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.8)]',
    },
    {
        id: 'symbols',
        label: 'Symbols',
        description: 'Map portfolio symbols to Yahoo Finance tickers and manage excluded symbols.',
        icon: ArrowLeftRight,
        color: 'text-blue-500 dark:text-blue-400',
        badgeBg: 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20',
        accentDot: 'bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.8)]',
    },
    {
        id: 'overrides',
        label: 'Overrides',
        description: 'Manually override price, sector, asset type, and metadata for specific symbols.',
        icon: SlidersHorizontal,
        color: 'text-emerald-500 dark:text-emerald-400',
        badgeBg: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20',
        accentDot: 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]',
    },
    {
        id: 'advanced',
        label: 'Advanced Settings',
        description: 'Benchmark comparisons, IBKR Flex Query sync, external API keys, and cache.',
        icon: Settings2,
        color: 'text-purple-500 dark:text-purple-400',
        badgeBg: 'bg-purple-500/10 text-purple-600 dark:text-purple-400 border border-purple-500/20',
        accentDot: 'bg-purple-500 shadow-[0_0_8px_rgba(168,85,247,0.8)]',
    },
    {
        id: 'account',
        label: 'Profile & Security',
        description: 'Manage user profile, login credentials, and session security.',
        icon: UserCircle,
        color: 'text-cyan-500 dark:text-cyan-400',
        badgeBg: 'bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border border-cyan-500/20',
        accentDot: 'bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.8)]',
    },
];

export const inputClassName = "w-full h-9 rounded-control border border-input bg-background text-foreground px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ring-offset-background transition-colors";
export const compactInputClassName = "w-full h-7 rounded-control border border-input bg-background text-foreground px-2.5 text-xs outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ring-offset-background transition-colors";
export const labelClassName = "section-label block mb-1.5";
export const cardClassName = "card-standard p-6 relative overflow-hidden";
export const sectionTitleClassName = "text-base font-semibold text-foreground flex items-center gap-2";
export const primaryButtonClassName = "h-9 px-3.5 rounded-control bg-primary text-primary-foreground hover:bg-primary-hover text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ring-offset-background disabled:opacity-50 disabled:pointer-events-none transition-colors inline-flex items-center justify-center gap-2 cursor-pointer";
