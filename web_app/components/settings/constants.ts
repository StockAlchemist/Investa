import { Users, ArrowLeftRight, SlidersHorizontal, Settings2, Palette, UserCircle } from 'lucide-react';
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
    },
    {
        id: 'symbols',
        label: 'Symbols',
        description: 'Map portfolio symbols to Yahoo Finance tickers and manage excluded symbols.',
        icon: ArrowLeftRight,
    },
    {
        id: 'overrides',
        label: 'Overrides',
        description: 'Manually override price, sector, asset type, and metadata for specific symbols.',
        icon: SlidersHorizontal,
    },
    {
        id: 'advanced',
        label: 'Advanced',
        description: 'Benchmark comparisons, IBKR Flex Query sync, external API keys, and cache.',
        icon: Settings2,
    },
    {
        id: 'appearance',
        label: 'Appearance',
        description: 'Light, dark, or follow the device theme.',
        icon: Palette,
    },
    {
        id: 'account',
        label: 'Profile & Security',
        description: 'Manage user profile, login credentials, and session security.',
        icon: UserCircle,
    },
];

export const inputClassName = "w-full h-9 rounded-control border border-input bg-background text-foreground px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ring-offset-background transition-colors";
export const compactInputClassName = "w-full h-7 rounded-control border border-input bg-background text-foreground px-2.5 text-xs outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ring-offset-background transition-colors";
export const labelClassName = "section-label block mb-1.5";
export const cardClassName = "card-standard p-6 relative overflow-hidden";
/* A Settings card is an ordinary Investa card: its head is the same 10px
   uppercase section label every other tab uses, plus at most a count badge and
   one action. No icon, no title bar, no per-card accent colour. */
export const sectionTitleClassName = "section-label flex items-center gap-2 min-w-0";
export const cardHeadClassName = "flex items-center gap-2.5 mb-4";
export const countBadgeClassName = "text-[11px] font-bold tabular-nums text-primary-ink bg-primary/12 rounded-full px-2 py-0.5 shrink-0";
export const cardHintClassName = "text-xs text-muted-foreground ml-auto shrink-0 hidden sm:block";
export const primaryButtonClassName = "h-9 px-3.5 rounded-control bg-primary text-primary-foreground hover:bg-primary-hover text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ring-offset-background disabled:opacity-50 disabled:pointer-events-none transition-colors inline-flex items-center justify-center gap-2 cursor-pointer";
export const secondaryButtonClassName = "h-9 px-3.5 rounded-control border border-border bg-background text-foreground hover:bg-muted text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ring-offset-background disabled:opacity-50 disabled:pointer-events-none transition-colors inline-flex items-center justify-center gap-2 cursor-pointer";
/* Destructive stays semantic — red means "this removes data", never decoration. */
export const destructiveButtonClassName = "h-9 px-3.5 rounded-control bg-down text-white hover:opacity-90 text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-down focus-visible:ring-offset-2 ring-offset-background disabled:opacity-50 disabled:pointer-events-none transition-opacity inline-flex items-center justify-center gap-2 cursor-pointer";
/* Inset panel inside a card — border only, 12px radius. */
export const insetClassName = "card-inset p-4";
export const chipClassName = "inline-flex items-center gap-2 h-8 px-3 rounded-control border border-border bg-background text-xs font-semibold text-muted-foreground transition-colors";
export const chipActiveClassName = "inline-flex items-center gap-2 h-8 px-3 rounded-control border border-primary/25 bg-primary/12 text-xs font-semibold text-primary-ink transition-colors";
