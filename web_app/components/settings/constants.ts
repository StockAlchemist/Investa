import { Users, Map as MapIcon, Sliders, Settings as SettingsIcon, UserCircle } from 'lucide-react';
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
    "Dow Jones",
    "NASDAQ",
    "Russell 2000",
    "SPY (S&P 500 ETF)",
    "QQQ (Nasdaq 100 ETF)",
    "DIA (Dow Jones ETF)",
    "S&P 500 Total Return",
];

export const TABS: TabDefinition[] = [
    { id: 'accounts', label: 'Accounts', description: 'Account groups, per-account currency/cash/closure settings, and cash-yield assumptions.', icon: Users, color: 'text-muted-foreground' },
    { id: 'symbols', label: 'Symbols', description: 'Map portfolio symbols to their Yahoo Finance ticker and manage excluded symbols.', icon: MapIcon, color: 'text-muted-foreground' },
    { id: 'overrides', label: 'Overrides', description: 'Manually override price/metadata and DCF valuation inputs for specific symbols.', icon: Sliders, color: 'text-emerald-500 dark:text-emerald-400' },
    { id: 'advanced', label: 'Advanced Settings', description: 'Webhook integration, Interactive Brokers sync, and system cache.', icon: SettingsIcon, color: 'text-muted-foreground' },
    { id: 'account', label: 'Profile & Security', description: 'Manage your user profile, password, and login.', icon: UserCircle, color: 'text-muted-foreground' },
];

export const inputClassName = "w-full h-9 rounded-control border border-input bg-background text-foreground px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ring-offset-background transition-colors";
export const compactInputClassName = "w-full h-7 rounded-control border border-input bg-background text-foreground px-2.5 text-xs outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ring-offset-background transition-colors";
export const labelClassName = "section-label block mb-1.5";
export const cardClassName = "card-standard p-6 relative overflow-hidden";
export const sectionTitleClassName = "text-base font-semibold text-foreground flex items-center gap-2";
export const primaryButtonClassName = "h-9 px-3.5 rounded-control bg-primary text-primary-foreground hover:bg-primary-hover text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ring-offset-background disabled:opacity-50 disabled:pointer-events-none transition-colors inline-flex items-center justify-center gap-2 cursor-pointer";
