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
    { id: 'accounts', label: 'Accounts', description: 'Account groups, per-account currency/cash/closure settings, and cash-yield assumptions.', icon: Users, color: 'text-indigo-500 dark:text-indigo-400' },
    { id: 'symbols', label: 'Symbols', description: 'Map portfolio symbols to their Yahoo Finance ticker and manage excluded symbols.', icon: MapIcon, color: 'text-blue-500 dark:text-blue-400' },
    { id: 'overrides', label: 'Overrides', description: 'Manually override price/metadata and DCF valuation inputs for specific symbols.', icon: Sliders, color: 'text-emerald-500 dark:text-emerald-400' },
    { id: 'advanced', label: 'Advanced Settings', description: 'Webhook integration, Interactive Brokers sync, and system cache.', icon: SettingsIcon, color: 'text-zinc-500 dark:text-zinc-400' },
    { id: 'account', label: 'Profile & Security', description: 'Manage your user profile, password, and login.', icon: UserCircle, color: 'text-cyan-500 dark:text-cyan-400' },
];

export const inputClassName = "w-full rounded-xl border border-black/10 dark:border-white/10 bg-white/50 dark:bg-black/20 backdrop-blur-sm text-foreground shadow-sm focus:border-cyan-500 focus:ring-cyan-500/50 px-4 py-2.5 text-sm outline-none focus:ring-2 transition-all hover:border-black/20 dark:hover:border-white/20";
export const compactInputClassName = "w-full rounded-lg border border-black/10 dark:border-white/10 bg-white/60 dark:bg-black/30 text-foreground shadow-sm focus:border-cyan-500 focus:ring-cyan-500/40 px-3 py-2 text-sm outline-none focus:ring-2 transition-all hover:border-black/20 dark:hover:border-white/20";
export const labelClassName = "block text-[11px] font-bold text-muted-foreground mb-1.5 uppercase tracking-wider";
export const cardClassName = "bg-white/60 dark:bg-zinc-900/60 backdrop-blur-xl p-6 rounded-2xl border border-white/40 dark:border-white/10 shadow-lg relative overflow-hidden";
export const sectionTitleClassName = "text-lg font-bold text-foreground flex items-center gap-2";
export const primaryButtonClassName = "px-6 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-white rounded-xl font-medium shadow-md hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 focus:ring-offset-background disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 cursor-pointer";
