import {
    LayoutDashboard, PieChart, TrendingUp, ArrowLeftRight,
    DollarSign, BarChart3, Search, Star, Globe, Sparkles, Trophy,
    Settings, Layers,
} from 'lucide-react';

/**
 * The one place a destination is named.
 *
 * Before this table the sidebar and the tab menu disagreed about half the app:
 * `performance` was "Dashboard" in one and "Performance" in the other,
 * `allocation` was Portfolio or Allocation, `asset_change` was Performance or
 * Asset Change, `dividend` was Income or Dividend, `screener` was Screener or
 * Market, `ai_review` was AI Insights or AI Review. Six of twelve destinations
 * answered to two names depending on which control you reached them by.
 *
 * These labels match `AppSection` in `macos_app/Investa/App/MainView.swift`, so
 * the web sidebar, the header title, the command palette, the macOS sidebar and
 * the iOS tab bar all say the same word for the same screen.
 */
export interface NavItem {
    id: string;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
}

/** Portfolio — the user's own holdings. */
export const PRIMARY_NAV: readonly NavItem[] = [
    { id: 'performance', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'allocation', label: 'Portfolio', icon: PieChart },
    { id: 'asset_change', label: 'Performance', icon: TrendingUp },
    { id: 'transactions', label: 'Transactions', icon: ArrowLeftRight },
    { id: 'dividend', label: 'Income', icon: DollarSign },
    { id: 'capital_gains', label: 'Capital Gains', icon: BarChart3 },
] as const;

/** Research — the market outside the portfolio. */
export const SECONDARY_NAV: readonly NavItem[] = [
    { id: 'screener', label: 'Screener', icon: Search },
    { id: 'buffett_rank', label: 'Rankings', icon: Trophy },
    { id: 'strategies', label: 'Strategies', icon: Layers },
    { id: 'watchlist', label: 'Watchlist', icon: Star },
    { id: 'markets', label: 'Markets', icon: Globe },
    { id: 'ai_review', label: 'AI Insights', icon: Sparkles },
] as const;

export const SETTINGS_NAV: NavItem = { id: 'settings', label: 'Settings', icon: Settings };

export const ALL_NAV: readonly NavItem[] = [...PRIMARY_NAV, ...SECONDARY_NAV, SETTINGS_NAV];

/** id → label, for headers, titles and confirm() text. */
export const NAV_LABELS: Record<string, string> = Object.fromEntries(
    ALL_NAV.map(item => [item.id, item.label]),
);

export function navLabel(id: string): string {
    return NAV_LABELS[id] ?? id;
}
