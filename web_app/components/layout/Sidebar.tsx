'use client';

import { cn } from '@/lib/utils';
import { useTheme } from 'next-themes';
import {
  LayoutDashboard, PieChart, TrendingUp, ArrowLeftRight,
  DollarSign, BarChart3, Search, Star, Globe, Sparkles,
  Settings, ChevronLeft, Sun, Moon, LogOut,
} from 'lucide-react';

const PRIMARY_NAV = [
  { id: 'performance',   label: 'Dashboard',     icon: LayoutDashboard },
  { id: 'allocation',    label: 'Portfolio',      icon: PieChart },
  { id: 'asset_change',  label: 'Performance',    icon: TrendingUp },
  { id: 'transactions',  label: 'Transactions',   icon: ArrowLeftRight },
  { id: 'dividend',      label: 'Income',         icon: DollarSign },
  { id: 'capital_gains', label: 'Capital Gains',  icon: BarChart3 },
] as const;

const SECONDARY_NAV = [
  { id: 'screener',   label: 'Screener',    icon: Search },
  { id: 'watchlist',  label: 'Watchlist',   icon: Star },
  { id: 'markets',    label: 'Markets',     icon: Globe },
  { id: 'ai_review',  label: 'AI Insights', icon: Sparkles },
] as const;

interface SidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
  user: { username: string } | null;
  onLogout: () => void;
  collapsed: boolean;
  onToggle: () => void;
  onUserClick: () => void;
  dayChangePct?: number;
}

export function Sidebar({
  activeTab, onTabChange, user, onLogout, collapsed, onToggle, onUserClick, dayChangePct,
}: SidebarProps) {
  const { resolvedTheme, setTheme } = useTheme();

  function NavItem({ id, label, icon: Icon }: { id: string; label: string; icon: React.ComponentType<{ className?: string }> }) {
    const active = activeTab === id;
    return (
      <button
        onClick={() => onTabChange(id)}
        title={collapsed ? label : undefined}
        className={cn(
          'group/item relative flex items-center w-full rounded-lg text-sm font-medium transition-all duration-150',
          collapsed ? 'h-9 justify-center' : 'h-9 px-3 gap-2.5',
          active
            ? 'bg-primary/10 text-primary'
            : 'text-muted-foreground hover:bg-muted hover:text-foreground',
        )}
      >
        {active && (
          <span className="absolute left-0 inset-y-[6px] w-[2px] bg-primary rounded-r-full" />
        )}
        <Icon className="shrink-0 w-4 h-4" />
        {!collapsed && <span className="truncate">{label}</span>}
        {collapsed && (
          <span className="pointer-events-none absolute left-full ml-2 z-50 px-2 py-1 rounded-md bg-popover border border-border text-xs font-medium shadow-lg whitespace-nowrap opacity-0 group-hover/item:opacity-100 transition-opacity duration-150 delay-200">
            {label}
          </span>
        )}
      </button>
    );
  }

  return (
    <aside
      className={cn(
        'relative hidden md:flex flex-col h-screen border-r border-border bg-card shrink-0 transition-[width] duration-300 ease-in-out overflow-visible',
        collapsed ? 'w-14' : 'w-[216px]',
      )}
    >
      {/* Logo */}
      <div
        className={cn(
          'flex items-center h-[52px] border-b border-border shrink-0',
          collapsed ? 'flex-col justify-center gap-0.5' : 'px-4 gap-2.5',
        )}
      >
        <img src="logo-dark.png?v=5" alt="Investa" className="w-7 h-7 rounded-lg shrink-0 hidden dark:block" />
        <img src="logo.png?v=5"      alt="Investa" className="w-7 h-7 rounded-lg shrink-0 dark:hidden" />
        {!collapsed && (
          <div className="min-w-0 overflow-hidden">
            <div className="text-sm font-bold text-foreground leading-none">Investa</div>
            <div className="text-[9px] text-muted-foreground/50 font-semibold tracking-[0.15em] uppercase mt-0.5">
              StockAlchemist
            </div>
          </div>
        )}
        {collapsed && dayChangePct !== undefined && (
          <span className={cn(
            'text-[9px] font-bold tabular-nums leading-none',
            dayChangePct >= 0 ? 'text-emerald-500' : 'text-red-500',
          )}>
            {dayChangePct >= 0 ? '+' : ''}{dayChangePct.toFixed(1)}%
          </span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto overflow-x-hidden px-2 py-3 space-y-0.5 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        {PRIMARY_NAV.map(item => <NavItem key={item.id} {...item} />)}
        <div className="my-2 border-t border-border" />
        {SECONDARY_NAV.map(item => <NavItem key={item.id} {...item} />)}
      </nav>

      {/* Bottom utilities */}
      <div className="border-t border-border px-2 py-2 space-y-0.5 shrink-0">
        <NavItem id="settings" label="Settings" icon={Settings} />

        <button
          onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
          title={collapsed ? (resolvedTheme === 'dark' ? 'Light mode' : 'Dark mode') : undefined}
          className={cn(
            'flex items-center w-full h-9 rounded-lg text-sm font-medium text-muted-foreground hover:bg-muted hover:text-foreground transition-all duration-150',
            collapsed ? 'justify-center' : 'px-3 gap-2.5',
          )}
        >
          {resolvedTheme === 'dark'
            ? <Sun className="w-4 h-4 shrink-0" />
            : <Moon className="w-4 h-4 shrink-0" />}
          {!collapsed && (
            <span>{resolvedTheme === 'dark' ? 'Light mode' : 'Dark mode'}</span>
          )}
        </button>

        {/* User row */}
        <div
          onClick={onUserClick}
          className={cn(
            'flex items-center h-9 rounded-lg hover:bg-muted transition-all duration-150 cursor-pointer',
            collapsed ? 'justify-center' : 'px-3 gap-2.5',
          )}
        >
          <div className="w-6 h-6 rounded-full bg-primary/15 text-primary text-xs font-bold flex items-center justify-center shrink-0 select-none">
            {user?.username?.[0]?.toUpperCase() ?? 'U'}
          </div>
          {!collapsed && (
            <>
              <span className="text-sm font-medium text-foreground truncate flex-1 min-w-0">
                {user?.username}
              </span>
              <button
                onClick={e => { e.stopPropagation(); onLogout(); }}
                title="Sign out"
                className="p-1 rounded text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
              >
                <LogOut className="w-3.5 h-3.5" />
              </button>
            </>
          )}
        </div>
      </div>

      {/* Collapse / expand toggle */}
      <button
        onClick={onToggle}
        className="absolute -right-3 top-[66px] z-20 w-6 h-6 rounded-full bg-card border border-border shadow-sm flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-muted transition-all"
        title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        <ChevronLeft
          className={cn('w-3 h-3 transition-transform duration-300', collapsed && 'rotate-180')}
        />
      </button>
    </aside>
  );
}
