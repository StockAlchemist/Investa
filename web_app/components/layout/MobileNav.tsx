'use client';

import { useEffect } from 'react';
import { cn } from '@/lib/utils';
import {
  LayoutDashboard, PieChart, TrendingUp, ArrowLeftRight,
  DollarSign, BarChart3, Search, Star, Globe, Sparkles,
  LogOut, X,
} from 'lucide-react';
import { StockSearchBar } from '@/components/StockSearchBar';

const ALL_NAV = [
  { id: 'performance',   label: 'Dashboard',     icon: LayoutDashboard, section: 'main' },
  { id: 'allocation',    label: 'Portfolio',      icon: PieChart,        section: 'main' },
  { id: 'asset_change',  label: 'Performance',    icon: TrendingUp,      section: 'main' },
  { id: 'transactions',  label: 'Transactions',   icon: ArrowLeftRight,  section: 'main' },
  { id: 'dividend',      label: 'Income',         icon: DollarSign,      section: 'main' },
  { id: 'capital_gains', label: 'Capital Gains',  icon: BarChart3,       section: 'main' },
  { id: 'screener',      label: 'Screener',       icon: Search,          section: 'tools' },
  { id: 'watchlist',     label: 'Watchlist',      icon: Star,            section: 'tools' },
  { id: 'markets',       label: 'Markets',        icon: Globe,           section: 'tools' },
  { id: 'ai_review',     label: 'AI Insights',    icon: Sparkles,        section: 'tools' },
] as const;

interface MobileNavProps {
  isOpen: boolean;
  onClose: () => void;
  activeTab: string;
  onTabChange: (tab: string) => void;
  user: { username: string; alias?: string } | null;
  onLogout: () => void;
  onUserClick: () => void;
  currency: string;
}

export function MobileNav({
  isOpen, onClose, activeTab, onTabChange, user, onLogout, onUserClick, currency,
}: MobileNavProps) {
  // Lock body scroll when open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => { document.body.style.overflow = ''; };
  }, [isOpen]);

  // Close on Escape key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const handleTabSelect = (tab: string) => {
    onTabChange(tab);
    onClose();
  };

  return (
    <>
      {/* Backdrop */}
      <div
        className={cn(
          'fixed inset-0 z-50 bg-black/60 backdrop-blur-sm transition-opacity duration-300 md:hidden',
          isOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none',
        )}
        onClick={onClose}
      />

      {/* Drawer */}
      <div
        className={cn(
          'fixed top-0 left-0 z-50 h-dvh w-72 flex flex-col bg-white dark:bg-zinc-900 border-r border-border shadow-2xl transition-transform duration-300 ease-in-out md:hidden',
          isOpen ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between h-[52px] border-b border-border px-4 shrink-0">
          <div className="flex items-center gap-2.5">
            <img src="logo-dark.png?v=5" alt="Investa" className="w-7 h-7 rounded-lg hidden dark:block" />
            <img src="logo.png?v=5"      alt="Investa" className="w-7 h-7 rounded-lg dark:hidden" />
            <div>
              <div className="text-sm font-bold text-foreground leading-none">Investa</div>
              <div className="text-[9px] text-muted-foreground/50 font-semibold tracking-[0.15em] uppercase mt-0.5">StockAlchemist</div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="h-7 w-7 flex items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground transition-all"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Stock search */}
        <div className="px-3 py-2.5 border-b border-border shrink-0">
          <StockSearchBar currency={currency} placeholder="Search symbol…" fullWidth />
        </div>

        {/* Nav */}
        <nav className="flex-1 min-h-0 overflow-y-auto px-3 py-3 space-y-0.5">
          <p className="section-label px-2 mb-2">Portfolio</p>
          {ALL_NAV.filter(n => n.section === 'main').map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => handleTabSelect(id)}
              className={cn(
                'relative flex items-center w-full h-10 px-3 gap-3 rounded-lg text-sm font-medium transition-all duration-150',
                activeTab === id
                  ? 'bg-blue-500/15 text-blue-600 dark:text-blue-400 font-semibold'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground',
              )}
            >
              {activeTab === id && (
                <span className="absolute left-0 inset-y-[8px] w-[3px] bg-blue-500 rounded-r-full" />
              )}
              <Icon className="w-4 h-4 shrink-0" />
              {label}
            </button>
          ))}

          <div className="my-2 border-t border-border" />
          <p className="section-label px-2 mb-2">Tools</p>
          {ALL_NAV.filter(n => n.section === 'tools').map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => handleTabSelect(id)}
              className={cn(
                'relative flex items-center w-full h-10 px-3 gap-3 rounded-lg text-sm font-medium transition-all duration-150',
                activeTab === id
                  ? 'bg-blue-500/15 text-blue-600 dark:text-blue-400 font-semibold'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground',
              )}
            >
              {activeTab === id && (
                <span className="absolute left-0 inset-y-[8px] w-[3px] bg-blue-500 rounded-r-full" />
              )}
              <Icon className="w-4 h-4 shrink-0" />
              {label}
            </button>
          ))}
        </nav>

        {/* Bottom utilities */}
        <div className="border-t border-border px-3 pt-3 pb-3 space-y-0.5 shrink-0" style={{ paddingBottom: 'max(0.75rem, env(safe-area-inset-bottom))' }}>
          {/* User row */}
          <div
            onClick={() => { onUserClick(); onClose(); }}
            className="flex items-center h-10 px-3 gap-3 rounded-lg hover:bg-muted transition-all duration-150 cursor-pointer"
          >
            <div className="w-6 h-6 rounded-full bg-primary/15 text-primary text-xs font-bold flex items-center justify-center shrink-0 select-none">
              {(user?.alias || user?.username)?.[0]?.toUpperCase() ?? 'U'}
            </div>
            <span className="text-sm font-medium text-foreground truncate flex-1">{user?.alias || user?.username}</span>
            <button
              onClick={e => { e.stopPropagation(); onLogout(); onClose(); }}
              title="Sign out"
              className="p-1 rounded text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
            >
              <LogOut className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
