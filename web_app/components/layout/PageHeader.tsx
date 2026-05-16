'use client';

import dynamic from 'next/dynamic';
import { cn } from '@/lib/utils';
import { Search, Menu } from 'lucide-react';
import AccountSelector from '@/components/AccountSelector';
import CurrencySelector from '@/components/CurrencySelector';

const MarketIndicesBox  = dynamic(() => import('@/components/MarketIndicesBox'),  { ssr: false });
const LayoutConfigurator = dynamic(() => import('@/components/LayoutConfigurator'));

const TAB_LABELS: Record<string, string> = {
  performance:   'Dashboard',
  allocation:    'Portfolio',
  asset_change:  'Performance',
  transactions:  'Transactions',
  dividend:      'Income',
  capital_gains: 'Capital Gains',
  screener:      'Screener',
  watchlist:     'Watchlist',
  markets:       'Markets',
  ai_review:     'AI Insights',
  settings:      'Settings',
};

interface PageHeaderProps {
  activeTab: string;
  currency: string;
  onCurrencyChange: (c: string) => void;
  availableAccounts: string[];
  selectedAccounts: string[];
  onAccountsChange: (a: string[]) => void;
  accountGroups?: Record<string, string[]>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  indices?: Record<string, any>;
  visibleItems: string[];
  onVisibleItemsChange: (items: string[]) => void;
  onCommandPaletteOpen: () => void;
  fxRate?: number;
  availableCurrencies?: string[];
  isFetching?: boolean;
  onIndexClick?: () => void;
  onMobileMenuOpen?: () => void;
  isMarketOpen?: boolean;
  lastUpdated?: Date | null;
}

export function PageHeader({
  activeTab,
  currency,
  onCurrencyChange,
  availableAccounts,
  selectedAccounts,
  onAccountsChange,
  accountGroups,
  indices,
  visibleItems,
  onVisibleItemsChange,
  onCommandPaletteOpen,
  fxRate,
  availableCurrencies,
  isFetching,
  onIndexClick,
  onMobileMenuOpen,
  isMarketOpen,
  lastUpdated,
}: PageHeaderProps) {
  return (
    <header className="sticky top-0 z-40 flex items-center h-[52px] border-b border-border bg-background/95 backdrop-blur-sm supports-[backdrop-filter]:bg-background/90 shrink-0 px-3 sm:px-5 gap-2 sm:gap-3">

      {/* Mobile: logo + hamburger */}
      <div className="flex items-center gap-2 md:hidden">
        <img src="logo-dark.png?v=5" alt="Investa" className="w-6 h-6 rounded-md hidden dark:block" />
        <img src="logo.png?v=5"      alt="Investa" className="w-6 h-6 rounded-md dark:hidden" />
        <span className="text-sm font-bold text-foreground">Investa</span>
      </div>

      {/* Desktop: page title */}
      <h1 className="hidden md:block text-sm font-semibold text-foreground shrink-0 select-none">
        {TAB_LABELS[activeTab] ?? activeTab}
      </h1>

      {/* Market status + last updated */}
      <div className="hidden md:flex items-center gap-2 shrink-0">
        {isMarketOpen !== undefined && (
          <span className={cn(
            'flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full border',
            isMarketOpen
              ? 'text-emerald-600 dark:text-emerald-400 border-emerald-500/30 bg-emerald-500/10'
              : 'text-muted-foreground border-border bg-muted/40',
          )}>
            <span className={cn(
              'w-1.5 h-1.5 rounded-full',
              isMarketOpen ? 'bg-emerald-500 animate-pulse' : 'bg-muted-foreground/50',
            )} />
            {isMarketOpen ? 'Live' : 'Closed'}
          </span>
        )}
        {lastUpdated && (
          <span className="text-[10px] text-muted-foreground/60 font-medium tabular-nums">
            {lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>
        )}
      </div>

      <div className="flex-1" />

      {/* Controls row */}
      <div className="flex items-center gap-1 sm:gap-1.5">

        {/* Search / command palette */}
        <button
          onClick={onCommandPaletteOpen}
          className="hidden sm:flex items-center gap-2 h-7 px-2.5 rounded-md border border-border bg-muted/40 text-muted-foreground text-xs hover:bg-muted hover:text-foreground transition-all"
        >
          <Search className="w-3 h-3 shrink-0" />
          <span className="hidden md:inline">Search</span>
          <kbd className="hidden lg:inline px-1.5 py-0.5 rounded bg-background/80 border border-border text-[10px] font-mono leading-none">
            ⌘K
          </kbd>
        </button>

        {/* Market indices ticker — hidden below xl */}
        {indices && (
          <div className="hidden xl:block">
            <MarketIndicesBox
              indices={indices}
              isFetching={isFetching ?? false}
              onClick={onIndexClick ?? (() => {})}
            />
          </div>
        )}

        {/* Separator */}
        <span className="hidden sm:block w-px h-4 bg-border" />

        {/* Dashboard widget configurator */}
        {activeTab === 'performance' && (
          <div className="hidden sm:block">
            <LayoutConfigurator
              visibleItems={visibleItems}
              onVisibleItemsChange={onVisibleItemsChange}
              variant="ghost"
            />
          </div>
        )}

        <CurrencySelector
          currentCurrency={currency}
          onChange={onCurrencyChange}
          fxRate={fxRate}
          availableCurrencies={availableCurrencies}
          align="right"
        />

        <AccountSelector
          availableAccounts={availableAccounts}
          selectedAccounts={selectedAccounts}
          onChange={onAccountsChange}
          accountGroups={accountGroups}
          align="right"
        />

        {/* Mobile: hamburger / nav trigger */}
        {onMobileMenuOpen && (
          <button
            onClick={onMobileMenuOpen}
            className="md:hidden h-7 w-7 flex items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground transition-all"
          >
            <Menu className="w-4 h-4" />
          </button>
        )}
      </div>
    </header>
  );
}
