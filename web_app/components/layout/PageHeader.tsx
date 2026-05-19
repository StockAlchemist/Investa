'use client';

import dynamic from 'next/dynamic';
import { cn, formatCompactNumber } from '@/lib/utils';
import { Menu, ArrowUpRight, ArrowDownRight, Eye, EyeOff } from 'lucide-react';
import AccountSelector from '@/components/AccountSelector';
import CurrencySelector from '@/components/CurrencySelector';
import { StockSearchBar } from '@/components/StockSearchBar';

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
  onCommandPaletteOpen?: () => void;
  fxRate?: number;
  availableCurrencies?: string[];
  isFetching?: boolean;
  onIndexClick?: () => void;
  onMobileMenuOpen?: () => void;
  isMarketOpen?: boolean;
  lastUpdated?: Date | null;
  marketValue?: number | null;
  dayChangePct?: number | null;
  showClosed?: boolean;
  onShowClosedChange?: (v: boolean) => void;
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
  marketValue,
  dayChangePct,
  showClosed,
  onShowClosedChange,
}: PageHeaderProps) {
  const hasKpi = marketValue != null;
  const dayPositive = (dayChangePct ?? 0) >= 0;

  return (
    <header
      className={cn(
        'sticky top-0 z-40 flex items-center h-[52px] shrink-0 px-3 sm:px-5 gap-2 sm:gap-3',
        // Glass backdrop with subtle gradient
        'border-b border-border/60',
        'bg-gradient-to-r from-background/85 via-background/75 to-background/85',
        'backdrop-blur-xl supports-[backdrop-filter]:bg-background/60',
      )}
    >

      {/* Mobile: logo + name */}
      <div className="flex items-center gap-2 md:hidden">
        <img src="logo-dark.png?v=5" alt="Investa" className="w-6 h-6 rounded-md hidden dark:block" />
        <img src="logo.png?v=5"      alt="Investa" className="w-6 h-6 rounded-md dark:hidden" />
        <span className="text-sm font-bold text-foreground">Investa</span>
      </div>

      {/* Desktop: page title */}
      <h1 className="hidden md:block text-sm font-semibold text-foreground shrink-0 select-none truncate max-w-[120px] lg:max-w-none">
        {TAB_LABELS[activeTab] ?? activeTab}
      </h1>

      {/* ── Mini KPI: portfolio value + day change ── */}
      {hasKpi && (
        <>
          <span className="hidden lg:block w-px h-5 bg-border/70" />
          <div className="hidden lg:flex items-baseline gap-2 shrink-0 select-none">
            <span className="text-sm font-bold tabular-nums text-foreground leading-none">
              {formatCompactNumber(marketValue!, currency)}
            </span>
            {dayChangePct != null && (
              <span
                className={cn(
                  'flex items-center gap-0.5 text-[11px] font-bold tabular-nums leading-none px-1.5 py-0.5 rounded-full',
                  dayPositive
                    ? 'text-emerald-600 dark:text-emerald-400 bg-emerald-500/10'
                    : 'text-red-600 dark:text-red-400 bg-red-500/10',
                )}
              >
                {dayPositive
                  ? <ArrowUpRight className="w-3 h-3" />
                  : <ArrowDownRight className="w-3 h-3" />}
                {dayPositive ? '+' : ''}{dayChangePct.toFixed(2)}%
              </span>
            )}
          </div>
        </>
      )}

      {/* ── Market status + last updated ── */}
      {(isMarketOpen !== undefined || lastUpdated) && (
        <>
          <span className="hidden xl:block w-px h-5 bg-border/70" />
          <div className="hidden xl:flex items-center gap-2 shrink-0">
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
              <span className="text-[10px] text-muted-foreground/70 font-medium tabular-nums">
                {lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            )}
          </div>
        </>
      )}

      <div className="flex-1 min-w-0" />

      {/* ── Right cluster: controls ── */}
      <div className="flex items-center gap-1 sm:gap-1.5">

        {/* Stock symbol search */}
        <div className="hidden sm:block">
          <StockSearchBar currency={currency} />
        </div>

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

        <span className="hidden sm:block w-px h-4 bg-border/70" />

        {/* Dashboard widget configurator */}
        {activeTab === 'performance' && (
          <LayoutConfigurator
            visibleItems={visibleItems}
            onVisibleItemsChange={onVisibleItemsChange}
            variant="ghost"
          />
        )}

        <CurrencySelector
          currentCurrency={currency}
          onChange={onCurrencyChange}
          fxRate={fxRate}
          availableCurrencies={availableCurrencies}
          align="right"
        />

        {/* Global Show/Hide Closed Positions toggle */}
        {onShowClosedChange && (
          <button
            onClick={() => onShowClosedChange(!showClosed)}
            title={showClosed ? 'Hide closed positions' : 'Show closed positions'}
            aria-pressed={!!showClosed}
            className={cn(
              'flex flex-row items-center gap-1.5 p-3 py-2 px-2 h-[44px] rounded-2xl transition-all duration-300 group bg-transparent',
              'font-semibold tracking-tight',
              showClosed ? 'ring-2 ring-cyan-500/20' : 'text-cyan-500',
            )}
          >
            {showClosed
              ? <EyeOff className="w-3.5 h-3.5 text-cyan-500" />
              : <Eye className="w-3.5 h-3.5 text-cyan-500" />}
            <span className="hidden md:inline bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent font-bold uppercase text-[14px]">
              Closed
            </span>
          </button>
        )}

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
