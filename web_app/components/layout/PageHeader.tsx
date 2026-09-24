'use client';

import dynamic from 'next/dynamic';
import { cn, formatCompactNumber } from '@/lib/utils';
import { ArrowUpRight, ArrowDownRight, Eye, EyeOff } from 'lucide-react';
import AccountSelector from '@/components/AccountSelector';
import CurrencySelector from '@/components/CurrencySelector';
import { StockSearchBar } from '@/components/StockSearchBar';
import { navLabel } from '@/lib/navigation';

const MarketIndicesBox  = dynamic(() => import('@/components/MarketIndicesBox'),  { ssr: false });
const LayoutConfigurator = dynamic(() => import('@/components/LayoutConfigurator'));

// Labels come from lib/navigation — see the note there on why.

interface PageHeaderProps {
  activeTab: string;
  currency: string;
  onCurrencyChange: (c: string) => void;
  availableAccounts: string[];
  selectedAccounts: string[];
  onAccountsChange: (a: string[]) => void;
  accountGroups?: Record<string, string[]>;
  closedAccounts?: string[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  indices?: Record<string, any>;
  visibleItems: string[];
  onVisibleItemsChange: (items: string[]) => void;
  layoutItems?: { id: string; title: string }[];
  layoutSectionTitle?: string;
  onCommandPaletteOpen?: () => void;
  fxRate?: number;
  availableCurrencies?: string[];
  isFetching?: boolean;
  onIndexClick?: () => void;
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
  closedAccounts,
  indices,
  visibleItems,
  onVisibleItemsChange,
  layoutItems,
  layoutSectionTitle,
  fxRate,
  availableCurrencies,
  isFetching,
  onIndexClick,
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
        // Desktop only. Below `md` the phone shell draws its own navigation bar
        // and control bar (components/layout/mobile/), which mirror the native
        // iPhone app's toolbar + GlobalControlBar.
        'sticky top-0 z-40 hidden md:flex items-center h-16 shrink-0 px-3 sm:px-6 gap-2 sm:gap-3',
        // Paper, lightly frosted so content scrolls under it; a hairline rule
        // only — no gradient.
        'border-b border-border/70 bg-background/90 backdrop-blur-md',
      )}
    >

      {/* Desktop: page title */}
      <h1 className="hidden md:block page-title text-[28px] leading-8 text-foreground shrink-0 select-none whitespace-nowrap">
        {navLabel(activeTab)}
      </h1>

      {/* ── Mini KPI: portfolio value + day change. From 2xl up only: with the
           serif title, the toolbar controls come first at laptop widths. ── */}
      {hasKpi && (
        <>
          <span className="hidden 2xl:block w-px h-5 bg-border/70" />
          <div className="hidden 2xl:flex items-baseline gap-2 shrink-0 select-none">
            <span className="text-sm font-semibold tabular-nums text-foreground leading-none">
              {formatCompactNumber(marketValue!, currency, true)}
            </span>
            {dayChangePct != null && (
              <span
                className={cn(
                  'flex items-center gap-0.5 text-xs font-semibold tabular-nums leading-none px-1.5 py-1 rounded-md',
                  dayPositive ? 'text-up bg-up-tint' : 'text-down bg-down-tint',
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
          <span className="hidden 2xl:block w-px h-5 bg-border/70" />
          <div className="hidden 2xl:flex items-center gap-2 shrink-0">
            {isMarketOpen !== undefined && (
              <span className={cn(
                'flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded-md',
                isMarketOpen ? 'text-up bg-up-tint' : 'text-muted-foreground bg-muted',
              )}>
                <span className={cn(
                  'w-1.5 h-1.5 rounded-full',
                  isMarketOpen ? 'bg-up' : 'bg-muted-foreground/50',
                )} />
                {isMarketOpen ? 'Live' : 'Closed'}
              </span>
            )}
            {lastUpdated && (
              <span className="text-xs text-muted-foreground tabular-nums">
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
          <div className="hidden 2xl:block">
            <MarketIndicesBox
              indices={indices}
              isFetching={isFetching ?? false}
              onClick={onIndexClick ?? (() => {})}
            />
          </div>
        )}

        <span className="hidden sm:block w-px h-4 bg-border/70" />

        {/* Tab layout configurator */}
        {layoutItems && layoutItems.length > 0 && (
          <LayoutConfigurator
            visibleItems={visibleItems}
            onVisibleItemsChange={onVisibleItemsChange}
            items={layoutItems}
            sectionTitle={layoutSectionTitle}
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
            aria-label={showClosed ? 'Hide closed positions' : 'Show closed positions'}
            aria-pressed={!!showClosed}
            className={cn(
              'flex items-center gap-1.5 h-9 px-2.5 rounded-control text-[13px] font-medium border transition-colors',
              showClosed
                ? 'bg-primary-tint text-primary-ink border-primary/25'
                : 'bg-card border-border text-ink-2 hover:text-foreground hover:border-input',
            )}
          >
            {showClosed
              ? <EyeOff className="w-3.5 h-3.5" />
              : <Eye className="w-3.5 h-3.5" />}
            <span className="hidden md:inline">Closed</span>
          </button>
        )}

        <AccountSelector
          availableAccounts={availableAccounts}
          selectedAccounts={selectedAccounts}
          onChange={onAccountsChange}
          accountGroups={accountGroups}
          closedAccounts={closedAccounts}
          align="right"
        />
      </div>
    </header>
  );
}
