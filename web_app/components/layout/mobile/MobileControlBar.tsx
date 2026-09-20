'use client';

/**
 * The phone's control bar — the web twin of `GlobalControlBar.compactBar`
 * (App/GlobalControlBar.swift), sitting under the navigation bar on every tab.
 *
 * Same anatomy, same order as the native bar: a horizontal scroller holding the
 * account, layout and show-closed controls; then the market-status dot and
 * last-updated time; refresh; the search field; the currency menu; and the
 * host's trailing control (Settings) at the right edge.
 *
 * The bar carries no tab name and no headline KPI. The native bar hides both at
 * compact width (`showsSectionTitle`) because the tab bar below already names
 * the tab, and the web header hides its own title below `md` for the same
 * reason.
 *
 * While the search field is focused it takes over the whole bar and the other
 * controls hide — the standard iOS search pattern, and what stops the expanded
 * field from shoving the currency menu off-screen.
 */

import { useState } from 'react';
import {
  Building2, SlidersHorizontal, Eye, EyeOff, RefreshCw, Loader2, Settings as SettingsIcon, ChevronsUpDown,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { StockSearchBar } from '@/components/StockSearchBar';
import { MobileMenu, MenuToggleRow, MenuSectionHeader } from './MobileMenu';

interface LayoutItem { id: string; title: string; group?: string }

interface MobileControlBarProps {
  currency: string;
  onCurrencyChange: (c: string) => void;
  availableCurrencies?: string[];
  availableAccounts: string[];
  selectedAccounts: string[];
  onAccountsChange: (a: string[]) => void;
  accountGroups?: Record<string, string[]>;
  accountGroupOrder?: string[];
  closedAccounts?: string[];
  showClosed: boolean;
  onShowClosedChange: (v: boolean) => void;
  layoutItems?: LayoutItem[];
  layoutSectionTitle?: string;
  visibleItems: string[];
  onVisibleItemsChange: (items: string[]) => void;
  isMarketOpen?: boolean;
  lastUpdated?: Date | null;
  isRefreshing?: boolean;
  onRefresh: () => void;
  onOpenSettings: () => void;
}

const DEFAULT_CURRENCIES = ['USD', 'THB', 'EUR', 'GBP', 'JPY', 'CNY'];

/** One icon-sized tap target — the bar's controls are all 32×32, as on iOS. */
const CONTROL = 'flex h-8 w-8 items-center justify-center rounded-control shrink-0 transition-colors active:bg-foreground/10';

export function MobileControlBar({
  currency, onCurrencyChange, availableCurrencies,
  availableAccounts, selectedAccounts, onAccountsChange, accountGroups = {}, accountGroupOrder = [], closedAccounts = [],
  showClosed, onShowClosedChange,
  layoutItems, layoutSectionTitle, visibleItems, onVisibleItemsChange,
  isMarketOpen, lastUpdated, isRefreshing, onRefresh, onOpenSettings,
}: MobileControlBarProps) {
  const [searchActive, setSearchActive] = useState(false);

  // Open accounts first, then closed, each alphabetical — the native ordering,
  // which is itself the web selector's.
  const closedSet = new Set(closedAccounts);
  const individuals = availableAccounts
    .filter(a => a !== 'All Accounts')
    .sort((a, b) => {
      const ac = closedSet.has(a), bc = closedSet.has(b);
      if (ac !== bc) return ac ? 1 : -1;
      return a.localeCompare(b, undefined, { numeric: true });
    });

  // Groups missing from the saved order (newly created) go last rather than
  // being dropped — matching the native `orderedGroups`.
  const groupNames = accountGroupOrder.length > 0 ? [...accountGroupOrder] : Object.keys(accountGroups).sort();
  for (const name of Object.keys(accountGroups).sort()) {
    if (!groupNames.includes(name)) groupNames.push(name);
  }
  const orderedGroups = groupNames.filter(n => accountGroups[n]).map(n => ({ name: n, accounts: accountGroups[n] }));

  const sameSet = (a: string[], b: string[]) =>
    a.length > 0 && a.length === b.length && a.every(x => b.includes(x));

  const toggleAccount = (account: string) => {
    onAccountsChange(
      selectedAccounts.includes(account)
        ? selectedAccounts.filter(a => a !== account)
        : [...selectedAccounts, account],
    );
  };

  const toggleLayoutItem = (id: string) => {
    onVisibleItemsChange(
      visibleItems.includes(id) ? visibleItems.filter(i => i !== id) : [...visibleItems, id],
    );
  };

  // Group the layout items by their `group` in first-seen order, as the native
  // `groupedItems(_:)` does.
  const layoutGroups: { label?: string; items: LayoutItem[] }[] = [];
  const groupIndex = new Map<string, number>();
  for (const item of layoutItems ?? []) {
    const key = item.group ?? '__none';
    const idx = groupIndex.get(key);
    if (idx === undefined) {
      groupIndex.set(key, layoutGroups.length);
      layoutGroups.push({ label: item.group, items: [item] });
    } else {
      layoutGroups[idx].items.push(item);
    }
  }

  const currencies = availableCurrencies?.length ? availableCurrencies : DEFAULT_CURRENCIES;

  return (
    <div role="toolbar" aria-label="Portfolio controls" className="ios-bar sticky top-0 z-30 shrink-0 border-b border-border/60 md:hidden">
      <div className="flex items-center gap-2 py-1 pr-3">
        {!searchActive && (
          <>
            {/* Scroller: account · layout · show-closed.
                It takes the slack (`flex-1`), as the native `ScrollView` does
                ahead of its `Spacer`.

                A 390pt phone cannot fit all nine controls at once — which is
                why the native bar makes these three a `ScrollView` rather than
                a plain `HStack`. The trailing fade says so: a control cut off
                at the edge with no fade reads as broken, the same control under
                a fade reads as "swipe for more", which is what it is. */}
            <div className="flex min-w-0 flex-1 items-center gap-3 overflow-x-auto pl-3 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden [mask-image:linear-gradient(to_right,black_calc(100%-20px),transparent)]">
              <MobileMenu
                ariaLabel="Accounts"
                minWidth={220}
                maxHeight={440}
                align="left"
                className={CONTROL}
                label={<Building2 className="h-[18px] w-[18px]" />}
              >
                <MenuToggleRow
                  title="All Accounts"
                  isOn={selectedAccounts.length === 0}
                  dismissOnTap
                  onSelect={() => onAccountsChange([])}
                />
                {orderedGroups.length > 0 && (
                  <>
                    <MenuSectionHeader>Groups</MenuSectionHeader>
                    {orderedGroups.map(group => (
                      <MenuToggleRow
                        key={group.name}
                        title={group.name}
                        isOn={sameSet(selectedAccounts, group.accounts)}
                        dismissOnTap
                        onSelect={() => onAccountsChange([...group.accounts])}
                      />
                    ))}
                    <MenuSectionHeader>Individual</MenuSectionHeader>
                  </>
                )}
                {individuals.map(account => (
                  <MenuToggleRow
                    key={account}
                    title={account}
                    isOn={selectedAccounts.includes(account)}
                    trailing={closedSet.has(account) ? 'Closed' : undefined}
                    onSelect={() => toggleAccount(account)}
                  />
                ))}
              </MobileMenu>

              {layoutItems && layoutItems.length > 0 && (
                <MobileMenu
                  ariaLabel={layoutSectionTitle ?? 'Layout'}
                  minWidth={240}
                  maxHeight={440}
                  align="left"
                  className={CONTROL}
                  label={<SlidersHorizontal className="h-[18px] w-[18px]" />}
                >
                  <MenuSectionHeader>{layoutSectionTitle ?? 'Elements'}</MenuSectionHeader>
                  {layoutGroups.map((group, i) => (
                    <div key={group.label ?? i}>
                      {group.label && <MenuSectionHeader>{group.label}</MenuSectionHeader>}
                      {group.items.map(item => (
                        <MenuToggleRow
                          key={item.id}
                          title={item.title}
                          isOn={visibleItems.includes(item.id)}
                          onSelect={() => toggleLayoutItem(item.id)}
                        />
                      ))}
                    </div>
                  ))}
                </MobileMenu>
              )}

              <button
                type="button"
                onClick={() => onShowClosedChange(!showClosed)}
                aria-label={showClosed ? 'Hide closed positions' : 'Show closed positions'}
                aria-pressed={showClosed}
                className={cn(CONTROL, showClosed ? 'text-foreground' : 'text-muted-foreground')}
              >
                {showClosed
                  ? <Eye className="h-[18px] w-[18px]" />
                  : <EyeOff className="h-[18px] w-[18px]" />}
              </button>
            </div>

            {/* Market status: a coloured dot carries open/closed without the
                word, then the time the data was last refreshed. */}
            {isMarketOpen !== undefined && (
              <div
                className="flex shrink-0 items-center gap-1"
                aria-label={isMarketOpen ? 'Market open' : 'Market closed'}
              >
                <span className={cn(
                  'h-[7px] w-[7px] rounded-full',
                  isMarketOpen ? 'animate-pulse bg-up' : 'bg-muted-foreground/60',
                )} />
                {lastUpdated && (
                  <span className="text-[11px] font-medium tabular-nums text-muted-foreground">
                    {lastUpdated.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}
                  </span>
                )}
              </div>
            )}

            {isRefreshing ? (
              <span className={CONTROL}><Loader2 className="h-4 w-4 animate-spin text-muted-foreground" /></span>
            ) : (
              <button type="button" onClick={onRefresh} aria-label="Refresh" className={CONTROL}>
                <RefreshCw className="h-[18px] w-[18px]" />
              </button>
            )}
          </>
        )}

        {/* One instance across the active/inactive switch, so focus and typed
            text survive when the sibling controls show and hide. */}
        <div className={cn('flex min-w-0', searchActive ? 'flex-1 pl-3' : 'shrink-0')}>
          <StockSearchBar
            currency={currency}
            collapsible
            fillExpanded
            onActiveChange={setSearchActive}
          />
        </div>

        {!searchActive && (
          <>
            <MobileMenu
              ariaLabel="Display currency"
              minWidth={130}
              align="right"
              className={cn(CONTROL, 'w-auto gap-1 px-1.5 text-sm font-semibold')}
              label={<><span>{currency}</span><ChevronsUpDown className="h-3.5 w-3.5 text-muted-foreground" /></>}
            >
              {currencies.map(c => (
                <MenuToggleRow
                  key={c}
                  title={c}
                  isOn={c === currency}
                  dismissOnTap
                  onSelect={() => onCurrencyChange(c)}
                />
              ))}
            </MobileMenu>

            {/* Settings sits at the right edge as itself, where a "•••" menu
                used to hold it alongside sign-out. Signing out stays a tap
                further in, on the Settings hub. */}
            <button type="button" onClick={onOpenSettings} aria-label="Settings" className={CONTROL}>
              <SettingsIcon className="h-[18px] w-[18px]" />
            </button>
          </>
        )}
      </div>
    </div>
  );
}
