'use client';

import { useState, useEffect, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { useQuery, useMutation, useQueryClient, keepPreviousData } from '@tanstack/react-query';
import {
  fetchSummary,
  fetchHoldings,
  fetchTransactions,
  fetchAssetChange,
  fetchCapitalGains,
  fetchDividends,
  fetchRiskMetrics,
  fetchAttribution,
  fetchDividendCalendar,
  fetchHistory,
  fetchWatchlist,
  fetchSettings,
  updateSettings,
  SettingsUpdate,
  fetchPortfolioHealth,
  fetchProjectedIncome,
  fetchMarketStatus,
  PerformanceData
} from '@/lib/api';
import { cn } from '@/lib/utils';
import { INITIAL_VISIBLE_ITEMS, TAB_THEMES } from '@/lib/dashboard_constants';
import Dashboard from '@/components/Dashboard';
import HoldingsTable from '@/components/HoldingsTable';
import { EmptyState } from '@/components/EmptyState';
import AppShellSkeleton from '@/components/skeletons/AppShellSkeleton';
import { Sidebar } from '@/components/layout/Sidebar';
import { PageHeader } from '@/components/layout/PageHeader';
import { MobileNav } from '@/components/layout/MobileNav';
import dynamic from 'next/dynamic';
import { useTheme } from 'next-themes';
import { Home as HomeIcon, Activity, Settings as SettingsIcon, Moon, Sun } from 'lucide-react';

const PerformanceGraph = dynamic(() => import('@/components/PerformanceGraph'), {
  loading: () => <div className="h-[400px] bg-card border border-border/50 rounded-2xl mb-6 animate-pulse" />,
  ssr: false,
});
const TransactionsTable    = dynamic(() => import('@/components/TransactionsTable'));
const Allocation           = dynamic(() => import('@/components/Allocation'));
const AssetChange          = dynamic(() => import('@/components/AssetChange'));
const CapitalGains         = dynamic(() => import('@/components/CapitalGains'));
const UnrealizedTaxView    = dynamic(() => import('@/components/UnrealizedTaxView'));
const DividendComponent    = dynamic(() => import('@/components/Dividend'));
const DividendCalendar     = dynamic(() => import('@/components/DividendCalendar'));
const IncomeProjector      = dynamic(() => import('@/components/IncomeProjector').then(mod => mod.IncomeProjector));
const Settings             = dynamic(() => import('@/components/Settings'));
const CommandPalette       = dynamic(() => import('@/components/CommandPalette'));
const Watchlist            = dynamic(() => import('@/components/Watchlist'));
const ScreenerView         = dynamic(() => import('@/components/ScreenerView'));
const PortfolioAIReview    = dynamic(() => import('@/components/PortfolioAIReview'));
const IndexGraphModal      = dynamic(() => import('@/components/IndexGraphModal'), { ssr: false });
const MarketsTab           = dynamic(() => import('@/components/MarketsTab'), { ssr: false });
const RiskMetrics          = dynamic(() => import('@/components/RiskMetrics'), { ssr: false });

export default function Home() {
  const { theme, setTheme } = useTheme();
  const { user, isLoading: authLoading, logout } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!authLoading && !user) {
      if (typeof window !== 'undefined' && window.location.protocol === 'file:') {
        window.location.href = 'login.html';
      } else {
        router.push('/login');
      }
    }
  }, [user, authLoading, router]);

  const [selectedAccounts, setSelectedAccounts]     = useState<string[]>([]);
  const [currency, setCurrency]                     = useState('USD');
  const [activeTab, setActiveTab]                   = useState('performance');
  const [showClosed, setShowClosed]                 = useState(false);
  const [backgroundFetchLevel, setBackgroundFetchLevel] = useState(0);
  const [mounted, setMounted]                       = useState(false);
  const [settingsInitialTab, setSettingsInitialTab] = useState<'overrides' | 'account' | undefined>(undefined);
  const [sidebarCollapsed, setSidebarCollapsed]     = useState(false);
  const [isIndexGraphModalOpen, setIsIndexGraphModalOpen] = useState(false);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen]   = useState(false);
  const [isMobileNavOpen, setIsMobileNavOpen]             = useState(false);
  const [benchmarks, setBenchmarks]                 = useState<string[]>(['S&P 500', 'Dow Jones', 'NASDAQ']);
  const [graphPeriod, setGraphPeriod]               = useState('1y');
  const [graphView, setGraphView]                   = useState<'return' | 'value' | 'drawdown'>('return');
  const [graphCustomFromDate, setGraphCustomFromDate] = useState(() => {
    const d = new Date(); d.setFullYear(d.getFullYear() - 1); return d.toISOString().split('T')[0];
  });
  const [graphCustomToDate, setGraphCustomToDate]   = useState(() => new Date().toISOString().split('T')[0]);
  const [capitalGainsDates, setCapitalGainsDates]   = useState<{ from?: string; to?: string }>({});
  const [visibleItems, setVisibleItems]             = useState<string[]>(INITIAL_VISIBLE_ITEMS);

  const handleUserIconClick = () => { setSettingsInitialTab('account'); setActiveTab('settings'); };
  const handleTabChange = (tab: string) => {
    if (tab === 'settings') setSettingsInitialTab(undefined);
    setActiveTab(tab);
  };

  // Hydrate all state from localStorage in one effect
  useEffect(() => {
    try {
      const savedAccounts      = localStorage.getItem('investa_selected_accounts');
      const savedCurrency      = localStorage.getItem('investa_currency');
      const savedTab           = localStorage.getItem('investa_active_tab');
      const savedShowClosed    = localStorage.getItem('investa_show_closed');
      const savedBenchmarks    = localStorage.getItem('investa_graph_benchmarks');
      const savedVisibleItems  = localStorage.getItem('investa_dashboard_visible_items');
      const savedGraphPeriod   = localStorage.getItem('investa_graph_period');
      const savedGraphView     = localStorage.getItem('investa_graph_view');
      const savedSidebarState  = localStorage.getItem('investa_sidebar_collapsed');

      if (savedAccounts)     setSelectedAccounts(JSON.parse(savedAccounts));
      if (savedCurrency)     setCurrency(savedCurrency);
      if (savedTab)          setActiveTab(savedTab);
      if (savedShowClosed)   setShowClosed(savedShowClosed === 'true');
      if (savedBenchmarks)   setBenchmarks(JSON.parse(savedBenchmarks));
      if (savedVisibleItems) { const p = JSON.parse(savedVisibleItems); if (p.length > 0) setVisibleItems(p); }
      if (savedGraphPeriod)  setGraphPeriod(savedGraphPeriod);
      if (savedGraphView && ['return', 'value', 'drawdown'].includes(savedGraphView)) {
        setGraphView(savedGraphView as 'return' | 'value' | 'drawdown');
      }
      if (savedSidebarState !== null) setSidebarCollapsed(savedSidebarState === 'true');
    } catch (e) {
      console.error('Failed to hydrate state from localStorage', e);
    } finally {
      setMounted(true);
    }
  }, []);

  useEffect(() => {
    if (!mounted) return;
    const t1 = setTimeout(() => setBackgroundFetchLevel(1), 3000);
    const t2 = setTimeout(() => setBackgroundFetchLevel(2), 8000);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [mounted]);

  // Persist sidebar state
  useEffect(() => {
    if (mounted) localStorage.setItem('investa_sidebar_collapsed', String(sidebarCollapsed));
  }, [sidebarCollapsed, mounted]);

  const queryClient = useQueryClient();

  const settingsQuery = useQuery({
    queryKey: ['settings', user?.username],
    queryFn: fetchSettings,
    staleTime: 5 * 60 * 1000,
    enabled: !!user,
  });

  const settingsMutation = useMutation({
    mutationFn: updateSettings,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['settings', user?.username] }),
  });

  // Persist + sync settings to server
  useEffect(() => {
    localStorage.setItem('investa_currency',                       currency);
    localStorage.setItem('investa_active_tab',                     activeTab);
    localStorage.setItem('investa_show_closed',                    showClosed.toString());
    localStorage.setItem('investa_selected_accounts',              JSON.stringify(selectedAccounts));
    localStorage.setItem('investa_graph_period',                   graphPeriod);
    localStorage.setItem('investa_graph_view',                     graphView);
    if (visibleItems.length > 0) localStorage.setItem('investa_dashboard_visible_items', JSON.stringify(visibleItems));
    if (benchmarks.length > 0)   localStorage.setItem('investa_graph_benchmarks',        JSON.stringify(benchmarks));

    if (!settingsQuery.data) return;
    const id = setTimeout(() => {
      const updates: Partial<SettingsUpdate> = {};
      const s = settingsQuery.data;
      if (s.display_currency !== currency)        updates.display_currency    = currency;
      if (s.active_tab !== activeTab)             updates.active_tab          = activeTab;
      if (s.show_closed !== showClosed)           updates.show_closed         = showClosed;
      const eq = (a: string[] | undefined | null, b: string[]) =>
        !!a && a.length === b.length && a.every((v, i) => v === b[i]);
      if (!eq(s.selected_accounts, selectedAccounts))               updates.selected_accounts = selectedAccounts;
      if (!eq(s.visible_items, visibleItems) && visibleItems.length > 0) updates.visible_items = visibleItems;
      if (!eq(s.benchmarks, benchmarks) && benchmarks.length > 0)  updates.benchmarks = benchmarks;
      if (Object.keys(updates).length > 0) settingsMutation.mutate(updates);
    }, 1000);
    return () => clearTimeout(id);
  }, [currency, activeTab, showClosed, benchmarks, selectedAccounts, visibleItems, settingsQuery.data]);

  // Command palette keyboard shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); setIsCommandPaletteOpen(p => !p); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const handleNavigate = (tab: string) => { setActiveTab(tab); window.scrollTo({ top: 0, behavior: 'smooth' }); };

  // ── Queries ──────────────────────────────────────────────────────────────
  const marketStatusQuery = useQuery({
    queryKey: ['marketStatus'],
    queryFn: fetchMarketStatus,
    staleTime: 60 * 1000,
    refetchInterval: 5 * 60 * 1000,
    enabled: !!user,
  });
  const isMarketOpen = marketStatusQuery.data?.is_open ?? false;

  const summaryQuery = useQuery({
    queryKey: ['summary', user?.username, currency, selectedAccounts, showClosed],
    queryFn: ({ signal }) => fetchSummary(currency, selectedAccounts, showClosed, signal),
    staleTime: 5 * 60 * 1000,
    refetchInterval: isMarketOpen ? 60 * 1000 : false,
    placeholderData: keepPreviousData,
    enabled: !!user,
  });

  const holdingsQuery = useQuery({
    queryKey: ['holdings', user?.username, currency, selectedAccounts, showClosed],
    queryFn: ({ signal }) => fetchHoldings(currency, selectedAccounts, showClosed, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: !!user,
  });

  const transactionsQuery = useQuery({
    queryKey: ['transactions', user?.username, selectedAccounts],
    queryFn: ({ signal }) => fetchTransactions(selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: !!user && (activeTab === 'transactions' || backgroundFetchLevel >= 2),
  });

  const assetChangeQuery = useQuery({
    queryKey: ['assetChange', user?.username, currency, selectedAccounts, benchmarks, showClosed],
    queryFn: ({ signal }) => fetchAssetChange(currency, selectedAccounts, benchmarks, showClosed, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: !!user && (activeTab === 'asset_change' || backgroundFetchLevel >= 2),
  });

  const capitalGainsQuery = useQuery({
    queryKey: ['capitalGains', user?.username, currency, selectedAccounts, capitalGainsDates.from, capitalGainsDates.to],
    queryFn: ({ signal }) => fetchCapitalGains(currency, selectedAccounts, capitalGainsDates.from, capitalGainsDates.to, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: !!user && (activeTab === 'capital_gains' || backgroundFetchLevel >= 2),
  });

  const dividendsQuery = useQuery({
    queryKey: ['dividends', user?.username, currency, selectedAccounts],
    queryFn: ({ signal }) => fetchDividends(currency, selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: !!user && (activeTab === 'dividend' || backgroundFetchLevel >= 2),
  });

  const riskMetricsQuery = useQuery({
    queryKey: ['riskMetrics', user?.username, currency, selectedAccounts, showClosed],
    queryFn: ({ signal }) => fetchRiskMetrics(currency, selectedAccounts, showClosed, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: !!user && (activeTab === 'performance' || backgroundFetchLevel >= 1),
  });

  const attributionQuery = useQuery({
    queryKey: ['attribution', user?.username, currency, selectedAccounts, showClosed],
    queryFn: ({ signal }) => fetchAttribution(currency, selectedAccounts, false, showClosed, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: !!user && (activeTab === 'performance' || backgroundFetchLevel >= 1),
  });

  const dividendCalendarQuery = useQuery({
    queryKey: ['dividendCalendar', user?.username, selectedAccounts],
    queryFn: ({ signal }) => fetchDividendCalendar(selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: !!user && (activeTab === 'dividend' || backgroundFetchLevel >= 2),
  });

  const historySparklineQuery = useQuery({
    queryKey: ['history', user?.username, currency, selectedAccounts, 'sparkline'],
    queryFn: ({ signal }) => fetchHistory(currency, selectedAccounts, '1d', [], '5m', undefined, undefined, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: !!user && backgroundFetchLevel >= 1,
  });

  const graphInterval = useMemo(() => {
    if (graphPeriod === '1d') return '2m';
    if (graphPeriod === '5d') return '15m';
    if (graphPeriod === '1m') return '1d';
    return '1d';
  }, [graphPeriod]);

  const graphFromDate = graphPeriod === 'custom' ? graphCustomFromDate : undefined;
  const graphToDate   = graphPeriod === 'custom' ? graphCustomToDate   : undefined;

  const historyQuery = useQuery({
    queryKey: ['history', user?.username, currency, selectedAccounts, graphPeriod, benchmarks, graphInterval, graphFromDate, graphToDate],
    queryFn: ({ signal }) => fetchHistory(currency, selectedAccounts, graphPeriod, benchmarks, graphInterval, graphFromDate, graphToDate, signal),
    placeholderData: keepPreviousData,
    staleTime: 5 * 60 * 1000,
    refetchInterval: isMarketOpen && (graphPeriod === '1d' || graphPeriod === '5d') ? 60000 : false,
    enabled: !!user && (activeTab === 'performance' || backgroundFetchLevel >= 1),
  });

  const watchlistQuery = useQuery({
    queryKey: ['watchlist', user?.username, currency, 1],
    queryFn: ({ signal }) => fetchWatchlist(currency, 1, signal),
    staleTime: 1 * 60 * 1000,
    enabled: !!user && (activeTab === 'watchlist' || activeTab === 'markets' || backgroundFetchLevel >= 2),
  });

  const portfolioHealthQuery = useQuery({
    queryKey: ['portfolioHealth', user?.username, currency, selectedAccounts, showClosed],
    queryFn: ({ signal }) => fetchPortfolioHealth(currency, selectedAccounts, showClosed, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: !!user && (activeTab === 'performance' || backgroundFetchLevel >= 1),
  });

  const incomeProjectionQuery = useQuery({
    queryKey: ['incomeProjection', user?.username, currency, selectedAccounts],
    queryFn: ({ signal }) => fetchProjectedIncome(currency, selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: !!user && (activeTab === 'dividend' || backgroundFetchLevel >= 2),
  });

  // ── Derived data ──────────────────────────────────────────────────────────
  const summary          = summaryQuery.data;
  const holdings         = holdingsQuery.data || [];
  const transactions     = transactionsQuery.data || [];
  const assetChangeData  = assetChangeQuery.data || null;
  const capitalGainsData = capitalGainsQuery.data || null;
  const dividendData     = dividendsQuery.data || null;
  const availableAccounts = (summary?.metrics?._available_accounts as string[]) || [];
  const graphData        = historyQuery.data || [];
  const graphLoading     = historyQuery.isFetching;

  // ── Tab content ───────────────────────────────────────────────────────────
  const renderTabContent = () => {
    switch (activeTab) {
      case 'performance':
        if (!summaryQuery.isLoading && !summaryQuery.data && summaryQuery.isFetched) {
          return <EmptyState onNavigate={handleNavigate} />;
        }
        return (
          <>
            <Dashboard
              summary={summary || { metrics: null, account_metrics: null }}
              currency={currency}
              history={historySparklineQuery.data || []}
              isLoading={summaryQuery.isLoading && !summaryQuery.data}
              isRefreshing={summaryQuery.isFetching || historySparklineQuery.isFetching}
              riskMetrics={riskMetricsQuery.data || {}}
              riskMetricsLoading={riskMetricsQuery.isLoading && !riskMetricsQuery.data}
              portfolioHealth={portfolioHealthQuery.data || null}
              attributionData={attributionQuery.data}
              attributionLoading={attributionQuery.isLoading && !attributionQuery.data}
              holdings={holdings}
              visibleItems={visibleItems}
              accounts={selectedAccounts}
              themeColor={currentTheme.color}
              showClosed={showClosed}
              // Risk metrics rendered separately below the performance graph.
              excludeFromAnalytics={['riskMetrics']}
            />
            <PerformanceGraph
              currency={currency}
              accounts={selectedAccounts}
              benchmarks={benchmarks}
              onBenchmarksChange={setBenchmarks}
              period={graphPeriod}
              onPeriodChange={setGraphPeriod}
              view={graphView}
              onViewChange={setGraphView}
              data={graphData}
              loading={graphLoading}
              customFromDate={graphCustomFromDate}
              onCustomFromDateChange={setGraphCustomFromDate}
              customToDate={graphCustomToDate}
              onCustomToDateChange={setGraphCustomToDate}
            />
            {visibleItems.includes('riskMetrics') && (
              <RiskMetrics
                metrics={riskMetricsQuery.data || {}}
                portfolioHealth={portfolioHealthQuery.data || null}
                isLoading={riskMetricsQuery.isLoading && !riskMetricsQuery.data}
                isRefreshing={riskMetricsQuery.isFetching}
              />
            )}
          </>
        );

      case 'watchlist':
        return <Watchlist currency={currency} />;

      case 'screener':
        return null;

      case 'ai_review':
        return <PortfolioAIReview currency={currency} accounts={selectedAccounts} />;

      case 'transactions':
        return <TransactionsTable transactions={transactions} isLoading={transactionsQuery.isPending && !transactionsQuery.data} />;

      case 'markets':
        return !summary?.metrics?.indices ? (
          <p className="text-muted-foreground text-sm">Market data unavailable.</p>
        ) : (
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          <MarketsTab
            indices={summary.metrics.indices as any}
            onIndexClick={() => setIsIndexGraphModalOpen(true)}
            portfolioSymbols={holdings.map(h => h.Symbol).filter(Boolean)}
            watchlistSymbols={(watchlistQuery.data || []).map((w: any) => w.Symbol).filter(Boolean)}
          />
        );

      case 'allocation':
        if (!holdingsQuery.isLoading && holdings.length === 0 && holdingsQuery.isFetched) {
          return <EmptyState onNavigate={handleNavigate} />;
        }
        return (
          <div className="space-y-6">
            <HoldingsTable
              holdings={holdings}
              currency={currency}
              isLoading={holdingsQuery.isLoading && !holdingsQuery.data}
              showClosed={showClosed}
              onToggleShowClosed={setShowClosed}
            />
            <Allocation holdings={holdings} currency={currency} />
          </div>
        );

      case 'asset_change':
        return <AssetChange data={assetChangeData} currency={currency} isLoading={assetChangeQuery.isPending && !assetChangeQuery.data} />;

      case 'capital_gains':
        return (
          <div className="space-y-6 p-4">
            <UnrealizedTaxView holdings={holdings} currency={currency} />
            <CapitalGains
              data={capitalGainsData}
              currency={currency}
              onDateRangeChange={(from, to) => setCapitalGainsDates({ from, to })}
              isLoading={capitalGainsQuery.isPending && !capitalGainsQuery.data}
            />
          </div>
        );

      case 'dividend':
        return (
          <div className="space-y-6">
            <DividendComponent
              data={dividendData}
              currency={currency}
              expectedDividends={summary?.metrics?.est_annual_income_display as number}
              dividendYield={summary?.metrics?.dividend_yield_pct as number}
              isLoading={dividendsQuery.isPending && !dividendsQuery.data}
            >
              <IncomeProjector
                data={incomeProjectionQuery.data || null}
                isLoading={incomeProjectionQuery.isLoading && !incomeProjectionQuery.data}
                currency={currency}
              />
              <DividendCalendar
                events={dividendCalendarQuery.data || []}
                isLoading={dividendCalendarQuery.isLoading && !dividendCalendarQuery.data}
                currency={currency}
              />
            </DividendComponent>
          </div>
        );

      case 'settings':
        return (
          <Settings
            settings={settingsQuery.data || null}
            holdings={holdings}
            availableAccounts={availableAccounts}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            initialTab={settingsInitialTab as any}
          />
        );

      default:
        return (
          <div className="p-8 text-center text-muted-foreground">
            <p className="text-lg font-medium">Coming Soon</p>
            <p className="text-sm mt-2">The {activeTab} tab is under construction.</p>
          </div>
        );
    }
  };

  const { resolvedTheme } = useTheme();

  if (!mounted || authLoading || !user) return <AppShellSkeleton />;

  const currentTheme = TAB_THEMES[activeTab] || TAB_THEMES.performance;

  return (
    <div className="flex h-screen overflow-hidden bg-background selection:bg-indigo-500/20 selection:text-indigo-500">

      {/* Ambient background glows */}
      <div className="fixed inset-0 z-[-1] pointer-events-none overflow-hidden">
        <div className={cn(
          'absolute -top-[30%] -left-[15%] w-[70%] h-[70%] rounded-full blur-[120px] transition-all duration-[1500ms] animate-pulse-glow opacity-20',
          currentTheme.bgGlow,
        )} />
        <div className={cn(
          'absolute top-[20%] -right-[25%] w-[60%] h-[60%] rounded-full blur-[100px] transition-all duration-[1500ms] opacity-10',
          currentTheme.bgGlow,
        )} />
      </div>
      <div className="fixed inset-0 z-[-2] bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-indigo-900/10 via-background to-background pointer-events-none" />

      {/* ── Sidebar (desktop) ── */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={handleTabChange}
        user={user}
        onLogout={logout}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(c => !c)}
        onUserClick={handleUserIconClick}
        dayChangePct={summary?.metrics?.day_change_pct as number | undefined}
      />

      {/* ── Mobile navigation drawer ── */}
      <MobileNav
        isOpen={isMobileNavOpen}
        onClose={() => setIsMobileNavOpen(false)}
        activeTab={activeTab}
        onTabChange={handleTabChange}
        user={user}
        onLogout={logout}
        onUserClick={handleUserIconClick}
        currency={currency}
      />

      {/* ── Main content ── */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">

        <PageHeader
          activeTab={activeTab}
          currency={currency}
          onCurrencyChange={setCurrency}
          availableAccounts={availableAccounts}
          selectedAccounts={selectedAccounts}
          onAccountsChange={setSelectedAccounts}
          accountGroups={settingsQuery.data?.account_groups}
          indices={summary?.metrics?.indices as Record<string, unknown> | undefined}
          visibleItems={visibleItems}
          onVisibleItemsChange={setVisibleItems}
          onCommandPaletteOpen={() => setIsCommandPaletteOpen(true)}
          fxRate={summary?.metrics?.exchange_rate_to_display as number | undefined}
          availableCurrencies={settingsQuery.data?.available_currencies}
          isFetching={summaryQuery.isFetching}
          onIndexClick={() => setIsIndexGraphModalOpen(true)}
          isMarketOpen={isMarketOpen}
          lastUpdated={summaryQuery.dataUpdatedAt ? new Date(summaryQuery.dataUpdatedAt) : null}
          onMobileMenuOpen={() => setIsMobileNavOpen(true)}
          marketValue={summary?.metrics?.market_value ?? null}
          dayChangePct={summary?.metrics?.day_change_percent ?? null}
        />

        {/* Scrollable content area */}
        <main className="flex-1 overflow-y-auto pb-20 md:pb-8">
          <div className="max-w-[1440px] mx-auto px-4 sm:px-6 py-5 sm:py-6">
            {renderTabContent()}
            <div className={activeTab === 'screener' ? 'block' : 'hidden'}>
              <ScreenerView currency={currency} />
            </div>
          </div>
        </main>
      </div>

      {/* ── Modals ── */}
      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
        onNavigate={handleNavigate}
        currency={currency}
      />
      <IndexGraphModal
        isOpen={isIndexGraphModalOpen}
        onClose={() => setIsIndexGraphModalOpen(false)}
        benchmarks={benchmarks}
        currentIndices={summary?.metrics?.indices}
      />

      {/* ── Mobile bottom nav ── */}
      <div
        className={cn("fixed bottom-0 left-0 right-0 border-t border-border px-4 py-3 flex justify-between items-center text-[10px] font-bold uppercase tracking-widest md:hidden z-50 transition-all duration-300", isMobileNavOpen && "hidden")}
        style={{ backgroundColor: 'var(--menu-solid)' }}
      >
        <div
          onClick={() => { setActiveTab('performance'); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
          className={cn(
            'flex flex-col items-center flex-1 cursor-pointer transition-colors',
            activeTab !== 'settings' && activeTab !== 'markets' && activeTab !== 'screener'
              ? 'text-indigo-600 dark:text-indigo-400'
              : 'text-slate-500 hover:text-indigo-600 dark:hover:text-indigo-400',
          )}
        >
          <HomeIcon className="w-5 h-5" /><span className="mt-1">Home</span>
        </div>
        <div
          onClick={() => { setActiveTab('markets'); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
          className={cn(
            'flex flex-col items-center flex-1 cursor-pointer transition-colors',
            activeTab === 'markets' ? 'text-indigo-600 dark:text-indigo-400' : 'text-slate-500 hover:text-indigo-600 dark:hover:text-indigo-400',
          )}
        >
          <Activity className="w-5 h-5" /><span className="mt-1">Indices</span>
        </div>
        <div
          onClick={() => { setActiveTab('settings'); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
          className={cn(
            'flex flex-col items-center flex-1 cursor-pointer transition-colors',
            activeTab === 'settings' ? 'text-indigo-600 dark:text-indigo-400' : 'text-slate-500 hover:text-indigo-600 dark:hover:text-indigo-400',
          )}
        >
          <SettingsIcon className="w-5 h-5" /><span className="mt-1">Settings</span>
        </div>
        <div
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          className="flex flex-col items-center flex-1 cursor-pointer transition-colors text-slate-500 hover:text-indigo-600 dark:hover:text-indigo-400"
        >
          {mounted && resolvedTheme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          <span className="mt-1">{mounted && resolvedTheme === 'dark' ? 'Light' : 'Dark'}</span>
        </div>
      </div>
    </div>
  );
}
