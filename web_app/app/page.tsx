'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { useQuery, keepPreviousData } from '@tanstack/react-query';
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
  fetchPortfolioHealth,
  fetchCorrelationMatrix,
  fetchProjectedIncome,
  PerformanceData
} from '@/lib/api';
import { cn } from '@/lib/utils';
import { AreaChart, Area, YAxis, ResponsiveContainer } from 'recharts';
import { DEFAULT_ITEMS } from '@/lib/dashboard_constants';
// import { CURRENCY_SYMBOLS } from '@/lib/utils';
import Dashboard from '@/components/Dashboard';
import HoldingsTable from '@/components/HoldingsTable';
import AccountSelector from '@/components/AccountSelector';
import CurrencySelector from '@/components/CurrencySelector';
import TabNavigation from '@/components/TabNavigation';
import dynamic from 'next/dynamic';

const PerformanceGraph = dynamic(() => import('@/components/PerformanceGraph'), {
  loading: () => <div className="h-[400px] bg-card rounded-xl border border-border mb-6 animate-pulse" />,
  ssr: false
});

const TransactionsTable = dynamic(() => import('@/components/TransactionsTable'));
const Allocation = dynamic(() => import('@/components/Allocation'));
const AssetChange = dynamic(() => import('@/components/AssetChange'));
const CapitalGains = dynamic(() => import('@/components/CapitalGains'));
const DividendComponent = dynamic(() => import('@/components/Dividend'));
const DividendCalendar = dynamic(() => import('@/components/DividendCalendar'));
const IncomeProjector = dynamic(() => import('@/components/IncomeProjector').then(mod => mod.IncomeProjector));
const Settings = dynamic(() => import('@/components/Settings'));
const CommandPalette = dynamic(() => import('@/components/CommandPalette'));
const CorrelationMatrix = dynamic(() => import('@/components/CorrelationMatrix').then(mod => mod.CorrelationMatrix));
const PortfolioHealthComponent = dynamic(() => import('@/components/PortfolioHealth').then(mod => mod.PortfolioHealthComponent));
const Watchlist = dynamic(() => import('@/components/Watchlist'));
const ScreenerView = dynamic(() => import('@/components/ScreenerView'));


import { useTheme } from 'next-themes';
import { Home as HomeIcon, BarChart3, Settings as SettingsIcon, Moon, Sun, LogOut } from 'lucide-react';
const LayoutConfigurator = dynamic(() => import('@/components/LayoutConfigurator'));

// Static import for ThemeToggle since it's in the sidebar always visible? OR lazy load it? 
// It's small, let's lazy load if it's not critical. But wait, I removed it from the chunk above.
// I need to add ThemeToggle back. Step 8 showed it was imported.
// I will keep ThemeToggle as dynamic too.
const ThemeToggle = dynamic(() => import('@/components/ThemeToggle'));


export default function Home() {
  const { theme, setTheme } = useTheme();
  const { user, isLoading: authLoading, logout } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

  const [selectedAccounts, setSelectedAccounts] = useState<string[]>([]);
  const [currency, setCurrency] = useState('USD');
  const [activeTab, setActiveTab] = useState('performance');
  const [backgroundFetchEnabled, setBackgroundFetchEnabled] = useState(false);

  // Trigger background fetch after initial load
  useEffect(() => {
    // Start background fetch immediately after mount to speed up data availability
    // while still allowing the main UI to paint first.
    // We use a small timeout to let the event loop clear the render task.
    const timer = setTimeout(() => {
      setBackgroundFetchEnabled(true);
    }, 100);
    return () => clearTimeout(timer);
  }, []);


  // Lazy init benchmarks from localStorage
  // Initialize with defaults to match server-side rendering, avoiding hydration mismatch.
  const [benchmarks, setBenchmarks] = useState<string[]>(['S&P 500', 'Dow Jones', 'NASDAQ']);

  // Load benchmarks from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem('investa_graph_benchmarks');
      if (saved) {
        setBenchmarks(JSON.parse(saved));
      }
    } catch (e) {
      console.error("Failed to load benchmarks", e);
    }
  }, []);

  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);

  // Lazy init showClosed
  const [showClosed, setShowClosed] = useState(() => {
    if (typeof window === 'undefined') return false;
    return localStorage.getItem('investa_show_closed') === 'true';
  });

  const [capitalGainsDates, setCapitalGainsDates] = useState<{ from?: string, to?: string }>({});
  const [correlationPeriod, setCorrelationPeriod] = useState('1y');

  // Lazy init visibleItems
  // Initialize with defaults to match server-side rendering, avoiding hydration mismatch.
  // We update from localStorage in useEffect.
  const [visibleItems, setVisibleItems] = useState<string[]>(DEFAULT_ITEMS.map(i => i.id));

  // Load visibleItems from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem('investa_dashboard_visible_items');
      if (saved) {
        setVisibleItems(JSON.parse(saved));
      }
    } catch (e) {
      console.error("Failed to load visible items", e);
    }
  }, []);

  // Persist visibility to localStorage
  useEffect(() => {
    if (visibleItems.length > 0) {
      localStorage.setItem('investa_dashboard_visible_items', JSON.stringify(visibleItems));
    }
  }, [visibleItems]);


  // Persist showClosed to localStorage
  useEffect(() => {
    localStorage.setItem('investa_show_closed', showClosed.toString());
  }, [showClosed]);

  // Persist benchmarks to localStorage
  useEffect(() => {
    localStorage.setItem('investa_graph_benchmarks', JSON.stringify(benchmarks));
  }, [benchmarks]);

  // Command Palette Keyboard Listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsCommandPaletteOpen(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Handler for navigation from Command Palette
  const handleNavigate = (tab: string) => {
    setActiveTab(tab);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Queries
  const summaryQuery = useQuery({
    queryKey: ['summary', currency, selectedAccounts],
    queryFn: ({ signal }) => fetchSummary(currency, selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
  });

  const holdingsQuery = useQuery({
    queryKey: ['holdings', currency, selectedAccounts, showClosed],
    queryFn: ({ signal }) => fetchHoldings(currency, selectedAccounts, showClosed, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
  });

  // --- PRIORITIZATION LOGIC ---
  // Ensure Summary and Holdings fetch FIRST before triggering others
  // We consider them "loaded" if they have data or successfully fetched.
  // Using isFetched or data check.
  const isHighPriorityLoaded = summaryQuery.isSuccess && holdingsQuery.isSuccess;


  const transactionsQuery = useQuery({
    queryKey: ['transactions', selectedAccounts],
    queryFn: ({ signal }) => fetchTransactions(selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: (activeTab === 'transactions' || backgroundFetchEnabled) && isHighPriorityLoaded,
  });

  const assetChangeQuery = useQuery({
    queryKey: ['assetChange', currency, selectedAccounts, benchmarks],
    queryFn: ({ signal }) => fetchAssetChange(currency, selectedAccounts, benchmarks, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: (activeTab === 'asset_change' || backgroundFetchEnabled) && isHighPriorityLoaded,
  });

  const capitalGainsQuery = useQuery({
    queryKey: ['capitalGains', currency, selectedAccounts, capitalGainsDates.from, capitalGainsDates.to],
    queryFn: ({ signal }) => fetchCapitalGains(currency, selectedAccounts, capitalGainsDates.from, capitalGainsDates.to, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: (activeTab === 'capital_gains' || backgroundFetchEnabled) && isHighPriorityLoaded,
  });

  const dividendsQuery = useQuery({
    queryKey: ['dividends', currency, selectedAccounts],
    queryFn: ({ signal }) => fetchDividends(currency, selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: (activeTab === 'dividend' || backgroundFetchEnabled) && isHighPriorityLoaded,
  });

  const riskMetricsQuery = useQuery({
    queryKey: ['riskMetrics', currency, selectedAccounts],
    queryFn: ({ signal }) => fetchRiskMetrics(currency, selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: isHighPriorityLoaded,
  });

  const attributionQuery = useQuery({
    queryKey: ['attribution', currency, selectedAccounts],
    queryFn: ({ signal }) => fetchAttribution(currency, selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: isHighPriorityLoaded,
  });

  const dividendCalendarQuery = useQuery({
    queryKey: ['dividendCalendar', selectedAccounts],
    queryFn: ({ signal }) => fetchDividendCalendar(selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: (activeTab === 'dividend' || backgroundFetchEnabled) && isHighPriorityLoaded,
  });

  const historySparklineQuery = useQuery({
    queryKey: ['history', currency, selectedAccounts, 'sparkline'],
    queryFn: ({ signal }) => fetchHistory(currency, selectedAccounts, '1d', [], '5m', undefined, undefined, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    // enabled: isHighPriorityLoaded, // Allow history to fetch in parallel with summary
  });

  const watchlistQuery = useQuery({
    queryKey: ['watchlist', currency, 1],
    queryFn: ({ signal }) => fetchWatchlist(currency, 1, signal),
    staleTime: 1 * 60 * 1000,
    // No keepPreviousData needed for watchlist as it's not dependent on accounts
    enabled: (activeTab === 'watchlist' || backgroundFetchEnabled) && isHighPriorityLoaded,
  });

  const settingsQuery = useQuery({
    queryKey: ['settings'],
    queryFn: fetchSettings,
    staleTime: 5 * 60 * 1000,
  });

  const portfolioHealthQuery = useQuery({
    queryKey: ['portfolioHealth', currency, selectedAccounts],
    queryFn: ({ signal }) => fetchPortfolioHealth(currency, selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: (activeTab === 'analytics' || backgroundFetchEnabled) && isHighPriorityLoaded,
  });

  const correlationMatrixQuery = useQuery({
    queryKey: ['correlationMatrix', correlationPeriod, selectedAccounts],
    queryFn: ({ signal }) => fetchCorrelationMatrix(correlationPeriod, selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: (activeTab === 'analytics' || backgroundFetchEnabled) && isHighPriorityLoaded,
  });

  const incomeProjectionQuery = useQuery({
    queryKey: ['incomeProjection', currency, selectedAccounts],
    queryFn: ({ signal }) => fetchProjectedIncome(currency, selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: (activeTab === 'dividend' || backgroundFetchEnabled) && isHighPriorityLoaded,
  });

  const summary = summaryQuery.data;
  const holdings = holdingsQuery.data || [];
  const transactions = transactionsQuery.data || [];
  const assetChangeData = assetChangeQuery.data || null;
  const capitalGainsData = capitalGainsQuery.data || null;
  const dividendData = dividendsQuery.data || null;
  const loading = summaryQuery.isPending; // Use isPending to show initial load, but fetch status for background

  const availableAccounts = (summary?.metrics?._available_accounts as string[]) || [];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'performance':
        return (
          <>
            <Dashboard
              summary={summary || { metrics: null, account_metrics: null }}
              currency={currency}
              history={historySparklineQuery.data || []}
              isLoading={summaryQuery.isLoading && !summaryQuery.data} // Only show skeleton if no data
              riskMetrics={riskMetricsQuery.data || {}}
              riskMetricsLoading={riskMetricsQuery.isLoading && !riskMetricsQuery.data}
              attributionData={attributionQuery.data}
              attributionLoading={attributionQuery.isLoading && !attributionQuery.data}
              holdings={holdings}
              visibleItems={visibleItems}
            />
            <PerformanceGraph
              currency={currency}
              accounts={selectedAccounts}
              benchmarks={benchmarks}
              onBenchmarksChange={setBenchmarks}
            />
            <HoldingsTable
              holdings={holdings}
              currency={currency}
              isLoading={holdingsQuery.isLoading && !holdingsQuery.data}
              showClosed={showClosed}
              onToggleShowClosed={setShowClosed}
            />
          </>
        );
      case 'watchlist':
        return <Watchlist currency={currency} />;
      case 'screener':
        return null;

      case 'transactions':
        return <TransactionsTable transactions={transactions} />;
      case 'markets':
        return (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold tracking-tight text-foreground">Market Indices</h2>
              <div className="text-xs text-muted-foreground">7D Trend</div>
            </div>
            {!summary?.metrics?.indices ? (
              <p className="text-muted-foreground">Market data unavailable.</p>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.values(summary.metrics.indices).map((index: any) => (
                  <div key={index.name} className="flex flex-col p-5 rounded-2xl bg-card border border-border bg-gradient-to-b from-card to-card/50 shadow-sm hover:shadow-md hover:border-cyan-500/30 transition-all duration-300 group">
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <span className="text-muted-foreground text-[10px] font-bold uppercase tracking-widest">{index.name}</span>
                        <h3 className="text-2xl font-bold text-foreground mt-1 tabular-nums">
                          {index.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </h3>
                      </div>
                      <div className={`flex flex-col items-end font-bold tabular-nums ${index.change >= 0 ? "text-emerald-500" : "text-rose-500"}`}>
                        <div className="flex items-center text-lg">
                          {index.change >= 0 ? "+" : ""}{index.change.toFixed(2)}
                        </div>
                        <div className="text-sm">
                          {index.changesPercentage.toFixed(2)}%
                        </div>
                      </div>
                    </div>

                    {index.sparkline && index.sparkline.length > 1 && (
                      <div className="h-16 w-full mt-2 filter drop-shadow-sm opacity-90 group-hover:opacity-100 transition-opacity">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={index.sparkline.map((v: number) => ({ value: v }))}>
                            <defs>
                              <linearGradient id={`splitFill-${index.name.replace(/[^a-zA-Z]/g, '')}`} x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={index.change >= 0 ? "#10b981" : "#ef4444"} stopOpacity={0.25} />
                                <stop offset="95%" stopColor={index.change >= 0 ? "#10b981" : "#ef4444"} stopOpacity={0} />
                              </linearGradient>
                            </defs>
                            <YAxis hide domain={['dataMin', 'dataMax']} />
                            <Area
                              type="monotone"
                              dataKey="value"
                              stroke={index.change >= 0 ? "#10b981" : "#ef4444"}
                              fill={`url(#splitFill-${index.name.replace(/[^a-zA-Z]/g, '')})`}
                              strokeWidth={2.5}
                              dot={false}
                              isAnimationActive={false}
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      case 'allocation':
        return <Allocation holdings={holdings} currency={currency} />;
      case 'asset_change':
        return <AssetChange data={assetChangeData} currency={currency} />;
      case 'capital_gains':
        return (
          <CapitalGains
            data={capitalGainsData}
            currency={currency}
            onDateRangeChange={(from, to) => setCapitalGainsDates({ from, to })}
          />
        );
      case 'analytics':
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold leading-none tracking-tight bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent w-fit">Analytics & Risk</h2>

            <PortfolioHealthComponent
              data={portfolioHealthQuery.data || null}
              isLoading={portfolioHealthQuery.isLoading && !portfolioHealthQuery.data}
            />

            <div className="h-[600px]">
              <CorrelationMatrix
                data={correlationMatrixQuery.data || null}
                isLoading={correlationMatrixQuery.isLoading && !correlationMatrixQuery.data}
                period={correlationPeriod}
                onPeriodChange={setCorrelationPeriod}
              />
            </div>
          </div>
        );
      case 'dividend':
        return (
          <div className="space-y-6">
            <DividendComponent data={dividendData} currency={currency} expectedDividends={summary?.metrics?.est_annual_income_display as number}>
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
          />
        );
      default:
        return (
          <div className="p-8 text-center text-gray-500 dark:text-gray-400">
            <p className="text-lg font-medium">Coming Soon</p>
            <p className="text-sm mt-2">The {activeTab} tab is under construction.</p>
          </div>
        );
    }
  };

  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);
  }, []);

  if (authLoading || !user) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-pulse text-muted-foreground">Loading Investa...</div>
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-background pb-20 selection:bg-cyan-500/20 selection:text-cyan-500">
      <div className="fixed inset-0 z-[-1] bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-indigo-900/20 via-background to-background pointer-events-none" />

      {/* Sidebar - Desktop */}
      <aside className="fixed left-0 top-0 bottom-0 w-[72px] flex flex-col items-center py-6 border-r border-border bg-background/40 backdrop-blur-2xl z-[60] hidden md:flex transition-all duration-300">
        <div className="flex-1 flex flex-col items-center gap-6">
          <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} side="right" />

          <CurrencySelector
            currentCurrency={currency}
            onChange={setCurrency}
            fxRate={summary?.metrics?.exchange_rate_to_display}
            side="right"
            availableCurrencies={settingsQuery.data?.available_currencies}
          />
        </div>

        <div className="mt-auto flex flex-col items-center gap-4 pb-4">
          <button
            onClick={() => user && logout()}
            className="p-2 text-muted-foreground hover:text-rose-500 hover:bg-rose-500/10 rounded-xl transition-all duration-300"
            title="Log Out"
          >
            <LogOut className="w-5 h-5" />
          </button>
          <ThemeToggle />
        </div>

      </aside>

      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
        onNavigate={handleNavigate}
      />

      <header className="sticky top-0 z-50 w-full border-b border-border bg-background/60 backdrop-blur-xl supports-[backdrop-filter]:bg-background/60">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 md:pl-[72px] py-3 sm:py-4 flex justify-between items-center gap-4 sm:gap-8">
          <div className="flex items-center gap-4">
            {/* Logo and App Title */}
            <div className="flex items-center gap-2 sm:gap-3 transition-all duration-300">
              {/* logo-dark.png is the dark mode logo, logo.png is light mode */}
              {/* Using display classes for instant switching based on next-themes class */}
              <img
                src="/logo.png?v=5"
                alt="Investa Logo"
                className="w-12 h-12 rounded-xl shadow-lg shadow-cyan-500/20 block dark:hidden"
              />
              <img
                src="/logo-dark.png?v=5"
                alt="Investa Logo"
                className="w-12 h-12 rounded-xl shadow-lg shadow-cyan-500/20 hidden dark:block"
              />
              <div className="hidden sm:flex flex-col -space-y-0.5">
                <h1 className="text-xl md:text-2xl font-bold tracking-tight text-foreground leading-none">
                  Investa
                </h1>
                <span className="text-[10px] text-muted-foreground font-medium tracking-wider">
                  by StockAlchemist
                </span>
              </div>
            </div>
          </div>


          <div className="flex items-center gap-4">
            {summary?.metrics?.indices && Object.values(summary.metrics.indices).map((index: any) => (
              <div key={index.name} className="hidden lg:flex flex-col items-start px-3 py-1.5 rounded-xl bg-card/40 border border-border/50 hover:bg-accent/10 hover:border-border transition-all duration-300 group overflow-hidden relative min-w-[120px]">
                <div className="flex items-center justify-between w-full gap-2">
                  <span className="text-[9px] font-bold uppercase tracking-wider text-muted-foreground group-hover:text-foreground transition-colors">{index.name}</span>
                  <span className={`text-[9px] font-bold tabular-nums ${(index.change || 0) >= 0 ? "text-emerald-500" : "text-rose-500"}`}>
                    {(index.changesPercentage || 0).toFixed(2)}%
                  </span>
                </div>
                <div className="text-xs font-bold text-foreground tabular-nums">
                  {index.price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) ?? "0.00"}
                </div>

                {index.sparkline && index.sparkline.length > 1 && (
                  <div className="absolute bottom-0 left-0 right-0 h-[3px] pointer-events-none opacity-40 group-hover:opacity-100 transition-opacity">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={index.sparkline.map((v: number) => ({ value: v }))}>
                        <YAxis hide domain={['dataMin', 'dataMax']} />
                        <Area
                          type="monotone"
                          dataKey="value"
                          stroke={index.change >= 0 ? "#10b981" : "#ef4444"}
                          fill="transparent"
                          strokeWidth={1.5}
                          dot={false}
                          isAnimationActive={false}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            ))}

            <div className="h-6 w-px bg-border hidden md:block" />

            <div className="md:hidden flex items-center gap-2">
              <CurrencySelector
                currentCurrency={currency}
                onChange={setCurrency}
                fxRate={summary?.metrics?.exchange_rate_to_display}
                availableCurrencies={settingsQuery.data?.available_currencies}
              />
              <AccountSelector
                availableAccounts={availableAccounts}
                selectedAccounts={selectedAccounts}
                onChange={setSelectedAccounts}
                accountGroups={settingsQuery.data?.account_groups}
              />
              {activeTab === 'performance' && (
                <LayoutConfigurator
                  visibleItems={visibleItems}
                  onVisibleItemsChange={setVisibleItems}
                />
              )}
              <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} side="bottom" />
            </div>

            <div className="hidden md:block">
              <AccountSelector
                availableAccounts={availableAccounts}
                selectedAccounts={selectedAccounts}
                onChange={setSelectedAccounts}
                accountGroups={settingsQuery.data?.account_groups}
              />
            </div>

            {activeTab === 'performance' && (
              <div className="hidden md:block">
                <LayoutConfigurator
                  visibleItems={visibleItems}
                  onVisibleItemsChange={setVisibleItems}
                />
              </div>
            )}
          </div>
        </div>

      </header >

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6 md:pl-[72px] transition-all duration-300">


        {renderTabContent()}
        <div className={activeTab === 'screener' ? 'block' : 'hidden'}>
          <ScreenerView currency={currency} />
        </div>
      </div>

      {/* Bottom Nav (Visual only for now) */}
      {/* Bottom Nav */}
      <div
        className="fixed bottom-0 left-0 right-0 border-t border-gray-200 dark:border-gray-700 px-4 py-3 flex justify-between items-center text-[10px] font-bold uppercase tracking-widest text-gray-500 dark:text-gray-400 md:hidden z-50"
        style={{ backgroundColor: 'var(--menu-solid)' }}
      >
        <div
          className={`flex flex-col items-center flex-1 cursor-pointer transition-colors ${activeTab !== 'settings' && activeTab !== 'markets' ? 'text-cyan-600 dark:text-cyan-400' : 'hover:text-cyan-600 dark:hover:text-cyan-400'}`}
          onClick={() => { setActiveTab('performance'); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
        >
          <HomeIcon className="w-5 h-5" />
          <span className="mt-1">Home</span>
        </div>
        <div
          className={`flex flex-col items-center flex-1 cursor-pointer transition-colors ${activeTab === 'markets' ? 'text-cyan-600 dark:text-cyan-400' : 'hover:text-cyan-600 dark:hover:text-cyan-400'}`}
          onClick={() => { setActiveTab('markets'); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
        >
          <BarChart3 className="w-5 h-5" />
          <span className="mt-1">Markets</span>
        </div>
        <div
          className={`flex flex-col items-center flex-1 cursor-pointer transition-colors ${activeTab === 'settings' ? 'text-cyan-600 dark:text-cyan-400' : 'hover:text-cyan-600 dark:hover:text-cyan-400'}`}
          onClick={() => { setActiveTab('settings'); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
        >
          <SettingsIcon className="w-5 h-5" />
          <span className="mt-1">Settings</span>
        </div>
        <div
          className="flex flex-col items-center flex-1 cursor-pointer transition-colors hover:text-cyan-600 dark:hover:text-cyan-400"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        >
          {mounted && theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          <span className="mt-1">{mounted && theme === 'dark' ? 'Light' : 'Dark'}</span>
        </div>
      </div>
    </main >
  );
}

