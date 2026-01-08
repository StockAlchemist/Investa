'use client';

import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
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
  PerformanceData
} from '@/lib/api';
// import { CURRENCY_SYMBOLS } from '@/lib/utils';
import Dashboard from '@/components/Dashboard';
import HoldingsTable from '@/components/HoldingsTable';
import AccountSelector from '@/components/AccountSelector';
import CurrencySelector from '@/components/CurrencySelector';
import TabNavigation from '@/components/TabNavigation';
import TransactionsTable from '@/components/TransactionsTable';
import PerformanceGraph from '@/components/PerformanceGraph';
import Allocation from '@/components/Allocation';
import AssetChange from '@/components/AssetChange';
import CapitalGains from '@/components/CapitalGains';
import DividendComponent from '@/components/Dividend';
import RiskMetricsComponent from '@/components/RiskMetrics';
import AttributionChart from '@/components/AttributionChart';
import DividendCalendar from '@/components/DividendCalendar';
import { IncomeProjector } from '@/components/IncomeProjector';
import ThemeToggle from '@/components/ThemeToggle';
import Settings from '@/components/Settings';
import CommandPalette from '@/components/CommandPalette';
import { CorrelationMatrix } from '@/components/CorrelationMatrix';
import { PortfolioHealthComponent } from '@/components/PortfolioHealth';
import Watchlist from '@/components/Watchlist';

import { useTheme } from 'next-themes';
import { Home as HomeIcon, BarChart3, Settings as SettingsIcon, Moon, Sun } from 'lucide-react';

export default function Home() {
  const { theme, setTheme } = useTheme();
  const [selectedAccounts, setSelectedAccounts] = useState<string[]>([]);
  const [currency, setCurrency] = useState('USD');
  const [activeTab, setActiveTab] = useState('performance');
  const [benchmarks, setBenchmarks] = useState<string[]>(['S&P 500', 'Dow Jones', 'NASDAQ']);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const [showClosed, setShowClosed] = useState(false);
  const [capitalGainsDates, setCapitalGainsDates] = useState<{ from?: string, to?: string }>({});

  // Initialize showClosed from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('investa_show_closed');
    if (saved) setShowClosed(saved === 'true');
  }, []);

  // Persist showClosed to localStorage
  useEffect(() => {
    localStorage.setItem('investa_show_closed', showClosed.toString());
  }, [showClosed]);

  const [benchmarksInitialized, setBenchmarksInitialized] = useState(false);

  // Initialize benchmarks from localStorage
  useEffect(() => {
    const savedBenchmarks = localStorage.getItem('investa_graph_benchmarks');
    if (savedBenchmarks) {
      try {
        const parsed = JSON.parse(savedBenchmarks);
        if (Array.isArray(parsed)) {
          setBenchmarks(parsed);
        }
      } catch (e) {
        console.error("Failed to parse saved benchmarks", e);
      }
    }
    setBenchmarksInitialized(true);
  }, []);

  // Persist benchmarks to localStorage
  useEffect(() => {
    if (benchmarksInitialized) {
      localStorage.setItem('investa_graph_benchmarks', JSON.stringify(benchmarks));
    }
  }, [benchmarks, benchmarksInitialized]);

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
    queryFn: () => fetchSummary(currency, selectedAccounts),
    staleTime: 5 * 60 * 1000,
  });

  const holdingsQuery = useQuery({
    queryKey: ['holdings', currency, selectedAccounts, showClosed],
    queryFn: () => fetchHoldings(currency, selectedAccounts, showClosed),
    enabled: activeTab === 'performance' || activeTab === 'allocation',
    staleTime: 5 * 60 * 1000,
  });

  const transactionsQuery = useQuery({
    queryKey: ['transactions', selectedAccounts],
    queryFn: () => fetchTransactions(selectedAccounts),
    enabled: activeTab === 'transactions',
    staleTime: 5 * 60 * 1000,
  });

  const assetChangeQuery = useQuery({
    queryKey: ['assetChange', currency, selectedAccounts, benchmarks],
    queryFn: () => fetchAssetChange(currency, selectedAccounts, benchmarks),
    enabled: activeTab === 'asset_change',
    staleTime: 5 * 60 * 1000,
  });

  const capitalGainsQuery = useQuery({
    queryKey: ['capitalGains', currency, selectedAccounts, capitalGainsDates.from, capitalGainsDates.to],
    queryFn: () => fetchCapitalGains(currency, selectedAccounts, capitalGainsDates.from, capitalGainsDates.to),
    enabled: activeTab === 'capital_gains',
    staleTime: 5 * 60 * 1000,
  });

  const dividendsQuery = useQuery({
    queryKey: ['dividends', currency, selectedAccounts],
    queryFn: () => fetchDividends(currency, selectedAccounts),
    enabled: activeTab === 'dividend',
    staleTime: 5 * 60 * 1000,
  });

  const riskMetricsQuery = useQuery({
    queryKey: ['riskMetrics', currency, selectedAccounts],
    queryFn: () => fetchRiskMetrics(currency, selectedAccounts),
    enabled: activeTab === 'performance',
    staleTime: 5 * 60 * 1000,
  });

  const attributionQuery = useQuery({
    queryKey: ['attribution', currency, selectedAccounts],
    queryFn: () => fetchAttribution(currency, selectedAccounts),
    enabled: activeTab === 'performance',
    staleTime: 5 * 60 * 1000,
  });

  const dividendCalendarQuery = useQuery({
    queryKey: ['dividendCalendar', selectedAccounts],
    queryFn: () => fetchDividendCalendar(selectedAccounts),
    enabled: activeTab === 'dividend',
    staleTime: 5 * 60 * 1000,
  });

  const historySparklineQuery = useQuery({
    queryKey: ['history', currency, selectedAccounts, 'sparkline'],
    queryFn: () => fetchHistory(currency, selectedAccounts, '1d', [], '5m'),
    enabled: activeTab === 'performance',
    staleTime: 5 * 60 * 1000,
  });

  const watchlistQuery = useQuery({
    queryKey: ['watchlist', currency],
    queryFn: () => fetchWatchlist(currency),
    staleTime: 1 * 60 * 1000,
  });

  const settingsQuery = useQuery({
    queryKey: ['settings'],
    queryFn: fetchSettings,
    staleTime: 5 * 60 * 1000,
  });

  const summary = summaryQuery.data;
  const holdings = holdingsQuery.data || [];
  const transactions = transactionsQuery.data || [];
  const assetChangeData = assetChangeQuery.data || null;
  const capitalGainsData = capitalGainsQuery.data || null;
  const dividendData = dividendsQuery.data || null;
  const loading = summaryQuery.isLoading;

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
              isLoading={summaryQuery.isLoading}
            />
            <div className="mb-6">
              <RiskMetricsComponent
                metrics={riskMetricsQuery.data || {}}
                isLoading={riskMetricsQuery.isLoading}
              />
            </div>
            <PerformanceGraph
              currency={currency}
              accounts={selectedAccounts}
              benchmarks={benchmarks}
              onBenchmarksChange={setBenchmarks}
            />
            {attributionQuery.data && (
              <div className="mb-6">
                <AttributionChart
                  data={attributionQuery.data}
                  isLoading={attributionQuery.isLoading}
                  currency={currency}
                />
              </div>
            )}
            <HoldingsTable
              holdings={holdings}
              currency={currency}
              isLoading={holdingsQuery.isLoading}
              showClosed={showClosed}
              onToggleShowClosed={setShowClosed}
              watchlist={watchlistQuery.data}
            />
          </>
        );
      case 'watchlist':
        return <Watchlist currency={currency} />;
      case 'transactions':
        return <TransactionsTable transactions={transactions} />;
      case 'markets':
        return (
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-foreground">Market Indices</h2>
            {!summary?.metrics?.indices ? (
              <p className="text-muted-foreground">Market data unavailable.</p>
            ) : (
              <div className="grid grid-cols-1 gap-4">
                {Object.values(summary.metrics.indices).map((index: { name: string; price: number; change: number; changesPercentage: number }) => (
                  <div key={index.name} className="flex items-center justify-between p-4 rounded-xl bg-card border border-border">
                    <span className="font-medium text-foreground text-lg">{index.name}</span>
                    <div className="flex flex-col items-end">
                      <span className="text-foreground font-medium text-lg">{index.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                      <span className={`text-sm ${index.change >= 0 ? "text-emerald-500" : "text-rose-500"}`}>
                        {index.change >= 0 ? "+" : ""}{index.change.toFixed(2)} ({index.changesPercentage.toFixed(2)}%)
                      </span>
                    </div>
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
            <h2 className="text-xl font-bold text-foreground">Analytics & Risk</h2>

            <PortfolioHealthComponent currency={currency} accounts={selectedAccounts} />

            <div className="h-[600px]">
              <CorrelationMatrix currency={currency} accounts={selectedAccounts} />
            </div>
          </div>
        );
      case 'dividend':
        return (
          <div className="space-y-6">
            <DividendComponent data={dividendData} currency={currency} expectedDividends={summary?.metrics?.est_annual_income_display as number}>
              <IncomeProjector currency={currency} accounts={selectedAccounts} />
              <DividendCalendar
                events={dividendCalendarQuery.data || []}
                isLoading={dividendCalendarQuery.isLoading}
                currency={currency}
              />
            </DividendComponent>
          </div>
        );
      case 'settings':
        return <Settings />;
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
          />
        </div>

        <div className="mt-auto space-y-4 pb-4">
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
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={mounted && resolvedTheme === 'dark' ? "/logo-dark.png" : "/logo.png"} alt="Investa Logo" className="w-12 h-12 rounded-xl shadow-lg shadow-cyan-500/20" />
              <h1 className="text-xl md:text-2xl font-bold tracking-tight text-foreground flex items-center gap-3">
                <span className="hidden sm:block">Investa</span>
                <span className="hidden md:inline-flex items-center rounded-md border border-white/10 bg-white/5 px-2 py-0.5 text-xs font-medium text-muted-foreground">
                  ⌘K
                </span>
              </h1>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {summary?.metrics?.indices && Object.values(summary.metrics.indices).map((index: { name: string; price: number; change: number; changesPercentage: number }) => (
              <div key={index.name} className="hidden md:flex items-center space-x-2 text-xs font-medium px-3 py-1.5 rounded-full bg-card border border-border hover:bg-accent/10 transition-colors">
                <span className="text-muted-foreground">{index.name}</span>
                <span className="text-foreground">{index.price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) ?? "0.00"}</span>
                <span className={(index.change || 0) >= 0 ? "text-emerald-500" : "text-rose-500"}>
                  {(index.change || 0) >= 0 ? "+" : ""}{(index.change || 0).toFixed(2)} ({(index.changesPercentage || 0).toFixed(2)}%)
                </span>
              </div>
            ))}

            <div className="h-6 w-px bg-border hidden md:block" />

            <div className="md:hidden flex items-center gap-1">
              <CurrencySelector
                currentCurrency={currency}
                onChange={setCurrency}
                fxRate={summary?.metrics?.exchange_rate_to_display}
              />
              <AccountSelector
                availableAccounts={availableAccounts}
                selectedAccounts={selectedAccounts}
                onChange={setSelectedAccounts}
                accountGroups={settingsQuery.data?.account_groups}
              />
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
          </div>
        </div>

      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6 md:pl-[72px] transition-all duration-300">


        {loading ? (
          <div className="p-4 text-center text-gray-500">Loading...</div>
        ) : (
          renderTabContent()
        )}
      </div>

      {/* Bottom Nav (Visual only for now) */}
      {/* Bottom Nav */}
      <div className="fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-4 py-3 flex justify-between items-center text-[10px] font-bold uppercase tracking-widest text-gray-500 dark:text-gray-400 md:hidden z-50">
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
          {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          <span className="mt-1">{theme === 'dark' ? 'Light' : 'Dark'}</span>
        </div>
      </div>
    </main >
  );
}
