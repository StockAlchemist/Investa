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
  fetchDividendCalendar
} from '@/lib/api';
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
import Settings from '@/components/Settings';
import CommandPalette from '@/components/CommandPalette';

import { useTheme } from 'next-themes';

export default function Home() {
  const [selectedAccounts, setSelectedAccounts] = useState<string[]>([]);
  const [currency, setCurrency] = useState('USD');
  const [activeTab, setActiveTab] = useState('performance');
  const [benchmarks, setBenchmarks] = useState<string[]>([]);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);

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
    queryKey: ['holdings', currency, selectedAccounts],
    queryFn: () => fetchHoldings(currency, selectedAccounts),
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
    queryKey: ['capitalGains', currency, selectedAccounts],
    queryFn: () => fetchCapitalGains(currency, selectedAccounts),
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

  const summary = summaryQuery.data;
  const holdings = holdingsQuery.data || [];
  const transactions = transactionsQuery.data || [];
  const assetChangeData = assetChangeQuery.data || null;
  const capitalGainsData = capitalGainsQuery.data || null;
  const dividendData = dividendsQuery.data || null;
  const loading = summaryQuery.isLoading;

  const availableAccounts = summary?.metrics?._available_accounts || [];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'performance':
        return (
          <>
            {summary && <Dashboard summary={summary} currency={currency} />}
            <RiskMetricsComponent
              metrics={riskMetricsQuery.data || {}}
              isLoading={riskMetricsQuery.isLoading}
            />
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
            <HoldingsTable holdings={holdings} currency={currency} />
          </>
        );
      case 'transactions':
        return <TransactionsTable transactions={transactions} />;
      case 'allocation':
        return <Allocation holdings={holdings} currency={currency} />;
      case 'asset_change':
        return <AssetChange data={assetChangeData} currency={currency} />;
      case 'capital_gains':
        return <CapitalGains data={capitalGainsData} currency={currency} />;
      case 'dividend':
        return (
          <div className="space-y-6">
            <DividendComponent data={dividendData} currency={currency} expectedDividends={summary?.metrics?.est_annual_income_display}>
              <DividendCalendar
                events={dividendCalendarQuery.data || []}
                isLoading={dividendCalendarQuery.isLoading}
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
    setMounted(true);
  }, []);

  return (
    <main className="min-h-screen bg-gray-50 dark:bg-gray-900 pb-20">
      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
        onNavigate={handleNavigate}
      />

      <header className="bg-white dark:bg-slate-900 shadow-sm sticky top-0 z-50 border-b border-slate-200 dark:border-slate-800 opacity-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            {/* Light Theme Logo */}
            <img
              src="/logo.png"
              alt="Investa Logo"
              className="h-10 w-auto dark:hidden"
            />
            {/* Dark Theme Logo */}
            <img
              src="/logo-dark.png"
              alt="Investa Logo"
              className="h-10 w-auto hidden dark:block"
            />
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
              Investa
              <span className="hidden md:inline-flex items-center rounded-md bg-gray-100 px-2 py-1 text-xs font-medium text-gray-600 ring-1 ring-inset ring-gray-500/10 dark:bg-gray-800 dark:text-gray-400 dark:ring-gray-700">
                ⌘K
              </span>
            </h1>
          </div>
          <div className="flex items-center space-x-4">
            {summary?.metrics?.indices && Object.values(summary.metrics.indices).map((index: any) => (
              <div key={index.name} className="hidden md:flex items-center space-x-2 text-sm font-medium text-gray-600 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 px-3 py-1.5 rounded-md">
                <span className="font-bold">{index.name}</span>
                <span>{index.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                <span className={index.change >= 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}>
                  {index.change >= 0 ? "+" : ""}{index.change.toFixed(2)} ({index.changesPercentage.toFixed(2)}%)
                </span>
              </div>
            ))}
            {summary?.metrics?.exchange_rate_to_display && (
              <div className="text-sm font-medium text-gray-600 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 px-3 py-1.5 rounded-md">
                1 USD = {summary.metrics.exchange_rate_to_display.toFixed(2)} {currency}
              </div>
            )}
            <CurrencySelector currentCurrency={currency} onChange={setCurrency} />
          </div>
        </div>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6">
        <div className="mb-6">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-4">
            {/* Removed overview text */}
            <AccountSelector
              availableAccounts={availableAccounts}
              selectedAccounts={selectedAccounts}
              onChange={setSelectedAccounts}
            />
          </div>
        </div>

        {loading ? (
          <div className="p-4 text-center text-gray-500">Loading...</div>
        ) : (
          renderTabContent()
        )}
      </div>

      {/* Bottom Nav (Visual only for now) */}
      <div className="fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-6 py-3 flex justify-between items-center text-xs text-gray-500 dark:text-gray-400 md:hidden">
        <div className="flex flex-col items-center text-blue-600 dark:text-blue-400">
          <span className="text-xl">🏠</span>
          <span className="mt-1">Home</span>
        </div>
        <div className="flex flex-col items-center">
          <span className="text-xl">📊</span>
          <span className="mt-1">Markets</span>
        </div>
        <div className="flex flex-col items-center cursor-pointer hover:text-blue-600 dark:hover:text-blue-400" onClick={() => setActiveTab('settings')}>
          <span className="text-xl">⚙️</span>
          <span className="mt-1">Settings</span>
        </div>
      </div>
    </main >
  );
}
