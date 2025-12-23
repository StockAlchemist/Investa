'use client';

import { useEffect, useState } from 'react';
import { fetchSummary, fetchHoldings, fetchTransactions, fetchHistory, fetchAssetChange, fetchCapitalGains, fetchDividends, PortfolioSummary, Holding, Transaction, PerformanceData, AssetChangeData, CapitalGain, Dividend } from '@/lib/api';
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
import Settings from '@/components/Settings';

export default function Home() {
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [holdings, setHoldings] = useState<Holding[]>([]);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [assetChangeData, setAssetChangeData] = useState<AssetChangeData | null>(null);
  const [capitalGainsData, setCapitalGainsData] = useState<CapitalGain[] | null>(null);
  const [dividendData, setDividendData] = useState<Dividend[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [availableAccounts, setAvailableAccounts] = useState<string[]>([]);
  const [selectedAccounts, setSelectedAccounts] = useState<string[]>([]);
  const [currency, setCurrency] = useState('USD');
  const [activeTab, setActiveTab] = useState('performance');
  const [benchmarks, setBenchmarks] = useState<string[]>([]);

  // Effect for core summary data - triggered by currency or account changes
  useEffect(() => {
    async function loadSummary() {
      try {
        setLoading(true);
        const summaryData = await fetchSummary(currency, selectedAccounts);
        setSummary(summaryData);

        if (summaryData?.metrics?._available_accounts && availableAccounts.length === 0) {
          setAvailableAccounts(summaryData.metrics._available_accounts);
        }
        setLoading(false);
      } catch (error) {
        console.error('Error loading summary:', error);
        setLoading(false);
      }
    }
    loadSummary();
  }, [selectedAccounts, currency]);

  // Effect for Holdings (Performance & Allocation tabs)
  useEffect(() => {
    if (activeTab === 'performance' || activeTab === 'allocation') {
      async function loadHoldings() {
        try {
          const data = await fetchHoldings(currency, selectedAccounts);
          setHoldings(data);
        } catch (error) {
          console.error('Error loading holdings:', error);
        }
      }
      loadHoldings();
    }
  }, [selectedAccounts, currency, activeTab]);

  // Effect for Transactions
  useEffect(() => {
    if (activeTab === 'transactions') {
      async function loadTransactions() {
        try {
          const data = await fetchTransactions(selectedAccounts);
          setTransactions(data);
        } catch (error) {
          console.error('Error loading transactions:', error);
        }
      }
      loadTransactions();
    }
  }, [selectedAccounts, activeTab]);

  // Effect for Capital Gains
  useEffect(() => {
    if (activeTab === 'capital_gains') {
      async function loadCapitalGains() {
        try {
          const data = await fetchCapitalGains(currency, selectedAccounts);
          setCapitalGainsData(data);
        } catch (error) {
          console.error('Error loading capital gains:', error);
        }
      }
      loadCapitalGains();
    }
  }, [selectedAccounts, currency, activeTab]);

  // Effect for Dividends
  useEffect(() => {
    if (activeTab === 'dividend') {
      async function loadDividends() {
        try {
          const data = await fetchDividends(currency, selectedAccounts);
          setDividendData(data);
        } catch (error) {
          console.error('Error loading dividends:', error);
        }
      }
      loadDividends();
    }
  }, [selectedAccounts, currency, activeTab]);

  // Effect for benchmark-dependent data (Asset Change)
  useEffect(() => {
    if (activeTab === 'asset_change') {
      async function loadAssetChange() {
        try {
          const data = await fetchAssetChange(currency, selectedAccounts, benchmarks);
          setAssetChangeData(data);
        } catch (error) {
          console.error('Error loading asset change data:', error);
        }
      }
      loadAssetChange();
    }
  }, [selectedAccounts, currency, activeTab, benchmarks]);

  // Initial load to get accounts if we don't have them? 
  // The useEffect above handles it.

  const renderTabContent = () => {
    switch (activeTab) {
      case 'performance':
        return (
          <>
            {summary && <Dashboard summary={summary} currency={currency} />}
            <PerformanceGraph
              currency={currency}
              accounts={selectedAccounts}
              benchmarks={benchmarks}
              onBenchmarksChange={setBenchmarks}
            />
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
        return <DividendComponent data={dividendData} currency={currency} expectedDividends={summary?.metrics?.est_annual_income_display} />;
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

  return (
    <main className="min-h-screen bg-gray-50 dark:bg-gray-900 pb-20">
      <header className="bg-white dark:bg-slate-900 shadow-sm sticky top-0 z-50 border-b border-slate-200 dark:border-slate-800 opacity-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <img src="/logo.png" alt="Investa Logo" className="h-10 w-auto" />
            <h1 className="text-xl font-bold text-gray-900 dark:text-white">Investa</h1>
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
          <AccountSelector
            availableAccounts={availableAccounts}
            selectedAccounts={selectedAccounts}
            onChange={setSelectedAccounts}
          />
        </div>

        {loading ? (
          <div className="p-4 text-center text-gray-500">Loading...</div>
        ) : (
          renderTabContent()
        )}
      </div>

      {/* Bottom Nav (Visual only for now) */}
      <div className="fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-6 py-3 flex justify-between items-center text-xs text-gray-500 dark:text-gray-400">
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
    </main>
  );
}
