'use client';

import { useState, useEffect, useMemo, useRef } from 'react';
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
  fetchProjection,
  fetchAttribution,
  fetchDividendCalendar,
  fetchEarningsCalendar,
  fetchHistory,
  fetchWatchlist,
  fetchSettings,
  updateSettings,
  type Settings as SettingsType,
  SettingsUpdate,
  fetchPortfolioHealth,
  fetchProjectedIncome,
  fetchMarketStatus,
  fetchIndices,
  fetchHeadline,
  PortfolioSummary
} from '@/lib/api';
import { cn } from '@/lib/utils';
import type { MarketIndex } from '@/components/MarketsTab';
import { INITIAL_VISIBLE_ITEMS, TAB_THEMES } from '@/lib/dashboard_constants';
import { TAB_LAYOUT_ITEMS, TAB_INITIAL_VISIBLE, TAB_SECTION_LABELS } from '@/lib/layout_registry';
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
import { useStockModal } from '@/context/StockModalContext';

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
const BuffettRankView      = dynamic(() => import('@/components/BuffettRankView'));
const StrategiesView       = dynamic(() => import('@/components/StrategiesView'));
const IndexGraphModal      = dynamic(() => import('@/components/IndexGraphModal'), { ssr: false });
const MarketsTab           = dynamic(() => import('@/components/MarketsTab'), { ssr: false });
const RiskMetrics          = dynamic(() => import('@/components/RiskMetrics'), { ssr: false });
const ProjectionCard       = dynamic(() => import('@/components/ProjectionCard'), { ssr: false });
const SectorAttribution    = dynamic(() => import('@/components/AttributionChart').then(m => ({ default: m.SectorAttribution })), { ssr: false });
const TopContributors      = dynamic(() => import('@/components/AttributionChart').then(m => ({ default: m.TopContributors })), { ssr: false });
const StockDetailView      = dynamic(() => import('@/components/StockDetailModal'), {
  loading: () => <div className="h-[600px] bg-card border border-border/50 rounded-2xl animate-pulse" />,
  ssr: false,
});

const TAB_NAMES: Record<string, string> = {
  performance: 'Dashboard',
  allocation: 'Portfolio',
  asset_change: 'Performance',
  transactions: 'Transactions',
  dividend: 'Income',
  capital_gains: 'Capital Gains',
  screener: 'Screener',
  buffett_rank: 'Rankings',
  strategies: 'Strategies',
  watchlist: 'Watchlist',
  markets: 'Markets',
  ai_review: 'AI Insights',
  settings: 'Settings',
};

export default function AuthenticatedDashboard() {
  const { theme, setTheme } = useTheme();
  const { user, logout } = useAuth();
  const { selectedSymbol, modalCurrency, closeStockDetail, goBack, canGoBack } = useStockModal();

  const [selectedAccounts, setSelectedAccounts]     = useState<string[]>([]);
  const [currency, setCurrency]                     = useState('USD');
  const [activeTab, setActiveTab]                   = useState('performance');
  const [showClosed, setShowClosed]                 = useState(false);
  const [backgroundFetchLevel, setBackgroundFetchLevel] = useState(0);
  const [mounted, setMounted]                       = useState(false);
  const [settingsInitialTab, setSettingsInitialTab] = useState<'overrides' | 'account' | undefined>(undefined);
  const [sidebarCollapsed, setSidebarCollapsed]     = useState(false);
  const [isIndexGraphModalOpen, setIsIndexGraphModalOpen] = useState(false);
  const [indexGraphFocus, setIndexGraphFocus]             = useState<string | null>(null);
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
  const [tabLayouts, setTabLayouts]                 = useState<Record<string, string[]>>({});

  const handleUserIconClick = () => { closeStockDetail(); setSettingsInitialTab('account'); setActiveTab('settings'); };
  const handleTabChange = (tab: string) => {
    closeStockDetail();
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
      if (savedVisibleItems) {
        const p = JSON.parse(savedVisibleItems);
        if (Array.isArray(p) && p.length > 0) {
          if (!p.includes('performanceGraph')) {
            const idx = p.indexOf('portfolioDonut');
            if (idx >= 0) p.splice(idx + 1, 0, 'performanceGraph');
            else p.push('performanceGraph');
          }
          const topSections = ['portfolioHero', 'todayStrip', 'dashboardEvents', 'dashboardInsights'];
          const missingTop = topSections.filter(id => !p.includes(id));
          if (missingTop.length > 0) p.unshift(...missingTop);
          setVisibleItems(p);
        }
      }
      // Hydrate per-tab layouts
      const loadedLayouts: Record<string, string[]> = {};
      for (const tabId of Object.keys(TAB_LAYOUT_ITEMS)) {
        if (tabId === 'performance') continue;
        const saved = localStorage.getItem(`investa_tab_layout_${tabId}`);
        if (saved) {
          try { 
            const arr = JSON.parse(saved); 
            if (Array.isArray(arr) && arr.length > 0) {
              if (tabId === 'allocation' && arr.includes('allocationCharts')) {
                const idx = arr.indexOf('allocationCharts');
                arr.splice(idx, 1, 'concentrationKpis', 'categoryDrift', 'stockDrift', 'rebalanceHelper', 'treemap', 'donutCharts');
              }
              if (tabId === 'allocation' && !arr.includes('holdingsHeatmap')) {
                const tIdx = arr.indexOf('treemap');
                arr.splice(tIdx >= 0 ? tIdx + 1 : arr.length, 0, 'holdingsHeatmap');
              }
              if (tabId === 'dividend' && arr.includes('annualDividends') && !arr.includes('incomeKpis')) {
                arr.unshift('incomeKpis', 'topPayers', 'byAccount');
              }
              if (tabId === 'capital_gains' && arr.includes('capitalGainsTable')) {
                const idx = arr.indexOf('capitalGainsTable');
                arr.splice(idx, 1, 'capitalGainsKpis', 'annualCapitalGains', 'capitalGainsTransactions');
              }
              loadedLayouts[tabId] = arr; 
            } 
          } catch {}
        }
      }
      setTabLayouts(loadedLayouts);
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
    onSuccess: (_, variables) => {
      // Optimistically update React Query cache in-memory to prevent invalidation refetch loops
      queryClient.setQueryData(['settings', user?.username], (old: SettingsType | undefined) => {
        if (!old) return old;
        return { ...old, ...variables };
      });
    },
  });

  const mutateSettingsRef = useRef(settingsMutation.mutate);
  mutateSettingsRef.current = settingsMutation.mutate;

  // Track the last synced server settings to prevent re-sending identical updates
  const lastSyncedRef = useRef<Record<string, unknown>>({});
  const isInitialSyncDone = useRef(false);

  // Initialize lastSyncedRef once settingsQuery.data is first loaded
  useEffect(() => {
    if (!settingsQuery.data || isInitialSyncDone.current) return;
    isInitialSyncDone.current = true;
    const s = settingsQuery.data;
    lastSyncedRef.current = {
      display_currency: s.display_currency,
      active_tab: s.active_tab,
      show_closed: s.show_closed,
      selected_accounts: s.selected_accounts,
      visible_items: s.visible_items,
      benchmarks: s.benchmarks,
    };
  }, [settingsQuery.data]);

  useEffect(() => {
    if (!mounted) return;
    try {
      localStorage.setItem('investa_currency',                       currency);
      localStorage.setItem('investa_active_tab',                     activeTab);
      localStorage.setItem('investa_show_closed',                    showClosed.toString());
      localStorage.setItem('investa_selected_accounts',              JSON.stringify(selectedAccounts));
      localStorage.setItem('investa_graph_period',                   graphPeriod);
      localStorage.setItem('investa_graph_view',                     graphView);
      if (visibleItems.length > 0) localStorage.setItem('investa_dashboard_visible_items', JSON.stringify(visibleItems));
      if (benchmarks.length > 0)   localStorage.setItem('investa_graph_benchmarks',        JSON.stringify(benchmarks));
      for (const [tabId, items] of Object.entries(tabLayouts)) {
        if (items.length > 0) localStorage.setItem(`investa_tab_layout_${tabId}`, JSON.stringify(items));
      }
    } catch (e) {
      console.warn('localStorage quota exceeded, skipping persistence', e);
    }

    if (!isInitialSyncDone.current) return;

    const id = setTimeout(() => {
      const updates: Partial<SettingsUpdate> = {};
      const last = lastSyncedRef.current;

      const arrayEqual = (a?: unknown, b?: string[] | null) => {
        const arrA = (Array.isArray(a) ? a : []) as string[];
        const arrB = b || [];
        return arrA.length === arrB.length && arrA.every((v, i) => v === arrB[i]);
      };

      if (last.display_currency !== currency) updates.display_currency = currency;
      if (last.active_tab !== activeTab) updates.active_tab = activeTab;
      if (last.show_closed !== showClosed) updates.show_closed = showClosed;
      if (!arrayEqual(last.selected_accounts, selectedAccounts)) updates.selected_accounts = selectedAccounts;
      if (!arrayEqual(last.visible_items, visibleItems) && visibleItems.length > 0) updates.visible_items = visibleItems;
      if (!arrayEqual(last.benchmarks, benchmarks) && benchmarks.length > 0) updates.benchmarks = benchmarks;

      if (Object.keys(updates).length > 0) {
        lastSyncedRef.current = {
          ...lastSyncedRef.current,
          ...updates,
        };
        mutateSettingsRef.current(updates);
      }
    }, 1500);

    return () => clearTimeout(id);
  }, [mounted, currency, activeTab, showClosed, benchmarks, selectedAccounts, visibleItems, tabLayouts, graphPeriod, graphView]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); setIsCommandPaletteOpen(p => !p); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const handleNavigate = (tab: string) => {
    closeStockDetail();
    setActiveTab(tab);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // ── Queries ──────────────────────────────────────────────────────────────
  const marketStatusQuery = useQuery({
    queryKey: ['marketStatus'],
    queryFn: fetchMarketStatus,
    staleTime: 60 * 1000,
    refetchInterval: 5 * 60 * 1000,
    enabled: !!user,
  });
  const isMarketOpen = marketStatusQuery.data?.is_open ?? false;

  const indicesQuery = useQuery({
    queryKey: ['indices'],
    queryFn: ({ signal }) => fetchIndices(signal),
    staleTime: 60 * 1000,
    refetchInterval: isMarketOpen ? 2 * 60 * 1000 : false,
    enabled: !!user,
  });

  const cachedInitialSummary = useMemo(() => {
    if (typeof window !== 'undefined' && user?.username) {
      try {
        const saved = localStorage.getItem(`investa_cached_summary_${user.username}`);
        if (saved) return JSON.parse(saved);
      } catch {}
    }
    return undefined;
  }, [user?.username]);

  const cachedInitialHoldings = useMemo(() => {
    if (typeof window !== 'undefined' && user?.username) {
      try {
        const saved = localStorage.getItem(`investa_cached_holdings_${user.username}`);
        if (saved) return JSON.parse(saved);
      } catch {}
    }
    return undefined;
  }, [user?.username]);

  const headlineQuery = useQuery({
    queryKey: ['headline', user?.username, currency, selectedAccounts],
    queryFn: ({ signal }) => fetchHeadline(currency, selectedAccounts, signal),
    staleTime: 60 * 1000,
    refetchInterval: isMarketOpen ? 60 * 1000 : false,
    placeholderData: (prev) => prev ?? (cachedInitialSummary?.metrics ? { metrics: cachedInitialSummary.metrics } : undefined),
    enabled: !!user,
  });

  const summaryQuery = useQuery({
    queryKey: ['summary', user?.username, currency, selectedAccounts, showClosed],
    queryFn: ({ signal }) => fetchSummary(currency, selectedAccounts, showClosed, signal),
    staleTime: 5 * 60 * 1000,
    refetchInterval: isMarketOpen ? 60 * 1000 : false,
    placeholderData: (prev) => prev ?? cachedInitialSummary,
    enabled: !!user,
  });

  const holdingsQuery = useQuery({
    queryKey: ['holdings', user?.username, currency, selectedAccounts, showClosed],
    queryFn: ({ signal }) => fetchHoldings(currency, selectedAccounts, showClosed, signal),
    staleTime: 5 * 60 * 1000,
    // Same cadence as the headline and the summary: every per-stock price on
    // this screen — the holdings table, the day's gainers and losers, the
    // heatmap — comes from here, and without a poll they sat on whatever the
    // session opened with until the tab lost and regained focus.
    refetchInterval: isMarketOpen ? 60 * 1000 : false,
    placeholderData: (prev) => prev ?? cachedInitialHoldings,
    enabled: !!user,
  });

  useEffect(() => {
    if (summaryQuery.data && user?.username) {
      try {
        localStorage.setItem(`investa_cached_summary_${user.username}`, JSON.stringify(summaryQuery.data));
      } catch {}
    }
  }, [summaryQuery.data, user?.username]);

  useEffect(() => {
    if (holdingsQuery.data && user?.username) {
      try {
        localStorage.setItem(`investa_cached_holdings_${user.username}`, JSON.stringify(holdingsQuery.data));
      } catch {}
    }
  }, [holdingsQuery.data, user?.username]);

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
    enabled: !!user && (activeTab === 'asset_change' || backgroundFetchLevel >= 1),
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

  const projectionQuery = useQuery({
    queryKey: ['projection', user?.username, currency, selectedAccounts],
    queryFn: ({ signal }) => fetchProjection(currency, selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: !!user && (activeTab === 'performance' || backgroundFetchLevel >= 1),
  });

  const riskMetricsQuery = useQuery({
    queryKey: ['riskMetrics', user?.username, currency, selectedAccounts, showClosed],
    queryFn: ({ signal }) => fetchRiskMetrics(currency, selectedAccounts, showClosed, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: !!user && (activeTab === 'performance' || activeTab === 'asset_change' || backgroundFetchLevel >= 1),
  });

  const attributionQuery = useQuery({
    queryKey: ['attribution', user?.username, currency, selectedAccounts, showClosed],
    queryFn: ({ signal }) => fetchAttribution(currency, selectedAccounts, false, showClosed, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: !!user && (activeTab === 'performance' || activeTab === 'asset_change' || backgroundFetchLevel >= 1),
  });

  const dividendCalendarQuery = useQuery({
    queryKey: ['dividendCalendar', user?.username, currency, selectedAccounts],
    queryFn: ({ signal }) => fetchDividendCalendar(currency, selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: !!user && (activeTab === 'dividend' || activeTab === 'performance' || backgroundFetchLevel >= 2),
  });

  const earningsCalendarQuery = useQuery({
    queryKey: ['earningsCalendar', user?.username, selectedAccounts],
    queryFn: ({ signal }) => fetchEarningsCalendar(selectedAccounts, signal),
    staleTime: 30 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: !!user && (activeTab === 'performance' || backgroundFetchLevel >= 2),
  });

  const historySparklineQuery = useQuery({
    queryKey: ['history', user?.username, currency, selectedAccounts, 'sparkline'],
    queryFn: ({ signal }) => fetchHistory(currency, selectedAccounts, '1d', [], '5m', undefined, undefined, signal),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    enabled: !!user && backgroundFetchLevel >= 1,
  });

  const historyWTDQuery = useQuery({
    queryKey: ['history', user?.username, currency, selectedAccounts, 'wtd'],
    queryFn: ({ signal }) => fetchHistory(currency, selectedAccounts, '5d', [], '15m', undefined, undefined, signal),
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

  const perfHistoryQuery = useQuery({
    queryKey: ['perf-history', user?.username, currency, selectedAccounts, benchmarks],
    queryFn: ({ signal }) => fetchHistory(currency, selectedAccounts, '1y', benchmarks, '1d', undefined, undefined, signal),
    placeholderData: keepPreviousData,
    staleTime: 5 * 60 * 1000,
    enabled: !!user && (activeTab === 'asset_change' || activeTab === 'performance' || backgroundFetchLevel >= 1),
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

  const headlineMetrics = headlineQuery.data?.metrics ?? null;
  const headlineFresher = headlineQuery.dataUpdatedAt > (summaryQuery.dataUpdatedAt || 0);
  let cardMetrics: PortfolioSummary['metrics'] = summary?.metrics ?? null;
  if (headlineMetrics && (headlineFresher || !cardMetrics)) {
    cardMetrics = { ...(cardMetrics || {}), ...headlineMetrics } as PortfolioSummary['metrics'];
  }
  const effectiveSummary: PortfolioSummary | undefined = summary
    ? { ...summary, metrics: cardMetrics }
    : (cardMetrics ? { metrics: cardMetrics, account_metrics: null } : undefined);
  const cardLoading = (summaryQuery.isLoading && !summary) && (headlineQuery.isLoading && !headlineMetrics);
  const cardRefreshing = summaryQuery.isFetching || headlineQuery.isFetching;
  const holdings         = holdingsQuery.data || [];
  const transactions     = transactionsQuery.data || [];
  const assetChangeData  = assetChangeQuery.data || null;
  const capitalGainsData = capitalGainsQuery.data || null;
  const dividendData     = dividendsQuery.data || null;
  const availableAccounts = (summary?.metrics?._available_accounts as string[]) || [];
  const closedAccounts = (() => {
    const dates = settingsQuery.data?.account_closure_dates;
    if (!dates) return [];
    const today = new Date().toISOString().slice(0, 10);
    return Object.entries(dates)
      .filter(([, d]) => d && d <= today)
      .map(([acc]) => acc);
  })();
  const graphData        = historyQuery.data || [];
  const graphLoading     = historyQuery.isFetching;
  const indices = (indicesQuery.data && Object.keys(indicesQuery.data).length > 0)
    ? indicesQuery.data
    : (summary?.metrics?.indices as Record<string, unknown> | undefined);

  // ── Tab content ───────────────────────────────────────────────────────────
  const getTabVisible = (tabId: string): string[] => {
    if (tabId === 'performance') return visibleItems;
    return tabLayouts[tabId] ?? TAB_INITIAL_VISIBLE[tabId] ?? [];
  };
  const setTabVisible = (tabId: string, items: string[]) => {
    if (tabId === 'performance') { setVisibleItems(items); return; }
    setTabLayouts(prev => ({ ...prev, [tabId]: items }));
  };
  const activeVisible = getTabVisible(activeTab);

  const renderTabContent = () => {
    switch (activeTab) {
      case 'performance':
        if (!summaryQuery.isLoading && !summaryQuery.data && summaryQuery.isFetched) {
          return <EmptyState onNavigate={handleNavigate} />;
        }
        return (
          <>
            <Dashboard
              summary={effectiveSummary || { metrics: null, account_metrics: null }}
              currency={currency}
              history={historySparklineQuery.data || []}
              wtdHistory={historyWTDQuery.data || []}
              isLoading={cardLoading}
              isRefreshing={cardRefreshing || historySparklineQuery.isFetching || historyWTDQuery.isFetching}
              riskMetrics={riskMetricsQuery.data || {}}
              riskMetricsLoading={riskMetricsQuery.isLoading && !riskMetricsQuery.data}
              portfolioHealth={portfolioHealthQuery.data || null}
              attributionData={attributionQuery.data}
              attributionLoading={attributionQuery.isLoading && !attributionQuery.data}
              dividendEvents={dividendCalendarQuery.data || []}
              earningsEvents={earningsCalendarQuery.data || []}
              longHistory={perfHistoryQuery.data || []}
              holdings={holdings}
              indices={indices}
              visibleItems={visibleItems}
              accounts={selectedAccounts}
              themeColor={currentTheme.color}
              showClosed={showClosed}
              excludeFromAnalytics={['riskMetrics', 'sectorContribution', 'topContributors', 'performanceGraph', 'projection']}
            />
            {visibleItems.includes('performanceGraph') && (
              <PerformanceGraph
                currency={currency}
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
                dayChange={cardMetrics?.day_change_display ?? null}
                dayChangePercent={cardMetrics?.day_change_percent ?? null}
              />
            )}
            {visibleItems.includes('projection') && (
              <ProjectionCard
                data={projectionQuery.data}
                isLoading={projectionQuery.isLoading && !projectionQuery.data}
                isRefreshing={projectionQuery.isFetching}
                currency={currency}
              />
            )}
            {visibleItems.includes('riskMetrics') && (
              <RiskMetrics
                metrics={riskMetricsQuery.data || {}}
                portfolioHealth={portfolioHealthQuery.data || null}
                isLoading={riskMetricsQuery.isLoading && !riskMetricsQuery.data}
                isRefreshing={riskMetricsQuery.isFetching}
              />
            )}
            {(visibleItems.includes('sectorContribution') || visibleItems.includes('topContributors')) && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 md:gap-5 mt-4 md:mt-5">
                {visibleItems.includes('sectorContribution') && (
                  <SectorAttribution
                    data={attributionQuery.data}
                    isLoading={attributionQuery.isLoading && !attributionQuery.data}
                    isRefreshing={attributionQuery.isFetching}
                    currency={currency}
                  />
                )}
                {visibleItems.includes('topContributors') && (
                  <TopContributors
                    data={attributionQuery.data}
                    isLoading={attributionQuery.isLoading && !attributionQuery.data}
                    isRefreshing={attributionQuery.isFetching}
                    currency={currency}
                    accounts={selectedAccounts}
                    showClosed={showClosed}
                  />
                )}
              </div>
            )}
          </>
        );

      case 'watchlist':
        return <Watchlist currency={currency} />;

      case 'screener':
        return null;

      case 'buffett_rank':
        return <BuffettRankView currency={currency} />;

      case 'strategies':
        return <StrategiesView currency={currency} defaultCapital={summary?.metrics?.market_value ?? undefined} />;

      case 'ai_review':
        return <PortfolioAIReview currency={currency} accounts={selectedAccounts} />;

      case 'transactions':
        return <TransactionsTable transactions={transactions} currency={currency} isLoading={transactionsQuery.isPending && !transactionsQuery.data} />;

      case 'markets':
        return !indices ? (
          <p className="text-muted-foreground text-sm">Market data unavailable.</p>
        ) : (
          <MarketsTab
            indices={indices as unknown as Record<string, MarketIndex>}
            onIndexClick={(idx) => { setIndexGraphFocus(idx.name ?? null); setIsIndexGraphModalOpen(true); }}
            holdings={holdings}
            currency={currency}
            portfolioSymbols={holdings.map(h => h.Symbol).filter(Boolean)}
            watchlistSymbols={(watchlistQuery.data || []).map((w) => w.Symbol).filter(Boolean)}
          />
        );

      case 'allocation':
        if (!holdingsQuery.isLoading && holdings.length === 0 && holdingsQuery.isFetched) {
          return <EmptyState onNavigate={handleNavigate} />;
        }
        return (
          <div className="space-y-6">
            {activeVisible.includes('holdingsTable') && (
              <HoldingsTable
                holdings={holdings}
                currency={currency}
                isLoading={holdingsQuery.isLoading && !holdingsQuery.data}
              />
            )}
            <Allocation holdings={holdings} currency={currency} visibleSections={activeVisible} />
          </div>
        );

      case 'asset_change':
        return <AssetChange
          data={assetChangeData}
          currency={currency}
          summary={summary}
          benchmarks={benchmarks}
          riskMetrics={riskMetricsQuery.data ?? null}
          history={perfHistoryQuery.data ?? null}
          historyLoading={perfHistoryQuery.isPending && !perfHistoryQuery.data}
          availableAccounts={availableAccounts}
          accountGroups={settingsQuery.data?.account_groups}
          closedAccounts={closedAccounts}
          attribution={attributionQuery.data ?? null}
          attributionLoading={attributionQuery.isLoading && !attributionQuery.data}
          attributionRefreshing={attributionQuery.isFetching}
          isLoading={assetChangeQuery.isPending && !assetChangeQuery.data}
          visibleSections={activeVisible}
        />;

      case 'capital_gains':
        return (
          <div className="space-y-6 p-4">
            {activeVisible.includes('unrealizedTax') && (
              <UnrealizedTaxView holdings={holdings} currency={currency} />
            )}
            <CapitalGains
              data={capitalGainsData}
              currency={currency}
              onDateRangeChange={(from, to) => setCapitalGainsDates({ from, to })}
              isLoading={capitalGainsQuery.isPending && !capitalGainsQuery.data}
              visibleSections={activeVisible}
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
              visibleSections={activeVisible}
            >
              {activeVisible.includes('incomeProjector') && (
                <IncomeProjector
                  data={incomeProjectionQuery.data || null}
                  isLoading={incomeProjectionQuery.isLoading && !incomeProjectionQuery.data}
                  currency={currency}
                />
              )}
              {activeVisible.includes('dividendCalendar') && (
                <DividendCalendar
                  events={dividendCalendarQuery.data || []}
                  isLoading={dividendCalendarQuery.isLoading && !dividendCalendarQuery.data}
                  currency={currency}
                />
              )}
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
            benchmarks={benchmarks}
            onBenchmarksChange={setBenchmarks}
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

  if (!mounted) return <AppShellSkeleton />;

  const currentTheme = TAB_THEMES[activeTab] || TAB_THEMES.performance;

  return (
    <div className="flex h-screen overflow-hidden bg-background selection:bg-primary/20 selection:text-primary-ink">

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
      <div className="fixed inset-0 z-[-2] bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/5 via-background to-background pointer-events-none" />

      {/* ── Sidebar (desktop) ── */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={handleTabChange}
        user={user}
        onLogout={logout}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(c => !c)}
        onUserClick={handleUserIconClick}
        dayChangePct={cardMetrics?.day_change_pct as number | undefined}
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
          closedAccounts={closedAccounts}
          indices={indices}
          visibleItems={activeVisible}
          onVisibleItemsChange={(items) => setTabVisible(activeTab, items)}
          layoutItems={TAB_LAYOUT_ITEMS[activeTab]}
          layoutSectionTitle={TAB_SECTION_LABELS[activeTab]}
          onCommandPaletteOpen={() => setIsCommandPaletteOpen(true)}
          fxRate={cardMetrics?.exchange_rate_to_display as number | undefined}
          availableCurrencies={settingsQuery.data?.available_currencies}
          isFetching={cardRefreshing}
          onIndexClick={() => { setIndexGraphFocus(null); setIsIndexGraphModalOpen(true); }}
          isMarketOpen={isMarketOpen}
          lastUpdated={Math.max(summaryQuery.dataUpdatedAt || 0, headlineQuery.dataUpdatedAt || 0) ? new Date(Math.max(summaryQuery.dataUpdatedAt || 0, headlineQuery.dataUpdatedAt || 0)) : null}
          onMobileMenuOpen={() => setIsMobileNavOpen(true)}
          marketValue={(cardMetrics?.market_value as number | undefined) ?? null}
          dayChangePct={(cardMetrics?.day_change_percent as number | undefined) ?? null}
          showClosed={showClosed}
          onShowClosedChange={setShowClosed}
        />

        {/* Scrollable content area */}
        <main className="flex-1 overflow-y-auto pb-20 md:pb-8">
          <div className="max-w-[1440px] mx-auto px-4 sm:px-6 py-5 sm:py-6">
            {selectedSymbol ? (
              <StockDetailView
                symbol={selectedSymbol}
                isOpen={true}
                onClose={closeStockDetail}
                onBack={goBack}
                previousViewName={canGoBack ? 'Previous Stock' : (TAB_NAMES[activeTab] || 'Dashboard')}
                currency={currency || modalCurrency}
              />
            ) : (
              <>
                {renderTabContent()}
                <div className={activeTab === 'screener' ? 'block' : 'hidden'}>
                  <ScreenerView currency={currency} />
                </div>
              </>
            )}
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
        currentIndices={indices as unknown as Record<string, MarketIndex> | undefined}
        focusIndex={indexGraphFocus}
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
          <Activity className="w-5 h-5" /><span className="mt-1">Markets</span>
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
