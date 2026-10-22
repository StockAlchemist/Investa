'use client';

import { useState, useEffect, useMemo, useRef } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
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
import { readView, writeView } from '@/lib/view_navigation';
import { portfolioKeys } from '@/lib/query_keys';
import { marketToday } from '@/lib/market_time';
import type { MarketIndex } from '@/components/MarketsTab';
import { INITIAL_VISIBLE_ITEMS, ACCENT_THEME } from '@/lib/dashboard_constants';
import { TAB_LAYOUT_ITEMS, TAB_INITIAL_VISIBLE, TAB_SECTION_LABELS } from '@/lib/layout_registry';
import Dashboard from '@/components/Dashboard';
import { originalDashboardLayout } from '@/lib/dashboard_layout';
import HoldingsTable from '@/components/HoldingsTable';
import { EmptyState } from '@/components/EmptyState';
import AppShellSkeleton from '@/components/skeletons/AppShellSkeleton';
import { Sidebar } from '@/components/layout/Sidebar';
import { PageHeader } from '@/components/layout/PageHeader';
import { MobileTopBar } from '@/components/layout/mobile/MobileTopBar';
import { MobileControlBar } from '@/components/layout/mobile/MobileControlBar';
import { MobileTabBar, isMoreTab } from '@/components/layout/mobile/MobileTabBar';
import { MobileMoreList } from '@/components/layout/mobile/MobileMoreList';
import dynamic from 'next/dynamic';
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
  const { user, logout } = useAuth();
  const { selectedSymbol, modalCurrency, openStockDetail, closeStockDetail, goBack, canGoBack } = useStockModal();

  const [selectedAccounts, setSelectedAccounts]     = useState<string[]>([]);
  const [currency, setCurrency]                     = useState('USD');
  const [activeTab, setActiveTab]                   = useState('performance');
  const [showClosed, setShowClosed]                 = useState(false);
  const [mounted, setMounted]                       = useState(false);
  const [settingsInitialTab, setSettingsInitialTab] = useState<'overrides' | 'account' | undefined>(undefined);
  const [sidebarCollapsed, setSidebarCollapsed]     = useState(false);
  const [isIndexGraphModalOpen, setIsIndexGraphModalOpen] = useState(false);
  const [indexGraphFocus, setIndexGraphFocus]             = useState<string | null>(null);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen]   = useState(false);
  // The phone's More screen. It is a screen, not a drawer: iOS's own More tab
  // lists the sections the tab bar has no room for, and stays selected while
  // you are inside one of them (see MobileTabBar).
  const [isMoreOpen, setIsMoreOpen]                       = useState(false);
  const [benchmarks, setBenchmarks]                 = useState<string[]>(['S&P 500', 'Dow Jones', 'NASDAQ']);
  const [graphPeriod, setGraphPeriod]               = useState('1y');
  const [graphView, setGraphView]                   = useState<'return' | 'value' | 'drawdown'>('return');
  // A year to today, on the market's clock.
  const [graphCustomFromDate, setGraphCustomFromDate] = useState(() => {
    const [y, m, d] = marketToday().split('-');
    return `${Number(y) - 1}-${m}-${d === '29' && m === '02' ? '28' : d}`;
  });
  const [graphCustomToDate, setGraphCustomToDate]   = useState(() => marketToday());
  const [capitalGainsDates, setCapitalGainsDates]   = useState<{ from?: string; to?: string }>({});
  const [visibleItems, setVisibleItems]             = useState<string[]>(INITIAL_VISIBLE_ITEMS);
  const restoredLayout = useRef(false);
  const didHydrateSelection = useRef(false);
  const [tabLayouts, setTabLayouts]                 = useState<Record<string, string[]>>({});

  const handleUserIconClick = () => { closeStockDetail(); setSettingsInitialTab('account'); setActiveTab('settings'); setIsMoreOpen(false); };
  const handleTabChange = (tab: string) => {
    closeStockDetail();
    if (tab === 'settings') setSettingsInitialTab(undefined);
    setActiveTab(tab);
    setIsMoreOpen(false);
  };


  // Hydrate all state from localStorage in one effect
  useEffect(() => {
    // Strict Mode repeats effect setup; rereading our own freshly persisted
    // defaults must not disguise a browser with no saved user layout.
    if (didHydrateSelection.current) return;
    didHydrateSelection.current = true;
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
          setVisibleItems(originalDashboardLayout(p));
          restoredLayout.current = true;
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
      const shared = readView(window.location.search);
      if (shared.section) setActiveTab(shared.section);
      if (shared.currency) setCurrency(shared.currency);
      if (shared.accounts) setSelectedAccounts(shared.accounts);
      if (shared.symbol) openStockDetail(shared.symbol, shared.currency);

    } catch (e) {
      console.error('Failed to hydrate state from localStorage', e);
    } finally {
      setMounted(true);
    }
  }, [openStockDetail]);

  useEffect(() => {
    if (mounted) localStorage.setItem('investa_sidebar_collapsed', String(sidebarCollapsed));
  }, [sidebarCollapsed, mounted]);

  const navigationReady = useRef(false);
  useEffect(() => {
    if (!mounted) return;
    const search = writeView({ section: activeTab, currency, accounts: selectedAccounts, symbol: selectedSymbol }, window.location.search);
    if (window.location.search !== search) {
      const url = `${window.location.pathname}${search}${window.location.hash}`;
      if (navigationReady.current) window.history.pushState(null, '', url);
      else window.history.replaceState(null, '', url);
    }
    navigationReady.current = true;
  }, [mounted, activeTab, currency, selectedAccounts, selectedSymbol]);

  useEffect(() => {
    const restore = () => {
      const shared = readView(window.location.search);
      if (shared.section) setActiveTab(shared.section);
      if (shared.currency) setCurrency(shared.currency);
      setSelectedAccounts(shared.accounts ?? []);
      closeStockDetail();
      if (shared.symbol) openStockDetail(shared.symbol, shared.currency);
      setIsMoreOpen(false);
    };
    window.addEventListener('popstate', restore);
    return () => window.removeEventListener('popstate', restore);
  }, [openStockDetail, closeStockDetail]);

  const queryClient = useQueryClient();

  const settingsQuery = useQuery({
    queryKey: ['settings', user?.username],
    queryFn: fetchSettings,
    staleTime: 5 * 60 * 1000,
    enabled: !!user && mounted,
  });

  const settingsMutation = useMutation({
    mutationFn: updateSettings,
    retry: 2,
    onSuccess: (_, variables) => {
      lastSyncedRef.current = { ...lastSyncedRef.current, ...variables };
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

  // Keep lastSyncedRef updated whenever settingsQuery.data is loaded or refreshed
  useEffect(() => {
    if (!settingsQuery.data) return;
    isInitialSyncDone.current = true;
    const s = settingsQuery.data;
    if (!restoredLayout.current) {
      if (Array.isArray(s.visible_items) && s.visible_items.length > 0) setVisibleItems(originalDashboardLayout(s.visible_items));
      restoredLayout.current = true;
    }
    lastSyncedRef.current = {
      display_currency: s.display_currency,
      show_closed: s.show_closed,
      selected_accounts: s.selected_accounts,
      visible_items: s.visible_items,
      benchmarks: s.benchmarks,
    };
  }, [settingsQuery.data]);

  // Persist UI state to localStorage immediately
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
  }, [mounted, currency, activeTab, showClosed, benchmarks, selectedAccounts, visibleItems, tabLayouts, graphPeriod, graphView]);

  // Debounced sync of persistent user configuration to backend settings
  useEffect(() => {
    if (!mounted || !isInitialSyncDone.current) return;

    const id = setTimeout(() => {
      const updates: Partial<SettingsUpdate> = {};
      const last = lastSyncedRef.current;

      const arrayEqual = (a?: unknown, b?: string[] | null) => {
        const arrA = (Array.isArray(a) ? a : []) as string[];
        const arrB = b || [];
        return arrA.length === arrB.length && arrA.every((v, i) => v === arrB[i]);
      };

      if (last.display_currency !== undefined && last.display_currency !== currency) updates.display_currency = currency;
      if (last.show_closed !== undefined && last.show_closed !== showClosed) updates.show_closed = showClosed;
      if (last.selected_accounts !== undefined && !arrayEqual(last.selected_accounts, selectedAccounts)) updates.selected_accounts = selectedAccounts;
      if (last.visible_items !== undefined && !arrayEqual(last.visible_items, visibleItems) && visibleItems.length > 0) updates.visible_items = visibleItems;
      if (last.benchmarks !== undefined && !arrayEqual(last.benchmarks, benchmarks) && benchmarks.length > 0) updates.benchmarks = benchmarks;

      if (Object.keys(updates).length > 0) {
        mutateSettingsRef.current(updates);
      }
    }, 1500);

    return () => clearTimeout(id);
  }, [mounted, currency, showClosed, benchmarks, selectedAccounts, visibleItems]);

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
    enabled: !!user && mounted,
  });
  const isMarketOpen = marketStatusQuery.data?.is_open ?? false;

  const indicesQuery = useQuery({
    queryKey: ['indices'],
    queryFn: ({ signal }) => fetchIndices(signal),
    staleTime: 60 * 1000,
    refetchInterval: isMarketOpen ? 2 * 60 * 1000 : false,
    enabled: !!user && mounted,
  });

  const headlineQuery = useQuery({
    queryKey: ['headline', user?.username, currency, selectedAccounts],
    queryFn: ({ signal }) => fetchHeadline(currency, selectedAccounts, signal),
    staleTime: 60 * 1000,
    refetchInterval: (query) => query.state.data?.freshness?.refreshing ? 2000 : (isMarketOpen && ['performance', 'allocation', 'markets'].includes(activeTab) ? 60 * 1000 : false),

    enabled: !!user && mounted,
  });

  const summaryQuery = useQuery({
    queryKey: ['summary', user?.username, currency, selectedAccounts, showClosed],
    queryFn: ({ signal }) => fetchSummary(currency, selectedAccounts, showClosed, signal),
    staleTime: 5 * 60 * 1000,
    refetchInterval: (query) => query.state.data?.freshness?.refreshing ? 2000 : (isMarketOpen && ['performance', 'allocation', 'markets'].includes(activeTab) ? 60 * 1000 : false),

    enabled: !!user && mounted,
  });

  const holdingsQuery = useQuery({
    queryKey: portfolioKeys.holdings(user?.username, currency, selectedAccounts, showClosed),
    queryFn: ({ signal }) => fetchHoldings(currency, selectedAccounts, showClosed, signal),
    staleTime: 5 * 60 * 1000,
    // Same cadence as the headline and the summary: every per-stock price on
    // this screen — the holdings table, the day's gainers and losers, the
    // heatmap — comes from here, and without a poll they sat on whatever the
    // session opened with until the tab lost and regained focus.
    refetchInterval: isMarketOpen && ['performance', 'allocation', 'markets'].includes(activeTab) ? 60 * 1000 : false,

    enabled: !!user && mounted,
  });

  const transactionsQuery = useQuery({
    queryKey: ['transactions', user?.username, selectedAccounts],
    queryFn: ({ signal }) => fetchTransactions(selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,

    enabled: !!user && mounted && (activeTab === 'transactions'),
  });

  const assetChangeQuery = useQuery({
    queryKey: ['assetChange', user?.username, currency, selectedAccounts, benchmarks, showClosed],
    queryFn: ({ signal }) => fetchAssetChange(currency, selectedAccounts, benchmarks, showClosed, signal),
    staleTime: 5 * 60 * 1000,

    enabled: !!user && mounted && (activeTab === 'asset_change'),
  });

  const capitalGainsQuery = useQuery({
    queryKey: ['capitalGains', user?.username, currency, selectedAccounts, capitalGainsDates.from, capitalGainsDates.to],
    queryFn: ({ signal }) => fetchCapitalGains(currency, selectedAccounts, capitalGainsDates.from, capitalGainsDates.to, signal),
    staleTime: 5 * 60 * 1000,

    enabled: !!user && mounted && (activeTab === 'capital_gains'),
  });

  const dividendsQuery = useQuery({
    queryKey: ['dividends', user?.username, currency, selectedAccounts],
    queryFn: ({ signal }) => fetchDividends(currency, selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,

    enabled: !!user && mounted && (activeTab === 'dividend'),
  });

  const projectionQuery = useQuery({
    queryKey: ['projection', user?.username, currency, selectedAccounts],
    queryFn: ({ signal }) => fetchProjection(currency, selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,

    enabled: !!user && mounted && (activeTab === 'performance' && visibleItems.includes('projection')),
  });

  const riskMetricsQuery = useQuery({
    queryKey: ['riskMetrics', user?.username, currency, selectedAccounts, showClosed],
    queryFn: ({ signal }) => fetchRiskMetrics(currency, selectedAccounts, showClosed, signal),
    staleTime: 5 * 60 * 1000,

    // The standalone YTD card also consumes /risk_metrics; summary may omit it.
    enabled: !!user && mounted && ((activeTab === 'performance' && visibleItems.some(id => ['riskMetrics', 'ytdReturn'].includes(id))) || activeTab === 'asset_change'),
  });

  const attributionQuery = useQuery({
    queryKey: ['attribution', user?.username, currency, selectedAccounts, showClosed],
    queryFn: ({ signal }) => fetchAttribution(currency, selectedAccounts, false, showClosed, signal),
    staleTime: 5 * 60 * 1000,

    enabled: !!user && mounted && ((activeTab === 'performance' && visibleItems.some(id => ['sectorContribution', 'topContributors'].includes(id))) || activeTab === 'asset_change'),
  });

  const dividendCalendarQuery = useQuery({
    queryKey: ['dividendCalendar', user?.username, currency, selectedAccounts],
    queryFn: ({ signal }) => fetchDividendCalendar(currency, selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,

    enabled: !!user && mounted && ((activeTab === 'performance' && visibleItems.includes('dashboardEvents')) || activeTab === 'dividend'),
  });

  const earningsCalendarQuery = useQuery({
    queryKey: ['earningsCalendar', user?.username, selectedAccounts],
    queryFn: ({ signal }) => fetchEarningsCalendar(selectedAccounts, signal),
    staleTime: 30 * 60 * 1000,

    enabled: !!user && mounted && (activeTab === 'performance' && visibleItems.includes('dashboardEvents')),
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
    queryKey: portfolioKeys.history(user?.username, currency, selectedAccounts, graphPeriod, benchmarks, graphInterval, graphFromDate, graphToDate),
    queryFn: ({ signal }) => fetchHistory(currency, selectedAccounts, graphPeriod, benchmarks, graphInterval, graphFromDate, graphToDate, signal),

    staleTime: 5 * 60 * 1000,
    refetchInterval: isMarketOpen && (graphPeriod === '1d' || graphPeriod === '5d') ? 60000 : false,
    enabled: !!user && mounted && (activeTab === 'performance' && visibleItems.includes('performanceGraph')),
  });

  const perfHistoryQuery = useQuery({
    queryKey: portfolioKeys.history(user?.username, currency, selectedAccounts, '1y', benchmarks, '1d'),
    queryFn: ({ signal }) => fetchHistory(currency, selectedAccounts, '1y', benchmarks, '1d', undefined, undefined, signal),

    staleTime: 5 * 60 * 1000,
    enabled: !!user && mounted && (activeTab === 'asset_change' || (activeTab === 'performance' && visibleItems.includes('portfolioHero'))),
  });

  const historySparklineQuery = useQuery({
    queryKey: portfolioKeys.history(user?.username, currency, selectedAccounts, '1d', [], '5m'),
    queryFn: ({ signal }) => fetchHistory(currency, selectedAccounts, '1d', [], '5m', undefined, undefined, signal),
    staleTime: 5 * 60 * 1000,
    enabled: !!user && mounted && activeTab === 'performance' && visibleItems.includes('portfolioHero'),
  });

  const historyWTDQuery = useQuery({
    queryKey: portfolioKeys.history(user?.username, currency, selectedAccounts, '5d', [], '15m'),
    queryFn: ({ signal }) => fetchHistory(currency, selectedAccounts, '5d', [], '15m', undefined, undefined, signal),
    staleTime: 5 * 60 * 1000,
    enabled: !!user && mounted && activeTab === 'performance' && visibleItems.includes('portfolioHero'),
  });

  const watchlistQuery = useQuery({
    queryKey: ['watchlist', user?.username, currency, 1],
    queryFn: ({ signal }) => fetchWatchlist(currency, 1, signal),
    staleTime: 1 * 60 * 1000,
    enabled: !!user && mounted && (activeTab === 'watchlist' || activeTab === 'markets'),
  });

  const portfolioHealthQuery = useQuery({
    queryKey: ['portfolioHealth', user?.username, currency, selectedAccounts, showClosed],
    queryFn: ({ signal }) => fetchPortfolioHealth(currency, selectedAccounts, showClosed, signal),
    staleTime: 5 * 60 * 1000,

    enabled: !!user && mounted && (activeTab === 'performance' && visibleItems.some(id => ['riskMetrics', 'dashboardInsights'].includes(id))),
  });

  const incomeProjectionQuery = useQuery({
    queryKey: ['incomeProjection', user?.username, currency, selectedAccounts],
    queryFn: ({ signal }) => fetchProjectedIncome(currency, selectedAccounts, signal),
    staleTime: 5 * 60 * 1000,

    enabled: !!user && mounted && (activeTab === 'dividend'),
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
  const cardRefreshing = summaryQuery.isFetching || headlineQuery.isFetching || !!summaryQuery.data?.freshness?.refreshing || !!headlineQuery.data?.freshness?.refreshing;
  const freshness = headlineFresher ? headlineQuery.data?.freshness : summary?.freshness;
  const lastUpdated = freshness?.calculated_at ? new Date(freshness.calculated_at * 1000) : null;
  const holdings         = holdingsQuery.data || [];
  const transactions     = transactionsQuery.data || [];
  const assetChangeData  = assetChangeQuery.data || null;
  const capitalGainsData = capitalGainsQuery.data || null;
  const dividendData     = dividendsQuery.data || null;
  const availableAccounts = useMemo(
    () => (summary?.metrics?._available_accounts as string[]) || [],
    [summary?.metrics?._available_accounts],
  );

  // Drop selected accounts that no longer exist (renamed, merged or split away).
  // The selection is restored verbatim from localStorage, so a rename leaves a
  // filter that matches nothing; the backend then has no transactions to work
  // with and the dashboard reads as empty. _available_accounts is computed from
  // the unfiltered transaction set, so it is authoritative even in that case.
  // Pruning to empty means "all accounts", which is the sane fallback.
  useEffect(() => {
    if (availableAccounts.length === 0 || selectedAccounts.length === 0) return;
    const known = new Set(availableAccounts);
    if (selectedAccounts.every(acc => known.has(acc))) return;
    const pruned = selectedAccounts.filter(acc => known.has(acc));
    console.warn(
      'Dropping unknown accounts from the saved selection:',
      selectedAccounts.filter(acc => !known.has(acc)),
    );
    setSelectedAccounts(pruned);
  }, [availableAccounts, selectedAccounts]);
  const closedAccounts = (() => {
    const dates = settingsQuery.data?.account_closure_dates;
    if (!dates) return [];
    // The market's day, as the backend decides closure (`get_est_today`).
    const today = marketToday();
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
        if (summaryQuery.isError && !summaryQuery.data && !headlineQuery.data?.metrics) {
          return <QueryFailure message="Could not load your portfolio." onRetry={() => void summaryQuery.refetch()} />;
        }
        if (!summaryQuery.isLoading && summaryQuery.data?.metrics === null && summaryQuery.isFetched) {
          return <EmptyState onNavigate={handleNavigate} />;
        }
        return (
          <>
            <Dashboard
              summary={effectiveSummary || { metrics: null, account_metrics: null }}
              currency={currency}
              history={historySparklineQuery.data || []}
              historyLoading={historySparklineQuery.isFetching && !historySparklineQuery.data}
              isMarketOpen={isMarketOpen}
              valuationTime={lastUpdated?.getTime() ?? (summaryQuery.dataUpdatedAt || headlineQuery.dataUpdatedAt)}
              wtdHistory={historyWTDQuery.data || []}
              longHistory={perfHistoryQuery.data || []}
              isLoading={cardLoading}
              isError={summaryQuery.isError || headlineQuery.isError}
              onRetry={() => void queryClient.refetchQueries({ queryKey: ['summary'] })}
              isRefreshing={cardRefreshing}
              riskMetrics={riskMetricsQuery.data || {}}
              riskMetricsLoading={riskMetricsQuery.isLoading && !riskMetricsQuery.data}
              portfolioHealth={portfolioHealthQuery.data || null}
              attributionData={attributionQuery.data}
              attributionLoading={attributionQuery.isLoading && !attributionQuery.data}
              dividendEvents={dividendCalendarQuery.data || []}
              earningsEvents={earningsCalendarQuery.data || []}
              holdings={holdings}
              indices={indices}
              visibleItems={visibleItems}
              accounts={selectedAccounts}
              themeColor={ACCENT_THEME.color}
              showClosed={showClosed}
              excludeFromAnalytics={['riskMetrics', 'sectorContribution', 'topContributors', 'performanceGraph', 'projection']}
            />
            {visibleItems.includes('performanceGraph') && (
              <PerformanceGraph currency={currency} period={graphPeriod} onPeriodChange={setGraphPeriod}
                view={graphView} onViewChange={setGraphView} data={graphData} loading={graphLoading}
                customFromDate={graphCustomFromDate} onCustomFromDateChange={setGraphCustomFromDate}
                customToDate={graphCustomToDate} onCustomToDateChange={setGraphCustomToDate}
                dayChange={cardMetrics?.day_change_display ?? null} dayChangePercent={cardMetrics?.day_change_percent ?? null} />
            )}
            <button type="button" className="my-4 rounded-control border border-border px-4 py-2 text-sm"
              aria-expanded={visibleItems.some(id => ['projection', 'riskMetrics', 'sectorContribution', 'topContributors'].includes(id))}
              onClick={() => {
                const detailed = ['projection', 'riskMetrics', 'sectorContribution', 'topContributors'];
                const expanded = visibleItems.some(id => detailed.includes(id));
                setVisibleItems(expanded ? visibleItems.filter(id => !detailed.includes(id)) : [...new Set([...visibleItems, ...detailed])]);
              }}>
              {visibleItems.some(id => ['projection', 'riskMetrics', 'sectorContribution', 'topContributors'].includes(id)) ? 'Hide detailed analytics' : 'Show detailed analytics'}
            </button>
            {visibleItems.includes('projection') && (
              <ProjectionCard
                data={projectionQuery.data}
                isLoading={projectionQuery.isLoading && !projectionQuery.data}
                isRefreshing={projectionQuery.isFetching}
                currency={currency}
                accounts={selectedAccounts}
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
        if (holdingsQuery.isError && !holdingsQuery.data) {
          return <QueryFailure message="Could not load your holdings." onRetry={() => void holdingsQuery.refetch()} />;
        }
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
          <div className="space-y-6">
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

  if (!mounted) return <AppShellSkeleton />;

  return (
    <div className="flex h-screen overflow-hidden bg-background selection:bg-primary/20 selection:text-primary-ink">

      {/* Ledger has no ambient glow or gradient wash: the paper ground is the
          background. */}

      {/* ── Sidebar (desktop) ── */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={handleTabChange}
        user={user}
        onLogout={logout}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(c => !c)}
        onUserClick={handleUserIconClick}
        onSearchClick={() => setIsCommandPaletteOpen(true)}
        dayChangePct={cardMetrics?.day_change_percent as number | undefined}
      />

      {/* ── Main content ── */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">

        {/* ── Phone shell: navigation bar + control bar (see macos_app
             MainView.phoneTabContent, which stacks exactly these two above the
             section and a Divider between them) ── */}
        <MobileTopBar
          indices={indices}
          onIndexClick={() => { setIndexGraphFocus(null); setIsIndexGraphModalOpen(true); }}
        />
        <MobileControlBar
          currency={currency}
          onCurrencyChange={setCurrency}
          availableCurrencies={settingsQuery.data?.available_currencies}
          availableAccounts={availableAccounts}
          selectedAccounts={selectedAccounts}
          onAccountsChange={setSelectedAccounts}
          accountGroups={settingsQuery.data?.account_groups}
          accountGroupOrder={settingsQuery.data?.account_group_order}
          closedAccounts={closedAccounts}
          showClosed={showClosed}
          onShowClosedChange={setShowClosed}
          layoutItems={TAB_LAYOUT_ITEMS[activeTab]}
          layoutSectionTitle={TAB_SECTION_LABELS[activeTab]}
          visibleItems={activeVisible}
          onVisibleItemsChange={(items) => setTabVisible(activeTab, items)}
          onResetLayout={() => setTabVisible(activeTab, [...(TAB_INITIAL_VISIBLE[activeTab] ?? [])])}
          isMarketOpen={isMarketOpen}
          lastUpdated={lastUpdated}
          isRefreshing={cardRefreshing}
          onRefresh={() => { void queryClient.refetchQueries({ type: 'active' }); }}
          onOpenSettings={() => handleTabChange('settings')}
        />

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
          onResetLayout={() => setTabVisible(activeTab, [...(TAB_INITIAL_VISIBLE[activeTab] ?? [])])}
          layoutItems={TAB_LAYOUT_ITEMS[activeTab]}
          layoutSectionTitle={TAB_SECTION_LABELS[activeTab]}
          onCommandPaletteOpen={() => setIsCommandPaletteOpen(true)}
          fxRate={cardMetrics?.exchange_rate_to_display as number | undefined}
          availableCurrencies={settingsQuery.data?.available_currencies}
          isFetching={cardRefreshing}
          onIndexClick={() => { setIndexGraphFocus(null); setIsIndexGraphModalOpen(true); }}
          isMarketOpen={isMarketOpen}
          lastUpdated={lastUpdated}
          marketValue={(cardMetrics?.market_value as number | undefined) ?? null}
          dayChangePct={(cardMetrics?.day_change_percent as number | undefined) ?? null}
          showClosed={showClosed}
          onShowClosedChange={setShowClosed}
        />

        {/* Scrollable content area. The phone reserves the tab bar's height at
            the bottom (49pt + the home indicator), the way a `TabView` insets
            its content's safe area. */}
        <main
          className="flex-1 overflow-y-auto md:pb-8"
          style={{ paddingBottom: 'calc(3.25rem + env(safe-area-inset-bottom, 0px))' }}
        >
          <div className="max-w-[1440px] mx-auto px-4 sm:px-6 py-5 sm:py-6">
            {selectedSymbol ? (
              <StockDetailView
                symbol={selectedSymbol}
                isOpen={true}
                onClose={closeStockDetail}
                onBack={goBack}
                previousViewName={canGoBack ? 'Previous Stock' : (TAB_NAMES[activeTab] || 'Dashboard')}
                currency={currency || modalCurrency}
                accounts={selectedAccounts}
                showClosed={showClosed}
              />
            ) : isMoreOpen ? (
              <div className="md:hidden">
                <MobileMoreList
                  activeTab={activeTab}
                  onTabChange={handleTabChange}
                  user={user}
                  onLogout={logout}
                />
              </div>
            ) : (
              <>
                {settingsMutation.isError && <QueryFailure message="Could not save your preferences." onRetry={() => {
                  if (settingsMutation.variables) settingsMutation.mutate(settingsMutation.variables);
                }} />}
                {renderTabContent()}
                {activeTab === 'screener' && <ScreenerView currency={currency} />}
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

      {/* ── Phone tab bar ── */}
      <MobileTabBar
        activeTab={activeTab}
        onTabChange={(tab) => { handleTabChange(tab); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
        onMoreClick={() => { closeStockDetail(); setIsMoreOpen(true); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
        moreActive={isMoreOpen || isMoreTab(activeTab)}
      />
    </div>
  );
}

function QueryFailure({ message, onRetry }: { message: string; onRetry: () => void }) {
  return <div role="alert" className="rounded-xl border border-border bg-card p-6">
    <p>{message}</p><button className="mt-3 rounded-control bg-primary px-4 py-2 text-primary-foreground" onClick={onRetry}>Retry</button>
  </div>;
}
