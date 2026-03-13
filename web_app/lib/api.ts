import { apiClient } from '../src/api/client';

const getApiBaseUrl = () => {
    if (process.env.NEXT_PUBLIC_API_URL) {
        return process.env.NEXT_PUBLIC_API_URL;
    }
    if (typeof window !== 'undefined') {
        // If serving via Tailscale (HTTPS/proxy), use relative path
        if (window.location.hostname.endsWith('ts.net')) {
            return '/api';
        }
        // Dynamically use the current hostname (e.g., 100.66.59.98) but port 8000
        return `http://${window.location.hostname}:8000/api`;
    }
    return 'http://localhost:8000/api';
};

export const API_BASE_URL = getApiBaseUrl();

export async function fetchCurrentUser(token: string): Promise<User> {
    const res = await fetch(`${API_BASE_URL}/auth/me`, {
        headers: {
            Authorization: `Bearer ${token}`,
        },
    });
    if (!res.ok) throw new Error('Failed to fetch user');
    return res.json();
}

export async function updateUserProfile(data: { alias: string }): Promise<User> {
    const res = await authFetch(`${API_BASE_URL}/auth/me`, {
        method: 'PATCH',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Failed to update user profile');
    return res.json();
}

export interface User {
    id: number;
    username: string;
    alias?: string;
    is_active: boolean;
    created_at: string;
}

const getAuthHeaders = () => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
    return token ? { 'Authorization': `Bearer ${token}` } : {};
};

export async function deleteUser(): Promise<StatusResponse> {
    const res = await authFetch(`${API_BASE_URL}/auth/me`, {
        method: "DELETE",
    });
    if (!res.ok) throw new Error('Failed to delete user');
    return res.json();
}

export async function changePassword(currentPassword: string, newPassword: string): Promise<StatusResponse> {
    const res = await authFetch(`${API_BASE_URL}/auth/change-password`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to change password');
    }
    return res.json();
}

const authFetch = async (url: string, options: RequestInit = {}) => {
    const headers = {
        ...getAuthHeaders(),
        ...(options.headers || {}),
    } as HeadersInit;

    // Auto-redirect on 401?
    // If we receive 401, we should probably clear token and redirect.
    // However, api.ts is not a react component. 
    // We can rely on the UI components (React Query) to handle errors, or dispatch a custom event.
    // For now, let's just pass the 401 through, and AuthContext or specific components will handle it.

    return fetch(url, { ...options, headers });
};

export interface PortfolioSummary {
    metrics: {
        market_value: number;
        day_change_display: number;
        day_change_percent: number;
        unrealized_gain: number;
        realized_gain: number;
        total_gain: number;
        total_return_pct: number;
        dividends: number;
        commissions: number;
        fx_gain_loss_display?: number;
        fx_gain_loss_pct?: number;
        annualized_twr?: number;
        cumulative_twr?: number;
        portfolio_mwr?: number; // Added Money-Weighted Return (IRR)
        cash_balance?: number; // Might not be directly in metrics, check account_metrics for Cash
        exchange_rate_to_display?: number;
        indices?: Record<string, {
            price: number;
            change: number;
            changesPercentage: number;
            name: string;
            [key: string]: unknown;
        }>;
        [key: string]: unknown;
    } | null;
    account_metrics: Record<string, unknown> | null;
    holdings_dict?: Record<string, any>;
}

export interface Lot {
    Date: string;
    Quantity: number;
    "Cost Basis": number;
    "Market Value": number;
    "Unreal. Gain": number;
    "Unreal. Gain %": number;
    [key: string]: unknown;
}

export interface Holding {
    Symbol: string;
    Quantity: number;
    Account?: string;
    Sector?: string;
    Industry?: string;
    "Day Change %"?: number;
    "Unreal. Gain %"?: number;
    "Total Return %"?: number;
    "IRR (%)"?: number;
    Country?: string;
    quoteType?: string;
    // Keys are dynamic based on currency, e.g., "Market Value (USD)"
    [key: string]: unknown;
    lots?: Lot[];
    sparkline_7d?: number[];
}

export interface Transaction {
    id?: number;
    Date: string;
    Account: string;
    Symbol: string;
    Type: string;
    Quantity: number;
    "Price/Share": number;
    Commission: number;
    "Total Amount": number;
    "Local Currency": string;
    "Split Ratio"?: number;
    Note?: string;
    "To Account"?: string;
    "Auto-add Cash"?: boolean;
    [key: string]: unknown;
}

export async function fetchSummary(currency: string = 'USD', accounts?: string[], showClosed?: boolean, signal?: AbortSignal): Promise<PortfolioSummary> {
    const { data, error } = await apiClient.GET("/api/summary", {
        params: {
            query: { currency, accounts: accounts || undefined, show_closed: showClosed }
        },
        signal,
        cache: 'no-store'
    });
    if (error) throw new Error('Failed to fetch summary');
    return data as unknown as PortfolioSummary;
}

export async function fetchMarketStatus(): Promise<{ is_open: boolean }> {
    const res = await authFetch(`${API_BASE_URL}/market_status`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch market status');
    return res.json();
}

export async function fetchHoldings(currency: string = 'USD', accounts?: string[], showClosed: boolean = false, signal?: AbortSignal): Promise<Holding[]> {
    const { data, error } = await apiClient.GET("/api/holdings", {
        params: {
            query: { currency, accounts: accounts || undefined, show_closed: showClosed }
        },
        signal,
        cache: 'no-store'
    });
    if (error) throw new Error('Failed to fetch holdings');
    return data as unknown as Holding[];
}

export interface PerformanceData {
    date: string;
    value: number;
    twr: number;
    drawdown?: number;
    abs_gain?: number;
    abs_roi?: number;
    cum_flow?: number;
    [key: string]: number | string | undefined; // Allow dynamic keys for benchmarks
}

export async function fetchTransactions(accounts?: string[], signal?: AbortSignal): Promise<Transaction[]> {
    const { data, error } = await apiClient.GET("/api/transactions", {
        params: {
            query: { accounts: accounts || undefined }
        },
        signal,
        cache: 'no-store'
    });
    if (error) throw new Error('Failed to fetch transactions');
    return data as unknown as Transaction[];
}

export interface StatusResponse {
    status: string;
    message?: string;
    id?: number;
    [key: string]: unknown;
}

export async function addTransaction(transaction: Transaction): Promise<StatusResponse> {
    const response = await authFetch(`${API_BASE_URL}/transactions`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(transaction),
    });
    if (!response.ok) {
        throw new Error(`Failed to add transaction: ${response.statusText}`);
    }
    return response.json();
}

export async function importIBKRPdf(file: File, autoAddCash: boolean = false): Promise<{ status: string, count: number, message: string }> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await authFetch(`${API_BASE_URL}/transactions/import_pdf?auto_add_cash=${autoAddCash}`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`Failed to import PDF: ${response.statusText}`);
    }
    return response.json();
}

export async function updateTransaction(id: number, transaction: Transaction): Promise<StatusResponse> {
    const response = await authFetch(`${API_BASE_URL}/transactions/${id}`, {
        method: "PUT",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(transaction),
    });
    if (!response.ok) {
        throw new Error(`Failed to update transaction: ${response.statusText}`);
    }
    return response.json();
}

export async function deleteTransaction(id: number): Promise<StatusResponse> {
    const response = await authFetch(`${API_BASE_URL}/transactions/${id}`, {
        method: "DELETE",
    });
    if (!response.ok) {
        throw new Error(`Failed to delete transaction: ${response.statusText}`);
    }
    return response.json();
}

export async function fetchHistory(
    currency: string = 'USD',
    accounts?: string[],
    period: string = '1y',
    benchmarks?: string[],
    interval: string = '1d',
    fromDate?: string,
    toDate?: string,
    signal?: AbortSignal
): Promise<PerformanceData[]> {
    const { data, error } = await apiClient.GET("/api/history", {
        params: {
            query: {
                currency,
                period,
                interval,
                accounts: accounts || undefined,
                benchmarks: benchmarks || undefined,
                from: fromDate,
                to: toDate
            }
        },
        signal,
        cache: 'no-store'
    });
    if (error) throw new Error('Failed to fetch history');
    return data as unknown as PerformanceData[];
}

export async function fetchMarketHistory(
    benchmarks: string[],
    period: string = '1y',
    interval: string = '1d',
    currency: string = 'USD',
    signal?: AbortSignal
): Promise<any[]> {
    const { data, error } = await apiClient.GET("/api/market_history", {
        params: {
            query: { period, interval, currency, benchmarks }
        },
        signal,
        cache: 'no-store'
    });
    if (error) throw new Error('Failed to fetch market history');
    return data as any[];
}

export interface StockHistoryData {
    date: string;
    value: number;
    volume: number;
    return_pct: number;
    [key: string]: number | string | undefined; // For benchmarks
}

export async function fetchStockHistory(
    symbol: string,
    period: string = '1y',
    interval: string = '1d',
    benchmarks?: string[],
    signal?: AbortSignal
): Promise<StockHistoryData[]> {
    const { data, error } = await apiClient.GET("/api/stock_history/{symbol}", {
        params: {
            path: { symbol },
            query: { period, interval, benchmarks: benchmarks || undefined }
        },
        signal,
        cache: 'no-store'
    });
    if (error) throw new Error('Failed to fetch stock history');
    return data as unknown as StockHistoryData[];
}

export interface AssetChangeData {
    [period: string]: {
        Date: string;
        [key: string]: unknown;
    }[];
}

export async function fetchAssetChange(
    currency: string = 'USD',
    accounts?: string[],
    benchmarks?: string[],
    showClosed?: boolean,
    signal?: AbortSignal
): Promise<AssetChangeData> {
    const { data, error } = await apiClient.GET("/api/asset_change", {
        params: {
            query: { currency, accounts: accounts || undefined, benchmarks: benchmarks || undefined, show_closed: showClosed }
        },
        signal,
        cache: 'no-store'
    });
    if (error) throw new Error('Failed to fetch asset change data');
    return data as unknown as AssetChangeData;
}

export interface CapitalGain {
    Date: string;
    Symbol: string;
    Account: string;
    Type: string;
    Quantity: number;
    "Avg Sale Price (Local)": number;
    "Total Proceeds (Local)": number;
    "Total Cost Basis (Local)": number;
    "Realized Gain (Local)": number;
    "Sale/Cover FX Rate": number;
    "Total Proceeds (Display)": number;
    "Total Cost Basis (Display)": number;
    "Realized Gain (Display)": number;
    LocalCurrency: string;
    original_tx_id: number;
    [key: string]: unknown;
}

export async function fetchCapitalGains(
    currency: string = 'USD',
    accounts?: string[],
    fromDate?: string,
    toDate?: string,
    signal?: AbortSignal
): Promise<CapitalGain[]> {
    const { data, error } = await apiClient.GET("/api/capital_gains", {
        params: {
            query: { currency, accounts: accounts || undefined, from: fromDate, to: toDate }
        },
        signal,
        cache: 'no-store'
    });
    if (error) throw new Error('Failed to fetch capital gains');
    return data as unknown as CapitalGain[];
}

export interface Dividend {
    Date: string;
    Symbol: string;
    Account: string;
    LocalCurrency: string;
    DividendAmountLocal: number;
    FXRateUsed: number;
    DividendAmountDisplayCurrency: number;
    [key: string]: unknown;
}

export async function fetchDividends(
    currency: string = 'USD',
    accounts?: string[],
    signal?: AbortSignal
): Promise<Dividend[]> {
    const { data, error } = await apiClient.GET("/api/dividends", {
        params: {
            query: { currency, accounts: accounts || undefined }
        },
        signal,
        cache: 'no-store'
    });
    if (error) throw new Error('Failed to fetch dividends');
    return data as unknown as Dividend[];
}
export interface ManualOverrideData {
    price: number;
    currency?: string;
    asset_type?: string;
    sector?: string;
    geography?: string;
    industry?: string;
    exchange?: string;
}

export type ManualOverride = number | ManualOverrideData;

export interface Settings {
    manual_overrides: Record<string, ManualOverride>;
    user_symbol_map: Record<string, string>;
    user_excluded_symbols: string[];
    account_currency_map: Record<string, string>;
    account_groups: Record<string, string[]>;
    available_currencies: string[];
    account_interest_rates: Record<string, number>;
    interest_free_thresholds: Record<string, number>;
    valuation_overrides: Record<string, any>;
    visible_items?: string[];
    benchmarks?: string[];
    show_closed?: boolean;
    display_currency?: string;
    selected_accounts?: string[];
    active_tab?: string;
    ibkr_token?: string;
    ibkr_query_id?: string;
}

export async function fetchSettings(): Promise<Settings> {
    const { data, error } = await apiClient.GET("/api/settings");
    if (error) throw new Error('Failed to fetch settings');
    return data as unknown as Settings;
}

export interface SettingsUpdate {
    manual_price_overrides?: Record<string, ManualOverride>;
    user_symbol_map?: Record<string, string>;
    user_excluded_symbols?: string[];
    account_groups?: Record<string, string[]>;
    account_currency_map?: Record<string, string>;
    available_currencies?: string[];
    account_interest_rates?: Record<string, number>;
    interest_free_thresholds?: Record<string, number>;
    valuation_overrides?: Record<string, any>;
    visible_items?: string[];
    benchmarks?: string[];
    show_closed?: boolean;
    display_currency?: string;
    selected_accounts?: string[];
    active_tab?: string;
    ibkr_token?: string;
    ibkr_query_id?: string;
}

export async function updateSettings(settings: SettingsUpdate): Promise<StatusResponse> {
    const { data, error } = await apiClient.POST("/api/settings/update", {
        body: settings as any
    });
    if (error) throw new Error(`Failed to update settings`);
    return data as unknown as StatusResponse;
}


export interface RiskMetrics {
    'Max Drawdown'?: number;
    'Volatility (Ann.)'?: number;
    'Sharpe Ratio'?: number;
    'Sortino Ratio'?: number;
}

export async function fetchRiskMetrics(currency: string = 'USD', accounts?: string[], showClosed?: boolean, signal?: AbortSignal): Promise<RiskMetrics> {
    const { data, error } = await apiClient.GET("/api/risk_metrics", {
        params: {
            query: { currency, accounts: accounts || undefined, show_closed: showClosed }
        },
        signal,
        cache: 'no-store'
    });
    if (error) throw new Error('Failed to fetch risk metrics');
    return data as unknown as RiskMetrics;
}

export interface AttributionData {
    sectors: {
        sector: string;
        gain: number;
        value: number;
        contribution: number;
    }[];
    stocks: {
        symbol: string;
        name: string;
        gain: number;
        value: number;
        sector: string;
        contribution: number;
    }[];
    total_gain: number;
}

export async function fetchAttribution(currency: string = 'USD', accounts?: string[], showAll: boolean = false, showClosed?: boolean, signal?: AbortSignal): Promise<AttributionData> {
    const { data, error } = await apiClient.GET("/api/attribution", {
        params: {
            query: { currency, accounts: accounts || undefined, show_all: showAll, show_closed: showClosed }
        },
        signal,
        cache: 'no-store'
    });
    if (error) throw new Error('Failed to fetch attribution');
    return data as unknown as AttributionData;
}

export interface DividendEvent {
    symbol: string;
    dividend_date: string;
    ex_dividend_date: string;
    amount: number;
    status: 'confirmed' | 'estimated'; // Added status
}

export async function fetchDividendCalendar(accounts?: string[], signal?: AbortSignal): Promise<DividendEvent[]> {
    const { data, error } = await apiClient.GET("/api/dividend_calendar", {
        params: {
            query: { accounts: accounts || undefined, _t: Date.now().toString() as any }
        },
        signal,
        cache: 'no-store'
    });
    if (error) throw new Error('Failed to fetch dividend calendar');
    return data as unknown as DividendEvent[];
}

export async function saveManualOverride(symbol: string, price: number | null): Promise<StatusResponse> {
    const response = await authFetch(`${API_BASE_URL}/settings/manual_overrides`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ symbol, price }),
    });
    if (!response.ok) {
        throw new Error(`Failed to save manual override: ${response.statusText}`);
    }
    return response.json();
}

export async function triggerRefresh(secret: string): Promise<StatusResponse> {
    const response = await authFetch(`${API_BASE_URL}/webhook/refresh`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ secret }),
    });
    if (!response.ok) {
        throw new Error(`Failed to trigger refresh: ${response.statusText}`);
    }
    return response.json();
}

export async function syncIbkr(): Promise<StatusResponse> {
    const response = await authFetch(`${API_BASE_URL}/sync/ibkr`, {
        method: "POST"
    });
    if (!response.ok) {
        const err = await response.json();
        throw new Error(err.message || err.detail || 'Failed to sync IBKR');
    }
    return response.json();
}

export async function fetchPendingIbkr(): Promise<Transaction[]> {
    const res = await authFetch(`${API_BASE_URL}/sync/ibkr/pending`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch pending transactions');
    return res.json();
}

export async function approveIbkr(ids: number[]): Promise<StatusResponse> {
    const response = await authFetch(`${API_BASE_URL}/sync/ibkr/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(ids),
    });
    if (!response.ok) throw new Error('Failed to approve transactions');
    return response.json();
}

export async function rejectIbkr(ids: number[]): Promise<StatusResponse> {
    const response = await authFetch(`${API_BASE_URL}/sync/ibkr/reject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(ids),
    });
    if (!response.ok) throw new Error('Failed to reject transactions');
    return response.json();
}

export interface ProjectedIncome {
    month: string;
    value: number;
    year_month: string;
    [key: string]: number | string; // Allow dynamic keys for stacked bar breakdown
}

export async function fetchProjectedIncome(
    currency: string = 'USD',
    accounts?: string[],
    signal?: AbortSignal
): Promise<ProjectedIncome[]> {
    const { data, error } = await apiClient.GET("/api/projected_income", {
        params: {
            query: { currency, accounts: accounts || undefined }
        },
        signal,
        cache: 'no-store'
    });
    if (error) throw new Error('Failed to fetch projected income');
    return data as unknown as ProjectedIncome[];
}

export interface HealthComponent {
    score: number;
    metric: number | string;
    label: string;
}

export interface PortfolioHealth {
    overall_score: number;
    rating: string;
    debug_error?: string;
    components: {
        diversification: HealthComponent;
        efficiency: HealthComponent;
        stability: HealthComponent;
    };
}

export async function fetchPortfolioHealth(
    currency: string = 'USD',
    accounts?: string[],
    showClosed?: boolean,
    signal?: AbortSignal
): Promise<PortfolioHealth | null> {
    const { data, error } = await apiClient.GET("/api/portfolio_health", {
        params: {
            query: { currency, accounts: accounts || undefined, show_closed: showClosed }
        },
        signal,
        cache: 'no-store'
    });
    if (error) {
        console.error("Failed to fetch portfolio health");
        return null;
    }
    return data as unknown as PortfolioHealth;
}

export interface WatchlistItem {
    Symbol: string;
    Note: string;
    AddedOn: string;
    Price: number | null;
    "Day Change": number | null;
    "Day Change %": number | null;
    Name: string | null;
    Currency: string | null;
    Sparkline: number[];
    "Market Cap"?: number | null;
    "PE Ratio"?: number | null;
    "Dividend Yield"?: number | null;
}


export interface WatchlistMeta {
    id: number;
    name: string;
    created_at: string;
}

export async function getWatchlists(signal?: AbortSignal): Promise<WatchlistMeta[]> {
    const { data, error } = await apiClient.GET("/api/watchlists", { signal, cache: 'no-store' });
    if (error) throw new Error('Failed to fetch watchlists');
    return data as unknown as WatchlistMeta[];
}

export async function createWatchlist(name: string): Promise<WatchlistMeta> {
    const { data, error } = await apiClient.POST("/api/watchlists", {
        body: { name } as any
    });
    if (error) throw new Error('Failed to create watchlist');
    return data as unknown as WatchlistMeta;
}

export async function renameWatchlist(id: number, name: string): Promise<StatusResponse> {
    const { data, error } = await apiClient.PUT("/api/watchlists/{watchlist_id}", {
        params: { path: { watchlist_id: id as unknown as string } as any },
        body: { name } as any
    });
    if (error) throw new Error('Failed to rename watchlist');
    return data as unknown as StatusResponse;
}

export async function deleteWatchlist(id: number): Promise<StatusResponse> {
    const { data, error } = await apiClient.DELETE("/api/watchlists/{watchlist_id}", {
        params: { path: { watchlist_id: id as unknown as string } as any }
    });
    if (error) throw new Error('Failed to delete watchlist');
    return data as unknown as StatusResponse;
}

export async function fetchWatchlist(currency: string = 'USD', watchlistId: number = 1, signal?: AbortSignal): Promise<WatchlistItem[]> {
    const { data, error } = await apiClient.GET("/api/watchlist", {
        params: {
            query: { currency, id: watchlistId } as any
        },
        signal,
        cache: 'no-store'
    });
    if (error) throw new Error('Failed to fetch watchlist');
    return data as unknown as WatchlistItem[];
}

export async function addToWatchlist(symbol: string, note: string = "", watchlistId: number = 1): Promise<StatusResponse> {
    const { data, error } = await apiClient.POST("/api/watchlist", {
        body: { symbol, note, watchlist_id: watchlistId } as any
    });
    if (error) {
        throw new Error(`Failed to add to watchlist`);
    }
    return data as unknown as StatusResponse;
}

export async function removeFromWatchlist(symbol: string, watchlistId: number = 1): Promise<StatusResponse> {
    const { data, error } = await apiClient.DELETE("/api/watchlist/{symbol}", {
        params: {
            path: { symbol },
            query: { id: watchlistId } as any
        }
    });
    if (error) {
        throw new Error(`Failed to remove from watchlist`);
    }
    return data as unknown as StatusResponse;
}

export async function updateHoldingTags(account: string, symbol: string, tags: string): Promise<StatusResponse> {
    const { data, error } = await apiClient.POST("/api/holdings/update_tags", {
        body: { account, symbol, tags }
    });
    if (error) {
        throw new Error('Failed to update holding tags');
    }
    return data as unknown as StatusResponse;
}

// --- Fundamentals, Financials, and Ratios ---

export interface Fundamentals {
    symbol: string;
    longName?: string;
    shortName?: string;
    longBusinessSummary?: string;
    website?: string;
    sector?: string;
    industry?: string;
    marketCap?: number;
    trailingPE?: number;
    forwardPE?: number;
    dividendYield?: number;
    beta?: number;
    fiftyTwoWeekHigh?: number;
    fiftyTwoWeekLow?: number;
    averageVolume?: number;
    regularMarketPrice?: number;
    currency?: string;
    exchange?: string;
    netExpenseRatio?: number;
    etf_data?: {
        top_holdings: { symbol: string; name: string; percent: number }[];
        sector_weightings: Record<string, number>;
        asset_classes: Record<string, number>;
    };
    [key: string]: any;
}

export interface FinancialStatement {
    columns: string[];
    index: string[];
    data: (number | null)[][];
}

export interface FinancialsResponse {
    financials: FinancialStatement;
    balance_sheet: FinancialStatement;
    cashflow: FinancialStatement;
    shareholders_equity?: FinancialStatement;
}

export interface FinancialRatio {
    Period: string;
    [key: string]: number | string | null;
}

export interface RatiosResponse {
    historical: FinancialRatio[];
    valuation: Record<string, number | null>;
}

export interface IntrinsicValueModel {
    intrinsic_value?: number;
    error?: string;
    model: string;
    parameters: Record<string, any>;
    mc?: {
        bear: number;
        base: number;
        bull: number;
        std_dev: number;
        histogram?: { price: number; count: number }[];
    };
}

export interface IntrinsicValueResponse {
    current_price: number | null;
    models: {
        dcf: IntrinsicValueModel;
        graham: IntrinsicValueModel;
    };
    average_intrinsic_value?: number;
    range?: {
        bear: number;
        bull: number;
    };
    margin_of_safety_pct?: number;
    valuation_note?: string;
}

export async function fetchFundamentals(symbol: string, force: boolean = false): Promise<Fundamentals> {
    const params = new URLSearchParams();
    if (force) params.append('force', 'true');
    const res = await authFetch(`${API_BASE_URL}/fundamentals/${symbol}?${params.toString()}`);
    if (!res.ok) throw new Error(`Failed to fetch fundamentals for ${symbol}`);
    return res.json();
}

export async function fetchFinancials(symbol: string, periodType: 'annual' | 'quarterly' = 'annual', force: boolean = false): Promise<FinancialsResponse> {
    const params = new URLSearchParams({ period_type: periodType });
    if (force) params.append('force', 'true');
    const res = await authFetch(`${API_BASE_URL}/financials/${symbol}?${params.toString()}`);
    if (!res.ok) throw new Error(`Failed to fetch financials for ${symbol}`);
    return res.json();
}

export async function fetchRatios(symbol: string, force: boolean = false): Promise<RatiosResponse> {
    const params = new URLSearchParams();
    if (force) params.append('force', 'true');
    const res = await authFetch(`${API_BASE_URL}/ratios/${symbol}?${params.toString()}`);
    if (!res.ok) throw new Error(`Failed to fetch ratios for ${symbol}`);
    return res.json();
}

export async function fetchIntrinsicValue(symbol: string, force: boolean = false): Promise<IntrinsicValueResponse> {
    const params = new URLSearchParams();
    if (force) params.append('force', 'true');
    const res = await authFetch(`${API_BASE_URL}/intrinsic_value/${symbol}?${params.toString()}`);
    if (!res.ok) throw new Error(`Failed to fetch intrinsic value for ${symbol}`);
    return res.json();
}

export interface StockAnalysisResponse {
    scorecard?: {
        moat: number;
        financial_strength: number;
        predictability: number;
        growth: number;
    };
    analysis?: {
        moat: string;
        financial_strength: string;
        predictability: string;
        growth_perspective: string;
    };
    summary?: string;
    error?: string;
}

export async function fetchStockAnalysis(symbol: string, force: boolean = false): Promise<StockAnalysisResponse> {
    const { data, error } = await apiClient.GET("/api/stock-analysis/{symbol}", {
        params: { path: { symbol }, query: { force } as any },
        cache: 'no-store'
    });
    if (error) throw new Error(`Failed to fetch AI analysis for ${symbol}`);
    return data as unknown as StockAnalysisResponse;
}

export async function clearCache(): Promise<StatusResponse> {
    const { data, error } = await apiClient.POST("/api/clear_cache", {});
    if (error) {
        throw new Error(`Failed to clear cache`);
    }
    return data as unknown as StatusResponse;
}

// --- Screener API ---

export interface ScreenerResult {
    symbol: string;
    name: string;
    price: number;
    intrinsic_value: number | null;
    margin_of_safety: number | null;
    pe_ratio: number | null;
    market_cap: number | null;
    sector: string | null;
    has_ai_review: boolean;
}

export interface ScreenerRequest {
    universe_type: string;
    universe_id: string | null;
    manual_symbols: string[];
    fast_mode?: boolean;
}

export async function runScreener(request: ScreenerRequest): Promise<ScreenerResult[]> {
    const { data, error } = await apiClient.POST("/api/screener/run", {
        body: request as any
    });
    if (error) throw new Error('Failed to run stock screen');
    return data as unknown as ScreenerResult[];
}

export async function fetchScreenerReview(symbol: string, force: boolean = false): Promise<StockAnalysisResponse> {
    const { data, error } = await apiClient.POST("/api/screener/review/{symbol}", {
        params: { path: { symbol }, query: { force } as any },
        cache: 'no-store'
    });
    if (error) throw new Error(`Failed to fetch AI review for ${symbol}`);
    return data as unknown as StockAnalysisResponse;
}

export async function fetchPortfolioAIReview(currency: string = 'USD', accounts?: string[], refresh: boolean = false, signal?: AbortSignal): Promise<any> {
    const { data, error } = await apiClient.POST("/api/portfolio/ai_review", {
        params: { query: { currency, accounts: accounts || undefined, refresh: refresh } },
        signal,
        cache: 'no-store'
    });
    if (error) throw new Error('Failed to fetch portfolio AI review');
    return data;
}
