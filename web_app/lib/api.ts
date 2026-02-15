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
    [key: string]: unknown;
}

export async function fetchSummary(currency: string = 'USD', accounts?: string[], signal?: AbortSignal): Promise<PortfolioSummary> {
    const params = new URLSearchParams({ currency });
    if (accounts) {
        accounts.forEach(acc => params.append('accounts', acc));
    }
    const res = await authFetch(`${API_BASE_URL}/summary?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch summary');
    return res.json();
}

export async function fetchMarketStatus(): Promise<{ is_open: boolean }> {
    const res = await authFetch(`${API_BASE_URL}/market_status`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch market status');
    return res.json();
}

export async function fetchHoldings(currency: string = 'USD', accounts?: string[], showClosed: boolean = false, signal?: AbortSignal): Promise<Holding[]> {
    const params = new URLSearchParams({ currency });
    if (accounts) {
        accounts.forEach(acc => params.append('accounts', acc));
    }
    if (showClosed) {
        params.append('show_closed', 'true');
    }
    const res = await authFetch(`${API_BASE_URL}/holdings?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch holdings');
    return res.json();
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
    const params = new URLSearchParams();
    if (accounts) {
        accounts.forEach(acc => params.append('accounts', acc));
    }
    const res = await authFetch(`${API_BASE_URL}/transactions?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch transactions');
    return res.json();
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
    const params = new URLSearchParams({ currency, period, interval });
    if (accounts) {
        accounts.forEach(acc => params.append('accounts', acc));
    }
    if (benchmarks) {
        benchmarks.forEach(b => params.append('benchmarks', b));
    }
    if (fromDate) params.append('from', fromDate);
    if (toDate) params.append('to', toDate);

    const res = await authFetch(`${API_BASE_URL}/history?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch history');
    return res.json();
}

export async function fetchMarketHistory(
    benchmarks: string[],
    period: string = '1y',
    interval: string = '1d',
    currency: string = 'USD',
    signal?: AbortSignal
): Promise<any[]> {
    const params = new URLSearchParams({ period, interval, currency });
    benchmarks.forEach(b => params.append('benchmarks', b));

    const res = await authFetch(`${API_BASE_URL}/market_history?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch market history');
    return res.json();
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
    const params = new URLSearchParams({ period, interval });
    if (benchmarks) {
        benchmarks.forEach(b => params.append('benchmarks', b));
    }

    const res = await authFetch(`${API_BASE_URL}/stock_history/${symbol}?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch stock history');
    return res.json();
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
    signal?: AbortSignal
): Promise<AssetChangeData> {
    const params = new URLSearchParams({ currency });
    if (accounts) {
        accounts.forEach(acc => params.append('accounts', acc));
    }
    if (benchmarks) {
        benchmarks.forEach(b => params.append('benchmarks', b));
    }
    const res = await authFetch(`${API_BASE_URL}/asset_change?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch asset change data');
    return res.json();
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
    const params = new URLSearchParams({ currency });
    if (accounts) {
        accounts.forEach(acc => params.append('accounts', acc));
    }
    if (fromDate) params.append('from', fromDate);
    if (toDate) params.append('to', toDate);
    const res = await authFetch(`${API_BASE_URL}/capital_gains?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch capital gains');
    return res.json();
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
    const params = new URLSearchParams({ currency });
    if (accounts) {
        accounts.forEach(acc => params.append('accounts', acc));
    }
    const res = await authFetch(`${API_BASE_URL}/dividends?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch dividends');
    return res.json();
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
    const res = await authFetch(`${API_BASE_URL}/settings`);
    if (!res.ok) throw new Error('Failed to fetch settings');
    return res.json();
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
    const response = await authFetch(`${API_BASE_URL}/settings/update`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(settings),
    });
    if (!response.ok) {
        throw new Error(`Failed to update settings: ${response.statusText}`);
    }
    return response.json();
}


export interface RiskMetrics {
    'Max Drawdown'?: number;
    'Volatility (Ann.)'?: number;
    'Sharpe Ratio'?: number;
    'Sortino Ratio'?: number;
}

export async function fetchRiskMetrics(currency: string = 'USD', accounts?: string[], signal?: AbortSignal): Promise<RiskMetrics> {
    const params = new URLSearchParams({ currency });
    if (accounts) {
        accounts.forEach(acc => params.append('accounts', acc));
    }
    const res = await authFetch(`${API_BASE_URL}/risk_metrics?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch risk metrics');
    return res.json();
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

export async function fetchAttribution(currency: string = 'USD', accounts?: string[], showAll: boolean = false, signal?: AbortSignal): Promise<AttributionData> {
    const params = new URLSearchParams({ currency });
    if (accounts) {
        accounts.forEach(acc => params.append('accounts', acc));
    }
    if (showAll) {
        params.append('show_all', 'true');
    }
    const res = await authFetch(`${API_BASE_URL}/attribution?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch attribution');
    return res.json();
}

export interface DividendEvent {
    symbol: string;
    dividend_date: string;
    ex_dividend_date: string;
    amount: number;
    status: 'confirmed' | 'estimated'; // Added status
}

export async function fetchDividendCalendar(accounts?: string[], signal?: AbortSignal): Promise<DividendEvent[]> {
    const params = new URLSearchParams();
    if (accounts) {
        accounts.forEach(acc => params.append('accounts', acc));
    }
    params.append('_t', Date.now().toString());
    const res = await authFetch(`${API_BASE_URL}/dividend_calendar?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch dividend calendar');
    return res.json();
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
    const params = new URLSearchParams({ currency });
    if (accounts) {
        accounts.forEach(acc => params.append('accounts', acc));
    }
    const res = await authFetch(`${API_BASE_URL}/projected_income?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch projected income');
    return res.json();
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
    signal?: AbortSignal
): Promise<PortfolioHealth | null> {
    const params = new URLSearchParams({ currency });
    if (accounts) {
        accounts.forEach(acc => params.append('accounts', acc));
    }
    const res = await authFetch(`${API_BASE_URL}/portfolio_health?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) {
        console.error("Failed to fetch portfolio health");
        return null;
    }
    return res.json();
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
    const res = await authFetch(`${API_BASE_URL}/watchlists`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch watchlists');
    return res.json();
}

export async function createWatchlist(name: string): Promise<WatchlistMeta> {
    const res = await authFetch(`${API_BASE_URL}/watchlists`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
    });
    if (!res.ok) throw new Error('Failed to create watchlist');
    return res.json();
}

export async function renameWatchlist(id: number, name: string): Promise<StatusResponse> {
    const res = await authFetch(`${API_BASE_URL}/watchlists/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
    });
    if (!res.ok) throw new Error('Failed to rename watchlist');
    return res.json();
}

export async function deleteWatchlist(id: number): Promise<StatusResponse> {
    const res = await authFetch(`${API_BASE_URL}/watchlists/${id}`, {
        method: "DELETE",
    });
    if (!res.ok) throw new Error('Failed to delete watchlist');
    return res.json();
}

export async function fetchWatchlist(currency: string = 'USD', watchlistId: number = 1, signal?: AbortSignal): Promise<WatchlistItem[]> {
    const params = new URLSearchParams({ currency, id: watchlistId.toString() });
    const res = await authFetch(`${API_BASE_URL}/watchlist?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch watchlist');
    return res.json();
}

export async function addToWatchlist(symbol: string, note: string = "", watchlistId: number = 1): Promise<StatusResponse> {
    const response = await authFetch(`${API_BASE_URL}/watchlist`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ symbol, note, watchlist_id: watchlistId }),
    });
    if (!response.ok) {
        throw new Error(`Failed to add to watchlist: ${response.statusText}`);
    }
    return response.json();
}

export async function removeFromWatchlist(symbol: string, watchlistId: number = 1): Promise<StatusResponse> {
    const response = await authFetch(`${API_BASE_URL}/watchlist/${symbol}?id=${watchlistId}`, {
        method: "DELETE",
    });
    if (!response.ok) {
        throw new Error(`Failed to remove from watchlist: ${response.statusText}`);
    }
    return response.json();
}

export async function updateHoldingTags(account: string, symbol: string, tags: string): Promise<StatusResponse> {
    const response = await authFetch(`${API_BASE_URL}/holdings/update_tags`, {

        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ account, symbol, tags }),
    });
    if (!response.ok) {
        throw new Error('Failed to update holding tags');
    }
    return response.json();
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
    const params = new URLSearchParams();
    if (force) params.append('force', 'true');
    const res = await authFetch(`${API_BASE_URL}/stock-analysis/${symbol}?${params.toString()}`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`Failed to fetch AI analysis for ${symbol}`);
    return res.json();
}

export async function clearCache(): Promise<StatusResponse> {
    const response = await authFetch(`${API_BASE_URL}/clear_cache`, {
        method: "POST",
    });
    if (!response.ok) {
        throw new Error(`Failed to clear cache: ${response.statusText}`);
    }
    return response.json();
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
    const res = await authFetch(`${API_BASE_URL}/screener/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request)
    });
    if (!res.ok) throw new Error('Failed to run stock screen');
    return res.json();
}

export async function fetchScreenerReview(symbol: string, force: boolean = false): Promise<StockAnalysisResponse> {
    const params = new URLSearchParams();
    if (force) params.append('force', 'true');
    const res = await authFetch(`${API_BASE_URL}/screener/review/${symbol}?${params.toString()}`, {
        method: "POST",
        cache: 'no-store'
    });
    if (!res.ok) throw new Error(`Failed to fetch AI review for ${symbol}`);
    return res.json();
}

export async function fetchPortfolioAIReview(currency: string = 'USD', accounts?: string[], refresh: boolean = false, signal?: AbortSignal): Promise<any> {
    const params = new URLSearchParams({ currency });
    if (accounts) {
        accounts.forEach(acc => params.append('accounts', acc));
    }
    if (refresh) {
        params.append('refresh', 'true');
    }
    const res = await authFetch(`${API_BASE_URL}/portfolio/ai_review?${params.toString()}`, {
        method: 'POST', // Use POST as defined in backend
        signal,
        cache: 'no-store'
    });
    if (!res.ok) throw new Error('Failed to fetch portfolio AI review');
    return res.json();
}
