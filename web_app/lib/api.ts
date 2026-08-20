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

export class SessionExpiredError extends Error {
    constructor() {
        super('Session expired');
        this.name = 'SessionExpiredError';
    }
}

export async function fetchCurrentUser(): Promise<User | null> {
    // Auth rides in the httpOnly cookie (apiClient sends credentials).
    const { data, response } = await apiClient.GET("/api/auth/me");
    if (response.status === 401 || !response.ok || !data) return null;
    return data as unknown as User;
}

export async function logoutRequest(): Promise<void> {
    // Clears the httpOnly auth cookie server-side. Best-effort.
    try {
        await apiClient.POST("/api/auth/logout");
    } catch {
        // Ignore — local session is cleared regardless.
    }
}

export async function updateUserProfile(data: { alias: string }): Promise<User> {
    const { data: user, error } = await apiClient.PATCH("/api/auth/me", {
        body: data,
    });
    if (error) throw new Error('Failed to update user profile');
    return user as unknown as User;
}

export interface User {
    id: number;
    username: string;
    alias?: string;
    is_active: boolean;
    created_at: string;
}

export async function deleteUser(): Promise<StatusResponse> {
    const { data, error } = await apiClient.DELETE("/api/auth/me");
    if (error) throw new Error('Failed to delete user');
    return data as unknown as StatusResponse;
}

export async function changePassword(currentPassword: string, newPassword: string): Promise<StatusResponse> {
    const { data, error } = await apiClient.POST("/api/auth/change-password", {
        body: { current_password: currentPassword, new_password: newPassword },
    });
    if (error) throw new Error((error as { detail?: string }).detail || 'Failed to change password');
    return data as unknown as StatusResponse;
}

const authFetch = async (url: string, options: RequestInit = {}) => {
    // Auth rides in the httpOnly cookie — include credentials on every call.
    const response = await fetch(url, { ...options, credentials: 'include' });

    if (response.status === 401 && typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('auth:expired'));
    }

    return response;
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
        taxes?: number;
        fx_gain_loss_display?: number;
        fx_gain_loss_pct?: number;
        annualized_twr?: number;
        cumulative_twr?: number;
        portfolio_mwr?: number; // Added Money-Weighted Return (IRR)
        dividend_return_cumulative?: number;
        dividend_return_annualized?: number;
        dividend_yield_pct?: number;
        cash_balance?: number; // Might not be directly in metrics, check account_metrics for Cash
        exchange_rate_to_display?: number;
        max_drawdown?: number;
        volatility_ann?: number;
        sharpe_ratio?: number;
        beta?: number;
        ytd_return?: number;
        // True when every account in the current selection has a closure date <= today.
        // The backend sets rate-of-return metrics to null in that case to avoid the
        // residual-dividend TWR inflation bug; the frontend renders "Closed" instead.
        all_selected_closed?: boolean;
        // Subset of the selected accounts that are closed (closure date <= today).
        closed_accounts?: string[];
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
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- pre-existing; typed cleanup tracked separately
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
    fx_rate?: number;
    // Keys are dynamic based on currency, e.g., "Market Value (USD)"
    [key: string]: unknown;
    lots?: Lot[];
    sparkline_7d?: number[];
    sparkline_1m?: number[];
    ai_score?: number;
    intrinsic_value?: number;
    margin_of_safety?: number;
    has_ai_review?: boolean;
    ai_sentiment?: number;
    ai_catalysts?: { event: string, date: string, impact: string }[];
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
        signal
    });
    if (error) throw new Error('Failed to fetch summary');
    return data as unknown as PortfolioSummary;
}

export async function fetchMarketStatus(): Promise<{ is_open: boolean }> {
    const { data, error } = await apiClient.GET("/api/market_status");
    if (error) throw new Error('Failed to fetch market status');
    return data as unknown as { is_open: boolean };
}

// Header index quotes (Dow / Nasdaq / S&P), served off the /summary critical
// path. May be slow on a cold cache, so it's fetched as its own query.
export async function fetchIndices(signal?: AbortSignal): Promise<Record<string, unknown>> {
    const { data, error } = await apiClient.GET("/api/indices", { signal });
    if (error) throw new Error('Failed to fetch indices');
    return data as unknown as Record<string, unknown>;
}

// Fast headline metrics (total value, day change, …) for the top card. Skips the
// expensive historical/TWR work in /summary so the card renders/updates first.
export async function fetchHeadline(currency: string = 'USD', accounts?: string[], signal?: AbortSignal): Promise<{ metrics: Record<string, unknown> | null }> {
    const { data, error } = await apiClient.GET("/api/summary/headline", {
        params: { query: { currency, accounts: accounts || undefined } },
        signal,
    });
    if (error) throw new Error('Failed to fetch headline summary');
    return data as unknown as { metrics: Record<string, unknown> | null };
}

export async function fetchHoldings(currency: string = 'USD', accounts?: string[], showClosed: boolean = false, signal?: AbortSignal): Promise<Holding[]> {
    const { data, error } = await apiClient.GET("/api/holdings", {
        params: {
            query: { currency, accounts: accounts || undefined, show_closed: showClosed }
        },
        signal
    });
    if (error) throw new Error('Failed to fetch holdings');
    return data as unknown as Holding[];
}

export type HoldingReturnPeriod = '1m' | '3m' | '6m' | '1y' | 'ytd';
// Map of holding symbol -> per-period price return (%), e.g. { AAPL: { '1m': 5.2, ... } }.
// Values may be null when there isn't enough price history for a window.
export type HoldingReturns = Record<string, Partial<Record<HoldingReturnPeriod, number | null>>>;

// Per-holding price returns over fixed windows, used by the performance heatmap.
// Uses authFetch (not the typed apiClient) because this endpoint isn't part of
// the generated openapi schema. Passing `symbols` limits compute to the current
// holdings; extra/missing symbols in the response are simply ignored by callers.
export async function fetchHoldingReturns(symbols?: string[], signal?: AbortSignal): Promise<HoldingReturns> {
    const params = new URLSearchParams();
    (symbols || []).forEach((s) => params.append('symbols', s));
    const qs = params.toString();
    const url = `${API_BASE_URL}/holdings/returns${qs ? `?${qs}` : ''}`;
    const response = await authFetch(url, { signal });
    if (!response.ok) throw new Error('Failed to fetch holding returns');
    return (await response.json()) as HoldingReturns;
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
        signal
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
    const { data, error, response } = await apiClient.POST("/api/transactions", {
        body: transaction as never,
    });
    if (error) throw new Error(`Failed to add transaction: ${response.statusText}`);
    return data as unknown as StatusResponse;
}

// Stays on authFetch: multipart upload — openapi-fetch would need a custom
// bodySerializer for FormData, which buys nothing here.
export async function parseDocument(file: File): Promise<{ status: string, transactions: Transaction[], count: number, message: string }> {
    const formData = new FormData();
    formData.append("file", file);

    const url = `${API_BASE_URL}/transactions/parse_document`;

    const response = await authFetch(url, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`Failed to parse document: ${response.statusText}`);
    }
    return response.json();
}

export async function addTransactionsBatch(transactions: Transaction[], autoAddCash: boolean = false): Promise<StatusResponse> {
    const { data, error, response } = await apiClient.POST("/api/transactions/batch", {
        body: { transactions, auto_add_cash: autoAddCash } as never,
    });
    if (error) throw new Error(`Failed to add transactions: ${response.statusText}`);
    return data as unknown as StatusResponse;
}

export async function updateTransaction(id: number, transaction: Transaction): Promise<StatusResponse> {
    const { data, error, response } = await apiClient.PUT("/api/transactions/{transaction_id}", {
        params: { path: { transaction_id: id } },
        body: transaction as never,
    });
    if (error) throw new Error(`Failed to update transaction: ${response.statusText}`);
    return data as unknown as StatusResponse;
}

export async function deleteTransaction(id: number): Promise<StatusResponse> {
    const { data, error, response } = await apiClient.DELETE("/api/transactions/{transaction_id}", {
        params: { path: { transaction_id: id } },
    });
    if (error) throw new Error(`Failed to delete transaction: ${response.statusText}`);
    return data as unknown as StatusResponse;
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
        signal
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
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- pre-existing; typed cleanup tracked separately
): Promise<any[]> {
    const { data, error } = await apiClient.GET("/api/market_history", {
        params: {
            query: { period, interval, currency, benchmarks }
        },
        signal
    });
    if (error) throw new Error('Failed to fetch market history');
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- pre-existing; typed cleanup tracked separately
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
        signal
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
        signal
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
        signal
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
    TaxAmountLocal?: number;
    TaxAmountDisplayCurrency?: number;
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
        signal
    });
    if (error) throw new Error('Failed to fetch dividends');
    return data as unknown as Dividend[];
}

export interface EarningsDate {
    date: string;
    eps_estimate?: number | null;
    eps_actual?: number | null;
    surprise_pct?: number | null;
}

export async function fetchEarningsDates(
    symbol: string,
    signal?: AbortSignal
): Promise<EarningsDate[]> {
    const { data, error } = await apiClient.GET("/api/earnings_dates/{symbol}", {
        params: { path: { symbol } },
        signal,
    });
    if (error) throw new Error('Failed to fetch earnings dates');
    return data as unknown as EarningsDate[];
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
    account_cash_mode_map: Record<string, string>;
    account_closure_dates?: Record<string, string>;
    account_groups: Record<string, string[]>;
    account_group_order?: string[];
    available_currencies: string[];
    account_interest_rates: Record<string, number>;
    interest_free_thresholds: Record<string, number>;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- pre-existing; typed cleanup tracked separately
    valuation_overrides: Record<string, any>;
    visible_items?: string[];
    benchmarks?: string[];
    show_closed?: boolean;
    display_currency?: string;
    selected_accounts?: string[];
    active_tab?: string;
    ibkr_token?: string;
    ibkr_query_id?: string;
    target_allocation?: Record<string, Record<string, number>>;
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
    account_group_order?: string[];
    account_currency_map?: Record<string, string>;
    account_cash_mode_map?: Record<string, string>;
    account_closure_dates?: Record<string, string>;
    available_currencies?: string[];
    account_interest_rates?: Record<string, number>;
    interest_free_thresholds?: Record<string, number>;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- pre-existing; typed cleanup tracked separately
    valuation_overrides?: Record<string, any>;
    visible_items?: string[];
    benchmarks?: string[];
    show_closed?: boolean;
    display_currency?: string;
    selected_accounts?: string[];
    active_tab?: string;
    ibkr_token?: string;
    ibkr_query_id?: string;
    target_allocation?: Record<string, Record<string, number>>;
}

export async function updateSettings(settings: SettingsUpdate): Promise<StatusResponse> {
    const { data, error } = await apiClient.POST("/api/settings/update", {
        body: settings as never
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
        signal
    });
    if (error) throw new Error('Failed to fetch risk metrics');
    return data as unknown as RiskMetrics;
}

export interface BenchmarkStat {
    name: string;
    alpha: number;            // annualized %
    beta: number;
    r2: number;               // 0-1
    tracking_error: number;   // annualized %
    information_ratio: number;
    excess_return: number;    // cumulative %
}

export async function fetchBenchmarkScoreboard(currency: string = 'USD', accounts?: string[], benchmarks?: string[], period: string = 'all', signal?: AbortSignal): Promise<BenchmarkStat[]> {
    const { data, error } = await apiClient.GET("/api/benchmark_scoreboard", {
        params: {
            query: { currency, accounts: accounts || undefined, benchmarks: benchmarks || undefined, period }
        },
        signal
    });
    if (error) throw new Error('Failed to fetch benchmark scoreboard');
    return ((data as { scoreboard?: BenchmarkStat[] })?.scoreboard) ?? [];
}

export interface ProjectionHorizon {
    years: number;
    median_value: number;
    median_return_pct: number;
    expected_value: number;
    p10: number;
    p25: number;
    p75: number;
    p90: number;
}

export interface Projection {
    available: boolean;
    current_value?: number;
    annual_return_pct?: number;
    annual_volatility_pct?: number;
    currency?: string;
    horizons?: ProjectionHorizon[];
}

export async function fetchProjection(currency: string = 'USD', accounts?: string[], signal?: AbortSignal): Promise<Projection> {
    const { data, error } = await apiClient.GET("/api/projection", {
        params: {
            query: { currency, accounts: accounts || undefined }
        },
        signal
    });
    if (error) throw new Error('Failed to fetch projection');
    return data as unknown as Projection;
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
        signal
    });
    if (error) throw new Error('Failed to fetch attribution');
    return data as unknown as AttributionData;
}

export interface DividendEvent {
    symbol: string;
    /** Company name; absent for synthetic rows such as cash interest. */
    name?: string | null;
    dividend_date: string;
    ex_dividend_date: string;
    amount: number;
    status: 'confirmed' | 'estimated'; // Added status
    /**
     * IANA zone of the exchange this date belongs to — count "days from now"
     * against this, not the browser's clock (see lib/market_time.ts).
     */
    market_timezone?: string | null;
}

export async function fetchDividendCalendar(currency: string = 'USD', accounts?: string[], signal?: AbortSignal): Promise<DividendEvent[]> {
    const { data, error } = await apiClient.GET("/api/dividend_calendar", {
        params: {
            query: { currency, accounts: accounts || undefined, _t: Date.now().toString() as never }
        },
        signal
    });
    if (error) throw new Error('Failed to fetch dividend calendar');
    return data as unknown as DividendEvent[];
}

export interface EarningsEvent {
    symbol: string;
    name?: string | null;
    earnings_date: string;
    /** Set only when the company announced a window rather than an exact day. */
    earnings_date_end?: string;
    /** 'reported' is a quarter already printed — the date is in the past. */
    status: 'confirmed' | 'estimated' | 'reported';
    eps_estimate?: number | null;
    eps_year_ago?: number | null;
    /** Reported ('reported' only); null while Yahoo has yet to attach the figure. */
    eps_actual?: number | null;
    /** Beat/miss vs consensus in percent, derived server-side from actual/estimate. */
    surprise_pct?: number | null;
    /** IANA zone of the reporting exchange — see DividendEvent.market_timezone. */
    market_timezone?: string | null;
}

/** The next dividend for a single symbol, per share (see Fundamentals.upcoming_events). */
export interface UpcomingDividend {
    symbol: string;
    name?: string | null;
    dividend_date: string;
    ex_dividend_date?: string | null;
    amount_per_share: number;
    frequency_months?: number | null;
    status: 'confirmed' | 'estimated';
    /** IANA zone of the paying exchange — see DividendEvent.market_timezone. */
    market_timezone?: string | null;
}

export async function fetchEarningsCalendar(accounts?: string[], signal?: AbortSignal): Promise<EarningsEvent[]> {
    const { data, error } = await apiClient.GET("/api/earnings_calendar", {
        params: {
            query: { accounts: accounts || undefined, _t: Date.now().toString() as never }
        },
        signal
    });
    if (error) throw new Error('Failed to fetch earnings calendar');
    return data as unknown as EarningsEvent[];
}

export async function saveManualOverride(symbol: string, price: number | null): Promise<StatusResponse> {
    const { data, error, response } = await apiClient.POST("/api/settings/manual_overrides", {
        body: { symbol, price } as never,
    });
    if (error) throw new Error(`Failed to save manual override: ${response.statusText}`);
    return data as unknown as StatusResponse;
}

export async function triggerRefresh(secret: string): Promise<StatusResponse> {
    const { data, error, response } = await apiClient.POST("/api/webhook/refresh", {
        body: { secret } as never,
    });
    if (error) throw new Error(`Failed to trigger refresh: ${response.statusText}`);
    return data as unknown as StatusResponse;
}

export async function syncIbkr(): Promise<StatusResponse> {
    const { data, error } = await apiClient.POST("/api/sync/ibkr");
    if (error) {
        const err = error as { message?: string; detail?: string };
        throw new Error(err.message || err.detail || 'Failed to sync IBKR');
    }
    return data as unknown as StatusResponse;
}

export async function fetchPendingIbkr(): Promise<Transaction[]> {
    const { data, error } = await apiClient.GET("/api/sync/ibkr/pending");
    if (error) throw new Error('Failed to fetch pending transactions');
    return data as unknown as Transaction[];
}

export async function approveIbkr(ids: number[]): Promise<StatusResponse> {
    const { data, error } = await apiClient.POST("/api/sync/ibkr/approve", {
        body: ids as never,
    });
    if (error) throw new Error('Failed to approve transactions');
    return data as unknown as StatusResponse;
}

export async function rejectIbkr(ids: number[]): Promise<StatusResponse> {
    const { data, error } = await apiClient.POST("/api/sync/ibkr/reject", {
        body: ids as never,
    });
    if (error) throw new Error('Failed to reject transactions');
    return data as unknown as StatusResponse;
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
        signal
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
        signal
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
    /** Forward annual dividend per share, used to settle the yield's encoding. */
    "Dividend Rate"?: number | null;
    /** Yahoo's trailingAnnualDividendYield — always a fraction. */
    "Trailing Dividend Yield"?: number | null;
    ai_score?: number | null;
    intrinsic_value?: number | null;
    margin_of_safety?: number | null;
    has_ai_review?: boolean;
    ai_sentiment?: number | null;
    ai_catalysts?: { event: string, date: string, impact: string }[] | null;
}


export interface WatchlistMeta {
    id: number;
    name: string;
    created_at: string;
}

export async function getWatchlists(signal?: AbortSignal): Promise<WatchlistMeta[]> {
    const { data, error } = await apiClient.GET("/api/watchlists", { signal });
    if (error) throw new Error('Failed to fetch watchlists');
    return data as unknown as WatchlistMeta[];
}

export async function createWatchlist(name: string): Promise<WatchlistMeta> {
    const { data, error } = await apiClient.POST("/api/watchlists", {
        body: { name } as never
    });
    if (error) throw new Error('Failed to create watchlist');
    return data as unknown as WatchlistMeta;
}

export async function renameWatchlist(id: number, name: string): Promise<StatusResponse> {
    const { data, error } = await apiClient.PUT("/api/watchlists/{watchlist_id}", {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- pre-existing; typed cleanup tracked separately
        params: { path: { watchlist_id: id as unknown as string } as any },
        body: { name } as never
    });
    if (error) throw new Error('Failed to rename watchlist');
    return data as unknown as StatusResponse;
}

export async function deleteWatchlist(id: number): Promise<StatusResponse> {
    const { data, error } = await apiClient.DELETE("/api/watchlists/{watchlist_id}", {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- pre-existing; typed cleanup tracked separately
        params: { path: { watchlist_id: id as unknown as string } as any }
    });
    if (error) throw new Error('Failed to delete watchlist');
    return data as unknown as StatusResponse;
}

export async function fetchWatchlist(currency: string = 'USD', watchlistId: number = 1, signal?: AbortSignal): Promise<WatchlistItem[]> {
    const { data, error } = await apiClient.GET("/api/watchlist", {
        params: {
            query: { currency, id: watchlistId } as never
        },
        signal
    });
    if (error) throw new Error('Failed to fetch watchlist');
    return data as unknown as WatchlistItem[];
}

export async function addToWatchlist(symbol: string, note: string = "", watchlistId: number = 1): Promise<StatusResponse> {
    const { data, error } = await apiClient.POST("/api/watchlist", {
        body: { symbol, note, watchlist_id: watchlistId } as never
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
            query: { id: watchlistId } as never
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
    /** Forward annual dividend per share; pairs with a price to settle the
     *  fraction-vs-percent encoding of `dividendYield`. */
    dividendRate?: number;
    trailingAnnualDividendRate?: number;
    /** Always a fraction, unlike `dividendYield`. */
    trailingAnnualDividendYield?: number;
    currentPrice?: number;
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
    /** Valuation / earnings / profitability / market readings, derived
     *  server-side by the same code that builds the heatmap payload — the field
     *  names are the heatmap's, so lib/metrics.ts reads both. */
    key_metrics?: Record<string, number | null>;
    /** Earnings / dividend events, derived server-side from this same blob. */
    upcoming_events?: {
        earnings: EarningsEvent | null;
        /** The quarter just reported (last 5 days), with the printed EPS. */
        recent_earnings: EarningsEvent | null;
        dividend: UpcomingDividend | null;
    };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- pre-existing; typed cleanup tracked separately
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
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- pre-existing; typed cleanup tracked separately
    parameters: Record<string, any>;
    mc?: {
        bear: number;
        base: number;
        bull: number;
        std_dev: number;
        histogram?: { price: number; count: number }[];
    };
}

/**
 * `ok` — models agree closely enough to trust the blend.
 * `low_confidence` — models disagree by more than the blended value.
 * `clamped` — raw output fell outside the credible band vs price.
 * `ineligible` — sub-$1 or micro-cap; per-share maths is dominated by noise.
 * `no_model` — no model could value the company; `average_intrinsic_value` is null.
 * `nav` — ETF/fund valued at net asset value.
 */
export type ValuationStatus =
    | "ok"
    | "low_confidence"
    | "clamped"
    | "ineligible"
    | "no_model"
    | "nav";

export interface IntrinsicValueResponse {
    current_price: number | null;
    models: {
        dcf: IntrinsicValueModel;
        graham: IntrinsicValueModel;
        /** Earnings Power Value — the no-growth floor. Reported, never blended. */
        epv?: IntrinsicValueModel;
    };
    /** Null when the backend refuses to value the company; check valuation_status. */
    average_intrinsic_value?: number | null;
    range?: {
        bear: number;
        bull: number;
    };
    margin_of_safety_pct?: number;
    valuation_note?: string;
    valuation_status?: ValuationStatus;
    /** Spread between contributing models, as % of the blended value. */
    model_spread_pct?: number | null;
    /** Normalized weight actually applied to each contributing model. */
    model_weights?: Record<string, number>;
    /** Value of current earning power with zero growth. */
    earnings_power_floor?: number;
}

export interface SymbolSearchResult {
    symbol: string;
    name: string;
    type: string;
}

export async function fetchSymbolSearch(q: string): Promise<SymbolSearchResult[]> {
    const { data, error } = await apiClient.GET("/api/search", {
        params: { query: { q } }
    });
    if (error) return [];
    return data as unknown as SymbolSearchResult[];
}

export interface MarketNewsItem {
    title: string;
    summary: string;
    url: string;
    thumbnail: string | null;
    provider: string;
    pub_date: string;
    symbol?: string | null;
}

export async function fetchMarketNews(limit = 20): Promise<MarketNewsItem[]> {
    const { data, error } = await apiClient.GET("/api/markets/news", {
        params: { query: { limit } }
    });
    if (error) return [];
    return data as unknown as MarketNewsItem[];
}

export async function fetchStockNews(symbols: string[], limit = 30): Promise<MarketNewsItem[]> {
    if (!symbols.length) return [];
    const { data, error } = await apiClient.GET("/api/markets/news", {
        params: { query: { symbols: symbols.join(','), limit } }
    });
    if (error) return [];
    return data as unknown as MarketNewsItem[];
}

// S&P 500 Heatmap
//
// Unit convention, mirrored by the Swift client: `change_pct` is percent points
// (it comes straight off the quote), ratio-style fields are raw numbers, and
// every other percentage — returns, growth, margins, yield, float short — is a
// **fraction** (0.15 = 15%). Ratios expressed against equity (`debt_equity`,
// `lt_debt_equity`) follow Yahoo and are percent points.
export interface SP500HeatmapItem {
    symbol: string;
    name: string;
    sector: string;
    sub_industry: string;
    price: number;
    market_cap: number | null;

    // Performance
    change_pct: number | null;
    week_change_pct?: number | null;
    month_change_pct?: number | null;
    mtd_change_pct?: number | null;
    "3m_change_pct"?: number | null;
    "6m_change_pct"?: number | null;
    ytd_change_pct?: number | null;
    "1y_change_pct"?: number | null;
    "3y_change_pct"?: number | null;
    "5y_change_pct"?: number | null;
    "10y_change_pct"?: number | null;
    /** Zero or below: the price cannot exceed its own 52-week high. */
    drawdown_52w?: number | null;
    /** Zero or above: the price cannot fall below its own 52-week low. */
    gain_from_52w_low?: number | null;

    // Valuation
    pe_ratio: number | null;
    forward_pe?: number | null;
    peg_ratio?: number | null;
    ps_ratio?: number | null;
    pb_ratio?: number | null;
    p_fcf?: number | null;
    ev_ebitda?: number | null;
    ev_sales?: number | null;
    dividend_yield: number | null;

    // Earnings & sales
    eps_ttm?: number | null;
    eps_qoq?: number | null;
    eps_growth_3y?: number | null;
    eps_growth_5y?: number | null;
    eps_surprise?: number | null;
    sales_ttm?: number | null;
    sales_qoq?: number | null;
    sales_growth_3y?: number | null;
    sales_growth_5y?: number | null;

    // Profitability & balance sheet
    roa?: number | null;
    roe?: number | null;
    roic?: number | null;
    gross_margin?: number | null;
    operating_margin?: number | null;
    net_margin?: number | null;
    quick_ratio?: number | null;
    current_ratio?: number | null;
    lt_debt_equity?: number | null;
    debt_equity?: number | null;

    // Market & sentiment
    relative_volume?: number | null;
    float_short?: number | null;
    /** Yahoo consensus: 1 (strong buy) .. 5 (sell). */
    analyst_recom?: number | null;
    /** Days until the next report; negative once it has happened. */
    earnings_days?: number | null;
}

export async function fetchSP500Heatmap(signal?: AbortSignal): Promise<SP500HeatmapItem[]> {
    const url = `${API_BASE_URL}/sp500/heatmap`;
    const response = await authFetch(url, { signal });
    if (!response.ok) throw new Error('Failed to fetch S&P 500 heatmap');
    return (await response.json()) as SP500HeatmapItem[];
}

export async function fetchFundamentals(symbol: string, force: boolean = false): Promise<Fundamentals> {
    const { data, error } = await apiClient.GET("/api/fundamentals/{symbol}", {
        params: { path: { symbol }, query: { force: force || undefined } },
    });
    if (error) throw new Error(`Failed to fetch fundamentals for ${symbol}`);
    return data as unknown as Fundamentals;
}

export async function fetchFinancials(symbol: string, periodType: 'annual' | 'quarterly' = 'annual', force: boolean = false): Promise<FinancialsResponse> {
    const { data, error } = await apiClient.GET("/api/financials/{symbol}", {
        params: { path: { symbol }, query: { period_type: periodType, force: force || undefined } },
    });
    if (error) throw new Error(`Failed to fetch financials for ${symbol}`);
    return data as unknown as FinancialsResponse;
}

/**
 * `periodType` shapes the historical series only. Quarterly measures the same
 * ratios on trailing-twelve-month flows at each quarter end, so they stay
 * comparable with the annual series and simply arrive four times as often.
 */
export async function fetchRatios(
    symbol: string,
    periodType: 'annual' | 'quarterly' = 'quarterly',
    force: boolean = false,
): Promise<RatiosResponse> {
    const { data, error } = await apiClient.GET("/api/ratios/{symbol}", {
        params: { path: { symbol }, query: { period_type: periodType, force: force || undefined } },
    });
    if (error) throw new Error(`Failed to fetch ratios for ${symbol}`);
    return data as unknown as RatiosResponse;
}

export async function fetchIntrinsicValue(symbol: string, force: boolean = false): Promise<IntrinsicValueResponse> {
    const { data, error } = await apiClient.GET("/api/intrinsic_value/{symbol}", {
        params: { path: { symbol }, query: { force: force || undefined } },
    });
    if (error) throw new Error(`Failed to fetch intrinsic value for ${symbol}`);
    return data as unknown as IntrinsicValueResponse;
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
    sentiment?: number;
    catalysts?: { event: string, date: string, impact: string }[];
    ai_review: string;
    optimizations?: {
        // New types (quality/value framing): add, trim, exit, monitor, tax_efficiency
        // Legacy types kept for backward-compat with old cached reviews.
        type: 'add' | 'trim' | 'exit' | 'monitor' | 'tax_efficiency'
        | 'tax_loss_harvesting' | 'rebalancing' | 'diversification';
        title: string;
        description: string;
        symbol: string;
        action: 'Add' | 'Trim' | 'Sell' | 'Buy' | 'Hold' | 'Monitor' | 'Swap';
        priority: 'High' | 'Medium' | 'Low';
    }[];
    error?: string;
}

export interface ChatMessage {
    role: 'user' | 'ai';
    text: string;
}

export async function sendChatMessage(message: string, history: ChatMessage[] = []): Promise<string> {
    const { data, error } = await apiClient.POST("/api/chat/message", {
        body: { message, history } as never,
    });
    if (error) throw new Error('Failed to send message to AI');
    return (data as unknown as { response: string }).response;
}

export async function fetchStockAnalysis(symbol: string, force: boolean = false): Promise<StockAnalysisResponse> {
    const { data, error } = await apiClient.GET("/api/stock-analysis/{symbol}", {
        params: { path: { symbol }, query: { force } as never }
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
        body: request as never
    });
    if (error) throw new Error('Failed to run stock screen');
    return data as unknown as ScreenerResult[];
}

export async function runNarrativeSearch(prompt: string): Promise<ScreenerResult[]> {
    const { data, error } = await apiClient.POST("/api/screener/narrative", {
        body: { prompt } as never,
    });
    if (error) throw new Error('Failed to run narrative search');
    return data as unknown as ScreenerResult[];
}

export async function fetchScreenerReview(symbol: string, force: boolean = false): Promise<StockAnalysisResponse> {
    const { data, error } = await apiClient.POST("/api/screener/review/{symbol}", {
        params: { path: { symbol }, query: { force } as never }
    });
    if (error) throw new Error(`Failed to fetch AI review for ${symbol}`);
    return data as unknown as StockAnalysisResponse;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- pre-existing; typed cleanup tracked separately
export async function fetchPortfolioAIReview(currency: string = 'USD', accounts?: string[], refresh: boolean = false, signal?: AbortSignal): Promise<any> {
    const { data, error } = await apiClient.POST("/api/portfolio/ai_review", {
        params: { query: { currency, accounts: accounts || undefined, refresh: refresh } },
        signal
    });
    if (error) throw new Error('Failed to fetch portfolio AI review');
    return data;
}

// --- Buffett / value ranking -------------------------------------------------
// Served from dated snapshots written by the batch worker, not computed per
// request: a full ranking run takes minutes over ~5,500 filers.

/** Which valuation model a company was scored under. */
export type BuffettModel = 'generic' | 'bank' | 'insurer' | 'reit';

export interface BuffettRankRow {
    symbol: string;
    cik: string | null;
    name: string | null;
    model: BuffettModel;
    rank: number | null;
    composite_score: number | null;
    quality_score: number | null;
    /** Null for banks, insurers and REITs, which are valued on multiples rather than a DCF. */
    value_score: number | null;
    /** Coverage-derived multiplier in [0.5, 1]; it can only ever demote. */
    confidence: number | null;
    coverage: number | null;
    returns_on_capital: number | null;
    financial_strength: number | null;
    predictability: number | null;
    growth: number | null;
    capital_allocation: number | null;
    price: number | null;
    market_cap: number | null;
    /** The two scored value inputs. There is no DCF-derived field any more. */
    earnings_yield: number | null;
    fcf_yield: number | null;
    period_count: number | null;
    latest_period: string | null;
}

/** A company kept out of the ranking, with the reasons it failed. */
export interface BuffettExclusion {
    symbol: string;
    cik: string | null;
    name: string | null;
    model: BuffettModel;
    reasons: string;
    period_count: number | null;
    coverage: number | null;
}

export interface BuffettRankRun {
    run_id: number;
    started_at: string | null;
    finished_at: string | null;
    universe_size: number | null;
    ranked_count: number | null;
    excluded_count: number | null;
}

export async function fetchBuffettRankRun(signal?: AbortSignal): Promise<BuffettRankRun | null> {
    const { data, error, response } = await apiClient.GET("/api/buffett-rank/latest", { signal });
    // 404 means no run has completed yet — a normal state on a fresh install,
    // not an error the UI should shout about.
    if (response.status === 404) return null;
    if (error) throw new Error('Failed to fetch ranking run');
    return data as unknown as BuffettRankRun;
}

/** One page of the ranking, plus how many rows match the active filters. */
export interface BuffettRankPage {
    total: number;
    rows: BuffettRankRow[];
}

export async function fetchBuffettRankings(
    limit: number = 100,
    offset: number = 0,
    model?: BuffettModel,
    search?: string,
    signal?: AbortSignal
): Promise<BuffettRankPage> {
    // `search` is applied server-side across the whole run. Filtering the
    // returned page instead would only ever search the rows already loaded,
    // which misses everything past rank ~100.
    const { data, error } = await apiClient.GET("/api/buffett-rank", {
        params: {
            query: {
                limit,
                offset,
                model: model || undefined,
                search: search?.trim() ? search.trim() : undefined,
            }
        },
        signal
    });
    if (error) throw new Error('Failed to fetch rankings');
    return data as unknown as BuffettRankPage;
}

export interface BuffettExclusionPage {
    total: number;
    rows: BuffettExclusion[];
}

export async function fetchBuffettExclusions(
    limit: number = 100,
    offset: number = 0,
    search?: string,
    signal?: AbortSignal
): Promise<BuffettExclusionPage> {
    const { data, error } = await apiClient.GET("/api/buffett-rank/exclusions", {
        params: {
            query: {
                limit,
                offset,
                search: search?.trim() ? search.trim() : undefined,
            }
        },
        signal
    });
    if (error) throw new Error('Failed to fetch ranking exclusions');
    return data as unknown as BuffettExclusionPage;
}

/** One measured metric from a company's record. */
export interface TrackRecordItem {
    key: string;
    label: string;
    unit: string;
    value: number | null;
    /** Preformatted by the backend so the three clients cannot format it differently. */
    display: string | null;
    /** Why a metric is unmeasurable, when it is knowably so (e.g. a stock split). */
    note: string | null;
    higher_is_better: boolean;
}

export interface TrackRecordGroup {
    key: string;
    title: string;
    items: TrackRecordItem[];
}

/**
 * Today's multiple against the company's own history. Absent entirely when the
 * local price store is too shallow to support a band.
 */
export interface TrackRecordBand {
    metric: string;
    label: string;
    current: number;
    median: number;
    p25: number;
    p75: number;
    low: number;
    high: number;
    /** 0 = cheapest it has ever been, 100 = dearest. */
    percentile: number;
    observations: number;
    display: string;
    median_display: string;
    /** "dearer than usual for this company" — a comparison, never advice. */
    summary: string;
}

/** How one metric behaved peak-to-trough in a downturn. */
export interface TrackRecordStressItem {
    metric: string;
    label: string;
    peak_year: number;
    trough_year: number;
    change_pct: number;
    /** "-64%", formatted by the backend. */
    display: string;
    recovered_year: number | null;
    /** "back in 2022" / "not back to its peak", or null when it never fell. */
    recovery_display: string | null;
}

/**
 * One downturn. `covered` is false when the company has no filings spanning it —
 * "not listed then" and "did not fall" are opposite claims.
 */
export interface TrackRecordStress {
    key: string;
    label: string;
    covered: boolean;
    items: TrackRecordStressItem[];
}

/** A number the company changed after first reporting it. */
export interface TrackRecordRevision {
    concept: string;
    /** The statement row it belongs to, e.g. "Total Revenue". */
    label: string;
    period_end: string;
    original: number;
    current: number;
    change_pct: number;
    /** "$1.95bn → $4.41bn", formatted by the backend. */
    display: string;
    change_display: string;
    first_filed: string;
    restated_filed: string;
}

/**
 * The measured quality record: the same metrics the Buffett ranking scores on,
 * over the durability window, with the span they rest on.
 */
export interface TrackRecord {
    symbol: string;
    name: string | null;
    cik: string;
    model: BuffettModel;
    period_count: number;
    first_period: string | null;
    latest_period: string | null;
    window_years: number;
    coverage: number | null;
    /** Hard-gate failures — why the ranking excludes this company, if it does. */
    gate_failures: string[];
    rank: {
        run_id: number | null;
        rank: number | null;
        composite_score: number | null;
        quality_score: number | null;
        value_score: number | null;
        confidence: number | null;
        pillars: Record<string, number | null>;
    } | null;
    groups: TrackRecordGroup[];
    /** Revision history. `count` is the total; `items` is the largest handful. */
    revisions: { count: number; items: TrackRecordRevision[] };
    /** Behaviour in each downturn the filed history reaches. */
    stress: TrackRecordStress[];
    /** Today's multiples against this company's own record. */
    valuation_bands: TrackRecordBand[];
}

export async function fetchTrackRecord(
    symbol: string,
    signal?: AbortSignal
): Promise<TrackRecord | null> {
    const { data, error, response } = await apiClient.GET("/api/track-record/{symbol}", {
        params: { path: { symbol } },
        signal
    });
    // 404 is the normal answer for anything that does not file with the SEC —
    // every SET holding, every foreign listing. The panel hides itself.
    if (response.status === 404) return null;
    if (error) throw new Error(`Failed to fetch track record for ${symbol}`);
    return data as unknown as TrackRecord;
}

export async function fetchBuffettRankHistory(
    symbol: string,
    limit: number = 24,
    signal?: AbortSignal
): Promise<BuffettRankRow[]> {
    const { data, error } = await apiClient.GET("/api/buffett-rank/history/{symbol}", {
        params: { path: { symbol }, query: { limit } },
        signal
    });
    if (error) throw new Error(`Failed to fetch rank history for ${symbol}`);
    return data as unknown as BuffettRankRow[];
}

// --- Rule-based strategies --------------------------------------------------

/**
 * The market-trend indicator: one index's close against its moving average.
 *
 * **Advisory only** (`advisory_only` is always true). No strategy acts on it —
 * gating a stock book with this signal was measured and rejected. It is market
 * context, and the UI must not present it as an instruction.
 *
 * `state` and `provisional_state` are deliberately separate. `state` is the
 * active reading, fixed at the last completed month-end. `provisional_state`
 * is what the comparison would say if the month ended today — a preview of
 * next month's reading, never the current one.
 */
export interface TrendSignal {
    advisory_only: boolean;
    signal_symbol: string;
    /** Display name for the symbol, e.g. "S&P 500" — named by the backend so
     *  every client labels a market identically. */
    signal_name: string;
    /** Zone the payload's dates were reckoned in (always a market clock). */
    market_timezone: string;
    state: 'in' | 'out';
    sma_months: number;
    /** Month-end close that set the active signal. */
    decision_date: string;
    decision_close: number;
    sma: number;
    /** The month the active signal governs, as YYYY-MM. */
    governs_month: string;
    provisional_state: 'in' | 'out';
    provisional_sma: number;
    latest_close: number;
    latest_date: string;
    /** Close at which the next month-end decision flips. */
    flip_close: number;
    /** Signed distance of the latest close from `flip_close`, in percent. */
    distance_pct: number | null;
    would_flip: boolean;
    next_decision_date: string;
    history: Array<{ date: string; close: number; sma: number | null }>;
}

export interface StrategyBacktest {
    window?: string;
    cagr?: number;
    volatility?: number;
    max_drawdown?: number;
    sharpe?: number;
    train_cagr?: number;
    test_cagr?: number;
    long_window?: string;
    long_cagr?: number;
}

/**
 * One entry in the strategy catalogue.
 *
 * There is no `trend` sleeve and no leverage field: strategies hold individual
 * common stock only. Both omissions are enforced by the backend's tests.
 */
export interface StrategyDefinition {
    id: string;
    name: string;
    summary: string;
    sleeves: Record<string, number>;
    backtest: StrategyBacktest;
    risks: string[];
    is_default: boolean;
    ranking: {
        quality_weight: number;
        top_n: number;
        max_per_sector: number | null;
        sector_digits: number;
        rebalance: string;
    };
}

export interface StrategyPosition {
    symbol: string;
    name?: string | null;
    /** Always 'stock' — no fund or cash-proxy roles exist. */
    role: 'stock';
    weight: number;
    amount: number;
    price?: number | null;
    shares?: number | null;
    cost?: number | null;
    score?: number | null;
    industry?: string | null;
    note?: string | null;
}

export interface StrategySleeve {
    key: string;
    label: string;
    weight: number;
    amount: number;
    /** How many names the rule asks for, against how many the ranking supplied. */
    positions_requested?: number;
    positions_filled?: number;
    /** Sum of the position amounts. Below `amount` when the book is short. */
    amount_allocated?: number;
    positions: StrategyPosition[];
    run_id?: number | null;
    ranked_at?: string | null;
    /**
     * Where the `price` on each position came from. Membership always comes
     * from the ranking snapshot; prices are live quotes, falling back to the
     * snapshot's stored close when a quote is unavailable.
     */
    price_source?: 'live' | 'snapshot' | 'mixed';
}

export interface StrategyAllocation {
    strategy_id: string;
    name: string;
    capital: number;
    as_of: string;
    /** Age of the ranking snapshot in whole days; null if it cannot be dated. */
    ranking_age_days?: number | null;
    /**
     * True once the snapshot is old enough that the batch worker has probably
     * stopped. The endpoints keep serving the last good run either way, so
     * without this a dead worker is indistinguishable from a healthy one.
     */
    ranking_is_stale?: boolean;
    /**
     * True when the ranking produced fewer names than the rule calls for, so
     * some capital is deliberately left unallocated rather than the weights
     * being silently widened away from the backtested rule. A matching
     * `warnings` entry says how short and by how much.
     */
    is_short?: boolean;
    sleeves: StrategySleeve[];
    warnings: string[];
}

export async function fetchStrategies(signal?: AbortSignal): Promise<{
    strategies: StrategyDefinition[];
    default: string;
}> {
    const { data, error } = await apiClient.GET("/api/strategies", { signal });
    if (error) throw new Error('Failed to fetch strategies');
    return data as unknown as { strategies: StrategyDefinition[]; default: string };
}

/**
 * The markets the trend panel reads, broadest first — mirrors
 * `MARKET_SIGNAL_INDICES` in `src/strategies.py`.
 *
 * Both legs are ETFs rather than the raw indices so the two moving averages are
 * built from the same kind of series, making a crossing in one comparable to a
 * crossing in the other. The backend also ships each reading's display name, so
 * these labels are only a fallback for a payload that predates that field.
 */
export const MARKET_TREND_INDICES = [
    { symbol: 'SPY', label: 'S&P 500' },
    { symbol: 'QQQ', label: 'NASDAQ 100' },
] as const;

export async function fetchTrendSignal(
    symbol: string = 'SPY',
    smaMonths: number = 10,
    signal?: AbortSignal
): Promise<TrendSignal> {
    const { data, error } = await apiClient.GET("/api/trend-signal", {
        params: { query: { symbol, sma_months: smaMonths } },
        signal
    });
    if (error) throw new Error('Failed to fetch the trend signal');
    return data as unknown as TrendSignal;
}

export async function fetchStrategyAllocation(
    strategyId: string,
    capital: number,
    signal?: AbortSignal
): Promise<StrategyAllocation> {
    const { data, error } = await apiClient.GET("/api/strategies/{strategy_id}/allocation", {
        params: { path: { strategy_id: strategyId }, query: { capital } },
        signal
    });
    if (error) throw new Error('Failed to build the strategy allocation');
    return data as unknown as StrategyAllocation;
}

export interface OpenLot {
    lot_id: number;
    date: string;
    account: string;
    quantity: number;
    cost_per_share_local: number;
    cost_basis_display: number;
    market_value_display: number;
    unrealized_gain_display: number;
    unrealized_gain_pct: number;
    holding_period_days: number;
    tax_term: 'short_term' | 'long_term';
}

export interface ClosedTrade {
    sell_date: string;
    account: string;
    quantity_sold: number;
    sale_price: number;
    proceeds_display: number;
    cost_basis_display: number;
    realized_gain_display: number;
    original_tx_id?: number;
}

export interface StockPositionSummary {
    quantity: number;
    current_price: number;
    market_value: number;
    avg_cost_price: number;
    cost_basis: number;
    total_buy_cost: number;
    portfolio_weight_pct?: number | null;
}

export interface StockReturnAttribution {
    unrealized_gain: number;
    unrealized_gain_pct: number;
    realized_gain: number;
    lifetime_dividends: number;
    commissions: number;
    withholding_taxes: number;
    total_gain: number;
    total_return_pct: number;
    irr_pct?: number | null;
    twrr_pct?: number | null;
    indicated_annual_dividend: number;
    yield_on_cost_pct?: number | null;
    market_yield_pct?: number | null;
    fx_gain_loss: number;
    fx_gain_loss_pct: number;
}

export interface StockPositionData {
    symbol: string;
    display_currency: string;
    local_currency: string;
    fx_rate: number;
    has_position: boolean;
    summary?: StockPositionSummary | null;
    returns?: StockReturnAttribution | null;
    open_lots: OpenLot[];
    closed_trades: ClosedTrade[];
}

export async function fetchStockPosition(
    symbol: string,
    currency: string = 'USD',
    accounts?: string[],
    signal?: AbortSignal
): Promise<StockPositionData> {
    const params = new URLSearchParams({ currency });
    if (accounts && accounts.length) {
        accounts.forEach((a) => params.append('accounts', a));
    }
    const url = `${API_BASE_URL}/stock/${encodeURIComponent(symbol)}/position?${params.toString()}`;
    const response = await authFetch(url, { signal });
    if (!response.ok) throw new Error(`Failed to fetch stock position for ${symbol}`);
    return (await response.json()) as StockPositionData;
}

export interface StockPositionHistoryPoint {
    date: string;
    value: number;
    cost_basis: number;
    shares: number;
    unrealized_gain: number;
    unrealized_gain_pct: number;
    return_pct: number;
    [key: string]: number | string | undefined;
}

export async function fetchStockPositionHistory(
    symbol: string,
    currency: string = 'USD',
    period: string = '1y',
    accounts?: string[],
    benchmarks?: string[],
    fromDate?: string,
    toDate?: string,
    signal?: AbortSignal
): Promise<StockPositionHistoryPoint[]> {
    const params = new URLSearchParams({ currency, period });
    if (accounts && accounts.length) {
        accounts.forEach((a) => params.append('accounts', a));
    }
    if (benchmarks && benchmarks.length) {
        benchmarks.forEach((b) => params.append('benchmarks', b));
    }
    if (fromDate) params.append('from', fromDate);
    if (toDate) params.append('to', toDate);

    const url = `${API_BASE_URL}/stock/${encodeURIComponent(symbol)}/position_history?${params.toString()}`;
    const response = await authFetch(url, { signal });
    if (!response.ok) throw new Error(`Failed to fetch stock position history for ${symbol}`);
    return (await response.json()) as StockPositionHistoryPoint[];
}

