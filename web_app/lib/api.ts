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

const API_BASE_URL = getApiBaseUrl();

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
    const res = await fetch(`${API_BASE_URL}/summary?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch summary');
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
    const res = await fetch(`${API_BASE_URL}/holdings?${params.toString()}`, { signal, cache: 'no-store' });
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
    const res = await fetch(`${API_BASE_URL}/transactions?${params.toString()}`, { signal, cache: 'no-store' });
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
    const response = await fetch(`${API_BASE_URL}/transactions`, {
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
    const response = await fetch(`${API_BASE_URL}/transactions/${id}`, {
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
    const response = await fetch(`${API_BASE_URL}/transactions/${id}`, {
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

    const res = await fetch(`${API_BASE_URL}/history?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch history');
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
    const res = await fetch(`${API_BASE_URL}/asset_change?${params.toString()}`, { signal, cache: 'no-store' });
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
    const res = await fetch(`${API_BASE_URL}/capital_gains?${params.toString()}`, { signal, cache: 'no-store' });
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
    const res = await fetch(`${API_BASE_URL}/dividends?${params.toString()}`, { signal, cache: 'no-store' });
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
}

export async function fetchSettings(): Promise<Settings> {
    const res = await fetch(`${API_BASE_URL}/settings`);
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
}

export async function updateSettings(settings: SettingsUpdate): Promise<StatusResponse> {
    const response = await fetch(`${API_BASE_URL}/settings/update`, {
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
    const res = await fetch(`${API_BASE_URL}/risk_metrics?${params.toString()}`, { signal, cache: 'no-store' });
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

export async function fetchAttribution(currency: string = 'USD', accounts?: string[], signal?: AbortSignal): Promise<AttributionData> {
    const params = new URLSearchParams({ currency });
    if (accounts) {
        accounts.forEach(acc => params.append('accounts', acc));
    }
    const res = await fetch(`${API_BASE_URL}/attribution?${params.toString()}`, { signal, cache: 'no-store' });
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
    const res = await fetch(`${API_BASE_URL}/dividend_calendar?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch dividend calendar');
    return res.json();
}

export async function saveManualOverride(symbol: string, price: number | null): Promise<StatusResponse> {
    const response = await fetch(`${API_BASE_URL}/settings/manual_overrides`, {
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
    const response = await fetch(`${API_BASE_URL}/webhook/refresh`, {
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
export interface CorrelationData {
    assets: string[];
    correlation: { x: string; y: string; value: number }[];
}

export async function fetchCorrelationMatrix(
    period: string = '1y',
    accounts?: string[],
    signal?: AbortSignal
): Promise<CorrelationData> {
    const params = new URLSearchParams({ period });
    if (accounts) {
        accounts.forEach(acc => params.append('accounts', acc));
    }
    const res = await fetch(`${API_BASE_URL}/correlation?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch correlation matrix');
    return res.json();
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
    const res = await fetch(`${API_BASE_URL}/projected_income?${params.toString()}`, { signal, cache: 'no-store' });
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
    const res = await fetch(`${API_BASE_URL}/portfolio_health?${params.toString()}`, { signal, cache: 'no-store' });
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
}

export async function fetchWatchlist(currency: string = 'USD', signal?: AbortSignal): Promise<WatchlistItem[]> {
    const params = new URLSearchParams({ currency });
    const res = await fetch(`${API_BASE_URL}/watchlist?${params.toString()}`, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch watchlist');
    return res.json();
}

export async function addToWatchlist(symbol: string, note: string = ""): Promise<StatusResponse> {
    const response = await fetch(`${API_BASE_URL}/watchlist`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ symbol, note }),
    });
    if (!response.ok) {
        throw new Error(`Failed to add to watchlist: ${response.statusText}`);
    }
    return response.json();
}

export async function removeFromWatchlist(symbol: string): Promise<StatusResponse> {
    const response = await fetch(`${API_BASE_URL}/watchlist/${symbol}`, {
        method: "DELETE",
    });
    if (!response.ok) {
        throw new Error(`Failed to remove from watchlist: ${response.statusText}`);
    }
    return response.json();
}

export async function updateHoldingTags(account: string, symbol: string, tags: string): Promise<StatusResponse> {
    const response = await fetch(`${API_BASE_URL}/holdings/update_tags`, {
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

export async function fetchFundamentals(symbol: string): Promise<Fundamentals> {
    const res = await fetch(`${API_BASE_URL}/fundamentals/${symbol}`);
    if (!res.ok) throw new Error(`Failed to fetch fundamentals for ${symbol}`);
    return res.json();
}

export async function fetchFinancials(symbol: string, periodType: 'annual' | 'quarterly' = 'annual'): Promise<FinancialsResponse> {
    const params = new URLSearchParams({ period_type: periodType });
    const res = await fetch(`${API_BASE_URL}/financials/${symbol}?${params.toString()}`);
    if (!res.ok) throw new Error(`Failed to fetch financials for ${symbol}`);
    return res.json();
}

export async function fetchRatios(symbol: string): Promise<RatiosResponse> {
    const res = await fetch(`${API_BASE_URL}/ratios/${symbol}`);
    if (!res.ok) throw new Error(`Failed to fetch ratios for ${symbol}`);
    return res.json();
}
export async function clearCache(): Promise<StatusResponse> {
    const response = await fetch(`${API_BASE_URL}/clear_cache`, {
        method: "POST",
    });
    if (!response.ok) {
        throw new Error(`Failed to clear cache: ${response.statusText}`);
    }
    return response.json();
}
