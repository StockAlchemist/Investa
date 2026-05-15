import { memo, lazy, Suspense } from 'react';
import { PortfolioSummary, PerformanceData } from '../lib/api';
import { formatCurrency, cn } from '../lib/utils';
import { MetricCard } from './MetricCard'; // Use new component
import { COMPLEX_METRIC_IDS, DEFAULT_ITEMS } from '../lib/dashboard_constants';
import {
    Wallet,
    TrendingUp,
    TrendingDown,
    DollarSign,
    Percent,
    Activity,
    PiggyBank,
    Receipt,
    PieChart,
    Loader2,
    Zap
} from 'lucide-react';
import { Holding } from '@/lib/api';

// Lazy-load heavy analytics components — these pull in recharts and complex chart logic
const RiskMetrics = lazy(() => import('./RiskMetrics'));
const PortfolioDonut = lazy(() => import('./PortfolioDonut'));

// Named export wrappers for lazy loading
const SectorAttribution = lazy(() => import('./AttributionChart').then(m => ({ default: m.SectorAttribution })));
const TopContributors = lazy(() => import('./AttributionChart').then(m => ({ default: m.TopContributors })));

const AnalyticsFallback = () => (
    <div className="h-full rounded-2xl bg-muted/30 animate-pulse flex items-center justify-center min-h-[200px]">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground/40" />
    </div>
);

interface DashboardProps {
    summary: PortfolioSummary;
    currency: string;
    history?: PerformanceData[];
    isLoading?: boolean;
    isRefreshing?: boolean;
    isError?: boolean;
    onRetry?: () => void;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    riskMetrics?: any;
    riskMetricsLoading?: boolean;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    portfolioHealth?: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    attributionData?: any;
    attributionLoading?: boolean;
    holdings?: Holding[];
    visibleItems: string[];
    accounts?: string[];
    themeColor?: string;
    showClosed?: boolean;
}

function DashboardInner({
    summary,
    currency,
    history = [],
    isLoading = false,
    isRefreshing = false,
    isError = false,
    onRetry,
    riskMetrics = {},
    riskMetricsLoading = false,
    portfolioHealth = null,
    attributionData = null,
    attributionLoading = false,
    holdings = [],
    visibleItems,
    accounts,
    themeColor = 'cyan-500',
    showClosed = false
}: DashboardProps) {
    const m = summary?.metrics;
    const am = summary?.account_metrics;

    if (isError && !isLoading) {
        return (
            <div className="flex flex-col items-center justify-center p-12 gap-4">
                <p className="text-muted-foreground text-sm">Failed to load portfolio data.</p>
                {onRetry && (
                    <button
                        onClick={onRetry}
                        className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm hover:opacity-90 transition-opacity"
                    >
                        Retry
                    </button>
                )}
            </div>
        );
    }

    if (!m && !isLoading) {
        return (
            <div className="flex items-center justify-center p-12">
                <div className="animate-pulse flex flex-col items-center">
                    <div className="h-4 w-32 bg-slate-200 dark:bg-slate-700 rounded mb-4"></div>
                    <div className="h-8 w-48 bg-slate-200 dark:bg-slate-700 rounded"></div>
                </div>
            </div>
        );
    }

    // Prepare data helpers

    // Use the correctly aggregated cash balance from overall metrics
    const cashBalance = m?.cash_balance ?? null;
    const dayGL = m?.day_change_display ?? null;
    const dayGLPct = m?.day_change_percent ?? null;
    const unrealizedGL = m?.unrealized_gain ?? null;

    // Calculate Unrealized GL Percent safely
    let unrealizedGLPct: number | null = null;
    const costBasisHeld = m?.['cost_basis_held'] as number | undefined;

    if (m && m.unrealized_gain != null && costBasisHeld && costBasisHeld !== 0) {
        unrealizedGLPct = (m.unrealized_gain / costBasisHeld) * 100;
    } else if (m && m.unrealized_gain != null && (!costBasisHeld || costBasisHeld === 0)) {
        // If cost basis is 0 but we have unrealized gain, it's effectively infinite return.
        unrealizedGLPct = 0;
    }

    const fxGL = m?.fx_gain_loss_display ?? null;
    const fxGLPct = m?.fx_gain_loss_pct ?? null;

    const dayGLColor = dayGL !== null && dayGL >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500';
    const unrealizedGLColor = unrealizedGL !== null && unrealizedGL >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500';
    const fxGLColor = fxGL !== null && fxGL >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500';

    const totalGain = m?.total_gain ?? null;
    const realizedGain = m?.realized_gain ?? null;

    const totalReturnColor = totalGain !== null && totalGain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500';
    const realizedGainColor = realizedGain !== null && realizedGain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500';

    // Render helper
    const renderContent = (id: string, variant: 'card' | 'seamless' = 'card') => {
        switch (id) {
            case 'portfolioValue':
                return <MetricCard
                    title="Total Portfolio Value"
                    value={m?.market_value ?? 0}
                    valueClassName="text-2xl sm:text-4xl" // Slightly smaller generally, but hero
                    containerClassName="h-full"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    icon={Wallet}
                    accentColor={themeColor}
                    variant={variant}
                />;
            case 'dayGL':
                return <MetricCard
                    title="Day's Gain/Loss"
                    value={dayGL}
                    subValue={dayGLPct}
                    colorClass={dayGLColor}
                    valueClassName="text-xl sm:text-2xl"
                    subValueClassName={cn("", (dayGLPct ?? 0) >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400")}
                    containerClassName="h-full"
                    isHero={true}
                    currency={currency}
                    sparklineData={history.map(d => ({ value: d.twr }))}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    icon={(dayGL ?? 0) >= 0 ? TrendingUp : TrendingDown}
                    accentColor={themeColor}
                    variant={variant}
                />;
            case 'totalReturn':
                return <MetricCard
                    title="Total Return"
                    value={totalGain}
                    subValue={m?.total_return_pct}
                    colorClass={totalReturnColor}
                    valueClassName="text-xl sm:text-2xl"
                    subValueClassName={cn("", (m?.total_return_pct ?? 0) >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400")}
                    containerClassName="h-full"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    icon={Activity}
                    accentColor={themeColor}
                    variant={variant}
                />;
            case 'annualTWR':
                return <MetricCard
                    title="Total TWR"
                    value={m?.cumulative_twr !== undefined && m?.cumulative_twr !== null ?
                        `${Math.abs(m.cumulative_twr).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%` : '-'}
                    subValue={m?.annualized_twr !== undefined && m?.annualized_twr !== null ?
                        `${Math.abs(m.annualized_twr).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}% p.a.` : undefined}
                    isCurrency={false}
                    colorClass={m?.cumulative_twr && m.cumulative_twr >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500'}
                    subValueClassName={cn("", (m?.annualized_twr ?? 0) >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400")}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    icon={Percent}
                    accentColor={themeColor}
                    variant={variant}
                />;
            case 'mwr':
                return <MetricCard
                    title="IRR (MWR)"
                    value={m?.portfolio_mwr !== undefined && m?.portfolio_mwr !== null ? `${m.portfolio_mwr.toFixed(2)}%` : '-'}
                    subValue="p.a."
                    subValueClassName={cn("", (m?.portfolio_mwr ?? 0) >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400")}
                    isCurrency={false}
                    colorClass={m?.portfolio_mwr && m.portfolio_mwr >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500'}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    icon={Activity}
                    accentColor={themeColor}
                    variant={variant}
                />;
            case 'unrealizedGL':
                return <MetricCard
                    title="Unrealized G/L"
                    value={unrealizedGL}
                    subValue={unrealizedGLPct}
                    colorClass={unrealizedGLColor}
                    valueClassName="text-xl sm:text-2xl"
                    subValueClassName={cn("", (unrealizedGLPct ?? 0) >= 0 ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" : "bg-red-500/10 text-red-600 dark:text-red-400")}
                    containerClassName="h-full"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    icon={TrendingUp} // Or separate icon
                    accentColor={themeColor}
                    variant={variant}
                />;
            case 'fxGL':
                return <MetricCard
                    title="FX Gain/Loss"
                    value={fxGL}
                    subValue={fxGLPct}
                    colorClass={fxGLColor}
                    subValueClassName={cn("", (fxGLPct ?? 0) >= 0 ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" : "bg-red-500/10 text-red-600 dark:text-red-400")}
                    containerClassName="h-full"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    icon={DollarSign}
                    accentColor={themeColor}
                    variant={variant}
                />;
            case 'realizedGain':
                return <MetricCard
                    title="Realized Gain"
                    value={realizedGain}
                    colorClass={realizedGainColor}
                    valueClassName="text-xl sm:text-2xl"
                    containerClassName="h-full"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    icon={PiggyBank}
                    accentColor={themeColor}
                    variant={variant}
                />;
            case 'cashBalance':
                return <MetricCard
                    title="Cash Balance"
                    value={cashBalance}
                    currency={currency}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    icon={DollarSign}
                    valueClassName="text-xl sm:text-2xl"
                    accentColor={themeColor}
                    variant={variant}
                />;
            case 'ytdDividends':
                return <MetricCard
                    title="Total Dividends"
                    value={m?.dividends ?? 0}
                    valueClassName="text-xl sm:text-2xl"
                    containerClassName="h-full"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    icon={DollarSign} // Or create a generic dividend icon
                    accentColor={themeColor}
                    variant={variant}
                />;
            case 'dividendYield':
                return <MetricCard
                    title="Dividend Yield %"
                    value={m?.dividend_return_cumulative !== undefined && m?.dividend_return_cumulative !== null ?
                        `${Math.abs(m.dividend_return_cumulative).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%` : '-'}
                    subValue={m?.dividend_return_annualized !== undefined && m?.dividend_return_annualized !== null ?
                        `${Math.abs(m.dividend_return_annualized).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}% p.a.` : undefined}
                    isCurrency={false}
                    colorClass={m?.dividend_return_cumulative && m.dividend_return_cumulative >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500'}
                    subValueClassName={cn("", (m?.dividend_return_annualized ?? 0) >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400")}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    icon={Percent}
                    accentColor={themeColor}
                    variant={variant}
                />;
            case 'ytdReturn':
                return <MetricCard
                    title="YTD Return"
                    value={riskMetrics?.['YTD Return'] !== undefined && riskMetrics?.['YTD Return'] !== null ? `${(riskMetrics['YTD Return'] * 100).toFixed(2)}%` :
                        m?.ytd_return !== undefined && m?.ytd_return !== null ? `${m.ytd_return.toFixed(2)}%` : '-'}
                    isCurrency={false}
                    colorClass={(riskMetrics?.['YTD Return'] ?? m?.ytd_return ?? 0) >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500'}
                    isLoading={isLoading || riskMetricsLoading}
                    isRefreshing={isRefreshing}
                    icon={TrendingUp}
                    accentColor={themeColor}
                    variant={variant}
                />;
            case 'maxDrawdown':
                const maxDD = riskMetrics?.['Max Drawdown'] !== undefined ? riskMetrics['Max Drawdown'] * 100 : m?.max_drawdown;
                return <MetricCard
                    title="Max Drawdown"
                    value={maxDD !== undefined && maxDD !== null ? `${maxDD.toFixed(2)}%` : '-'}
                    isCurrency={false}
                    colorClass="text-red-600 dark:text-red-500"
                    isLoading={isLoading || riskMetricsLoading}
                    isRefreshing={isRefreshing}
                    icon={TrendingDown}
                    accentColor={themeColor}
                    variant={variant}
                />;
            case 'volatility':
                const vol = riskMetrics?.['Volatility (Ann.)'] !== undefined ? riskMetrics['Volatility (Ann.)'] * 100 : m?.volatility_ann;
                return <MetricCard
                    title="Volatility (Ann.)"
                    value={vol !== undefined && vol !== null ? `${vol.toFixed(2)}%` : '-'}
                    isCurrency={false}
                    isLoading={isLoading || riskMetricsLoading}
                    isRefreshing={isRefreshing}
                    icon={Activity}
                    accentColor={themeColor}
                    variant={variant}
                />;
            case 'sharpeRatio':
                const sharpe = riskMetrics?.['Sharpe Ratio'] ?? m?.sharpe_ratio;
                return <MetricCard
                    title="Sharpe Ratio"
                    value={sharpe !== undefined && sharpe !== null ? sharpe.toFixed(2) : '-'}
                    isCurrency={false}
                    colorClass={(sharpe ?? 0) >= 1 ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'}
                    isLoading={isLoading || riskMetricsLoading}
                    isRefreshing={isRefreshing}
                    icon={Zap}
                    accentColor={themeColor}
                    variant={variant}
                />;
            case 'beta':
                const beta = riskMetrics?.['Beta'] ?? m?.beta;
                return <MetricCard
                    title="Portfolio Beta"
                    value={beta !== undefined && beta !== null ? beta.toFixed(2) : '-'}
                    isCurrency={false}
                    colorClass={(beta ?? 1) > 1.2 ? 'text-amber-600 dark:text-amber-400' : 'text-emerald-600 dark:text-emerald-400'}
                    isLoading={isLoading || riskMetricsLoading}
                    isRefreshing={isRefreshing}
                    icon={Activity}
                    accentColor={themeColor}
                    variant={variant}
                />;
            case 'fees':
                return <MetricCard
                    title="Fees"
                    value={m?.commissions ?? 0}
                    colorClass="text-red-600 dark:text-red-500"
                    currency={currency}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    icon={Receipt}
                    accentColor={themeColor}
                    variant={variant}
                />;
            case 'riskMetrics':
                return <Suspense fallback={<AnalyticsFallback />}><RiskMetrics metrics={riskMetrics} portfolioHealth={portfolioHealth} isLoading={riskMetricsLoading!} isRefreshing={isRefreshing} /></Suspense>;
            case 'sectorContribution':
                return <Suspense fallback={<AnalyticsFallback />}><SectorAttribution data={attributionData} isLoading={attributionLoading!} isRefreshing={isRefreshing} currency={currency} /></Suspense>;
            case 'topContributors':
                return <Suspense fallback={<AnalyticsFallback />}><TopContributors data={attributionData} isLoading={attributionLoading!} isRefreshing={isRefreshing} currency={currency} accounts={accounts} showClosed={showClosed} /></Suspense>;
            case 'portfolioDonut':
                return (
                    <div className="metric-card card-shine h-full p-5 relative overflow-hidden group">
                        {/* Accent top bar */}
                        <div className="absolute top-0 left-5 right-5 h-[2px] rounded-full bg-cyan-500 opacity-50" />
                        <div className="h-full relative z-10">
                            <div className="flex justify-between items-start mb-4">
                                <div className="flex items-center gap-2">
                                    <h3 className="section-label">Portfolio Composition</h3>
                                    {isRefreshing && !isLoading && (
                                        <Loader2 className="w-2.5 h-2.5 animate-spin text-muted-foreground/40" />
                                    )}
                                </div>
                                <div className="p-2 rounded-xl bg-cyan-500/15 dark:bg-cyan-500/20 text-cyan-500 group-hover:scale-110 group-hover:rotate-3 transition-all duration-300">
                                    <PieChart className="w-4 h-4" />
                                </div>
                            </div>
                            <div className="h-[calc(100%-48px)]">
                                <Suspense fallback={<AnalyticsFallback />}>
                                    <PortfolioDonut holdings={holdings} currency={currency} />
                                </Suspense>
                            </div>
                        </div>
                    </div>
                );
            default:
                return null;
        }
    };

    const visibleScalarItems = DEFAULT_ITEMS.filter(item => visibleItems.includes(item.id) && !COMPLEX_METRIC_IDS.includes(item.id));
    const visibleComplexItems = DEFAULT_ITEMS.filter(item => visibleItems.includes(item.id) && COMPLEX_METRIC_IDS.includes(item.id));

    return (
        <div className="mb-4 md:mb-14 space-y-6 md:space-y-10">
            {/* Scalar Metrics Grid */}
            {visibleScalarItems.length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-2 lg:grid-cols-4 gap-3 md:gap-4">
                    {visibleScalarItems.map((item) => (
                        <div key={item.id} className={cn(item.colSpan, "w-full min-w-0")}>
                            {renderContent(item.id, 'seamless')}
                        </div>
                    ))}
                </div>
            )}

            {/* Complex/Tall Metrics Grid */}
            {visibleComplexItems.length > 0 && (
                <>
                    {/* Visual separator */}
                    <div className="flex items-center gap-3">
                        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-border to-transparent" />
                        <span className="section-label tracking-[0.2em]">Analytics</span>
                        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-border to-transparent" />
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
                        {visibleComplexItems.map((item) => (
                            <div key={item.id} className={item.colSpan}>
                                {renderContent(item.id)}
                            </div>
                        ))}
                    </div>
                </>
            )}
        </div>
    );
}

const Dashboard = memo(DashboardInner);
export default Dashboard;
