import { memo, lazy, Suspense, useState, useEffect, useRef, useMemo } from 'react';
import { PortfolioSummary, PerformanceData, DividendEvent } from '../lib/api';
import { formatCurrency, cn } from '../lib/utils';
import { MetricCard } from './MetricCard';
import { COMPLEX_METRIC_IDS, DEFAULT_ITEMS, TOP_SECTION_IDS } from '../lib/dashboard_constants';
import {
    Wallet, TrendingUp, TrendingDown, DollarSign, Percent,
    Activity, PiggyBank, Receipt, PieChart, Loader2, Zap,
    ArrowUpRight, ArrowDownRight,
} from 'lucide-react';
import { AreaChart, Area, ResponsiveContainer, YAxis } from 'recharts';
import { Holding } from '@/lib/api';
import { Skeleton } from '@/components/ui/skeleton';
import TodayStrip from './dashboard/TodayStrip';
import DashboardEvents from './dashboard/DashboardEvents';
import DashboardInsights from './dashboard/DashboardInsights';

const RiskMetrics       = lazy(() => import('./RiskMetrics'));
const PortfolioDonut    = lazy(() => import('./PortfolioDonut'));
const SectorAttribution = lazy(() => import('./AttributionChart').then(m => ({ default: m.SectorAttribution })));
const TopContributors   = lazy(() => import('./AttributionChart').then(m => ({ default: m.TopContributors })));

const AnalyticsFallback = () => (
    <div className="h-full rounded-xl bg-muted/30 animate-pulse flex items-center justify-center min-h-[200px]">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground/40" />
    </div>
);

interface DashboardProps {
    summary: PortfolioSummary;
    currency: string;
    history?: PerformanceData[];
    wtdHistory?: PerformanceData[];
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
    /** Complex widget IDs to exclude from the Analytics grid — useful when the
     *  parent wants to render them in a specific position (e.g. risk metrics
     *  rendered after the performance graph on the dashboard tab). */
    excludeFromAnalytics?: string[];
    /** Upcoming dividend events surfaced in the Events panel. */
    dividendEvents?: DividendEvent[];
    /** Longer (1y/daily) history used by the hero period selector. */
    longHistory?: PerformanceData[];
    /** Header index quotes, fetched separately from /summary. Falls back to
     *  summary.metrics.indices when not provided. */
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    indices?: Record<string, any>;
}

// ── Animated number hook ─────────────────────────────────────────────────────
function useAnimatedNumber(target: number, duration = 600): number {
    const [current, setCurrent] = useState(target);
    const prevRef = useRef(target);
    const rafRef  = useRef<number>(0);
    const startRef = useRef<number>(0);

    useEffect(() => {
        const from = prevRef.current;
        if (from === target) return;
        prevRef.current = target;
        const startTime = performance.now();
        startRef.current = startTime;

        const tick = (now: number) => {
            const elapsed = now - startTime;
            const t = Math.min(elapsed / duration, 1);
            const ease = 1 - Math.pow(1 - t, 3); // ease-out-cubic
            setCurrent(from + (target - from) * ease);
            if (t < 1) rafRef.current = requestAnimationFrame(tick);
        };
        cancelAnimationFrame(rafRef.current);
        rafRef.current = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(rafRef.current);
    }, [target, duration]);

    return current;
}

// ── Portfolio hero card ───────────────────────────────────────────────────────
interface HeroCardProps {
    marketValue: number | null | undefined;
    dayGL: number | null;
    dayGLPct: number | null;
    cumTWR: number | null | undefined;
    annTWR: number | null | undefined;
    irr: number | null | undefined;
    currency: string;
    isLoading: boolean;
    isRefreshing: boolean;
    themeColor: string;
    history?: PerformanceData[];      // intraday (1d / 5m)
    wtdHistory?: PerformanceData[];   // intraday (5d / 15m)
    longHistory?: PerformanceData[];  // 1y / daily
}

type HeroPeriod = 'day' | 'wtd' | 'mtd' | 'ytd' | '1y';

const HERO_PERIODS: { key: HeroPeriod; label: string }[] = [
    { key: 'day', label: '1D' },
    { key: 'wtd', label: 'WTD' },
    { key: 'mtd', label: 'MTD' },
    { key: 'ytd', label: 'YTD' },
    { key: '1y', label: '1Y' },
];

// Cutoff date (UTC midnight) for a given period — anything on/after this counts.
function periodCutoff(period: HeroPeriod, now: Date): Date {
    const d = new Date(now);
    d.setHours(0, 0, 0, 0);
    if (period === 'wtd') {
        // Monday-anchored week.
        const dow = (d.getDay() + 6) % 7; // 0..6, Monday = 0
        d.setDate(d.getDate() - dow);
    } else if (period === 'mtd') {
        d.setDate(1);
    } else if (period === 'ytd') {
        d.setMonth(0, 1);
    } else if (period === '1y') {
        d.setFullYear(d.getFullYear() - 1);
    } // 'day' handled by caller (uses intraday history)
    return d;
}

// Tiny intraday sparkline showing today's portfolio value path. Returns null if
// there isn't enough variation to draw a meaningful line.
function HeroSparkline({ history, positive }: { history: PerformanceData[]; positive: boolean }) {
    const series = history
        .filter(d => typeof d.value === 'number')
        .map(d => ({ value: d.value as number }));
    if (series.length < 2) return null;
    const values = series.map(s => s.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    if (max - min < Math.max(0.01, max * 0.0005)) return null; // ~flat — skip

    const stroke = positive ? '#10b981' : '#ef4444';
    const gradId = `hero-spark-${positive ? 'up' : 'dn'}`;
    return (
        <div className="h-12 w-full mt-3">
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={series} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
                    <defs>
                        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={stroke} stopOpacity={0.35} />
                            <stop offset="95%" stopColor={stroke} stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <YAxis hide domain={[(d: number) => d * 0.999, (d: number) => d * 1.001]} />
                    <Area
                        type="monotone"
                        dataKey="value"
                        stroke={stroke}
                        strokeWidth={2}
                        fill={`url(#${gradId})`}
                        dot={false}
                        isAnimationActive={false}
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
}

function StatPill({
    label, value, sub, isLoading,
}: {
    label: string;
    value: number | null | undefined;
    sub?: string;
    isLoading: boolean;
}) {
    if (value == null) return null;
    const positive = value >= 0;
    return (
        <div className="flex flex-col gap-1 min-w-0">
            <p className="section-label text-[10px] uppercase tracking-wider">{label}</p>
            {isLoading ? (
                <Skeleton className="h-6 w-16 rounded" />
            ) : (
                <p className={cn(
                    'text-lg sm:text-xl font-bold tabular-nums leading-none',
                    positive ? 'text-emerald-500' : 'text-red-500',
                )}>
                    {positive ? '+' : ''}{value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%
                </p>
            )}
            {sub && !isLoading && (
                <p className="text-[10px] text-muted-foreground tabular-nums leading-none">{sub}</p>
            )}
        </div>
    );
}

function PortfolioHeroCard({
    marketValue, dayGL, dayGLPct, cumTWR, annTWR, irr,
    currency, isLoading, isRefreshing, themeColor, history, wtdHistory, longHistory,
}: HeroCardProps) {
    const animatedValue  = useAnimatedNumber(marketValue ?? 0);
    const animatedDayGL  = useAnimatedNumber(dayGL ?? 0);
    const animatedDayPct = useAnimatedNumber(dayGLPct ?? 0);
    const positive = (dayGL ?? 0) >= 0;
    const hasPerf = cumTWR != null || irr != null;
    void themeColor;

    const [heroPeriod, setHeroPeriod] = useState<HeroPeriod>('day');
    // Freeze "now" at mount so re-renders don't shift period boundaries mid-session.
    const [now] = useState<number>(() => Date.now());

    // Derive the series + period return for the currently-selected window.
    const periodView = useMemo(() => {
        if (heroPeriod === 'day') {
            const series = (history ?? []).filter(d => typeof d.value === 'number').map(d => ({ value: d.value as number }));
            return { series, pct: dayGLPct ?? null, abs: dayGL ?? null };
        }
        const historyToUse = heroPeriod === 'wtd' ? wtdHistory : longHistory;
        const longRows = (historyToUse ?? []).filter(d => typeof d.value === 'number' && d.date);
        if (longRows.length === 0) return { series: [], pct: null, abs: null };
        const cutoff = periodCutoff(heroPeriod, new Date(now));
        const cutoffMs = cutoff.getTime();
        const sliced = longRows.filter(d => new Date(d.date).getTime() >= cutoffMs);
        // Anchor the period to the last point before the cutoff so the first
        // delta represents the move INTO the period (avoids a misleading flat
        // baseline if the cutoff falls between data points).
        const beforeIdx = longRows.findIndex(d => new Date(d.date).getTime() >= cutoffMs);
        const anchor = beforeIdx > 0 ? longRows[beforeIdx - 1] : (sliced[0] ?? longRows[0]);
        const tail = sliced.length > 0 ? sliced : longRows.slice(-1);
        const series = [anchor, ...tail].map(d => ({ value: d.value as number }));
        const startVal = anchor?.value as number;
        const endVal = tail[tail.length - 1]?.value as number;
        if (!startVal || !endVal) return { series, pct: null, abs: null };
        const abs = endVal - startVal;
        const pct = (abs / startVal) * 100;
        return { series, pct, abs };
    }, [heroPeriod, history, longHistory, dayGL, dayGLPct, now]);

    const periodPositive = (periodView.pct ?? 0) >= 0;
    const sparklinePositive = heroPeriod === 'day' ? positive : periodPositive;

    return (
        <div className="metric-card card-shine relative overflow-hidden p-5 sm:p-6">
            <div className="flex flex-wrap items-center gap-x-8 gap-y-4">

                {/* Left: main value + day change */}
                <div className="min-w-0 flex-1" style={{ minWidth: 'min(100%, 260px)' }}>
                    <div className="flex items-center gap-2 mb-2">
                        <Wallet className="w-3.5 h-3.5 text-muted-foreground/60 shrink-0" />
                        <span className="section-label">Total Portfolio Value</span>
                        {isRefreshing && !isLoading && (
                            <Loader2 className="w-3 h-3 animate-spin text-muted-foreground/40" />
                        )}
                    </div>

                    {isLoading ? (
                        <div className="space-y-2">
                            <Skeleton className="h-10 w-52 rounded-lg" />
                            <Skeleton className="h-5 w-36 rounded-lg" />
                        </div>
                    ) : (
                        <div className="flex items-baseline gap-4 flex-wrap">
                            <span className="text-3xl sm:text-4xl md:text-5xl font-black tabular-nums text-foreground leading-none tracking-tight whitespace-nowrap">
                                {formatCurrency(animatedValue, currency)}
                            </span>

                            {dayGL !== null && (
                                <div className={cn(
                                    'flex items-center gap-2 flex-wrap',
                                    positive ? 'text-emerald-500' : 'text-red-500',
                                )}>
                                    {positive
                                        ? <ArrowUpRight className="w-4 h-4 shrink-0" />
                                        : <ArrowDownRight className="w-4 h-4 shrink-0" />}
                                    <span className="text-lg font-semibold tabular-nums">
                                        {animatedDayGL >= 0 ? '+' : ''}{formatCurrency(animatedDayGL, currency)}
                                    </span>
                                    {dayGLPct !== null && (
                                        <span className={cn(
                                            'text-sm font-bold px-2.5 py-0.5 rounded-full',
                                            positive ? 'bg-emerald-500/10' : 'bg-red-500/10',
                                        )}>
                                            {animatedDayPct >= 0 ? '+' : ''}{animatedDayPct.toFixed(2)}%
                                        </span>
                                    )}
                                    <span className="text-sm text-muted-foreground font-normal">today</span>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Right: performance stats separated by vertical dividers */}
                {hasPerf && (
                    <div className="hidden sm:flex items-stretch gap-0 divide-x divide-border/50">
                        <div className="px-6 first:pl-0">
                            <StatPill label="Total TWR" value={cumTWR} isLoading={isLoading} />
                        </div>
                        {annTWR != null && (
                            <div className="px-6">
                                <StatPill label="Ann. TWR" value={annTWR} sub="p.a." isLoading={isLoading} />
                            </div>
                        )}
                        {irr != null && (
                            <div className="px-6">
                                <StatPill label="IRR (MWR)" value={irr} sub="p.a." isLoading={isLoading} />
                            </div>
                        )}
                    </div>
                )}
            </div>
            {/* Period selector + sparkline + period return */}
            {!isLoading && (
                <div className="mt-4">
                    <div className="flex flex-wrap items-center justify-between gap-3 mb-1.5">
                        <div className="inline-flex rounded-lg bg-secondary p-0.5">
                            {HERO_PERIODS.map(p => (
                                <button
                                    key={p.key}
                                    type="button"
                                    onClick={() => setHeroPeriod(p.key)}
                                    className={cn(
                                        'px-2.5 py-1 rounded-md text-xs font-semibold transition-all',
                                        heroPeriod === p.key
                                            ? 'bg-[#0097b2] text-white'
                                            : 'text-muted-foreground hover:text-foreground',
                                    )}
                                >
                                    {p.label}
                                </button>
                            ))}
                        </div>
                        {periodView.pct != null && periodView.abs != null && heroPeriod !== 'day' && (
                            <div className={cn(
                                'inline-flex items-baseline gap-2 tabular-nums',
                                periodPositive ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400',
                            )}>
                                <span className="text-sm font-bold">
                                    {periodPositive ? '+' : ''}{periodView.pct.toFixed(2)}%
                                </span>
                                <span className="text-xs text-muted-foreground/80 font-medium">
                                    ({periodPositive ? '+' : ''}{formatCurrency(periodView.abs, currency)})
                                </span>
                                <span className="text-[10px] uppercase tracking-wider text-muted-foreground/60 font-semibold">
                                    {HERO_PERIODS.find(p => p.key === heroPeriod)?.label}
                                </span>
                            </div>
                        )}
                    </div>
                    {periodView.series.length > 1 ? (
                        <HeroSparkline history={periodView.series as PerformanceData[]} positive={sparklinePositive} />
                    ) : heroPeriod !== 'day' ? (
                        // Only shown for explicitly selected longer periods — on 1D
                        // an empty series usually just means intraday history is
                        // still loading, where this message would mislead.
                        <p className="text-[11px] text-muted-foreground/60 mt-2">
                            Not enough history yet to chart this period.
                        </p>
                    ) : null}
                </div>
            )}
        </div>
    );
}

// ── Main dashboard component ──────────────────────────────────────────────────
function DashboardInner({
    summary,
    currency,
    history = [],
    wtdHistory = [],
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
    themeColor = 'indigo-500',
    showClosed = false,
    excludeFromAnalytics = [],
    dividendEvents = [],
    longHistory = [],
    indices,
}: DashboardProps) {
    const m  = summary?.metrics;

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
                <div className="animate-pulse flex flex-col items-center gap-3">
                    <div className="h-4 w-32 bg-muted rounded" />
                    <div className="h-8 w-48 bg-muted rounded" />
                </div>
            </div>
        );
    }

    const cashBalance    = m?.cash_balance ?? null;
    const dayGL          = m?.day_change_display ?? null;
    const dayGLPct       = m?.day_change_percent ?? null;
    const unrealizedGL   = m?.unrealized_gain ?? null;
    const fxGL           = m?.fx_gain_loss_display ?? null;
    const fxGLPct        = m?.fx_gain_loss_pct ?? null;
    const totalGain      = m?.total_gain ?? null;
    const realizedGain   = m?.realized_gain ?? null;

    // When every account in the current selection has a closure date <= today,
    // the backend gates rate-of-return metrics to null. The cards listed in
    // GATED_WHEN_CLOSED render "Closed" instead of "-" so the user understands
    // why the number is missing (vs. a transient calculation error).
    const allSelectedClosed = m?.all_selected_closed === true;
    const closedAccounts    = m?.closed_accounts ?? [];
    const GATED_WHEN_CLOSED = new Set(['totalReturn', 'annualTWR', 'mwr', 'ytdReturn', 'dividendYield']);

    let unrealizedGLPct: number | null = null;
    const costBasisHeld = m?.['cost_basis_held'] as number | undefined;
    if (m && m.unrealized_gain != null && costBasisHeld && costBasisHeld !== 0)
        unrealizedGLPct = (m.unrealized_gain / costBasisHeld) * 100;
    else if (m && m.unrealized_gain != null)
        unrealizedGLPct = 0;

    const pos = (v: number | null) => v !== null && v >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500';
    const subBadge = (v: number | null | undefined) =>
        v != null && v >= 0
            ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/20'
            : 'bg-red-500/10 text-red-700 dark:text-red-400 border-red-500/20';

    const renderContent = (id: string, variant: 'card' | 'seamless' = 'card') => {
        if (allSelectedClosed && GATED_WHEN_CLOSED.has(id)) {
            const titleMap: Record<string, string> = {
                totalReturn: 'Total Return',
                annualTWR: 'Total TWR',
                mwr: 'IRR (MWR)',
                ytdReturn: 'YTD Return',
                dividendYield: 'Dividend Yield',
            };
            return (
                <MetricCard
                    title={titleMap[id]}
                    value="Closed"
                    isCurrency={false}
                    colorClass="text-muted-foreground"
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    accentColor={themeColor}
                    variant={variant}
                />
            );
        }
        switch (id) {
            case 'portfolioValue':
            case 'dayGL':
                return null; // handled by hero card
            case 'totalReturn':
                return <MetricCard title="Total Return" value={totalGain} subValue={m?.total_return_pct} colorClass={pos(totalGain)} valueClassName="text-xl sm:text-2xl" containerClassName="h-full" isHero currency={currency} isLoading={isLoading} isRefreshing={isRefreshing} icon={Activity} accentColor={themeColor} variant={variant} />;
            case 'annualTWR':
                return <MetricCard title="Total TWR" value={m?.cumulative_twr != null ? `${Math.abs(m.cumulative_twr).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%` : '-'} subValue={m?.annualized_twr != null ? `${Math.abs(m.annualized_twr).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}% p.a.` : undefined} subValueClassName={subBadge(m?.annualized_twr)} isCurrency={false} colorClass={pos(m?.cumulative_twr ?? null)} isLoading={isLoading} isRefreshing={isRefreshing} icon={Percent} accentColor={themeColor} variant={variant} />;
            case 'mwr':
                return <MetricCard title="IRR (MWR)" value={m?.portfolio_mwr != null ? `${m.portfolio_mwr.toFixed(2)}%` : '-'} subValue="p.a." subValueClassName={subBadge(m?.portfolio_mwr)} isCurrency={false} colorClass={pos(m?.portfolio_mwr ?? null)} isLoading={isLoading} isRefreshing={isRefreshing} icon={Activity} accentColor={themeColor} variant={variant} />;
            case 'unrealizedGL':
                return <MetricCard title="Unrealized G/L" value={unrealizedGL} subValue={unrealizedGLPct} colorClass={pos(unrealizedGL)} valueClassName="text-xl sm:text-2xl" containerClassName="h-full" isHero currency={currency} isLoading={isLoading} isRefreshing={isRefreshing} icon={TrendingUp} accentColor={themeColor} variant={variant} />;
            case 'fxGL':
                return <MetricCard title="FX Gain/Loss" value={fxGL} subValue={fxGLPct} colorClass={pos(fxGL)} containerClassName="h-full" isHero currency={currency} isLoading={isLoading} isRefreshing={isRefreshing} icon={DollarSign} accentColor={themeColor} variant={variant} />;
            case 'realizedGain':
                return <MetricCard title="Realized Gain" value={realizedGain} colorClass={pos(realizedGain)} valueClassName="text-xl sm:text-2xl" containerClassName="h-full" isHero currency={currency} isLoading={isLoading} isRefreshing={isRefreshing} icon={PiggyBank} accentColor={themeColor} variant={variant} />;
            case 'cashBalance':
                return <MetricCard title="Cash Balance" value={cashBalance} currency={currency} isLoading={isLoading} isRefreshing={isRefreshing} icon={DollarSign} valueClassName="text-xl sm:text-2xl" accentColor={themeColor} variant={variant} />;
            case 'ytdDividends':
                return <MetricCard title="Total Dividends" value={m?.dividends ?? 0} valueClassName="text-xl sm:text-2xl" containerClassName="h-full" isHero currency={currency} isLoading={isLoading} isRefreshing={isRefreshing} icon={DollarSign} accentColor={themeColor} variant={variant} />;
            case 'dividendYield':
                return <MetricCard title="Dividend Yield" value={m?.dividend_return_cumulative != null ? `${Math.abs(m.dividend_return_cumulative).toFixed(2)}%` : '-'} subValue={m?.dividend_return_annualized != null ? `${Math.abs(m.dividend_return_annualized).toFixed(2)}% p.a.` : undefined} subValueClassName={subBadge(m?.dividend_return_annualized)} isCurrency={false} colorClass={pos(m?.dividend_return_cumulative ?? null)} isLoading={isLoading} isRefreshing={isRefreshing} icon={Percent} accentColor={themeColor} variant={variant} />;
            case 'ytdReturn':
                return <MetricCard title="YTD Return" value={riskMetrics?.['YTD Return'] != null ? `${(riskMetrics['YTD Return'] * 100).toFixed(2)}%` : m?.ytd_return != null ? `${m.ytd_return.toFixed(2)}%` : '-'} isCurrency={false} colorClass={pos((riskMetrics?.['YTD Return'] ?? m?.ytd_return ?? 0))} isLoading={isLoading || riskMetricsLoading} isRefreshing={isRefreshing} icon={TrendingUp} accentColor={themeColor} variant={variant} />;
            case 'maxDrawdown': {
                const maxDD = riskMetrics?.['Max Drawdown'] != null ? riskMetrics['Max Drawdown'] * 100 : m?.max_drawdown;
                return <MetricCard title="Max Drawdown" value={maxDD != null ? `${maxDD.toFixed(2)}%` : '-'} isCurrency={false} colorClass="text-red-600 dark:text-red-500" isLoading={isLoading || riskMetricsLoading} isRefreshing={isRefreshing} icon={TrendingDown} accentColor={themeColor} variant={variant} />;
            }
            case 'volatility': {
                const vol = riskMetrics?.['Volatility (Ann.)'] != null ? riskMetrics['Volatility (Ann.)'] * 100 : m?.volatility_ann;
                return <MetricCard title="Volatility (Ann.)" value={vol != null ? `${vol.toFixed(2)}%` : '-'} isCurrency={false} isLoading={isLoading || riskMetricsLoading} isRefreshing={isRefreshing} icon={Activity} accentColor={themeColor} variant={variant} />;
            }
            case 'sharpeRatio': {
                const sharpe = riskMetrics?.['Sharpe Ratio'] ?? m?.sharpe_ratio;
                return <MetricCard title="Sharpe Ratio" value={sharpe != null ? sharpe.toFixed(2) : '-'} isCurrency={false} colorClass={(sharpe ?? 0) >= 1 ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'} isLoading={isLoading || riskMetricsLoading} isRefreshing={isRefreshing} icon={Zap} accentColor={themeColor} variant={variant} />;
            }
            case 'beta': {
                const beta = riskMetrics?.['Beta'] ?? m?.beta;
                return <MetricCard title="Portfolio Beta" value={beta != null ? beta.toFixed(2) : '-'} isCurrency={false} colorClass={(beta ?? 1) > 1.2 ? 'text-amber-600 dark:text-amber-400' : 'text-emerald-600 dark:text-emerald-400'} isLoading={isLoading || riskMetricsLoading} isRefreshing={isRefreshing} icon={Activity} accentColor={themeColor} variant={variant} />;
            }
            case 'fees':
                return <MetricCard title="Fees" value={m?.commissions ?? 0} colorClass="text-red-600 dark:text-red-500" currency={currency} isLoading={isLoading} isRefreshing={isRefreshing} icon={Receipt} accentColor={themeColor} variant={variant} />;
            case 'taxes':
                return <MetricCard title="Taxes" value={m?.taxes ?? 0} colorClass="text-red-600 dark:text-red-500" currency={currency} isLoading={isLoading} isRefreshing={isRefreshing} icon={Receipt} accentColor={themeColor} variant={variant} />;
            case 'riskMetrics':
                return <Suspense fallback={<AnalyticsFallback />}><RiskMetrics metrics={riskMetrics} portfolioHealth={portfolioHealth} isLoading={riskMetricsLoading!} isRefreshing={isRefreshing} /></Suspense>;
            case 'sectorContribution':
                return <Suspense fallback={<AnalyticsFallback />}><SectorAttribution data={attributionData} isLoading={attributionLoading!} isRefreshing={isRefreshing} currency={currency} /></Suspense>;
            case 'topContributors':
                return <Suspense fallback={<AnalyticsFallback />}><TopContributors data={attributionData} isLoading={attributionLoading!} isRefreshing={isRefreshing} currency={currency} accounts={accounts} showClosed={showClosed} /></Suspense>;
            case 'portfolioDonut':
                return (
                    <div className="metric-card card-shine h-full p-5 relative overflow-hidden group">
                        <div className="absolute top-0 left-5 right-5 h-[2px] rounded-full bg-cyan-500 opacity-40" />
                        <div className="h-full relative z-10">
                            <div className="flex justify-between items-start mb-4">
                                <div className="flex items-center gap-2">
                                    <h3 className="section-label">Portfolio Composition</h3>
                                    {isRefreshing && !isLoading && <Loader2 className="w-2.5 h-2.5 animate-spin text-muted-foreground/40" />}
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

    // Scalar items that are NOT the hero (portfolio value / day G/L) and NOT the
    // top-page sections (hero, today strip, events, insights).
    const compactItems = DEFAULT_ITEMS.filter(
        item => visibleItems.includes(item.id)
            && !COMPLEX_METRIC_IDS.includes(item.id)
            && !TOP_SECTION_IDS.includes(item.id)
    );

    const visibleComplexItems = DEFAULT_ITEMS.filter(
        item => visibleItems.includes(item.id)
            && COMPLEX_METRIC_IDS.includes(item.id)
            && !excludeFromAnalytics.includes(item.id)
    );

    const showEvents = visibleItems.includes('dashboardEvents');
    const showInsights = visibleItems.includes('dashboardInsights');

    return (
        <div className="mb-4 md:mb-10 space-y-4 md:space-y-5">

            {/* ── Closed-account banner ── */}
            {allSelectedClosed && (
                <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 dark:bg-amber-500/15 px-4 py-3 text-sm text-amber-900 dark:text-amber-200 flex items-start gap-3">
                    <Activity className="w-4 h-4 mt-0.5 shrink-0" />
                    <div>
                        <div className="font-semibold">
                            {closedAccounts.length === 1
                                ? `${closedAccounts[0]} is marked closed.`
                                : `${closedAccounts.length} closed accounts selected: ${closedAccounts.join(', ')}.`}
                        </div>
                        <div className="text-xs opacity-80 mt-0.5">
                            Rate-of-return metrics (TWR, IRR, YTD return, dividend yield) are hidden for closed accounts to avoid misleading values from residual dividends.
                        </div>
                    </div>
                </div>
            )}

            {/* ── Hero card ── */}
            {visibleItems.includes('portfolioHero') && (
                <PortfolioHeroCard
                    marketValue={m?.market_value}
                    dayGL={dayGL}
                    dayGLPct={dayGLPct}
                    cumTWR={allSelectedClosed ? null : m?.cumulative_twr}
                    annTWR={allSelectedClosed ? null : m?.annualized_twr}
                    irr={allSelectedClosed ? null : m?.portfolio_mwr}
                    currency={currency}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    themeColor={themeColor}
                    history={history}
                    wtdHistory={wtdHistory}
                    longHistory={longHistory}
                />
            )}

            {/* ── Today panel: market context + movers ── */}
            {visibleItems.includes('todayStrip') && !isLoading && holdings.length > 0 && (
                <TodayStrip
                    holdings={holdings}
                    currency={currency}
                    portfolioDayChangePct={dayGLPct}
                    indices={indices ?? m?.indices}
                />
            )}

            {/* ── Upcoming events + actionable insights ── */}
            {!isLoading && holdings.length > 0 && (showEvents || showInsights) && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 md:gap-5">
                    {showEvents && <DashboardEvents events={dividendEvents} currency={currency} />}
                    {showInsights && <DashboardInsights holdings={holdings} currency={currency} />}
                </div>
            )}

            {/* ── Compact metric grid ── */}
            {compactItems.length > 0 && (
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3">
                    {compactItems.map(item => (
                        <div key={item.id} className="w-full min-w-0">
                            {renderContent(item.id, 'seamless')}
                        </div>
                    ))}
                </div>
            )}

            {/* ── Analytics widgets ── */}
            {visibleComplexItems.length > 0 && (
                <>
                    <div className="flex items-center gap-3 pt-1">
                        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-border to-transparent" />
                        <span className="section-label tracking-[0.2em]">Analytics</span>
                        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-border to-transparent" />
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-5">
                        {visibleComplexItems.map(item => (
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
