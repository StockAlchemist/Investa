import React, { useState, useEffect } from 'react';
import {
    Wallet,
    Calendar,
    LayoutDashboard,
    RotateCcw,
    TrendingUp,
    Scale,
    Globe,
    Receipt,
    DollarSign,
    Building2,
    Hash,
    Tag,
    PieChart as PieChartIcon,
    LineChart as LineChartIcon,
    BarChart3,
    Activity as LucideActivity,
    Sparkles,
    Shield,
    Zap,
    Target,
    ChevronRight
} from 'lucide-react';
import { StatCard } from '../components/StatCard';
import { FiftyTwoWeekCard } from '../components/FiftyTwoWeekCard';
import { UpcomingEventRow } from '../components/UpcomingEventRow';
import StockKeyMetrics from '../../StockKeyMetrics';
import { cn, formatCurrency, formatPercent as formatPercentShared } from '../../../lib/utils';
import { normalizeDividendYield, normalizeExpenseRatio } from '../../../lib/dividend';
import { formatCalendarDate } from '../../../lib/market_time';
import { fetchStockAnalysis, type StockAnalysisResponse } from '../../../lib/api';
import type { TabType } from '../types';

function formatPercentPoints(points: number | null | undefined): string {
    if (points === null || points === undefined || isNaN(points)) return "-";
    return `${points.toFixed(2)}%`;
}

function formatEventDate(iso: string): string {
    return formatCalendarDate(iso);
}

interface OverviewTabProps {
    symbol?: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- fundamentals payload
    fundamentals: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- intrinsicValue payload
    intrinsicValue: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- userPosition payload
    userPosition: any;
    currency: string;
    fxRate: number;
    loading: boolean;
    onRefreshData: () => void;
    onSelectTab?: (tab: TabType) => void;
}

export const OverviewTab: React.FC<OverviewTabProps> = ({
    symbol,
    fundamentals,
    intrinsicValue,
    userPosition,
    currency,
    fxRate,
    loading,
    onRefreshData,
    onSelectTab,
}) => {
    const [summaryExpanded, setSummaryExpanded] = useState(false);
    const [analysis, setAnalysis] = useState<StockAnalysisResponse | null>(null);

    const activeSymbol = symbol || fundamentals?.symbol;

    useEffect(() => {
        if (activeSymbol) {
            let active = true;
            fetchStockAnalysis(activeSymbol)
                .then(data => {
                    if (active && data && !data.error) {
                        setAnalysis(data);
                    }
                })
                .catch(err => {
                    console.debug("Overview AI analysis fetch (non-critical):", err);
                });
            return () => { active = false; };
        }
    }, [activeSymbol]);

    useEffect(() => {
        const handler = (e: Event) => {
            const ce = e as CustomEvent<{ symbol: string; analysis: StockAnalysisResponse }>;
            if (ce.detail?.symbol === activeSymbol && ce.detail.analysis) {
                setAnalysis(ce.detail.analysis);
            }
        };
        window.addEventListener('stock-analysis-updated', handler);
        return () => window.removeEventListener('stock-analysis-updated', handler);
    }, [activeSymbol]);

    if (!fundamentals) return null;

    const upcomingEarnings = fundamentals.upcoming_events?.earnings;
    const upcomingDividend = fundamentals.upcoming_events?.dividend;
    const reportedEarnings = fundamentals.upcoming_events?.recent_earnings;

    const getUpsidePercentage = (iv?: number) => {
        if (!iv || !intrinsicValue?.current_price) return null;
        return (iv / intrinsicValue.current_price) - 1;
    };

    const formatUpside = (upside: number | null) => {
        if (upside === null) return null;
        return formatPercentShared(upside);
    };

    const getUpsideColor = (upside: number | null) => {
        if (upside === null) return "";
        return upside > 0 ? "text-up font-bold" : "text-rose-500 font-bold";
    };

    const recommendedMethod = intrinsicValue?.recommended_method;
    const bestFitValue = recommendedMethod?.intrinsic_value;
    const bestFitUpside = getUpsidePercentage(bestFitValue);
    const blendedValue = intrinsicValue?.average_intrinsic_value;
    const blendedUpside = getUpsidePercentage(blendedValue);

    const getBestFitRange = () => {
        if (!recommendedMethod || !intrinsicValue?.models) return null;
        const key = recommendedMethod.method_key;
        if (key === 'dcf') return intrinsicValue.models.dcf?.mc;
        if (key === 'graham') return intrinsicValue.models.graham?.mc;
        if (key === 'ddm') return intrinsicValue.models.ddm?.mc;
        return null;
    };
    const bestFitRange = getBestFitRange();

    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {userPosition && (
                <div className="space-y-3">
                    <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold flex items-center gap-2">
                            <Wallet className="w-5 h-5 text-indigo-500" />
                            Your Position
                        </h3>
                        <div className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider bg-secondary/50 px-2 py-1 rounded-md">
                            Aggregated
                        </div>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2.5">
                        <StatCard
                            label="Quantity"
                            value={userPosition.Quantity.toLocaleString()}
                            icon={Hash}
                            color="text-indigo-500"
                        />
                        <StatCard
                            label="Avg Cost"
                            value={formatCurrency(userPosition["Avg Cost"], currency)}
                            icon={Tag}
                            color="text-slate-500"
                        />
                        <StatCard
                            label="Market Value"
                            value={formatCurrency(userPosition["Market Value"], currency)}
                            icon={PieChartIcon}
                            color="text-indigo-500"
                        />
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
                        <StatCard
                            label="Unrealized G/L"
                            value={formatCurrency(userPosition["Unreal. Gain"], currency)}
                            subValue={userPosition["Unreal. Gain %"] === Infinity ? "∞" : `${(userPosition["Unreal. Gain %"] || 0).toFixed(2)}%`}
                            subValueColor={(userPosition["Unreal. Gain %"] || 0) >= 0 ? "text-up" : "text-rose-500"}
                            valueColor={(userPosition["Unreal. Gain"] || 0) >= 0 ? "text-up" : "text-rose-500"}
                            icon={LucideActivity}
                            color={(userPosition?.["Unreal. Gain"] ?? 0) >= 0 ? "bg-up/12 text-up" : "bg-rose-500/10 text-rose-500"}
                        />
                        <StatCard
                            label="Total Return"
                            value={formatCurrency(userPosition["Total Gain"], currency)}
                            subValue={userPosition["Total Return %"] === Infinity ? "∞" : `${(userPosition["Total Return %"] || 0).toFixed(2)}%`}
                            subValueColor={(userPosition["Total Return %"] || 0) >= 0 ? "text-up" : "text-rose-500"}
                            valueColor={(userPosition["Total Return %"] || 0) >= 0 ? "text-up" : "text-rose-500"}
                            icon={TrendingUp}
                            color={(userPosition["Total Return %"] || 0) >= 0 ? "bg-up/12 text-up" : "bg-rose-500/10 text-rose-500"}
                            extra={
                                <p className="text-[11px] font-semibold text-amber-600 dark:text-amber-500/90 leading-tight">
                                    Divs: {formatCurrency(userPosition["Dividends"], currency)}
                                </p>
                            }
                        />
                        <StatCard
                            label="IRR %"
                            value={userPosition["IRR %"] === Infinity ? "∞" : `${(userPosition["IRR %"] || 0).toFixed(2)}%`}
                            icon={LineChartIcon}
                            valueColor={(userPosition["IRR %"] || 0) >= 0 ? "text-up" : "text-rose-500"}
                            color={(userPosition["IRR %"] || 0) >= 0 ? "bg-up/12 text-up" : "bg-rose-500/10 text-rose-500"}
                        />
                    </div>

                    <div className="h-px bg-border/40 w-full my-1" />
                </div>
            )}

            {(upcomingEarnings || upcomingDividend || reportedEarnings) && (
                <div className="space-y-3">
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                        <Calendar className="w-5 h-5 text-indigo-500" />
                        Upcoming Events
                    </h3>
                    <div className="bg-muted rounded-xl divide-y divide-border/50 overflow-hidden">
                        {reportedEarnings && (
                            <UpcomingEventRow
                                icon={BarChart3}
                                color="bg-violet-500/10 text-violet-500"
                                label="Latest Earnings"
                                status={reportedEarnings.status}
                                date={reportedEarnings.earnings_date}
                                timeZone={reportedEarnings.market_timezone}
                                detail={reportedEarnings.eps_actual != null
                                    ? [
                                        `EPS ${reportedEarnings.eps_actual.toFixed(2)}`,
                                        reportedEarnings.eps_estimate != null
                                            ? `vs ${reportedEarnings.eps_estimate.toFixed(2)} expected`
                                            : null,
                                        reportedEarnings.surprise_pct != null
                                            ? `${reportedEarnings.surprise_pct >= 0 ? '+' : ''}${reportedEarnings.surprise_pct.toFixed(1)}%`
                                            : null,
                                    ].filter(Boolean).join(' · ')
                                    : 'Figures not published yet'}
                                detailColor={reportedEarnings.surprise_pct == null
                                    ? undefined
                                    : reportedEarnings.surprise_pct >= 0
                                        ? 'text-up'
                                        : 'text-down'}
                            />
                        )}
                        {upcomingEarnings && (
                            <UpcomingEventRow
                                icon={BarChart3}
                                color="bg-violet-500/10 text-violet-500"
                                label="Next Earnings"
                                status={upcomingEarnings.status}
                                date={upcomingEarnings.earnings_date}
                                dateEnd={upcomingEarnings.earnings_date_end}
                                timeZone={upcomingEarnings.market_timezone}
                                detail={upcomingEarnings.eps_estimate != null
                                    ? `Est. EPS ${upcomingEarnings.eps_estimate.toFixed(2)}${upcomingEarnings.eps_year_ago != null
                                        ? ` vs ${upcomingEarnings.eps_year_ago.toFixed(2)} a year ago` : ''}`
                                    : undefined}
                            />
                        )}
                        {upcomingDividend && (
                            <UpcomingEventRow
                                icon={DollarSign}
                                color="bg-up/12 text-up"
                                label="Next Dividend"
                                status={upcomingDividend.status}
                                date={upcomingDividend.dividend_date}
                                timeZone={upcomingDividend.market_timezone}
                                detail={[
                                    `${formatCurrency(upcomingDividend.amount_per_share * fxRate, currency)} / share`,
                                    upcomingDividend.ex_dividend_date
                                        ? `ex-div ${formatEventDate(upcomingDividend.ex_dividend_date)}`
                                        : null,
                                ].filter(Boolean).join(' · ')}
                            />
                        )}
                    </div>
                </div>
            )}

            {/* AI Analysis Scores */}
            {analysis?.scorecard && (() => {
                const sc = analysis.scorecard;
                const validScores = [
                    sc.moat,
                    sc.financial_strength,
                    sc.predictability,
                    sc.growth,
                ].filter((v): v is number => typeof v === 'number' && !isNaN(v));

                const compositeScore = validScores.length > 0
                    ? validScores.reduce((acc, s) => acc + s, 0) / validScores.length
                    : null;

                const compositeTier = compositeScore != null
                    ? compositeScore >= 8.5 ? "Exceptional" : compositeScore >= 7.0 ? "Strong" : compositeScore >= 5.5 ? "Moderate" : "Weak"
                    : null;

                const getCompositeColor = (score: number) => {
                    if (score >= 8.5) return {
                        text: 'text-up',
                        bg: 'bg-up/12 dark:bg-up/12',
                        border: 'border-up/25 dark:border-emerald-500/25',
                    };
                    if (score >= 7.0) return {
                        text: 'text-indigo-600 dark:text-indigo-400',
                        bg: 'bg-indigo-500/10 dark:bg-indigo-500/15',
                        border: 'border-indigo-500/30 dark:border-indigo-500/25',
                    };
                    if (score >= 5.5) return {
                        text: 'text-amber-600 dark:text-amber-400',
                        bg: 'bg-amber-500/10 dark:bg-amber-500/15',
                        border: 'border-amber-500/30 dark:border-amber-500/25',
                    };
                    return {
                        text: 'text-down',
                        bg: 'bg-rose-500/10 dark:bg-rose-500/15',
                        border: 'border-rose-500/30 dark:border-rose-500/25',
                    };
                };

                const compStyle = compositeScore !== null ? getCompositeColor(compositeScore) : null;

                const pillars = [
                    {
                        id: 'moat',
                        label: 'Moat & Edge',
                        icon: Shield,
                        score: sc.moat,
                        color: 'text-blue-500 dark:text-blue-400',
                        bgColor: 'bg-blue-500/10 dark:bg-blue-500/20 text-blue-500 dark:text-blue-400',
                        barColor: 'from-blue-500 to-cyan-500',
                        badgeColor: 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/25',
                        getTier: (s: number) => s >= 9 ? 'Wide Moat' : s >= 7.5 ? 'Solid Moat' : s >= 5.5 ? 'Narrow Moat' : 'No Moat',
                    },
                    {
                        id: 'strength',
                        label: 'Financial Strength',
                        icon: Zap,
                        score: sc.financial_strength,
                        color: 'text-amber-500 dark:text-amber-400',
                        bgColor: 'bg-amber-500/10 dark:bg-amber-500/20 text-amber-500 dark:text-amber-400',
                        barColor: 'from-amber-500 to-orange-500',
                        badgeColor: 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/25',
                        getTier: (s: number) => s >= 9 ? 'Fortress' : s >= 7.5 ? 'Healthy' : s >= 5.5 ? 'Adequate' : 'Constrained',
                    },
                    {
                        id: 'predictability',
                        label: 'Predictability',
                        icon: Target,
                        score: sc.predictability,
                        color: 'text-up',
                        bgColor: 'bg-up/12 dark:bg-emerald-500/20 text-up',
                        barColor: 'from-emerald-500 to-teal-500',
                        badgeColor: 'bg-up/12 text-up border-emerald-500/25',
                        getTier: (s: number) => s >= 9 ? 'High Visibility' : s >= 7.5 ? 'Predictable' : s >= 5.5 ? 'Moderate' : 'Volatile',
                    },
                    {
                        id: 'growth',
                        label: 'Growth Pace',
                        icon: TrendingUp,
                        score: sc.growth,
                        color: 'text-purple-500 dark:text-purple-400',
                        bgColor: 'bg-purple-500/10 dark:bg-purple-500/20 text-purple-500 dark:text-purple-400',
                        barColor: 'from-purple-500 to-pink-500',
                        badgeColor: 'bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/25',
                        getTier: (s: number) => s >= 9 ? 'High Growth' : s >= 7.5 ? 'Solid Growth' : s >= 5.5 ? 'Moderate' : 'Sluggish',
                    },
                ];

                return (
                    <div className="relative overflow-hidden rounded-2xl sm:rounded-3xl p-4 sm:p-5 bg-gradient-to-br from-indigo-500/[0.03] via-card to-purple-500/[0.04] border border-border/70 shadow-xs space-y-3.5 sm:space-y-4">
                        {/* Ambient glow accent */}
                        <div className="pointer-events-none absolute -top-10 -right-10 w-36 h-36 bg-purple-500/10 dark:bg-purple-500/15 rounded-full blur-2xl" />

                        {/* Card Header */}
                        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2.5 sm:gap-3 relative z-10">
                            <div className="flex items-center justify-between sm:justify-start gap-2.5 min-w-0">
                                <div className="flex items-center gap-2 sm:gap-2.5 min-w-0">
                                    <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-xl bg-gradient-to-tr from-purple-600 to-indigo-500 text-white flex items-center justify-center shadow-xs shadow-purple-500/20 shrink-0">
                                        <Sparkles className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                                    </div>
                                    <h3 className="text-sm sm:text-base font-bold text-foreground tracking-tight whitespace-nowrap">
                                        AI Fundamental Health
                                    </h3>
                                </div>
                                <span className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-600 dark:text-purple-400 border border-purple-500/20 whitespace-nowrap shrink-0">
                                    <span className="w-1.5 h-1.5 rounded-full bg-purple-500 animate-pulse" />
                                    Gemini AI
                                </span>
                            </div>
                            {compositeScore !== null && compStyle && (
                                <div className={cn(
                                    "flex items-center justify-between sm:justify-start gap-2 px-3 py-1.5 rounded-xl border backdrop-blur-xs shadow-2xs whitespace-nowrap w-full sm:w-auto",
                                    compStyle.bg,
                                    compStyle.border
                                )}>
                                    <span className="text-[11px] font-medium text-muted-foreground">Composite Score:</span>
                                    <div className="flex items-center gap-1">
                                        <span className={cn("text-xs sm:text-sm font-black tabular-nums", compStyle.text)}>
                                            {compositeScore.toFixed(1)}
                                        </span>
                                        <span className="text-[10px] sm:text-[11px] font-bold text-muted-foreground/60">/10</span>
                                        {compositeTier && (
                                            <span className={cn("text-[10px] sm:text-[11px] font-bold ml-1", compStyle.text)}>
                                                · {compositeTier}
                                            </span>
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* 4 Pillar Cards */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5 sm:gap-3.5 relative z-10">
                            {pillars.map(p => {
                                const tier = p.score != null ? p.getTier(p.score) : null;
                                const pct = p.score != null ? Math.min(100, Math.max(0, (p.score / 10) * 100)) : 0;
                                return (
                                    <div
                                        key={p.id}
                                        onClick={() => onSelectTab && onSelectTab('analysis')}
                                        className={cn(
                                            "group p-3 sm:p-3.5 rounded-xl sm:rounded-2xl bg-card/90 dark:bg-card/70 border border-border/60 hover:border-border shadow-2xs hover:shadow-xs transition-all duration-200 flex flex-col justify-between gap-2.5",
                                            onSelectTab ? "cursor-pointer hover:bg-card" : ""
                                        )}
                                        title={p.label}
                                    >
                                        {/* Card Top: Icon & Status Badge */}
                                        <div className="flex items-center justify-between gap-1.5">
                                            <div className={cn("p-1.5 rounded-lg shrink-0 flex items-center justify-center", p.bgColor)}>
                                                <p.icon className={cn("w-3.5 h-3.5", p.color)} />
                                            </div>
                                            {tier && (
                                                <span className={cn(
                                                    "text-[9px] sm:text-[10px] font-bold px-1.5 py-0.5 rounded-md border leading-none shrink-0 whitespace-nowrap tracking-tight",
                                                    p.badgeColor
                                                )}>
                                                    {tier}
                                                </span>
                                            )}
                                        </div>

                                        {/* Card Middle: Pillar Label (Dedicated Row, Prevents Truncation) */}
                                        <div className="pt-0.5">
                                            <span className="text-xs font-semibold text-foreground/90 block leading-tight truncate">
                                                {p.label}
                                            </span>
                                        </div>

                                        {/* Card Bottom: Numeric Score & Progress Bar */}
                                        <div className="space-y-1.5 pt-0.5">
                                            <div className="flex items-baseline justify-between">
                                                <div className="flex items-baseline gap-1">
                                                    <span className={cn("text-2xl sm:text-3xl font-black tracking-tight tabular-nums", p.color)}>
                                                        {p.score != null ? p.score : '—'}
                                                    </span>
                                                    <span className="text-xs font-medium text-muted-foreground/60">/10</span>
                                                </div>
                                                {onSelectTab && (
                                                    <ChevronRight className="w-3.5 h-3.5 text-muted-foreground/30 group-hover:text-foreground/70 group-hover:translate-x-0.5 transition-all hidden sm:block" />
                                                )}
                                            </div>

                                            {/* Mini Progress Track */}
                                            <div className="w-full bg-muted/80 dark:bg-muted/40 h-1.5 rounded-full overflow-hidden">
                                                <div
                                                    className={cn("h-full rounded-full transition-all duration-700 bg-gradient-to-r", p.barColor)}
                                                    style={{ width: `${pct}%` }}
                                                />
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>

                        {/* Optional footer quick link to full AI analysis */}
                        {onSelectTab && (
                            <div className="flex items-center justify-between pt-1 border-t border-border/40 text-[11px] relative z-10">
                                <span className="text-muted-foreground">Detailed fundamental review & drivers</span>
                                <button
                                    onClick={() => onSelectTab('analysis')}
                                    className="font-bold text-indigo-600 dark:text-indigo-400 hover:text-indigo-700 dark:hover:text-indigo-300 flex items-center gap-1 transition-colors cursor-pointer group"
                                >
                                    <span>View Full Analysis</span>
                                    <ChevronRight className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
                                </button>
                            </div>
                        )}
                    </div>
                );
            })()}

            <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                    <LayoutDashboard className="w-5 h-5 text-indigo-500" />
                    Market Overview
                </h3>
                <button
                    onClick={onRefreshData}
                    disabled={loading}
                    className="flex items-center gap-1.5 text-[10px] font-bold text-cyan-600 hover:text-cyan-700 dark:text-cyan-400 dark:hover:text-cyan-300 transition-colors uppercase tracking-wider cursor-pointer"
                    title="Force Refresh Data"
                >
                    <RotateCcw className="w-3 h-3" />
                    Refresh Data
                </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full">
                {bestFitValue != null && bestFitValue > 0 && (
                    <StatCard
                        label={`Best-Fit: ${recommendedMethod?.name || 'Valuation Model'}`}
                        value={formatCurrency(bestFitValue * fxRate, currency)}
                        subValue={formatUpside(bestFitUpside)}
                        subValueColor={getUpsideColor(bestFitUpside)}
                        rangeMin={bestFitRange?.bear ? formatCurrency((bestFitRange.bear ?? 0) * fxRate, currency) : undefined}
                        rangeMax={bestFitRange?.bull ? formatCurrency((bestFitRange.bull ?? 0) * fxRate, currency) : undefined}
                        icon={Sparkles}
                        color="text-indigo-500 dark:text-indigo-400"
                    />
                )}
                {blendedValue != null && blendedValue > 0 && (
                    <StatCard
                        label={intrinsicValue.valuation_status === "nav" ? "Net Asset Value (NAV)" : "Blended Intrinsic Value"}
                        value={formatCurrency(blendedValue * fxRate, currency)}
                        subValue={formatUpside(blendedUpside)}
                        subValueColor={getUpsideColor(blendedUpside)}
                        rangeMin={intrinsicValue.range?.bear ? formatCurrency((intrinsicValue.range.bear ?? 0) * fxRate, currency) : undefined}
                        rangeMax={intrinsicValue.range?.bull ? formatCurrency((intrinsicValue.range.bull ?? 0) * fxRate, currency) : undefined}
                        icon={Scale}
                        color="text-indigo-500 dark:text-indigo-400"
                    />
                )}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
                <StatCard label="Market Cap" value={formatCurrency(fundamentals.marketCap)} icon={Globe} color="text-indigo-400" />
                <FiftyTwoWeekCard
                    low={fundamentals.fiftyTwoWeekLow}
                    high={fundamentals.fiftyTwoWeekHigh}
                    price={fundamentals.currentPrice ?? fundamentals.regularMarketPrice}
                    format={(v) => formatCurrency(v)}
                />
                {(() => {
                    const expensePct = normalizeExpenseRatio(
                        fundamentals.netExpenseRatio || fundamentals.expenseRatio || fundamentals.annualReportExpenseRatio
                    );
                    if (expensePct !== null) {
                        return (
                            <StatCard
                                label="Expense Ratio"
                                value={formatPercentPoints(expensePct)}
                                icon={Receipt}
                                color="text-orange-400"
                            />
                        );
                    }
                    return (
                        <StatCard
                            label="Dividend Yield"
                            value={formatPercentPoints(normalizeDividendYield({
                                rawYield: fundamentals.dividendYield,
                                dividendRate: fundamentals.dividendRate ?? fundamentals.trailingAnnualDividendRate,
                                price: fundamentals.currentPrice ?? fundamentals.regularMarketPrice,
                                trailingYield: fundamentals.trailingAnnualDividendYield,
                            }))}
                            icon={DollarSign}
                            color="text-amber-400"
                        />
                    );
                })()}
            </div>

            <StockKeyMetrics
                symbol={fundamentals.symbol}
                metrics={fundamentals.key_metrics}
                beta={fundamentals.beta}
                averageVolume={fundamentals.averageVolume}
            />

            {fundamentals.longBusinessSummary && (
                <div className="bg-muted rounded-2xl px-5 py-4">
                    <h3 className="text-base font-semibold mb-2 flex items-center gap-2">
                        <Building2 className="w-4 h-4 text-indigo-500" />
                        Business Summary
                    </h3>
                    <p className={cn(
                        "text-muted-foreground text-sm leading-relaxed whitespace-pre-wrap",
                        !summaryExpanded && "line-clamp-4"
                    )}>
                        {fundamentals.longBusinessSummary}
                    </p>
                    {fundamentals.longBusinessSummary.length > 320 && (
                        <button
                            onClick={() => setSummaryExpanded(v => !v)}
                            className="mt-2 text-[11px] font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400 hover:underline cursor-pointer"
                        >
                            {summaryExpanded ? 'Show less' : 'Read more'}
                        </button>
                    )}
                </div>
            )}
        </div>
    );
};
