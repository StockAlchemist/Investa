import React, { useState } from 'react';
import {
    Wallet,
    Calendar,
    LayoutDashboard,
    RotateCcw,
    TrendingUp,
    Scale,
    Anchor,
    Globe,
    Receipt,
    DollarSign,
    Building2,
    Hash,
    Tag,
    PieChart as PieChartIcon,
    LineChart as LineChartIcon,
    BarChart3,
    Activity as LucideActivity
} from 'lucide-react';
import { StatCard } from '../components/StatCard';
import { FiftyTwoWeekCard } from '../components/FiftyTwoWeekCard';
import { UpcomingEventRow } from '../components/UpcomingEventRow';
import StockKeyMetrics from '../../StockKeyMetrics';
import { cn, formatCurrency, formatPercent as formatPercentShared } from '../../../lib/utils';
import { normalizeDividendYield, normalizeExpenseRatio } from '../../../lib/dividend';
import { formatCalendarDate } from '../../../lib/market_time';

function formatPercentPoints(points: number | null | undefined): string {
    if (points === null || points === undefined || isNaN(points)) return "-";
    return `${points.toFixed(2)}%`;
}

function formatEventDate(iso: string): string {
    return formatCalendarDate(iso);
}

interface OverviewTabProps {
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
}

export const OverviewTab: React.FC<OverviewTabProps> = ({
    fundamentals,
    intrinsicValue,
    userPosition,
    currency,
    fxRate,
    loading,
    onRefreshData,
}) => {
    const [summaryExpanded, setSummaryExpanded] = useState(false);

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
        return upside > 0 ? "text-emerald-500 font-bold" : "text-rose-500 font-bold";
    };

    const dcfUpside = getUpsidePercentage(intrinsicValue?.models?.dcf?.intrinsic_value);
    const grahamUpside = getUpsidePercentage(intrinsicValue?.models?.graham?.intrinsic_value);
    const epvUpside = getUpsidePercentage(intrinsicValue?.models?.epv?.intrinsic_value);
    const valuationCardCount = [
        intrinsicValue?.models?.dcf?.intrinsic_value,
        intrinsicValue?.models?.graham?.intrinsic_value,
        intrinsicValue?.models?.epv?.intrinsic_value,
    ].filter(Boolean).length;

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
                            subValueColor={(userPosition["Unreal. Gain %"] || 0) >= 0 ? "text-emerald-500" : "text-rose-500"}
                            valueColor={(userPosition["Unreal. Gain"] || 0) >= 0 ? "text-emerald-500" : "text-rose-500"}
                            icon={LucideActivity}
                            color={(userPosition?.["Unreal. Gain"] ?? 0) >= 0 ? "bg-emerald-500/10 text-emerald-500" : "bg-rose-500/10 text-rose-500"}
                        />
                        <StatCard
                            label="Total Return"
                            value={formatCurrency(userPosition["Total Gain"], currency)}
                            subValue={userPosition["Total Return %"] === Infinity ? "∞" : `${(userPosition["Total Return %"] || 0).toFixed(2)}%`}
                            subValueColor={(userPosition["Total Return %"] || 0) >= 0 ? "text-emerald-500" : "text-rose-500"}
                            valueColor={(userPosition["Total Return %"] || 0) >= 0 ? "text-emerald-500" : "text-rose-500"}
                            icon={TrendingUp}
                            color={(userPosition["Total Return %"] || 0) >= 0 ? "bg-emerald-500/10 text-emerald-500" : "bg-rose-500/10 text-rose-500"}
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
                            valueColor={(userPosition["IRR %"] || 0) >= 0 ? "text-emerald-500" : "text-rose-500"}
                            color={(userPosition["IRR %"] || 0) >= 0 ? "bg-emerald-500/10 text-emerald-500" : "bg-rose-500/10 text-rose-500"}
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
                                        ? 'text-emerald-600 dark:text-emerald-400'
                                        : 'text-rose-600 dark:text-rose-400'}
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
                                color="bg-emerald-500/10 text-emerald-500"
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

            <div className={cn(
                'grid grid-cols-1 gap-2.5 w-full',
                valuationCardCount >= 3 ? 'sm:grid-cols-2 md:grid-cols-3' : 'sm:grid-cols-2',
            )}>
                {intrinsicValue?.models?.dcf?.intrinsic_value && (
                    <StatCard
                        label="DCF Intrinsic Value"
                        value={formatCurrency((intrinsicValue.models.dcf.intrinsic_value ?? 0) * fxRate, currency)}
                        subValue={formatUpside(dcfUpside)}
                        subValueColor={getUpsideColor(dcfUpside)}
                        rangeMin={formatCurrency((intrinsicValue.models.dcf.mc?.bear ?? 0) * fxRate, currency)}
                        rangeMax={formatCurrency((intrinsicValue.models.dcf.mc?.bull ?? 0) * fxRate, currency)}
                        icon={TrendingUp}
                        color="text-emerald-400"
                    />
                )}
                {intrinsicValue?.models?.graham?.intrinsic_value && (
                    <StatCard
                        label="Graham Intrinsic Value"
                        value={formatCurrency((intrinsicValue.models.graham.intrinsic_value ?? 0) * fxRate, currency)}
                        subValue={formatUpside(grahamUpside)}
                        subValueColor={getUpsideColor(grahamUpside)}
                        rangeMin={formatCurrency((intrinsicValue.models.graham.mc?.bear ?? 0) * fxRate, currency)}
                        rangeMax={formatCurrency((intrinsicValue.models.graham.mc?.bull ?? 0) * fxRate, currency)}
                        icon={Scale}
                        color="text-amber-400"
                    />
                )}
                {intrinsicValue?.models?.epv?.intrinsic_value && (
                    <StatCard
                        label="Earnings Power (no growth)"
                        value={formatCurrency((intrinsicValue.models.epv.intrinsic_value ?? 0) * fxRate, currency)}
                        subValue={formatUpside(epvUpside)}
                        subValueColor={getUpsideColor(epvUpside)}
                        icon={Anchor}
                        color="text-sky-400"
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
