import { useState, useEffect, useRef } from 'react';
import { PortfolioSummary, PerformanceData } from '../lib/api';
import { formatCurrency, cn } from '../lib/utils';
import { LineChart, Line, ResponsiveContainer } from 'recharts';
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { COMPLEX_METRIC_IDS, DEFAULT_ITEMS } from '../lib/dashboard_constants';

// Lazy component import logic handled by parent or standard import above
import RiskMetrics from './RiskMetrics';
import { SectorAttribution, TopContributors } from './AttributionChart';
import PortfolioDonut from './PortfolioDonut';
import { Holding } from '@/lib/api';

interface DashboardProps {
    summary: PortfolioSummary;
    currency: string;
    history?: PerformanceData[];
    isLoading?: boolean;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    riskMetrics?: any;
    riskMetricsLoading?: boolean;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    attributionData?: any;
    attributionLoading?: boolean;
    holdings?: Holding[];
    visibleItems: string[];
}

interface MetricCardProps {
    title: string;
    value: string | number;
    subValue?: number;
    isCurrency?: boolean;
    colorClass?: string;
    valueClassName?: string;
    containerClassName?: string;
    subValueClassName?: string;
    currency?: string;
    isHero?: boolean;
    isPercent?: boolean;
    vertical?: boolean;
    sparklineData?: { value: number }[];
    isLoading?: boolean;
}

const MetricCard = ({
    title,
    value,
    subValue,
    isCurrency = true,
    colorClass = '',
    valueClassName = 'text-xl sm:text-2xl',
    containerClassName = '',
    subValueClassName = '',
    vertical = false,
    sparklineData,
    currency = 'USD',
    isLoading = false
}: MetricCardProps) => (
    <Card className={cn(
        "h-full transition-all duration-300 relative overflow-hidden group",
        "hover:bg-accent/5 transition-colors",
        containerClassName
    )}>
        <CardContent className="h-full flex flex-col justify-center p-4 sm:p-6 relative">
            <p className="text-sm font-medium text-muted-foreground relative z-10 uppercase tracking-wider text-[10px]">{title}</p>

            <div className="mt-2 flex items-center gap-2 sm:gap-3 flex-wrap relative z-10">
                {isLoading ? (
                    <Skeleton className="h-8 w-32" />
                ) : (
                    <h3 className={cn("font-bold tracking-tight", colorClass || "text-foreground", valueClassName)}>
                        {value !== null && value !== undefined ? (isCurrency && typeof value === 'number' ? formatCurrency(value, currency) : value) : '-'}
                    </h3>
                )}
                {isLoading ? (
                    <Skeleton className="h-5 w-12 rounded-full" />
                ) : subValue !== undefined && subValue !== null && (
                    <Badge variant={subValue >= 0 ? "success" : "destructive"} className={cn("text-[9px] sm:text-[11px] font-bold px-1.5 sm:px-2 py-0.5", subValueClassName)}>
                        {subValue > 0 ? '+' : ''}{subValue.toFixed(2)}%
                    </Badge>
                )}
            </div>

            {!isLoading && sparklineData && sparklineData.length > 1 && (
                <div className="absolute inset-0 z-0 pointer-events-none opacity-30 group-hover:opacity-50 transition-opacity">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={sparklineData}>
                            <Line
                                type="monotone"
                                dataKey="value"
                                stroke={subValue && subValue >= 0 ? "#10b981" : "#f43f5e"}
                                strokeWidth={2}
                                dot={false}
                                isAnimationActive={false}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            )}
            {isLoading && (
                <div className="absolute bottom-0 left-0 right-0 h-10 px-6">
                    <Skeleton className="h-full w-full opacity-20" />
                </div>
            )}
        </CardContent>
    </Card>
);



export default function Dashboard({
    summary,
    currency,
    history = [],
    isLoading = false,
    riskMetrics = {},
    riskMetricsLoading = false,
    attributionData = null,
    attributionLoading = false,
    holdings = [],
    visibleItems
}: DashboardProps) {
    const m = summary?.metrics;
    const am = summary?.account_metrics;

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
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    // Use the correctly aggregated cash balance from overall metrics
    const cashBalance = m?.cash_balance || 0;
    const dayGL = (m?.day_change_display as number) || 0;
    const dayGLPct = (m?.day_change_percent as number) || 0;
    const unrealizedGL = (m?.unrealized_gain as number) || 0;
    const unrealizedGLPct = m ? ((m.unrealized_gain as number) / ((m.cost_basis_held as number) || 1)) * 100 : 0;
    const fxGL = (m?.fx_gain_loss_display as number) || 0;
    const fxGLPct = (m?.fx_gain_loss_pct as number) || 0;

    const dayGLColor = dayGL >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';
    const unrealizedGLColor = unrealizedGL >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';
    const fxGLColor = fxGL >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';

    const totalGain = (m?.total_gain as number) || 0;
    const realizedGain = (m?.realized_gain as number) || 0;

    const totalReturnColor = totalGain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';
    const realizedGainColor = realizedGain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';

    // Render helper same as before
    const renderContent = (id: string) => {
        switch (id) {
            case 'portfolioValue':
                return <MetricCard
                    title="Total Portfolio Value"
                    value={m?.market_value ?? 0}
                    valueClassName="text-3xl sm:text-5xl"
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                />;
            case 'dayGL':
                return <MetricCard
                    title="Day's Gain/Loss"
                    value={dayGL}
                    subValue={dayGLPct}
                    colorClass={dayGLColor}
                    valueClassName="text-2xl sm:text-3xl"
                    subValueClassName={cn("text-base sm:text-xl", dayGLPct >= 0 ? "bg-emerald-600 text-white dark:text-white hover:bg-emerald-700 border-none" : "bg-rose-600 text-white dark:text-white hover:bg-rose-700 border-none")}
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                    sparklineData={history.map(d => ({ value: d.twr }))}
                    isLoading={isLoading}
                />;
            case 'totalReturn':
                return <MetricCard
                    title="Total Return"
                    value={totalGain}
                    subValue={m?.total_return_pct}
                    colorClass={totalReturnColor}
                    valueClassName="text-2xl sm:text-3xl"
                    subValueClassName={cn("text-base sm:text-xl", (m?.total_return_pct || 0) >= 0 ? "bg-emerald-600 text-white dark:text-white hover:bg-emerald-700 border-none" : "bg-rose-600 text-white dark:text-white hover:bg-rose-700 border-none")}
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                />;
            case 'annualTWR':
                return <MetricCard
                    title="Annual TWR"
                    value={m?.annualized_twr !== undefined && m?.annualized_twr !== null ? `${m.annualized_twr.toFixed(2)}%` : '-'}
                    isCurrency={false}
                    colorClass={m?.annualized_twr && m.annualized_twr >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}
                    isLoading={isLoading}
                // Force height or min-width? No.
                />;
            case 'unrealizedGL':
                return <MetricCard
                    title="Unrealized G/L"
                    value={unrealizedGL}
                    subValue={unrealizedGLPct}
                    colorClass={unrealizedGLColor}
                    valueClassName="text-2xl sm:text-3xl"
                    subValueClassName={cn("text-base sm:text-xl", unrealizedGLPct >= 0 ? "bg-emerald-600 text-white dark:text-white hover:bg-emerald-700 border-none" : "bg-rose-600 text-white dark:text-white hover:bg-rose-700 border-none")}
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                />;
            case 'fxGL':
                return <MetricCard
                    title="FX Gain/Loss"
                    value={fxGL}
                    subValue={fxGLPct}
                    colorClass={fxGLColor}
                    subValueClassName={cn("text-base sm:text-xl", fxGLPct >= 0 ? "bg-emerald-600 text-white dark:text-white hover:bg-emerald-700 border-none" : "bg-rose-600 text-white dark:text-white hover:bg-rose-700 border-none")}
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                />;
            case 'realizedGain':
                return <MetricCard
                    title="Realized Gain"
                    value={realizedGain}
                    colorClass={realizedGainColor}
                    valueClassName="text-2xl sm:text-3xl"
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                />;
            case 'cashBalance':
                return <MetricCard
                    title="Cash Balance"
                    value={cashBalance}
                    currency={currency}
                    isLoading={isLoading}
                />;
            case 'ytdDividends':
                return <MetricCard
                    title="Total Dividends"
                    value={m?.dividends ?? 0}
                    valueClassName="text-2xl sm:text-3xl"
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                />;
            case 'fees':
                return <MetricCard
                    title="Fees"
                    value={m?.commissions ?? 0}
                    colorClass="text-rose-600 dark:text-rose-400"
                    currency={currency}
                    isLoading={isLoading}
                />;
            case 'riskMetrics':
                return <RiskMetrics metrics={riskMetrics} isLoading={riskMetricsLoading!} />;
            case 'sectorContribution':
                return <SectorAttribution data={attributionData} isLoading={attributionLoading!} currency={currency} />;
            case 'topContributors':
                return <TopContributors data={attributionData} isLoading={attributionLoading!} currency={currency} />;
            case 'portfolioDonut':
                return (
                    <Card className="h-full">
                        <CardContent className="h-full p-4 relative">
                            <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-2">Portfolio Composition</h3>
                            <div className="h-[calc(100%-24px)]">
                                <PortfolioDonut holdings={holdings} currency={currency} />
                            </div>
                        </CardContent>
                    </Card>
                );
            default:
                return null;
        }
    };

    const visibleScalarItems = DEFAULT_ITEMS.filter(item => visibleItems.includes(item.id) && !COMPLEX_METRIC_IDS.includes(item.id));
    const visibleComplexItems = DEFAULT_ITEMS.filter(item => visibleItems.includes(item.id) && COMPLEX_METRIC_IDS.includes(item.id));

    return (
        <div className="mb-10 space-y-6">
            {/* Scalar Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-2 lg:grid-cols-4 gap-3 md:gap-6">
                {visibleScalarItems.map((item) => (
                    <div key={item.id} className={cn(item.colSpan, "w-full min-w-0")}>
                        {renderContent(item.id)}
                    </div>
                ))}
            </div>

            {/* Complex/Tall Metrics Grid */}
            {visibleComplexItems.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {visibleComplexItems.map((item) => (
                        <div key={item.id} className={item.colSpan}>
                            {renderContent(item.id)}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
