import { useState, useEffect, useRef } from 'react';
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
    Loader2
} from 'lucide-react';

// Lazy component import logic handled by parent or standard import above
import RiskMetrics from './RiskMetrics';
import { SectorAttribution, TopContributors } from './AttributionChart';
import PortfolioDonut from './PortfolioDonut';
import { Holding } from '@/lib/api';
import { Card, CardContent } from "@/components/ui/card";

interface DashboardProps {
    summary: PortfolioSummary;
    currency: string;
    history?: PerformanceData[];
    isLoading?: boolean;
    isRefreshing?: boolean;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
}

export default function Dashboard({
    summary,
    currency,
    history = [],
    isLoading = false,
    riskMetrics = {},
    riskMetricsLoading = false,
    portfolioHealth = null,
    attributionData = null,
    attributionLoading = false,
    holdings = [],
    visibleItems,
    isRefreshing = false
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

    // Render helper same as before
    const renderContent = (id: string) => {
        switch (id) {
            case 'portfolioValue':
                return <MetricCard
                    title="Total Portfolio Value"
                    value={m?.market_value ?? 0}
                    valueClassName="text-2xl sm:text-4xl" // Slightly smaller generally, but hero
                    containerClassName="h-full bg-gradient-to-br from-card to-secondary/30"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    icon={Wallet}
                />;
            case 'dayGL':
                return <MetricCard
                    title="Day's Gain/Loss"
                    value={dayGL}
                    subValue={dayGLPct}
                    colorClass={dayGLColor}
                    valueClassName="text-xl sm:text-2xl"
                    subValueClassName={cn("", (dayGLPct ?? 0) >= 0 ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" : "bg-red-500/10 text-red-600 dark:text-red-400")}
                    containerClassName="h-full"
                    isHero={true}
                    currency={currency}
                    sparklineData={history.map(d => ({ value: d.twr }))}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    icon={(dayGL ?? 0) >= 0 ? TrendingUp : TrendingDown}
                />;
            case 'totalReturn':
                return <MetricCard
                    title="Total Return"
                    value={totalGain}
                    subValue={m?.total_return_pct}
                    colorClass={totalReturnColor}
                    valueClassName="text-xl sm:text-2xl"
                    subValueClassName={cn("", (m?.total_return_pct ?? 0) >= 0 ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" : "bg-red-500/10 text-red-600 dark:text-red-400")}
                    containerClassName="h-full"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    icon={Activity}
                />;
            case 'annualTWR':
                return <MetricCard
                    title="Annual TWR"
                    value={m?.annualized_twr !== undefined && m?.annualized_twr !== null ? `${m.annualized_twr.toFixed(2)}%` : '-'}
                    isCurrency={false}
                    colorClass={m?.annualized_twr && m.annualized_twr >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500'}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    icon={Percent}
                />;
            case 'mwr':
                return <MetricCard
                    title="IRR (MWR)"
                    value={m?.portfolio_mwr !== undefined && m?.portfolio_mwr !== null ? `${m.portfolio_mwr.toFixed(2)}%` : '-'}
                    isCurrency={false}
                    colorClass={m?.portfolio_mwr && m.portfolio_mwr >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500'}
                    isLoading={isLoading}
                    isRefreshing={isRefreshing}
                    icon={Activity}
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
                />;
            case 'riskMetrics':
                return <RiskMetrics metrics={riskMetrics} portfolioHealth={portfolioHealth} isLoading={riskMetricsLoading!} isRefreshing={isRefreshing} />;
            case 'sectorContribution':
                return <SectorAttribution data={attributionData} isLoading={attributionLoading!} isRefreshing={isRefreshing} currency={currency} />;
            case 'topContributors':
                return <TopContributors data={attributionData} isLoading={attributionLoading!} isRefreshing={isRefreshing} currency={currency} />;
            case 'portfolioDonut':
                return (
                    <Card className="h-full border-border hover:border-cyan-500/20 transition-colors relative overflow-hidden">
                        {isRefreshing && !isLoading && (
                            <div className="absolute top-2 right-2 z-20">
                                <Loader2 className="w-3 h-3 animate-spin text-cyan-500 opacity-70" />
                            </div>
                        )}
                        <CardContent className="h-full p-4 relative">
                            <div className="flex justify-between items-start mb-2">
                                <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest">Portfolio Composition</h3>
                                <div className="p-2 rounded-lg bg-secondary/50 text-muted-foreground">
                                    <PieChart className="w-4 h-4" />
                                </div>
                            </div>
                            <div className="h-[calc(100%-32px)]">
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
            <div className="grid grid-cols-2 md:grid-cols-2 lg:grid-cols-4 gap-3 md:gap-4">
                {visibleScalarItems.map((item) => (
                    <div key={item.id} className={cn(item.colSpan, "w-full min-w-0")}>
                        {renderContent(item.id)}
                    </div>
                ))}
            </div>

            {/* Complex/Tall Metrics Grid */}
            {visibleComplexItems.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
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
