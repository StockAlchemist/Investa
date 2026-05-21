'use client';
import React, { useEffect, useState } from 'react';
import { AssetChangeData, PortfolioSummary, RiskMetrics, PerformanceData, AttributionData } from '../lib/api';
import TabContentSkeleton from './skeletons/TabContentSkeleton';
import KpiStrip from './performance/KpiStrip';
import ReturnsChart from './performance/ReturnsChart';
import MonthlyHeatmap from './performance/MonthlyHeatmap';
import DrawdownTimeline from './performance/DrawdownTimeline';
import BenchmarkScoreboard from './performance/BenchmarkScoreboard';
import { SectorAttribution, TopContributors } from './AttributionChart';

interface AssetChangeProps {
    data: AssetChangeData | null;
    currency: string;
    summary?: PortfolioSummary | null;
    benchmarks?: string[];
    riskMetrics?: RiskMetrics | null;
    history?: PerformanceData[] | null;
    historyLoading?: boolean;
    attribution?: AttributionData | null;
    attributionLoading?: boolean;
    attributionRefreshing?: boolean;
    isLoading?: boolean;
    visibleSections?: string[];
}

export default function AssetChange({
    data,
    currency,
    summary = null,
    benchmarks = [],
    riskMetrics = null,
    history = null,
    historyLoading = false,
    attribution = null,
    attributionLoading = false,
    attributionRefreshing = false,
    isLoading,
    visibleSections,
}: AssetChangeProps) {
    const [mounted, setMounted] = useState(false);
    useEffect(() => setMounted(true), []);

    const show = (id: string) => !visibleSections || visibleSections.includes(id);

    if (isLoading) {
        return <TabContentSkeleton type="chart-only" />;
    }

    if (!data || Object.keys(data).length === 0) {
        return (
            <div className="metric-card p-12 text-center">
                <p className="font-medium text-sm text-muted-foreground">No asset change data available.</p>
                <p className="text-xs mt-1 text-muted-foreground/60">Please ensure your portfolio history is populated.</p>
            </div>
        );
    }

    if (!mounted) {
        return <TabContentSkeleton type="chart-only" />;
    }

    return (
        <div className="p-4 space-y-6">
            {show('kpiStrip') && <KpiStrip data={data} summary={summary} riskMetrics={riskMetrics} benchmarks={benchmarks} />}
            {show('returnsChart') && <ReturnsChart data={data} currency={currency} />}
            {show('monthlyHeatmap') && <MonthlyHeatmap data={data} />}

            {/* Risk view: drawdown + risk-adjusted benchmark stats */}
            {(show('drawdownTimeline') || show('benchmarkScoreboard')) && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {show('drawdownTimeline') && <DrawdownTimeline history={history} isLoading={historyLoading} />}
                    {show('benchmarkScoreboard') && <BenchmarkScoreboard history={history} isLoading={historyLoading} />}
                </div>
            )}

            {/* Attribution: what drove returns */}
            {(show('sectorAttribution') || show('topContributors')) && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {show('sectorAttribution') && (
                        <SectorAttribution
                            data={attribution}
                            isLoading={attributionLoading}
                            isRefreshing={attributionRefreshing}
                            currency={currency}
                        />
                    )}
                    {show('topContributors') && (
                        <TopContributors
                            data={attribution}
                            isLoading={attributionLoading}
                            isRefreshing={attributionRefreshing}
                            currency={currency}
                        />
                    )}
                </div>
            )}
        </div>
    );
}
