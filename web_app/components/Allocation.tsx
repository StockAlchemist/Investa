'use client';
import { Holding } from '../lib/api';
import AllocationDrift from './AllocationDrift';
import AllocationPieChart, { AggregatedSlice, PieBucketKey } from './portfolio/AllocationPieChart';
import ConcentrationKpiStrip from './portfolio/ConcentrationKpiStrip';
import HoldingsHeatmap from './HoldingsHeatmap';
import PortfolioTreemap from './portfolio/PortfolioTreemap';
import RebalanceHelper from './portfolio/RebalanceHelper';

interface AllocationProps {
    holdings: Holding[];
    currency: string;
    visibleSections?: string[];
}

function isUnknown(v: unknown): boolean {
    if (v == null) return true;
    const s = String(v).trim().toUpperCase();
    return s === '' || s === '-' || s === 'NONE' || s === 'NULL' || s === 'UNKNOWN'
        || s.startsWith('N/A') || s.startsWith('UNKNOWN');
}

function aggregate(holdings: Holding[], key: PieBucketKey, marketValueKey: string): AggregatedSlice[] {
    const aggregation: Record<string, number> = {};

    holdings.forEach(h => {
        const value = (h[marketValueKey] as number) || 0;
        let raw: unknown;
        if (key === 'Country') {
            raw = (h['geography'] as string) || (h['Country'] as string);
        } else {
            raw = h[key];
        }
        const category = isUnknown(raw) ? 'Unknown' : (raw as string);
        aggregation[category] = (aggregation[category] || 0) + value;
    });

    const sorted = Object.entries(aggregation)
        .map(([name, value]) => ({ name, value }))
        .sort((a, b) => b.value - a.value);

    const totalVal = sorted.reduce((sum, item) => sum + item.value, 0);
    const top: AggregatedSlice[] = [];
    const otherBuckets: string[] = [];
    let otherVal = 0;

    sorted.forEach(item => {
        if (totalVal > 0 && item.value / totalVal >= 0.02) {
            top.push(item);
        } else {
            otherVal += item.value;
            otherBuckets.push(item.name);
        }
    });

    if (otherVal > 0) {
        top.push({ name: 'Other', value: otherVal, sourceBuckets: otherBuckets });
    }

    return top;
}

export default function Allocation({ holdings, currency, visibleSections = [] }: AllocationProps) {
    if (!holdings || holdings.length === 0) {
        return <div className="p-4 text-center text-muted-foreground">No holdings data available.</div>;
    }

    const marketValueKey = `Market Value (${currency})`;
    const assetTypeData = aggregate(holdings, 'quoteType', marketValueKey);
    const sectorData    = aggregate(holdings, 'Sector',    marketValueKey);
    const industryData  = aggregate(holdings, 'Industry',  marketValueKey);
    const countryData   = aggregate(holdings, 'Country',   marketValueKey);

    return (
        <div className="p-4 space-y-6">
            {/* Concentration KPIs */}
            {visibleSections.includes('concentrationKpis') && (
                <ConcentrationKpiStrip holdings={holdings} currency={currency} />
            )}

            {/* Drift vs target — Asset Type / Sector / Country */}
            {visibleSections.includes('categoryDrift') && (
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                    <AllocationDrift
                        holdings={holdings}
                        currency={currency}
                        bucketKey="quoteType"
                        settingsBucket="quoteType"
                        title="Asset Type — drift vs target"
                        storageKey="allocation-target-quoteType"
                    />
                    <AllocationDrift
                        holdings={holdings}
                        currency={currency}
                        bucketKey="Sector"
                        settingsBucket="sector"
                        title="Sector — drift vs target"
                        storageKey="allocation-target-sector"
                    />
                    <AllocationDrift
                        holdings={holdings}
                        currency={currency}
                        bucketKey="Country"
                        settingsBucket="country"
                        title="Country — drift vs target"
                        storageKey="allocation-target-country"
                    />
                </div>
            )}

            {/* Per-stock drift — scrollable since this list can get long */}
            {visibleSections.includes('stockDrift') && (
                <AllocationDrift
                    holdings={holdings}
                    currency={currency}
                    bucketKey="Symbol"
                    settingsBucket="symbol"
                    title="Stocks — drift vs target"
                    storageKey="allocation-target-symbol"
                    scrollable
                />
            )}

            {/* Rebalance helper — suggested trades to close drift */}
            {visibleSections.includes('rebalanceHelper') && (
                <RebalanceHelper holdings={holdings} currency={currency} />
            )}

            {/* Treemap — spatial view of concentration */}
            {visibleSections.includes('treemap') && (
                <PortfolioTreemap holdings={holdings} currency={currency} />
            )}

            {/* Performance heatmap — tiles colored by period return (Finviz-style) */}
            {visibleSections.includes('holdingsHeatmap') && (
                <HoldingsHeatmap holdings={holdings} currency={currency} />
            )}

            {/* Donut charts with click drill-down */}
            {visibleSections.includes('donutCharts') && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <AllocationPieChart title="By Asset Type" data={assetTypeData} currency={currency} holdings={holdings} bucketKey="quoteType" />
                    <AllocationPieChart title="By Sector"     data={sectorData}    currency={currency} holdings={holdings} bucketKey="Sector"    />
                    <AllocationPieChart title="By Industry"   data={industryData}  currency={currency} holdings={holdings} bucketKey="Industry"  />
                    <AllocationPieChart title="By Country"    data={countryData}   currency={currency} holdings={holdings} bucketKey="Country"   />
                </div>
            )}
        </div>
    );
}
