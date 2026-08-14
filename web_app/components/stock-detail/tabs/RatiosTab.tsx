import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchRatios, fetchTrackRecord } from '../../../lib/api';
import { StatementPeriod } from '../../../lib/statement_chart';
import { cn } from '../../../lib/utils';
import { Skeleton } from '../../ui/skeleton';
import { RatioChart } from '../components/RatioChart';
import { TrackRecordPanel } from '../components/TrackRecordPanel';

interface RatiosTabProps {
    symbol: string;
    isOpen: boolean;
}

export const RatiosTab: React.FC<RatiosTabProps> = ({ symbol, isOpen }) => {
    const [ratioPeriod, setRatioPeriod] = useState<StatementPeriod>('quarterly');

    const ratiosQuery = useQuery({
        queryKey: ['stock-ratios', symbol, ratioPeriod],
        queryFn: () => fetchRatios(symbol, ratioPeriod),
        enabled: isOpen && !!symbol,
        staleTime: 30 * 60 * 1000,
    });
    const ratios = ratiosQuery.data ?? null;

    const trackRecordQuery = useQuery({
        queryKey: ['stock-track-record', symbol],
        queryFn: () => fetchTrackRecord(symbol),
        enabled: isOpen && !!symbol,
        staleTime: 60 * 60 * 1000,
    });
    const trackRecord = trackRecordQuery.data ?? null;

    const periodSwitch = (
        <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-[11px] text-muted-foreground">
                {ratioPeriod === 'quarterly'
                    ? 'Measured on the trailing twelve months at each quarter end, so these are the same ratios the annual view reports — sampled four times as often.'
                    : 'Measured on each filed fiscal year.'}
            </p>
            <div className="flex items-center p-0.5 rounded-full bg-muted/50 flex-shrink-0">
                {([
                    { id: 'quarterly', label: 'Quarterly' },
                    { id: 'annual', label: 'Annual' }
                ] as const).map(opt => (
                    <button
                        key={opt.id}
                        onClick={() => setRatioPeriod(opt.id)}
                        aria-pressed={ratioPeriod === opt.id}
                        className={cn(
                            "px-3 sm:px-4 py-1.5 rounded-full text-[10px] sm:text-xs font-bold transition-all whitespace-nowrap cursor-pointer",
                            ratioPeriod === opt.id
                                ? "bg-white dark:bg-zinc-800 text-foreground shadow-sm"
                                : "text-muted-foreground hover:text-foreground"
                        )}
                    >
                        {opt.label}
                    </button>
                ))}
            </div>
        </div>
    );

    if (ratiosQuery.isPending) {
        return (
            <div className="space-y-8 animate-in fade-in duration-500">
                {periodSwitch}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {[0, 1, 2, 3].map(i => <Skeleton key={i} className="h-64 rounded-2xl" />)}
                </div>
            </div>
        );
    }

    if (!ratios || !ratios.historical.length) {
        return (
            <div className="space-y-8 animate-in fade-in duration-500">
                {periodSwitch}
                {trackRecord
                    ? <TrackRecordPanel record={trackRecord} />
                    : <div className="text-center py-20 text-gray-500">No historical ratio data available.</div>}
            </div>
        );
    }

    const chartData = [...ratios.historical].reverse();

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {periodSwitch}
            {trackRecord && <TrackRecordPanel record={trackRecord} />}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <RatioChart
                    periodType={ratioPeriod}
                    data={chartData}
                    dataKey="Return on Equity (ROE) (%)"
                    title="Return on Equity"
                    color="#10b981"
                    suffix="%"
                />
                <RatioChart
                    periodType={ratioPeriod}
                    data={chartData}
                    dataKey="Gross Profit Margin (%)"
                    title="Gross Margin"
                    color="#06b6d4"
                    suffix="%"
                />
                <RatioChart
                    periodType={ratioPeriod}
                    data={chartData}
                    dataKey="Net Profit Margin (%)"
                    title="Net Margin"
                    color="#8b5cf6"
                    suffix="%"
                />
                <RatioChart
                    periodType={ratioPeriod}
                    data={chartData}
                    dataKey="Asset Turnover"
                    title="Asset Turnover"
                    color="#f59e0b"
                />
                <RatioChart
                    periodType={ratioPeriod}
                    data={chartData}
                    dataKey="Return on Invested Capital (ROIC) (%)"
                    title="Return on Invested Capital"
                    color="#ec4899"
                    suffix="%"
                />
                <RatioChart
                    periodType={ratioPeriod}
                    data={chartData}
                    dataKey="Free Cash Flow Margin (%)"
                    title="Free Cash Flow Margin"
                    color="#14b8a6"
                    suffix="%"
                />
                <RatioChart
                    periodType={ratioPeriod}
                    data={chartData}
                    dataKey="Diluted Shares Outstanding"
                    title="Diluted Shares Outstanding"
                    color="#64748b"
                    compact
                />
            </div>
        </div>
    );
};
