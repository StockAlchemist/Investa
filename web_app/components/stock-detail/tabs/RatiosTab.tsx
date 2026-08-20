import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchRatios, fetchTrackRecord, type FinancialRatio } from '../../../lib/api';
import { StatementPeriod } from '../../../lib/statement_chart';
import { cn } from '../../../lib/utils';
import { Skeleton } from '../../ui/skeleton';
import { RatioChart } from '../components/RatioChart';
import { TrackRecordPanel } from '../components/TrackRecordPanel';

interface RatiosTabProps {
    symbol: string;
    isOpen: boolean;
}

interface ChartItem {
    group: 'Valuation' | 'Profitability' | 'Balance Sheet' | 'Earnings & Sales';
    dataKey: string;
    title: string;
    color: string;
    suffix?: string;
    compact?: boolean;
}

const ALL_RATIO_CHARTS: ChartItem[] = [
    // 1. Valuation
    { group: 'Valuation', dataKey: 'P/E Ratio', title: 'Price to Earnings (P/E)', color: '#10b981' },
    { group: 'Valuation', dataKey: 'P/S Ratio', title: 'Price to Sales (P/S)', color: '#06b6d4' },
    { group: 'Valuation', dataKey: 'P/B Ratio', title: 'Price to Book (P/B)', color: '#8b5cf6' },
    { group: 'Valuation', dataKey: 'EV/EBITDA', title: 'EV / EBITDA', color: '#f59e0b' },
    { group: 'Valuation', dataKey: 'EV/Sales', title: 'EV / Sales', color: '#ec4899' },
    { group: 'Valuation', dataKey: 'P/FCF Ratio', title: 'Price to Free Cash Flow (P/FCF)', color: '#14b8a6' },
    { group: 'Valuation', dataKey: 'Dividend Yield (%)', title: 'Dividend Yield', color: '#10b981', suffix: '%' },

    // 2. Profitability
    { group: 'Profitability', dataKey: 'Return on Invested Capital (ROIC) (%)', title: 'Return on Invested Capital (ROIC)', color: '#ec4899', suffix: '%' },
    { group: 'Profitability', dataKey: 'Return on Equity (ROE) (%)', title: 'Return on Equity (ROE)', color: '#10b981', suffix: '%' },
    { group: 'Profitability', dataKey: 'Return on Assets (ROA) (%)', title: 'Return on Assets (ROA)', color: '#06b6d4', suffix: '%' },
    { group: 'Profitability', dataKey: 'Gross Profit Margin (%)', title: 'Gross Margin', color: '#8b5cf6', suffix: '%' },
    { group: 'Profitability', dataKey: 'Net Profit Margin (%)', title: 'Net Margin', color: '#f59e0b', suffix: '%' },
    { group: 'Profitability', dataKey: 'Free Cash Flow Margin (%)', title: 'Free Cash Flow Margin', color: '#14b8a6', suffix: '%' },

    // 3. Balance Sheet & Solvency
    { group: 'Balance Sheet', dataKey: 'Current Ratio', title: 'Current Ratio', color: '#10b981' },
    { group: 'Balance Sheet', dataKey: 'Quick Ratio', title: 'Quick Ratio', color: '#06b6d4' },
    { group: 'Balance Sheet', dataKey: 'Debt-to-Equity Ratio', title: 'Debt to Equity', color: '#f59e0b' },
    { group: 'Balance Sheet', dataKey: 'Long-Term Debt to Equity', title: 'LT Debt to Equity', color: '#8b5cf6' },
    { group: 'Balance Sheet', dataKey: 'Interest Coverage Ratio', title: 'Interest Coverage Ratio', color: '#ec4899' },
    { group: 'Balance Sheet', dataKey: 'Asset Turnover', title: 'Asset Turnover', color: '#06b6d4' },
    { group: 'Balance Sheet', dataKey: 'Diluted Shares Outstanding', title: 'Diluted Shares Outstanding', color: '#64748b', compact: true },

    // 4. Earnings & Sales
    { group: 'Earnings & Sales', dataKey: 'Diluted EPS', title: 'Diluted EPS ($)', color: '#10b981' },
    { group: 'Earnings & Sales', dataKey: 'Total Revenue', title: 'Total Revenue (Sales)', color: '#06b6d4', compact: true },
    { group: 'Earnings & Sales', dataKey: 'Revenue Growth YoY (%)', title: 'Revenue Growth YoY', color: '#8b5cf6', suffix: '%' },
    { group: 'Earnings & Sales', dataKey: 'EPS Growth YoY (%)', title: 'EPS Growth YoY', color: '#ec4899', suffix: '%' },
    { group: 'Earnings & Sales', dataKey: 'Operating Margin (%)', title: 'Operating Margin', color: '#f59e0b', suffix: '%' },
];

export const RatiosTab: React.FC<RatiosTabProps> = ({ symbol, isOpen }) => {
    const [ratioPeriod, setRatioPeriod] = useState<StatementPeriod>('quarterly');
    const [selectedCategory, setSelectedCategory] = useState<string>('All');

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
                                ? "bg-indigo-600 text-white shadow-sm"
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

    const chartData: FinancialRatio[] = [...ratios.historical].reverse();

    const categories = ['All', 'Valuation', 'Profitability', 'Balance Sheet', 'Earnings & Sales'];
    const filteredCharts = selectedCategory === 'All'
        ? ALL_RATIO_CHARTS
        : ALL_RATIO_CHARTS.filter(c => c.group === selectedCategory);

    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {periodSwitch}
            {trackRecord && <TrackRecordPanel record={trackRecord} />}

            {/* Category Filter Pills */}
            <div className="flex items-center gap-1.5 overflow-x-auto pb-1">
                {categories.map(cat => (
                    <button
                        key={cat}
                        type="button"
                        onClick={() => setSelectedCategory(cat)}
                        className={cn(
                            "px-3.5 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all cursor-pointer",
                            selectedCategory === cat
                                ? "bg-muted text-foreground border border-border shadow-sm font-bold"
                                : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                        )}
                    >
                        {cat}
                    </button>
                ))}
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {filteredCharts.map(item => (
                    <RatioChart
                        key={item.dataKey}
                        periodType={ratioPeriod}
                        data={chartData}
                        dataKey={item.dataKey}
                        title={item.title}
                        color={item.color}
                        suffix={item.suffix}
                        compact={item.compact}
                    />
                ))}
            </div>
        </div>
    );
};
