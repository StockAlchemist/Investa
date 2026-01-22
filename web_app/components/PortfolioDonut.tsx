'use client';

import React, { useState, useMemo, useEffect } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { Holding } from '@/lib/api';
import { formatCurrency, cn } from '@/lib/utils';
import StockIcon from './StockIcon';
import { ChevronDown, TrendingUp, DollarSign, Percent } from 'lucide-react';

interface PortfolioDonutProps {
    holdings: Holding[];
    currency: string;
}

const COLORS = [
    '#0ea5e9', // sky-500
    '#22c55e', // green-500
    '#eab308', // yellow-500
    '#f97316', // orange-500
    '#ef4444', // red-500
    '#8b5cf6', // violet-500
    '#ec4899', // pink-500
    '#14b8a6', // teal-500
];

const METRICS = [
    { id: 'value', label: 'Total Value', icon: DollarSign },
    { id: 'day_change', label: "Day's Change", icon: TrendingUp },
    { id: 'total_gain', label: 'Unrealized Gain', icon: TrendingUp },
];

const RADIAN = Math.PI / 180;

// Mapping for combining symbols
const SYMBOL_MAPPING: Record<string, string> = {
    'GOOGL': 'GOOG',
    // Add others if needed
};

// Custom label component
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, index, payload, forceAllLabels }: any) => {
    // If forceAllLabels is true, we skip the 3% threshold check.
    // We strictly hide 'Other' if needed, though for Accounts it usually doesn't exist.
    if ((!forceAllLabels && percent < 0.03) || payload.name === 'Other') return null;

    // Push labels further out to avoid crowding
    const radius = outerRadius + 42;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
        <foreignObject x={x - 24} y={y - 28} width={48} height={56} className="overflow-visible pointer-events-none">
            <div className="flex flex-col items-center justify-center w-full h-full transform transition-transform hover:scale-110">
                <div className="shadow-sm rounded-md bg-card p-0.5">
                    <StockIcon symbol={payload.name} size={24} />
                </div>
                <span className="text-[10px] font-bold mt-0.5 text-foreground drop-shadow-md bg-background/50 backdrop-blur-sm px-1.5 py-0.5 rounded-sm border border-border/50 max-w-[70px] truncate text-center">
                    {payload.name}
                </span>
            </div>
        </foreignObject>
    );
};

interface ChartData {
    name: string;
    value: number;
    dayChange: number;
    unrealizedGain: number; // For Total Gain % calc
    costBasis: number; // For Total Gain % calc
    percent: number;
    color: string;
    [key: string]: any;
}

interface SingleDonutProps {
    title: string;
    data: ChartData[];
    currency: string;
    totalValue: number;
    totalDayChange: number;
    totalCostBasis: number;
    totalUnrealizedGain: number;
    metric: string;
    setMetric: (m: string) => void;
    forceAllLabels?: boolean;
}

function SingleDonut({ title, data, currency, totalValue, totalDayChange, totalCostBasis, totalUnrealizedGain, metric, setMetric, forceAllLabels }: SingleDonutProps) {
    const [activeIndex, setActiveIndex] = useState<number | undefined>(undefined);

    const activeItem = activeIndex !== undefined && data[activeIndex] ? data[activeIndex] : null;

    // Calculate display values for active item
    const activeDayChange = activeItem ? activeItem.dayChange : 0;
    const activePrevValue = activeItem ? (activeItem.value - activeItem.dayChange) : 0;
    const activeDayChangePct = activePrevValue !== 0 ? (activeDayChange / activePrevValue) * 100 : 0;

    const activeUnrealizedGain = activeItem ? activeItem.unrealizedGain : 0;
    const activeCostBasis = activeItem ? activeItem.costBasis : 0;
    const activeTotalGainPct = activeCostBasis !== 0 ? (activeUnrealizedGain / activeCostBasis) * 100 : 0;

    // Calculate display values for TOTAL
    const totalPrevValue = totalValue - totalDayChange;
    const totalDayChangePct = totalPrevValue !== 0 ? (totalDayChange / totalPrevValue) * 100 : 0;
    const totalGainPct = totalCostBasis !== 0 ? (totalUnrealizedGain / totalCostBasis) * 100 : 0;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const onPieEnter = (_: any, index: number) => setActiveIndex(index);
    const onPieLeave = () => setActiveIndex(undefined);

    // Helper to get formatted value string for length calculation
    const getMainValueString = (isTotal: boolean) => {
        let val = 0;
        if (metric === 'value') val = isTotal ? totalValue : (activeItem?.value || 0);
        else if (metric === 'day_change') val = isTotal ? totalDayChange : activeDayChange;
        else if (metric === 'total_gain') val = isTotal ? totalUnrealizedGain : activeUnrealizedGain;

        const formatted = formatCurrency(val, currency);
        // Add sign character length if not 'value' metric (approximate)
        return (metric !== 'value' && val > 0) ? '+' + formatted : formatted;
    };

    const getFontSizeClass = (text: string) => {
        const len = text.length;
        if (len > 18) return "text-xs sm:text-sm md:text-base";
        if (len > 14) return "text-sm sm:text-base md:text-lg";
        if (len > 11) return "text-base sm:text-lg md:text-xl";
        return "text-lg sm:text-xl md:text-2xl";
    };

    // Helper to render main value
    const renderMainValue = (isTotal: boolean) => {
        if (metric === 'value') return formatCurrency(isTotal ? totalValue : (activeItem?.value || 0), currency);

        if (metric === 'day_change') {
            const val = isTotal ? totalDayChange : activeDayChange;
            const isPositive = val >= 0;
            return (
                <span className={isPositive ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-500"}>
                    {val > 0 ? '+' : ''}{formatCurrency(val, currency)}
                </span>
            );
        }

        if (metric === 'total_gain') {
            const val = isTotal ? totalUnrealizedGain : activeUnrealizedGain;
            const isPositive = val >= 0;
            return (
                <span className={isPositive ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-500"}>
                    {val > 0 ? '+' : ''}{formatCurrency(val, currency)}
                </span>
            );
        }
        return null;
    };

    // Helper to render subtitle
    const renderSubtitle = (isTotal: boolean) => {
        if (metric === 'value') {
            if (!isTotal) {
                // On hover: Show Allocation %
                // Increased font size
                return (
                    <span className="text-xl sm:text-2xl font-bold text-muted-foreground">
                        {((activeItem?.percent || 0) * 100).toFixed(1)}%
                    </span>
                );
            }
            // Total: Show Day Change $
            const val = totalDayChange;
            return (
                <span className={val >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-500"}>
                    {val > 0 ? '+' : ''}{formatCurrency(val, currency)}
                </span>
            );
        }
        if (metric === 'day_change') {
            // Show Day Change %
            const val = isTotal ? totalDayChangePct : activeDayChangePct;
            return (
                <span className={val >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-500"}>
                    {val > 0 ? '+' : ''}{val.toFixed(2)}%
                </span>
            );
        }
        if (metric === 'total_gain') {
            // Show Total Gain %
            const val = isTotal ? totalGainPct : activeTotalGainPct;
            return (
                <span className={val >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-500"}>
                    {val > 0 ? '+' : ''}{val.toFixed(2)}%
                </span>
            );
        }
        return null;
    };

    const mainActiveText = activeItem ? getMainValueString(false) : '';
    const mainTotalText = getMainValueString(true);

    return (
        <div className="relative h-full">
            <h4 className="absolute top-3 left-4 z-10 text-xs font-semibold text-muted-foreground uppercase tracking-tight">{title}</h4>
            <div className="relative w-full h-full min-h-[380px] md:min-h-[550px]">
                {/* Only render ResponsiveContainer when we have valid data, otherwise it might error with width -1 */}
                {(data && data.length > 0) ? (
                    <ResponsiveContainer width="100%" height="100%" debounce={50}>
                        <PieChart>
                            <Pie
                                data={data}
                                cx="50%"
                                cy="53%"
                                innerRadius="50%"
                                outerRadius="70%"
                                paddingAngle={2}
                                dataKey="value"
                                onMouseEnter={onPieEnter}
                                onMouseLeave={onPieLeave}
                                // Pass forceAllLabels through a closure or similar
                                label={(props) => renderCustomizedLabel({ ...props, forceAllLabels })}
                                labelLine={false}
                                isAnimationActive={false}
                            >
                                {data.map((entry, index) => (
                                    <Cell
                                        key={`cell-${index}`}
                                        fill={entry.color}
                                        strokeWidth={2}
                                        stroke={index === activeIndex ? "rgba(255,255,255,0.8)" : "transparent"}
                                        className="transition-all duration-300 outline-none"
                                    />
                                ))}
                            </Pie>
                        </PieChart>
                    </ResponsiveContainer>
                ) : (
                    <div className="flex items-center justify-center w-full h-full text-muted-foreground text-sm">
                        Loading...
                    </div>
                )}

                {/* Center Content Overlay */}
                <div className="absolute left-1/2 top-[53%] -translate-x-1/2 -translate-y-1/2 pointer-events-none">
                    <div className="flex flex-col items-center justify-center pointer-events-auto">
                        {activeItem ? (
                            <>
                                <div className="mb-1 transition-transform duration-300 transform scale-110">
                                    {activeItem.name !== 'Other' ? (
                                        <StockIcon symbol={activeItem.name} size={32} />
                                    ) : (
                                        <div className="w-8 h-8 rounded-full bg-slate-200 dark:bg-slate-700 flex items-center justify-center">
                                            <span className="text-[10px] font-bold text-muted-foreground">...</span>
                                        </div>
                                    )}
                                </div>
                                <span className="text-sm font-medium text-muted-foreground max-w-[100px] truncate text-center">{activeItem.name}</span>
                                <span className={cn(getFontSizeClass(mainActiveText), "font-bold tracking-tight text-foreground whitespace-nowrap")}>
                                    {renderMainValue(false)}
                                </span>
                                <span className={cn(
                                    "text-base sm:text-lg font-bold tracking-tight mt-0.5",
                                    (metric === 'day_change' || metric === 'total_gain') ? "" : ""
                                )}>
                                    {renderSubtitle(false)}
                                </span>
                            </>
                        ) : (
                            <>
                                <div className="relative group flex items-center justify-center gap-1 cursor-pointer">
                                    <span className="text-xs sm:text-sm font-semibold text-muted-foreground group-hover:text-foreground uppercase tracking-wider transition-colors">
                                        {METRICS.find(m => m.id === metric)?.label}
                                    </span>
                                    <ChevronDown className="w-2.5 h-2.5 text-muted-foreground group-hover:text-foreground transition-colors" />
                                    <select
                                        value={metric}
                                        onChange={(e) => setMetric(e.target.value)}
                                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                    >
                                        {METRICS.map(m => (
                                            <option key={m.id} value={m.id}>{m.label}</option>
                                        ))}
                                    </select>
                                </div>

                                <div className="flex flex-col items-center mt-1">
                                    <span className={cn(getFontSizeClass(mainTotalText), "font-bold tracking-tight text-foreground whitespace-nowrap")}>
                                        {renderMainValue(true)}
                                    </span>
                                    <span className={cn(
                                        "text-base sm:text-lg font-bold tracking-tight mt-0.5"
                                    )}>
                                        {renderSubtitle(true)}
                                    </span>
                                </div>
                            </>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default function PortfolioDonut({ holdings, currency }: PortfolioDonutProps) {
    // Hydration-safe state initialization
    const [holdingsMetric, setHoldingsMetric] = useState('value');
    const [accountsMetric, setAccountsMetric] = useState('value');
    const [isLoaded, setIsLoaded] = useState(false);

    // Initial load from local storage
    useEffect(() => {
        if (typeof window !== 'undefined') {
            const savedHoldings = localStorage.getItem('investa_donut_holdings_metric');
            const savedAccounts = localStorage.getItem('investa_donut_accounts_metric');

            if (savedHoldings) setHoldingsMetric(savedHoldings);
            if (savedAccounts) setAccountsMetric(savedAccounts);
            setIsLoaded(true);
        }
    }, []);

    // Persist changes
    useEffect(() => {
        if (isLoaded && typeof window !== 'undefined') {
            localStorage.setItem('investa_donut_holdings_metric', holdingsMetric);
        }
    }, [holdingsMetric, isLoaded]);

    useEffect(() => {
        if (isLoaded && typeof window !== 'undefined') {
            localStorage.setItem('investa_donut_accounts_metric', accountsMetric);
        }
    }, [accountsMetric, isLoaded]);

    // ... continue with getValue and data processing


    // Helper to get value
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const getValue = (holding: any, key: string) => {
        // First try exact key
        if (holding[key] !== undefined) return holding[key];
        // Try with currency suffix e.g. "Market Value (USD)"
        const keyWithCurrency = `${key} (${currency})`;
        if (holding[keyWithCurrency] !== undefined) return holding[keyWithCurrency];
        // Special case for specific columns if needed or partial match? 
        // For now, assume consistent naming from API.
        return 0;
    };

    const COLORS = [
        '#0ea5e9', // sky-500
        '#22c55e', // green-500
        '#eab308', // yellow-500
        '#f97316', // orange-500
        '#ef4444', // red-500
        '#8b5cf6', // violet-500
        '#ec4899', // pink-500
        '#14b8a6', // teal-500
        '#6366f1', // indigo-500
        '#84cc16', // lime-500
        '#d946ef', // fuchsia-500
        '#06b6d4', // cyan-500
    ];

    // --- Process Holdings Data ---
    const holdingsData = useMemo(() => {
        if (!holdings?.length) return [];

        // 1. Group by Canonical Symbol
        const grouped: Record<string, { value: number; dayChange: number; unrealizedGain: number; costBasis: number }> = {};

        holdings.forEach(h => {
            let symbol = h.Symbol.trim(); // Trim whitespace
            // Apply mapping
            if (SYMBOL_MAPPING[symbol]) symbol = SYMBOL_MAPPING[symbol];

            const val = (getValue(h, 'Market Value') as number) || 0;
            if (val <= 0) return; // Skip zero/negative value holdings?

            const change = (getValue(h, 'Day Change') as number) || 0;
            // For Total Gain, try 'Unreal. Gain' or fallback to 'Unrealized Gain'
            let gain = (getValue(h, 'Unreal. Gain') as number);
            if (gain === 0 && !h['Unreal. Gain']) gain = (getValue(h, 'Unrealized Gain') as number) || 0;

            let basis = (getValue(h, 'Cost Basis') as number);
            // If basis missing, try calculating
            if (!basis && val && gain) basis = val - gain;
            if (!basis) basis = 0; // Fallback

            if (!grouped[symbol]) grouped[symbol] = { value: 0, dayChange: 0, unrealizedGain: 0, costBasis: 0 };

            grouped[symbol].value += val;
            grouped[symbol].dayChange += change;
            grouped[symbol].unrealizedGain += gain;
            grouped[symbol].costBasis += basis;
        });

        // 2. Sort
        const sorted = Object.entries(grouped)
            .map(([name, data]) => ({ name, ...data }))
            .sort((a, b) => b.value - a.value);

        const totalVal = sorted.reduce((sum, item) => sum + item.value, 0);

        const topN = 12;
        const top = sorted.slice(0, topN);
        const other = sorted.slice(topN);

        const processed = top.map((h, i) => ({
            name: h.name,
            value: h.value,
            dayChange: h.dayChange,
            unrealizedGain: h.unrealizedGain,
            costBasis: h.costBasis,
            percent: h.value / totalVal,
            color: COLORS[i % COLORS.length]
        }));

        if (other.length > 0) {
            const otherVal = other.reduce((s, h) => s + h.value, 0);
            processed.push({
                name: 'Other',
                value: otherVal,
                dayChange: other.reduce((s, h) => s + h.dayChange, 0),
                unrealizedGain: other.reduce((s, h) => s + h.unrealizedGain, 0),
                costBasis: other.reduce((s, h) => s + h.costBasis, 0),
                percent: otherVal / totalVal,
                color: '#94a3b8'
            });
        }
        return processed;
    }, [holdings, currency]);

    // --- Process Accounts Data ---
    const accountsData = useMemo(() => {
        if (!holdings?.length) return [];

        // Group by Account
        const grouped: Record<string, { value: number; dayChange: number; unrealizedGain: number; costBasis: number }> = {};

        holdings.forEach(h => {
            const account = h.Account || 'Unknown';
            const val = (getValue(h, 'Market Value') as number) || 0;
            const change = (getValue(h, 'Day Change') as number) || 0;
            let gain = (getValue(h, 'Unreal. Gain') as number);
            if (gain === 0 && !h['Unreal. Gain']) gain = (getValue(h, 'Unrealized Gain') as number) || 0;
            let basis = (getValue(h, 'Cost Basis') as number);
            if (!basis) basis = val - gain;

            if (!grouped[account]) grouped[account] = { value: 0, dayChange: 0, unrealizedGain: 0, costBasis: 0 };
            grouped[account].value += val;
            grouped[account].dayChange += change;
            grouped[account].unrealizedGain += gain;
            grouped[account].costBasis += basis; // Estimate
        });

        // Convert and sort
        const sorted = Object.entries(grouped)
            .map(([name, data]) => ({ name, ...data }))
            .sort((a, b) => b.value - a.value);

        const totalVal = sorted.reduce((sum, item) => sum + item.value, 0);

        return sorted.map((acc, i) => ({
            name: acc.name,
            value: acc.value,
            dayChange: acc.dayChange,
            unrealizedGain: acc.unrealizedGain,
            costBasis: acc.costBasis,
            percent: acc.value / totalVal,
            color: COLORS[i % COLORS.length]
        }));
    }, [holdings, currency]);

    // Totals need to be calculated across all holdings
    const totalValue = useMemo(() => holdings.reduce((sum, h) => sum + ((getValue(h, 'Market Value') as number) || 0), 0), [holdings, currency]);
    const totalDayChange = useMemo(() => holdings.reduce((sum, h) => sum + ((getValue(h, 'Day Change') as number) || 0), 0), [holdings, currency]);

    // Calculate total cost basis and unrealized gain for correct total %
    const totalUnrealizedGain = useMemo(() => holdings.reduce((sum, h) => {
        let gain = getValue(h, 'Unreal. Gain') as number;
        if (gain === 0 && !h['Unreal. Gain']) gain = (getValue(h, 'Unrealized Gain') as number) || 0;
        return sum + gain;
    }, 0), [holdings, currency]);

    const totalCostBasis = useMemo(() => {
        // Try direct sum first
        const directSum = holdings.reduce((sum, h) => sum + ((getValue(h, 'Cost Basis') as number) || 0), 0);
        if (directSum > 0) return directSum;
        // Fallback: Market Value - Unrealized Gain
        return totalValue - totalUnrealizedGain;
    }, [holdings, currency, totalValue, totalUnrealizedGain]);

    return (
        <div className="h-full w-full grid grid-cols-1 md:grid-cols-2 gap-0 md:gap-4">
            <SingleDonut
                title="By Holding"
                data={holdingsData}
                currency={currency}
                totalValue={totalValue}
                totalDayChange={totalDayChange}
                totalCostBasis={totalCostBasis}
                totalUnrealizedGain={totalUnrealizedGain}
                metric={holdingsMetric}
                setMetric={setHoldingsMetric}
            />
            <div className="block md:hidden h-px bg-border/50 my-1" />
            <SingleDonut
                title="By Account"
                data={accountsData}
                currency={currency}
                totalValue={totalValue}
                totalDayChange={totalDayChange}
                totalCostBasis={totalCostBasis}
                totalUnrealizedGain={totalUnrealizedGain}
                metric={accountsMetric}
                setMetric={setAccountsMetric}
                forceAllLabels={true}
            />
        </div>
    );
}
