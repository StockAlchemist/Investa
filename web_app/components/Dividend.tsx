import React, { useState, useMemo } from 'react';

import type { Dividend } from '../lib/api';
import { formatCurrency, formatCompactNumber } from '../lib/utils';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

import dynamic from 'next/dynamic';
const StockDetailModal = dynamic(() => import('@/components/StockDetailModal'), { ssr: false });
import StockIcon from './StockIcon';
import TabContentSkeleton from './skeletons/TabContentSkeleton';
import { CircleDollarSign, CalendarClock, Percent } from 'lucide-react';
import { MetricCard } from './MetricCard';

interface DividendProps {
    data: Dividend[] | null;
    currency: string;
    expectedDividends?: number;
    dividendYield?: number;
    children?: React.ReactNode;
    isLoading?: boolean;
}

export default function Dividend({ data, currency, expectedDividends, dividendYield, children, isLoading }: DividendProps) {
    const [sortConfig, setSortConfig] = useState<{ key: keyof Dividend; direction: 'ascending' | 'descending' } | null>({ key: 'Date', direction: 'descending' });
    const [visibleRows, setVisibleRows] = useState(10);
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);

    // Group by Year for Chart
    const dividendsByYear = useMemo(() => {
        if (!data) return [];
        const groups: Record<string, number> = {};
        data.forEach(item => {
            const year = item.Date.substring(0, 4);
            groups[year] = (groups[year] || 0) + (item['DividendAmountDisplayCurrency'] || 0);
        });
        return Object.entries(groups)
            .map(([year, amount]) => ({ year, amount }))
            .sort((a, b) => a.year.localeCompare(b.year));
    }, [data]);

    // Sorting
    const sortedData = useMemo(() => {
        if (!data) return [];
        const sortableItems = [...data];
        if (sortConfig !== null) {
            sortableItems.sort((a, b) => {
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                const aValue = a[sortConfig.key] as any;
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                const bValue = b[sortConfig.key] as any;

                if (aValue < bValue) {
                    return sortConfig.direction === 'ascending' ? -1 : 1;
                }
                if (aValue > bValue) {
                    return sortConfig.direction === 'ascending' ? 1 : -1;
                }
                return 0;
            });
        }
        return sortableItems;
    }, [data, sortConfig]);

    if (isLoading) {
        return <TabContentSkeleton type="full" />;
    }

    if (!data) {
        return <div className="p-4 text-center text-muted-foreground">Loading dividend data...</div>;
    }



    // --- Calculations ---
    const totalDividends = data.reduce((sum, item) => sum + (item['DividendAmountDisplayCurrency'] || 0), 0);

    const requestSort = (key: keyof Dividend) => {
        let direction: 'ascending' | 'descending' = 'ascending';
        if (sortConfig && sortConfig.key === key && sortConfig.direction === 'ascending') {
            direction = 'descending';
        }
        setSortConfig({ key, direction });
    };

    const visibleData = sortedData.slice(0, visibleRows);

    const handleShowMore = () => {
        setVisibleRows(prev => prev + 20);
    };

    const handleShowAll = () => {
        setVisibleRows(sortedData.length);
    };

    return (
        <div className="space-y-8 md:space-y-12">


            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <MetricCard
                    title="Total Dividends"
                    value={totalDividends}
                    currency={currency}
                    colorClass="text-emerald-600 dark:text-emerald-400"
                    icon={CircleDollarSign}
                    accentColor="emerald-500"
                />

                {expectedDividends !== undefined && (
                    <MetricCard
                        title="Expected Dividends (Next 12M)"
                        value={expectedDividends}
                        currency={currency}
                        colorClass="text-cyan-600 dark:text-cyan-400"
                        icon={CalendarClock}
                    />
                )}

                {dividendYield !== undefined && (
                    <MetricCard
                        title="Annual Yield %"
                        value={`${dividendYield.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`}
                        isCurrency={false}
                        colorClass="text-purple-600 dark:text-purple-400"
                        icon={Percent}
                    />
                )}
            </div>

            {/* Injected Content (e.g. Dividend Calendar) */}
            {children}

            {/* Chart Section */}
            <div className="metric-card card-shine p-6 relative overflow-hidden group">
                <div className="absolute top-0 left-0 right-0 h-[2px] bg-emerald-500 opacity-80" />
                <h3 className="section-label mb-4 relative z-10">Annual Dividends</h3>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={dividendsByYear} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.1)" />
                            <XAxis dataKey="year" tick={{ fontSize: 12, fill: '#6b7280' }} axisLine={{ stroke: 'rgba(255,255,255,0.1)' }} />
                            <YAxis
                                tickFormatter={(val) => formatCompactNumber(val, currency)}
                                tick={{ fill: '#6b7280', fontSize: 12 }}
                                axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                                width={35}
                            />
                            <Bar dataKey="amount" fill="#3B82F6" name="Dividend Amount" radius={[4, 4, 0, 0]} />
                            <Tooltip
                                wrapperStyle={{ opacity: 1, zIndex: 1000 }}
                                contentStyle={{
                                    backgroundColor: 'transparent',
                                    borderRadius: '12px',
                                    border: 'none',
                                    boxShadow: 'none',
                                    color: 'var(--foreground)'
                                }}
                                content={({ active, payload, label }) => {
                                    if (active && payload && payload.length) {
                                        return (
                                            <div className="bg-background/98 backdrop-blur-2xl p-3 rounded-xl !opacity-100 border border-border/60 shadow-2xl">
                                                <p className="font-medium text-foreground mb-1">{label}</p>
                                                <div className="flex items-center gap-2 text-sm">
                                                    <span className="w-2 h-2 rounded-full bg-blue-500" />
                                                    <span className="text-muted-foreground">Dividend Amount:</span>
                                                    <span className="font-medium text-blue-500">
                                                        {formatCurrency(payload[0].value as number, currency)}
                                                    </span>
                                                </div>
                                            </div>
                                        );
                                    }
                                    return null;
                                }}
                                cursor={{ fill: 'var(--glass-hover)' }}
                            />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Transactions Table */}
            <div className="metric-card card-shine overflow-hidden relative group">
                <div className="absolute top-0 left-0 right-0 h-[2px] bg-emerald-500 opacity-80" />
                <div className="p-4 flex justify-between items-center relative z-10">
                    <h3 className="section-label">Dividend Transactions</h3>
                    <div className="text-xs font-medium text-muted-foreground/60">
                        {visibleData.length} / {sortedData.length} transactions
                    </div>
                </div>
                {/* Desktop Table View */}
                <div className="hidden md:block overflow-x-auto">
                    <table className="min-w-full">
                        <thead className="bg-muted/30 dark:bg-white/[0.03] backdrop-blur-md sticky top-0 z-10 font-semibold">
                            <tr>
                                {['Date', 'Symbol', 'Account', 'DividendAmountDisplayCurrency'].map((header) => (
                                    <th
                                        key={header}
                                        onClick={() => requestSort(header as keyof Dividend)}
                                        className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground cursor-pointer hover:bg-accent/10 transition-colors"
                                    >
                                        {header === 'DividendAmountDisplayCurrency' ? 'Amount' : header}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y-none">
                            {visibleData.length === 0 ? (
                                <tr>
                                    <td colSpan={4} className="px-6 py-12 text-center text-muted-foreground">
                                        No dividend history found for the selected criteria.
                                    </td>
                                </tr>
                            ) : (
                                visibleData.map((item, index) => (
                                    <tr key={index} className="hover:bg-accent/5 transition-colors">
                                        <td className="px-6 py-3 whitespace-nowrap text-sm text-foreground">{item.Date}</td>
                                        <td
                                            className="px-6 py-3 whitespace-nowrap text-sm font-medium text-foreground cursor-pointer hover:text-indigo-500 transition-colors"
                                            onClick={() => setSelectedSymbol(item.Symbol)}
                                        >
                                            <div className="flex items-center gap-2">
                                                <StockIcon symbol={item.Symbol} size={20} />
                                                {item.Symbol}
                                            </div>
                                        </td>
                                        <td className="px-6 py-3 whitespace-nowrap text-sm text-muted-foreground">{item.Account}</td>
                                        <td className="px-6 py-3 whitespace-nowrap text-sm text-right text-muted-foreground tabular-nums">
                                            {formatCurrency(item['DividendAmountDisplayCurrency'] || 0, currency)}
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>

                {/* Mobile Card View */}
                <div className="block md:hidden space-y-4 p-4">
                    {visibleData.map((item, index) => (
                        <div key={`mobile-div-${index}`} className="bg-muted/20 dark:bg-white/[0.03] backdrop-blur-md rounded-2xl p-4 border border-border/40 dark:border-white/[0.05]">
                            <div className="flex justify-between items-start mb-2">
                                <div>
                                    <h3
                                        className="text-lg font-bold text-foreground cursor-pointer hover:text-indigo-500 transition-colors flex items-center gap-2"
                                        onClick={() => setSelectedSymbol(item.Symbol)}
                                    >
                                        <StockIcon symbol={item.Symbol} size={24} />
                                        {item.Symbol}
                                    </h3>
                                    <div className="text-xs text-muted-foreground">{item.Date} • {item.Account}</div>
                                </div>
                                <div className="text-right">
                                    <div className="text-lg font-bold text-emerald-500">
                                        {formatCurrency(item['DividendAmountDisplayCurrency'] || 0, currency)}
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
                {visibleRows < sortedData.length && (
                    <div className="flex justify-center gap-4 p-4">
                        <button
                            onClick={handleShowMore}
                            className="px-4 py-2 bg-[#0097b2] text-white rounded-md hover:bg-[#0086a0] transition-colors text-sm font-medium"
                        >
                            Show More
                        </button>
                        <button
                            onClick={handleShowAll}
                            className="px-4 py-2 bg-card text-foreground rounded-md hover:bg-secondary transition-colors text-sm font-medium"
                        >
                            Show All
                        </button>
                    </div>
                )}
            </div>

            {/* Stock Detail Modal */}
            {selectedSymbol && (
                <StockDetailModal
                    symbol={selectedSymbol}
                    isOpen={!!selectedSymbol}
                    onClose={() => setSelectedSymbol(null)}
                    currency={currency}
                />
            )}
        </div>
    );
}
