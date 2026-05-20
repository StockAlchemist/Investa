import React, { useState, useMemo } from 'react';

import { CapitalGain } from '../lib/api';
import { formatCurrency } from '../lib/utils';
import { exportToCSV } from '../lib/export';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from 'recharts';

import dynamic from 'next/dynamic';
const StockDetailModal = dynamic(() => import('@/components/StockDetailModal'), { ssr: false });
import StockIcon from './StockIcon';
import TabContentSkeleton from './skeletons/TabContentSkeleton';
import { Scale, Search, X, Download, Calendar } from 'lucide-react';
import CapitalGainsKpiStrip from './capital-gains/CapitalGainsKpiStrip';

interface CapitalGainsProps {
    data: CapitalGain[] | null;
    currency: string;
    onDateRangeChange?: (fromDate?: string, toDate?: string) => void;
    isLoading?: boolean;
}

export default function CapitalGains({ data, currency, isLoading }: CapitalGainsProps) {
    const [selectedYear, setSelectedYear] = useState<string | null>(null);
    const [sortConfig, setSortConfig] = useState<{ key: keyof CapitalGain; direction: 'ascending' | 'descending' } | null>({ key: 'Date', direction: 'descending' });
    const [visibleRows, setVisibleRows] = useState(10);
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState('');

    // Group by Year for Chart (Always use full data for context)
    const gainsByYear = useMemo(() => {
        if (!data) return [];
        const groups: Record<string, number> = {};
        data.forEach(item => {
            const year = item.Date.substring(0, 4);
            groups[year] = (groups[year] || 0) + (item['Realized Gain (Display)'] || 0);
        });
        return Object.entries(groups)
            .map(([year, gain]) => ({ year, gain }))
            .sort((a, b) => a.year.localeCompare(b.year));
    }, [data]);

    // Filter data based on year selection + search query.
    const filteredData = useMemo(() => {
        if (!data) return [];
        const q = searchQuery.trim().toLowerCase();
        return data.filter(item => {
            const yearMatch = !selectedYear || item.Date.startsWith(selectedYear);
            const searchMatch = !q
                || item.Symbol?.toLowerCase().includes(q)
                || item.Account?.toLowerCase().includes(q);
            return yearMatch && searchMatch;
        });
    }, [data, selectedYear, searchQuery]);

    // Sorting (on filtered data)
    const sortedData = useMemo(() => {
        // eslint-disable-next-line
        let sortableItems = [...filteredData];
        if (sortConfig !== null) {
            sortableItems.sort((a, b) => {
                if (sortConfig.key === 'Realized Gain (Display)' && (sortConfig as any).isGainPct) {
                    const aCostBasis = a['Total Cost Basis (Display)'] || 0;
                    const bCostBasis = b['Total Cost Basis (Display)'] || 0;
                    const aGainPct = aCostBasis !== 0 ? (a['Realized Gain (Display)'] || 0) / aCostBasis : 0;
                    const bGainPct = bCostBasis !== 0 ? (b['Realized Gain (Display)'] || 0) / bCostBasis : 0;
                    return sortConfig.direction === 'ascending' ? aGainPct - bGainPct : bGainPct - aGainPct;
                }

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
    }, [filteredData, sortConfig]);

    if (isLoading) {
        return <TabContentSkeleton type="full" />;
    }

    if (!data || data.length === 0) {
        return (
            <div className="metric-card p-12 text-center">
                <Scale className="w-10 h-10 mx-auto mb-3 text-muted-foreground/40" />
                <p className="text-sm font-medium text-foreground">No realized capital gains yet</p>
                <p className="text-xs mt-1 text-muted-foreground/70">
                    Sales that close a position will appear here with their realized gain or loss.
                </p>
            </div>
        );
    }

    const requestSort = (key: keyof CapitalGain, isGainPct: boolean = false) => {
        let direction: 'ascending' | 'descending' = 'ascending';
        if (sortConfig && sortConfig.key === key && sortConfig.direction === 'ascending' && (sortConfig as any).isGainPct === isGainPct) {
            direction = 'descending';
        }
        setSortConfig({ key, direction, ...({ isGainPct } as any) });
    };

    const visibleData = sortedData.slice(0, visibleRows);

    const handleShowMore = () => {
        setVisibleRows(prev => prev + 20);
    };

    const handleShowAll = () => {
        setVisibleRows(sortedData.length);
    };

    const handleExport = () => {
        const scope = selectedYear ? `_${selectedYear}` : '';
        exportToCSV(sortedData as unknown as Record<string, unknown>[], `capital_gains${scope}.csv`);
    };

    const handleBarClick = (entry: any) => {
        // When clicking Bar directly, 'entry' is the data item itself (e.g. { year: '2023', gain: 100 })
        if (entry && entry.year) {
            const clickedYear = entry.year;
            if (selectedYear === clickedYear) {
                setSelectedYear(null); // Toggle off
            } else {
                setSelectedYear(clickedYear);
                setVisibleRows(10); // Reset pagination on filter change
            }
        }
    };

    return (
        <div className="space-y-8 md:space-y-12">

            {/* Consolidated KPIs (reflect the active year/search filter) */}
            <CapitalGainsKpiStrip data={filteredData} currency={currency} />

            {/* Annual Gains Chart */}
            <div className="metric-card p-6 relative overflow-hidden group">
                <div className="absolute top-0 left-0 right-0 h-[2px] bg-amber-500 opacity-80" />
                <h3 className="section-label mb-4 relative z-10">Annual Realized Gains</h3>
                <style>{`
                    .recharts-wrapper, .recharts-surface, .recharts-cartesian-grid, .recharts-layer {
                        outline: none !important;
                        box-shadow: none !important;
                    }
                    *:focus {
                        outline: none !important;
                        box-shadow: none !important;
                    }
                `}</style>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                            data={gainsByYear}
                            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                            className="outline-none focus:outline-none [&_.recharts-surface]:outline-none"
                        >
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                            <XAxis dataKey="year" tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }} axisLine={{ stroke: 'hsl(var(--border))' }} />
                            <YAxis
                                tickFormatter={(val) => new Intl.NumberFormat('en-US', { notation: "compact", compactDisplay: "short" }).format(val)}
                                tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 10 }}
                                axisLine={{ stroke: 'hsl(var(--border))' }}
                                width={35}
                            />
                            <Bar
                                dataKey="gain"
                                name="Realized Gain"
                                radius={[4, 4, 0, 0]}
                                onClick={handleBarClick}
                                cursor="pointer"
                            >
                                {gainsByYear.map((entry, index) => {
                                    const positive = entry.gain >= 0;
                                    const isSelected = selectedYear === entry.year;
                                    const isFaded = selectedYear != null && !isSelected;
                                    let fill: string;
                                    if (isFaded) fill = 'var(--glass-hover)';
                                    else if (positive) fill = isSelected ? '#059669' : '#10B981';
                                    else fill = isSelected ? '#dc2626' : '#ef4444';
                                    return <Cell key={`cell-${index}`} fill={fill} />;
                                })}
                            </Bar>
                            <Tooltip
                                wrapperStyle={{ opacity: 1, zIndex: 1000 }}
                                contentStyle={{ backgroundColor: 'transparent', border: 'none' }}
                                content={({ active, payload, label }) => {
                                    if (active && payload && payload.length) {
                                        return (
                                            <div className="bg-background/98 backdrop-blur-2xl p-3 rounded-xl !opacity-100 border border-border/60 shadow-2xl">
                                                <p className="font-medium text-foreground mb-1">{label}</p>
                                                <div className="flex items-center gap-2 text-sm">
                                                    <span className="w-2 h-2 rounded-full bg-emerald-500" />
                                                    <span className="text-muted-foreground">Realized Gain:</span>
                                                    <span className="font-medium text-emerald-600 dark:text-emerald-400">
                                                        {formatCurrency(payload[0].value as number, currency)}
                                                    </span>
                                                </div>
                                                <div className="mt-1 text-xs text-muted-foreground">Click to filter transactions</div>
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
            <div className="metric-card overflow-hidden relative group">
                <div className="absolute top-0 left-0 right-0 h-[2px] bg-amber-500 opacity-80" />
                <div className="p-4 flex flex-col lg:flex-row lg:items-center justify-between gap-3 relative z-10">
                    <div className="flex items-center gap-3 flex-wrap">
                        <h3 className="section-label">Realized Gain Transactions</h3>
                        <span className="text-[10px] font-medium text-muted-foreground/60 px-2 py-0.5 rounded bg-secondary/50">
                            {visibleData.length} / {sortedData.length}
                        </span>
                        {selectedYear && (
                            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-amber-500/10 text-amber-600 dark:text-amber-400 text-xs font-medium">
                                <Calendar className="w-3 h-3" />
                                <span className="font-bold">{selectedYear}</span>
                                <button
                                    onClick={() => setSelectedYear(null)}
                                    className="ml-0.5 hover:text-foreground"
                                    title="Clear year filter"
                                >
                                    <X className="w-3 h-3" />
                                </button>
                            </span>
                        )}
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="relative max-w-xs w-full sm:w-auto">
                            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground/60 pointer-events-none" />
                            <input
                                type="text"
                                placeholder="Search symbol or account..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="bg-card border-none text-foreground rounded-md pl-9 pr-8 py-2 text-sm w-full focus:ring-amber-500 focus:border-amber-500"
                            />
                            {searchQuery && (
                                <button
                                    onClick={() => setSearchQuery('')}
                                    className="absolute right-2 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
                                    title="Clear search"
                                >
                                    <X className="w-3.5 h-3.5" />
                                </button>
                            )}
                        </div>
                        <button
                            onClick={handleExport}
                            className="p-2 text-foreground bg-secondary rounded-lg hover:bg-accent/10 transition-all shrink-0"
                            title="Export filtered set to CSV"
                        >
                            <Download className="h-4 w-4" />
                        </button>
                    </div>
                </div>
                {/* Mobile Card View */}
                <div className="md:hidden space-y-4 p-4">
                    {visibleData.map((item, index) => (
                        <div key={`mobile-${index}`} className="bg-card p-4 rounded-lg">
                            <div className="flex justify-between items-start mb-2">
                                <div>
                                    <h3
                                        className="text-lg font-bold text-foreground cursor-pointer hover:text-cyan-500 transition-colors flex items-center gap-2"
                                        onClick={() => setSelectedSymbol(item.Symbol)}
                                    >
                                        <StockIcon symbol={item.Symbol} size={24} />
                                        {item.Symbol}
                                    </h3>
                                    <div className="text-xs text-muted-foreground">
                                        {item.Date} • {item.Account}
                                    </div>
                                </div>
                                <div className="text-right">
                                    <div className={`text-lg font-bold ${(item['Realized Gain (Display)'] || 0) >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500'}`}>
                                        {formatCurrency(item['Realized Gain (Display)'] || 0, currency)}
                                    </div>
                                    <div className="text-xs text-muted-foreground uppercase tracking-wider">
                                        {item.Type}
                                    </div>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-sm mt-3 pt-3">
                                <div className="flex justify-between">
                                    <span className="text-muted-foreground">Qty:</span>
                                    <span className="text-foreground font-medium">{item.Quantity}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-muted-foreground">Gain %:</span>
                                    {(() => {
                                        const costBasis = item['Total Cost Basis (Display)'] || 0;
                                        const gain = item['Realized Gain (Display)'] || 0;
                                        const gainPct = costBasis !== 0 ? (gain / costBasis) * 100 : 0;
                                        return (
                                            <span className={`font-medium ${gainPct >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500'}`}>
                                                {gainPct.toFixed(2)}%
                                            </span>
                                        );
                                    })()}
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-muted-foreground">Sold:</span>
                                    <span className="text-foreground font-medium whitespace-nowrap">
                                        {formatCurrency(item["Total Proceeds (Display)"] || 0, currency)}
                                    </span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-muted-foreground">Cost:</span>
                                    <span className="text-foreground font-medium whitespace-nowrap">
                                        {formatCurrency(item["Total Cost Basis (Display)"] || 0, currency)}
                                    </span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Desktop Table View */}
                <div className="hidden md:block overflow-x-auto">
                    <table className="min-w-full">
                        <thead className="bg-secondary sticky top-0 z-10 font-semibold">
                            <tr>
                                {['Date', 'Symbol', 'Account', 'Type', 'Quantity', 'Proceeds', 'Cost Basis', 'Realized Gain', 'Gain %'].map((header) => (
                                    <th
                                        key={header}
                                        onClick={() => {
                                            if (header === 'Gain %') {
                                                requestSort('Realized Gain (Display)', true);
                                            } else {
                                                requestSort(
                                                    header === 'Realized Gain' ? 'Realized Gain (Display)' :
                                                        header === 'Proceeds' ? 'Total Proceeds (Display)' :
                                                            header === 'Cost Basis' ? 'Total Cost Basis (Display)' :
                                                                header as keyof CapitalGain
                                                )
                                            }
                                        }}
                                        className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground cursor-pointer hover:bg-accent/10 transition-colors"
                                    >
                                        {header}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y-none">
                            {visibleData.length === 0 ? (
                                <tr>
                                    <td colSpan={9} className="px-6 py-12 text-center text-muted-foreground">
                                        {searchQuery
                                            ? `No realized gains match "${searchQuery}"${selectedYear ? ` in ${selectedYear}` : ''}.`
                                            : 'No realized capital gains found for the selected criteria.'}
                                    </td>
                                </tr>
                            ) : (
                                visibleData.map((item, index) => (
                                    <tr key={index} className="hover:bg-accent/5 transition-colors">
                                        <td className="px-6 py-3 whitespace-nowrap text-sm text-foreground">{item.Date}</td>
                                        <td
                                            className="px-6 py-3 whitespace-nowrap text-sm font-medium text-foreground cursor-pointer hover:text-cyan-500 transition-colors"
                                            onClick={() => setSelectedSymbol(item.Symbol)}
                                        >
                                            <div className="flex items-center gap-2">
                                                <StockIcon symbol={item.Symbol} size={20} />
                                                {item.Symbol}
                                            </div>
                                        </td>
                                        <td className="px-6 py-3 whitespace-nowrap text-sm text-muted-foreground">{item.Account}</td>
                                        <td className="px-6 py-3 whitespace-nowrap text-sm text-muted-foreground">{item.Type}</td>
                                        <td className="px-6 py-3 whitespace-nowrap text-sm text-muted-foreground tabular-nums">{item.Quantity}</td>
                                        <td className="px-6 py-3 whitespace-nowrap text-sm text-right text-muted-foreground tabular-nums">
                                            {formatCurrency(item["Total Proceeds (Display)"] || 0, currency)}
                                        </td>
                                        <td className="px-6 py-3 whitespace-nowrap text-sm text-right text-muted-foreground tabular-nums">
                                            {formatCurrency(item["Total Cost Basis (Display)"] || 0, currency)}
                                        </td>
                                        <td className={`px-6 py-3 whitespace-nowrap text-sm text-right font-medium tabular-nums ${(item['Realized Gain (Display)'] || 0) >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500'
                                            }`}>
                                            {formatCurrency(item['Realized Gain (Display)'] || 0, currency)}
                                        </td>
                                        <td className="px-6 py-3 whitespace-nowrap text-sm text-right font-medium tabular-nums">
                                            {(() => {
                                                const costBasis = item['Total Cost Basis (Display)'] || 0;
                                                const gain = item['Realized Gain (Display)'] || 0;
                                                const gainPct = costBasis !== 0 ? (gain / costBasis) * 100 : 0;
                                                return (
                                                    <span className={gainPct >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500'}>
                                                        {gainPct.toFixed(2)}%
                                                    </span>
                                                );
                                            })()}
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
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
