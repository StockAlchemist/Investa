import React, { useState, useMemo } from 'react';

import type { Dividend } from '../lib/api';
import { formatCurrency, formatCompactNumber, cn } from '../lib/utils';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, LabelList } from 'recharts';

import dynamic from 'next/dynamic';
const StockDetailModal = dynamic(() => import('@/components/StockDetailModal'), { ssr: false });
import StockIcon from './StockIcon';
import TabContentSkeleton from './skeletons/TabContentSkeleton';
import { Search, X, ChevronDown, ChevronUp, CircleDollarSign } from 'lucide-react';
import IncomeKpiStrip from './income/IncomeKpiStrip';
import TopPayers from './income/TopPayers';
import ByAccount from './income/ByAccount';

interface DividendProps {
    data: Dividend[] | null;
    currency: string;
    expectedDividends?: number;
    dividendYield?: number;
    children?: React.ReactNode;
    isLoading?: boolean;
    visibleSections?: string[];
}

type SortableKey = 'Date' | 'Symbol' | 'Account' | 'DividendAmountDisplayCurrency' | 'TaxAmountDisplayCurrency' | 'Net';

function SortIndicator({ active, direction }: { active: boolean; direction: 'asc' | 'desc' }) {
    if (!active) {
        return <ChevronDown className="w-3 h-3 opacity-30 group-hover:opacity-60 transition-opacity" />;
    }
    return direction === 'asc'
        ? <ChevronUp className="w-3 h-3 text-foreground" />
        : <ChevronDown className="w-3 h-3 text-foreground" />;
}

export default function Dividend({
    data, currency, expectedDividends, dividendYield, children, isLoading, visibleSections,
}: DividendProps) {
    const [sortConfig, setSortConfig] = useState<{ key: SortableKey; direction: 'asc' | 'desc' }>({ key: 'Date', direction: 'desc' });
    const [visibleRows, setVisibleRows] = useState(10);
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState('');

    const show = (id: string) => !visibleSections || visibleSections.includes(id);

    // Group by Year for chart, with YoY growth %.
    const dividendsByYear = useMemo(() => {
        const list = data ?? [];
        const groups: Record<string, number> = {};
        list.forEach(item => {
            const year = item.Date.substring(0, 4);
            groups[year] = (groups[year] || 0) + (item['DividendAmountDisplayCurrency'] || 0);
        });
        const arr = Object.entries(groups)
            .map(([year, amount]) => ({ year, amount }))
            .sort((a, b) => a.year.localeCompare(b.year));
        return arr.map((row, i) => ({
            ...row,
            yoyPct: i > 0 && arr[i - 1].amount > 0
                ? ((row.amount - arr[i - 1].amount) / arr[i - 1].amount) * 100
                : null,
        }));
    }, [data]);

    // Filter + sort for the transactions table.
    const filteredData = useMemo(() => {
        const list = data ?? [];
        const q = searchQuery.trim().toLowerCase();
        if (!q) return list;
        return list.filter(d =>
            d.Symbol?.toLowerCase().includes(q) || d.Account?.toLowerCase().includes(q),
        );
    }, [data, searchQuery]);

    const sortedData = useMemo(() => {
        const arr = [...filteredData];
        const dir = sortConfig.direction === 'asc' ? 1 : -1;
        arr.sort((a, b) => {
            if (sortConfig.key === 'Net') {
                const an = (a.DividendAmountDisplayCurrency || 0) - (a.TaxAmountDisplayCurrency || 0);
                const bn = (b.DividendAmountDisplayCurrency || 0) - (b.TaxAmountDisplayCurrency || 0);
                return (an - bn) * dir;
            }
            if (sortConfig.key === 'Date' || sortConfig.key === 'Symbol' || sortConfig.key === 'Account') {
                return String(a[sortConfig.key] ?? '').localeCompare(String(b[sortConfig.key] ?? '')) * dir;
            }
            const av = Number(a[sortConfig.key] ?? 0);
            const bv = Number(b[sortConfig.key] ?? 0);
            return (av - bv) * dir;
        });
        return arr;
    }, [filteredData, sortConfig]);

    if (isLoading) {
        return <TabContentSkeleton type="full" />;
    }

    if (!data || data.length === 0) {
        return (
            <div className="space-y-6">
                {children}
                <div className="metric-card p-12 text-center">
                    <CircleDollarSign className="w-10 h-10 mx-auto mb-3 text-muted-foreground/40" />
                    <p className="text-sm font-medium text-foreground">No dividend history yet</p>
                    <p className="text-xs mt-1 text-muted-foreground/70">
                        Dividend transactions you record (or import) will appear here.
                    </p>
                </div>
            </div>
        );
    }

    const requestSort = (key: SortableKey) => {
        setSortConfig(prev => ({
            key,
            direction: prev.key === key && prev.direction === 'desc' ? 'asc' : 'desc',
        }));
    };

    const sortableHeader = (label: string, fieldKey: SortableKey, align: 'left' | 'right' = 'left') => (
        <button
            type="button"
            onClick={() => requestSort(fieldKey)}
            className={cn(
                'group inline-flex items-center gap-1 hover:text-foreground transition-colors',
                align === 'right' && 'justify-end',
            )}
        >
            <span>{label}</span>
            <SortIndicator active={sortConfig.key === fieldKey} direction={sortConfig.direction} />
        </button>
    );

    const visibleData = sortedData.slice(0, visibleRows);

    const handleShowMore = () => setVisibleRows(prev => prev + 20);
    const handleShowAll = () => setVisibleRows(sortedData.length);

    return (
        <div className="space-y-8 md:space-y-10">
            {/* Enriched KPI strip */}
            <IncomeKpiStrip
                dividends={data}
                currency={currency}
                expectedDividends={expectedDividends}
                dividendYield={dividendYield}
            />

            {/* Injected: IncomeProjector + DividendCalendar */}
            {children}

            {/* Top payers + By account */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <TopPayers dividends={data} currency={currency} />
                <ByAccount dividends={data} currency={currency} />
            </div>

            {/* Annual dividends chart with YoY growth labels */}
            {show('annualDividends') && (
            <div className="metric-card card-shine p-6 relative overflow-hidden group">
                <div className="absolute top-0 left-0 right-0 h-[2px] bg-emerald-500 opacity-80" />
                <h3 className="section-label mb-4 relative z-10">Annual Dividends</h3>
                <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={dividendsByYear} margin={{ top: 24, right: 30, left: 20, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.1)" />
                            <XAxis dataKey="year" tick={{ fontSize: 12, fill: '#6b7280' }} axisLine={{ stroke: 'rgba(255,255,255,0.1)' }} />
                            <YAxis
                                tickFormatter={(val) => formatCompactNumber(val, currency)}
                                tick={{ fill: '#6b7280', fontSize: 12 }}
                                axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                                width={35}
                            />
                            <Bar dataKey="amount" fill="#10b981" name="Dividend Amount" radius={[4, 4, 0, 0]}>
                                <LabelList
                                    dataKey="yoyPct"
                                    position="top"
                                    formatter={(v: unknown) => {
                                        if (typeof v !== 'number' || !isFinite(v)) return '';
                                        return `${v > 0 ? '+' : ''}${v.toFixed(0)}%`;
                                    }}
                                    style={{ fontSize: 10, fontWeight: 700, fill: '#10b981' }}
                                />
                            </Bar>
                            <Tooltip
                                wrapperStyle={{ opacity: 1, zIndex: 1000 }}
                                contentStyle={{ backgroundColor: 'transparent', border: 'none', boxShadow: 'none' }}
                                content={({ active, payload, label }) => {
                                    if (active && payload && payload.length) {
                                        const row = payload[0].payload as { amount: number; yoyPct: number | null };
                                        return (
                                            <div className="bg-background/98 backdrop-blur-2xl p-3 rounded-xl border border-border/60 shadow-2xl">
                                                <p className="font-medium text-foreground mb-1">{label}</p>
                                                <div className="flex items-center gap-2 text-sm">
                                                    <span className="w-2 h-2 rounded-full bg-emerald-500" />
                                                    <span className="text-muted-foreground">Dividends:</span>
                                                    <span className="font-medium text-emerald-600 dark:text-emerald-400 tabular-nums">
                                                        {formatCurrency(row.amount, currency)}
                                                    </span>
                                                </div>
                                                {row.yoyPct != null && (
                                                    <div className="flex items-center gap-2 text-xs mt-1">
                                                        <span className="w-2 h-2 rounded-full bg-transparent" />
                                                        <span className="text-muted-foreground">YoY:</span>
                                                        <span className={cn(
                                                            'font-medium tabular-nums',
                                                            row.yoyPct >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400',
                                                        )}>
                                                            {row.yoyPct > 0 ? '+' : ''}{row.yoyPct.toFixed(1)}%
                                                        </span>
                                                    </div>
                                                )}
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
            )}

            {/* Dividend transactions table — sortable + searchable */}
            {show('dividendTransactions') && (
            <div className="metric-card card-shine overflow-hidden relative group">
                <div className="absolute top-0 left-0 right-0 h-[2px] bg-emerald-500 opacity-80" />
                <div className="p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 relative z-10">
                    <div className="flex items-center gap-3">
                        <h3 className="section-label">Dividend Transactions</h3>
                        <span className="text-[10px] font-medium text-muted-foreground/60 px-2 py-0.5 rounded bg-secondary/50">
                            {visibleData.length} / {sortedData.length}
                        </span>
                    </div>
                    <div className="relative max-w-xs w-full sm:w-auto">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground/60 pointer-events-none" />
                        <input
                            type="text"
                            placeholder="Search symbol or account..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="bg-card border-none text-foreground rounded-md pl-9 pr-8 py-2 text-sm w-full focus:ring-emerald-500 focus:border-emerald-500"
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
                </div>

                {/* Desktop Table View */}
                <div className="hidden md:block overflow-x-auto">
                    <table className="min-w-full">
                        <thead className="bg-muted/30 dark:bg-white/[0.03] backdrop-blur-md sticky top-0 z-10 font-semibold">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground">{sortableHeader('Date', 'Date')}</th>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground">{sortableHeader('Symbol', 'Symbol')}</th>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground">{sortableHeader('Account', 'Account')}</th>
                                <th className="px-6 py-3 text-right text-xs font-semibold text-muted-foreground">{sortableHeader('Gross', 'DividendAmountDisplayCurrency', 'right')}</th>
                                <th className="px-6 py-3 text-right text-xs font-semibold text-muted-foreground">{sortableHeader('Tax', 'TaxAmountDisplayCurrency', 'right')}</th>
                                <th className="px-6 py-3 text-right text-xs font-semibold text-muted-foreground">{sortableHeader('Net', 'Net', 'right')}</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y-none">
                            {visibleData.length === 0 ? (
                                <tr>
                                    <td colSpan={6} className="px-6 py-12 text-center text-muted-foreground">
                                        {searchQuery
                                            ? `No dividends match "${searchQuery}".`
                                            : 'No dividend history found for the selected criteria.'}
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
                                        <td className="px-6 py-3 whitespace-nowrap text-sm text-right text-emerald-500 tabular-nums">
                                            {formatCurrency(item['DividendAmountDisplayCurrency'] || 0, currency)}
                                        </td>
                                        <td className="px-6 py-3 whitespace-nowrap text-sm text-right text-red-500/80 tabular-nums">
                                            {item['TaxAmountDisplayCurrency'] ? formatCurrency(item['TaxAmountDisplayCurrency'], currency) : '-'}
                                        </td>
                                        <td className="px-6 py-3 whitespace-nowrap text-sm text-right text-foreground font-medium tabular-nums">
                                            {formatCurrency((item['DividendAmountDisplayCurrency'] || 0) - (item['TaxAmountDisplayCurrency'] || 0), currency)}
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>

                {/* Mobile Card View */}
                <div className="block md:hidden space-y-4 p-4">
                    {visibleData.length === 0 ? (
                        <p className="text-center text-sm text-muted-foreground py-8">
                            {searchQuery ? `No dividends match "${searchQuery}".` : 'No dividend history.'}
                        </p>
                    ) : visibleData.map((item, index) => (
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
                                    {item['TaxAmountDisplayCurrency'] ? (
                                        <div className="text-[10px] text-red-500/70">
                                            Tax: {formatCurrency(item['TaxAmountDisplayCurrency'], currency)}
                                        </div>
                                    ) : null}
                                    <div className="text-xs font-medium text-foreground">
                                        Net: {formatCurrency((item['DividendAmountDisplayCurrency'] || 0) - (item['TaxAmountDisplayCurrency'] || 0), currency)}
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
            )}

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
