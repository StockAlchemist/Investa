'use client';

import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import {
    fetchFundamentals,
    fetchFinancials,
    fetchRatios,
    Fundamentals,
    FinancialsResponse,
    RatiosResponse
} from '../lib/api';
import {
    X,
    LayoutDashboard,
    FileText,
    TrendingUp,
    Globe,
    Building2,
    Info,
    Calendar,
    DollarSign,
    BarChart3,
    Receipt,
    Scale,
    Users,
    Wallet
} from 'lucide-react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    AreaChart,
    Area
} from 'recharts';
import { cn } from "@/lib/utils";
import { Skeleton } from './ui/skeleton';
import { Badge } from './ui/badge';

interface StockDetailModalProps {
    symbol: string;
    isOpen: boolean;
    onClose: () => void;
    currency: string;
}

type TabType = 'overview' | 'financials' | 'ratios';

// Importance ranking for financial rows
const RANKING_CONFIG: Record<string, string[]> = {
    income: [
        'Total Revenue',
        'Cost Of Revenue',
        'Gross Profit',
        'Operating Expense',
        'Operating Income',
        'EBITDA',
        'EBIT',
        'Pretax Income',
        'Tax Provision',
        'Net Income Common Stockholders',
        'Net Income',
        'Normalized Income',
        'Basic EPS',
        'Diluted EPS'
    ],
    balance: [
        'Total Assets',
        'Current Assets',
        'Cash And Cash Equivalents',
        'Receivables',
        'Inventory',
        'Total Liabilities Net Minority Interest',
        'Current Liabilities',
        'Total Debt',
        'Net Debt',
        'Total Equity Gross Minority Interest',
        'Stockholders Equity',
        'Common Stock Equity',
        'Retained Earnings',
        'Working Capital',
        'Invested Capital',
        'Tangible Book Value'
    ],
    cash: [
        'Operating Cash Flow',
        'Investing Cash Flow',
        'Financing Cash Flow',
        'Capital Expenditure',
        'Free Cash Flow',
        'End Cash Position',
        'Net Income'
    ],
    equity: [
        'Total Equity Gross Minority Interest',
        'Stockholders Equity',
        'Common Stock Equity',
        'Retained Earnings',
        'Capital Stock',
        'Common Stock'
    ]
};

export default function StockDetailModal({ symbol, isOpen, onClose, currency }: StockDetailModalProps) {
    const [activeTab, setActiveTab] = useState<TabType>('overview');
    const [finType, setFinType] = useState<'income' | 'balance' | 'cash' | 'equity'>('income');
    const [fundamentals, setFundamentals] = useState<Fundamentals | null>(null);
    const [financials, setFinancials] = useState<FinancialsResponse | null>(null);
    const [ratios, setRatios] = useState<RatiosResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [logoError, setLogoError] = useState(false);
    useEffect(() => {
        if (isOpen && symbol) {
            setLogoError(false);
            loadData();
        }
    }, [isOpen, symbol]);

    const loadData = async () => {
        setLoading(true);
        setError(null);
        try {
            const [fundRes, finRes, ratioRes] = await Promise.all([
                fetchFundamentals(symbol),
                fetchFinancials(symbol),
                fetchRatios(symbol)
            ]);
            setFundamentals(fundRes);
            setFinancials(finRes);
            setRatios(ratioRes);
        } catch (err: any) {
            console.error(err);
            setError(err.message || "Failed to load stock details");
        } finally {
            setLoading(false);
        }
    };

    if (!isOpen) return null;

    const formatCurrency = (val: number | undefined) => {
        if (val === undefined || val === null) return '-';
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: fundamentals?.currency || 'USD',
            notation: val > 1000000 ? 'compact' : 'standard',
        }).format(val);
    };

    const formatPercent = (val: number | undefined) => {
        if (val === undefined || val === null) return '-';
        return `${val.toFixed(2)}%`;
    };

    const formatCompact = (val: number | undefined) => {
        if (val === undefined || val === null) return '-';
        return new Intl.NumberFormat('en-US', {
            notation: 'compact',
            maximumFractionDigits: 2
        }).format(val);
    };

    const getDomain = (url: string | undefined) => {
        if (!url) return null;
        try {
            // Add protocol if missing for URL parser
            const fullUrl = url.startsWith('http') ? url : `https://${url}`;
            return new URL(fullUrl).hostname.replace('www.', '');
        } catch {
            return null;
        }
    };

    const domain = getDomain(fundamentals?.website);
    const logoUrl = domain && !logoError ? `https://logo.clearbit.com/${domain}` : null;

    const renderOverview = () => {
        if (!fundamentals) return null;
        return (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <StatCard label="Market Cap" value={formatCurrency(fundamentals.marketCap)} icon={Globe} color="text-cyan-400" />
                    <StatCard label="P/E Ratio (TTM)" value={fundamentals.trailingPE?.toFixed(2)} icon={TrendingUp} color="text-emerald-400" />
                    <StatCard label="Dividend Yield" value={formatPercent(fundamentals.dividendYield)} icon={DollarSign} color="text-amber-400" />
                    <StatCard label="52W High" value={formatCurrency(fundamentals.fiftyTwoWeekHigh)} icon={TrendingUp} color="text-blue-400" />
                    <StatCard label="52W Low" value={formatCurrency(fundamentals.fiftyTwoWeekLow)} icon={TrendingUp} color="text-pink-400" className="rotate-180" />
                    <StatCard label="Beta" value={fundamentals.beta?.toFixed(2)} icon={Activity} color="text-purple-400" />
                </div>

                <div className="bg-muted/30 backdrop-blur-md rounded-2xl p-6 border border-border shadow-md">
                    <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                        <Building2 className="w-5 h-5 text-cyan-500" />
                        Business Summary
                    </h3>
                    <p className="text-muted-foreground text-sm leading-relaxed whitespace-pre-wrap">
                        {fundamentals.longBusinessSummary}
                    </p>
                </div>
            </div>
        );
    };

    const renderFinancials = () => {
        if (!financials) return null;

        let rawStatement;
        switch (finType) {
            case 'income': rawStatement = financials.financials; break;
            case 'balance': rawStatement = financials.balance_sheet; break;
            case 'cash': rawStatement = financials.cashflow; break;
            case 'equity': rawStatement = financials.shareholders_equity; break;
            default: rawStatement = financials.financials;
        }

        if (!rawStatement || !rawStatement.index) {
            return (
                <div className="flex flex-col items-center justify-center py-20 text-center text-muted-foreground">
                    <Info className="w-8 h-8 mb-2 opacity-20" />
                    <p>No data available for this statement.</p>
                </div>
            );
        }

        // Apply importance ranking
        const ranking = RANKING_CONFIG[finType] || [];
        const indexedData = rawStatement.index.map((label, idx) => ({
            label,
            data: rawStatement.data[idx]
        }));

        const sortedData = [...indexedData].sort((a, b) => {
            const idxA = ranking.indexOf(a.label);
            const idxB = ranking.indexOf(b.label);

            if (idxA !== -1 && idxB !== -1) return idxA - idxB;
            if (idxA !== -1) return -1;
            if (idxB !== -1) return 1;
            return 0;
        });

        const currentStatement = {
            ...rawStatement,
            index: sortedData.map(d => d.label),
            data: sortedData.map(d => d.data)
        };

        return (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="flex flex-nowrap overflow-x-auto no-scrollbar gap-2 mb-4 pb-2 -mx-4 px-4 sm:mx-0 sm:px-0">
                    {[
                        { id: 'income', label: 'Income', fullLabel: 'Income Statement', icon: Receipt },
                        { id: 'balance', label: 'Balance', fullLabel: 'Balance Sheet', icon: Scale },
                        { id: 'cash', label: 'Cash Flow', fullLabel: 'Cash Flow', icon: Wallet },
                        { id: 'equity', label: 'Equity', fullLabel: "Shareholders' Equity", icon: Users }
                    ].map((btn) => (
                        <button
                            key={btn.id}
                            onClick={() => setFinType(btn.id as any)}
                            className={cn(
                                "flex items-center gap-2 px-3 sm:px-4 py-2 rounded-full text-[10px] sm:text-xs font-bold transition-all border whitespace-nowrap flex-shrink-0",
                                finType === btn.id
                                    ? "bg-cyan-500 text-white border-cyan-500 shadow-lg shadow-cyan-500/20"
                                    : "bg-muted/50 text-muted-foreground border-border hover:bg-muted hover:text-foreground"
                            )}
                            title={btn.fullLabel}
                        >
                            <btn.icon className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                            <span className="hidden sm:inline">{btn.label}</span>
                        </button>
                    ))}
                </div>

                <div className="overflow-x-auto rounded-2xl border border-border bg-muted/20 backdrop-blur-md">
                    <table className="w-full text-sm text-left">
                        <thead className="text-xs uppercase bg-secondary/50 text-muted-foreground">
                            <tr>
                                <th className="px-6 py-4 font-semibold text-foreground sticky left-0 bg-card/80 backdrop-blur-md"></th>
                                {currentStatement.columns.map(col => (
                                    <th key={col} className="px-6 py-4 font-semibold text-center">{new Date(col).getFullYear()}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                            {currentStatement.index.map((item, idx) => (
                                <tr key={item} className="hover:bg-accent/10 transition-colors">
                                    <td className="px-6 py-4 font-medium text-foreground sticky left-0 bg-card/80 backdrop-blur-md min-w-[200px]">{item}</td>
                                    {currentStatement.data[idx].map((val, vIdx) => (
                                        <td key={vIdx} className="px-6 py-4 text-foreground text-right font-medium tabular-nums">
                                            {formatCompact(val as number)}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        );
    };

    const renderRatios = () => {
        if (!ratios || !ratios.historical.length) return <div className="text-center py-20 text-gray-500">No historical ratio data available.</div>;

        const chartData = [...ratios.historical].reverse();

        return (
            <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <RatioChart
                        data={chartData}
                        dataKey="ROE (%)"
                        title="Return on Equity"
                        color="#10b981"
                        suffix="%"
                    />
                    <RatioChart
                        data={chartData}
                        dataKey="Gross Profit Margin (%)"
                        title="Gross Margin"
                        color="#06b6d4"
                        suffix="%"
                    />
                    <RatioChart
                        data={chartData}
                        dataKey="Net Profit Margin (%)"
                        title="Net Margin"
                        color="#8b5cf6"
                        suffix="%"
                    />
                    <RatioChart
                        data={chartData}
                        dataKey="Asset Turnover"
                        title="Asset Turnover"
                        color="#f59e0b"
                    />
                </div>
            </div>
        );
    };

    return createPortal(
        <div className="fixed inset-0 z-[100] flex flex-col justify-end sm:justify-center items-center p-0 sm:p-4 isolate">
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

            <div className="relative bg-card w-full max-w-5xl h-[94vh] sm:h-auto sm:max-h-[90vh] rounded-t-[2.5rem] sm:rounded-[2rem] flex flex-col shadow-2xl overflow-hidden animate-in slide-in-from-bottom sm:zoom-in-95 duration-300">

                {/* Mobile Drag Handle */}
                <div className="sm:hidden w-full flex justify-center pt-3 pb-1 flex-shrink-0">
                    <div className="w-12 h-1.5 bg-border/50 rounded-full" />
                </div>

                {/* Sticky Header & Tabs Container */}
                <div className="sticky top-0 z-50 bg-card/98 backdrop-blur-xl border-b border-border flex-shrink-0 shadow-sm">
                    {/* Header */}
                    <div className="p-5 sm:p-8 pb-3 sm:pb-4 flex justify-between items-start relative">
                        <div className="hidden sm:block absolute top-0 right-0 w-64 h-64 bg-cyan-500/10 rounded-full blur-[100px] -mr-32 -mt-32" />

                        <div className="flex items-center gap-4 sm:gap-6 relative z-10 text-foreground">
                            <div className="w-10 h-10 sm:w-16 sm:h-16 rounded-xl sm:rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-lg sm:text-3xl font-bold shadow-lg shadow-cyan-500/20 text-white overflow-hidden flex-shrink-0">
                                {logoUrl ? (
                                    <img
                                        src={logoUrl}
                                        alt={fundamentals?.shortName || symbol}
                                        className="w-full h-full object-cover p-2 bg-white"
                                        onError={() => setLogoError(true)}
                                    />
                                ) : (
                                    fundamentals?.shortName?.[0] || symbol[0]
                                )}
                            </div>
                            <div>
                                <div className="flex items-center gap-2 sm:gap-3 mb-0.5 sm:mb-1">
                                    <h2 className="text-lg sm:text-3xl font-bold tracking-tight truncate max-w-[140px] sm:max-w-none">{fundamentals?.shortName || symbol}</h2>
                                    <Badge className="bg-secondary text-secondary-foreground border-none font-mono text-[9px] sm:text-xs">{symbol}</Badge>
                                </div>
                                <p className="text-muted-foreground flex items-center gap-1.5 sm:gap-2 text-[9px] sm:text-sm">
                                    <span className="font-semibold text-cyan-500">{fundamentals?.sector}</span>
                                    <span className="text-border">•</span>
                                    <span className="truncate max-w-[100px] sm:max-w-none">{fundamentals?.industry}</span>
                                </p>
                            </div>
                        </div>

                        <button
                            onClick={onClose}
                            className="p-1 px-2 -mr-1 hover:bg-muted rounded-full transition-colors text-muted-foreground hover:text-foreground relative z-20"
                            aria-label="Close modal"
                        >
                            <X className="w-5 h-5 sm:w-6 sm:h-6" />
                        </button>
                    </div>

                    {/* Tabs */}
                    <div className="px-4 sm:px-8 flex justify-around sm:justify-start gap-2 sm:gap-8 overflow-x-auto no-scrollbar">
                        <TabButton
                            active={activeTab === 'overview'}
                            onClick={() => setActiveTab('overview')}
                            icon={LayoutDashboard}
                            label="Overview"
                        />
                        <TabButton
                            active={activeTab === 'financials'}
                            onClick={() => setActiveTab('financials')}
                            icon={FileText}
                            label="Financials"
                        />
                        <TabButton
                            active={activeTab === 'ratios'}
                            onClick={() => setActiveTab('ratios')}
                            icon={BarChart3}
                            label="Ratios & Trends"
                        />
                    </div>
                </div>

                {/* Content Area */}
                <div className="flex-1 overflow-y-auto p-4 sm:p-8 custom-scrollbar">
                    {loading ? (
                        <div className="space-y-4">
                            <Skeleton className="h-40 w-full rounded-2xl" />
                            <div className="grid grid-cols-3 gap-4">
                                <Skeleton className="h-24 rounded-2xl" />
                                <Skeleton className="h-24 rounded-2xl" />
                                <Skeleton className="h-24 rounded-2xl" />
                            </div>
                            <Skeleton className="h-60 w-full rounded-2xl" />
                        </div>
                    ) : error ? (
                        <div className="flex flex-col items-center justify-center py-20 text-center">
                            <div className="w-16 h-16 bg-destructive/10 rounded-full flex items-center justify-center mb-4">
                                <Info className="w-8 h-8 text-destructive" />
                            </div>
                            <h3 className="text-xl font-bold mb-2">Something went wrong</h3>
                            <p className="text-muted-foreground max-w-md">{error}</p>
                            <button onClick={loadData} className="mt-6 px-6 py-2 bg-secondary hover:bg-muted rounded-full transition-colors">
                                Try Again
                            </button>
                        </div>
                    ) : (
                        <>
                            {activeTab === 'overview' && renderOverview()}
                            {activeTab === 'financials' && renderFinancials()}
                            {activeTab === 'ratios' && renderRatios()}
                        </>
                    )}
                </div>

                {/* Footer */}
                <div className="px-4 sm:px-8 py-3 sm:py-4 bg-muted/30 border-t border-border flex flex-col sm:flex-row justify-between items-center gap-2 text-[9px] sm:text-[10px] text-muted-foreground uppercase tracking-widest font-bold pb-[calc(0.75rem+env(safe-area-inset-bottom))] sm:pb-4">
                    <div className="flex gap-4">
                        <span>Exchange: {fundamentals?.exchange || 'Unknown'}</span>
                        <span>Currency: {fundamentals?.currency || 'USD'}</span>
                    </div>
                    <span className="text-center sm:text-right">Data by Yahoo Finance & Stock Alchemist</span>
                </div>
            </div>
        </div>,
        document.body
    );
}

function StatCard({ label, value, icon: Icon, color, className, rotate }: any) {
    return (
        <div className="bg-muted/30 backdrop-blur-md border border-border p-5 rounded-2xl flex items-center gap-4 transition-all hover:bg-muted/50 hover:border-accent group">
            <div className={cn("p-3 rounded-xl bg-card border border-border", color, rotate)}>
                <Icon className="w-5 h-5" />
            </div>
            <div>
                <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider mb-0.5">{label}</p>
                <p className="text-lg font-bold text-foreground tracking-tight">{value}</p>
            </div>
        </div>
    );
}

function TabButton({ active, onClick, icon: Icon, label }: any) {
    return (
        <button
            onClick={onClick}
            className={cn(
                "py-4 px-4 flex items-center gap-2 text-sm font-medium transition-all relative border-b-2 outline-none focus-visible:ring-2 focus-visible:ring-cyan-500/20",
                active ? "text-cyan-600 dark:text-cyan-400 border-cyan-600 dark:border-cyan-400" : "text-muted-foreground border-transparent hover:text-foreground"
            )}
        >
            <Icon className="w-5 h-5 sm:w-4 sm:h-4" />
            <span className="whitespace-nowrap hidden sm:inline">{label}</span>
        </button>
    );
}

function RatioChart({ data, dataKey, title, color, suffix = "" }: any) {
    const sanitizedId = `gradient-${dataKey.replace(/[^a-zA-Z0-9]/g, '')}`;
    return (
        <div className="bg-muted/30 backdrop-blur-md border border-border p-6 rounded-2xl">
            <h4 className="text-sm font-semibold text-muted-foreground mb-6 uppercase tracking-wider">{title}</h4>
            <div className="h-48 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data}>
                        <defs>
                            <linearGradient id={sanitizedId} x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                                <stop offset="95%" stopColor={color} stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-border" opacity={0.1} vertical={false} />
                        <XAxis
                            dataKey="Period"
                            axisLine={false}
                            tickLine={false}
                            tick={{ fontSize: 10 }}
                            className="fill-muted-foreground"
                            tickFormatter={(val) => new Date(val).getFullYear().toString()}
                        />
                        <YAxis
                            axisLine={false}
                            tickLine={false}
                            tick={{ fontSize: 10 }}
                            className="fill-muted-foreground"
                            tickFormatter={(val) => `${val}${suffix}`}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: 'var(--card)', border: '1px solid var(--border)', borderRadius: '12px', fontSize: '12px', color: 'var(--foreground)' }}
                            itemStyle={{ color: color }}
                            formatter={(val: any) => [`${val.toFixed(2)}${suffix}`, title]}
                        />
                        <Area
                            type="monotone"
                            dataKey={dataKey}
                            stroke={color}
                            strokeWidth={3}
                            fillOpacity={1}
                            fill={`url(#${sanitizedId})`}
                            animationDuration={1500}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}

const Activity = (props: any) => (
    <svg
        {...props}
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
    >
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
)
