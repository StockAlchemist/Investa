'use client';

import React, { useState, useEffect, useId } from 'react';
import { useTheme } from 'next-themes';
import { createPortal } from 'react-dom';
import {
    fetchFundamentals,
    fetchFinancials,
    fetchRatios,
    fetchIntrinsicValue,
    fetchStockAnalysis,
    Fundamentals,
    FinancialsResponse,
    RatiosResponse,
    IntrinsicValueResponse,
    StockAnalysisResponse
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
    Wallet,
    PieChart as PieChartIcon,
    List,
    HelpCircle,
    Sparkles,
    Shield,
    Zap,
    Target,
    Activity as LucideActivity,
    CheckCircle2
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
    Area,
    ReferenceLine,
    PieChart,
    Pie,
    Cell,
    Legend
} from 'recharts';
import { cn, formatPercent as formatPercentShared } from "@/lib/utils";
import { Skeleton } from './ui/skeleton';
import { Badge } from './ui/badge';
import StockIcon from './StockIcon';

interface StockDetailModalProps {
    symbol: string;
    isOpen: boolean;
    onClose: () => void;
    currency: string;
}

type TabType = 'overview' | 'financials' | 'ratios' | 'valuation' | 'holdings' | 'analysis';

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
    const [intrinsicValue, setIntrinsicValue] = useState<IntrinsicValueResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [viewingDistribution, setViewingDistribution] = useState<'dcf' | 'graham' | null>(null);
    const [analysis, setAnalysis] = useState<any>(null);
    const [analysisLoading, setAnalysisLoading] = useState(false);
    const [analysisError, setAnalysisError] = useState<string | null>(null);
    const { resolvedTheme } = useTheme();
    const isDarkMode = resolvedTheme === 'dark';

    useEffect(() => {
        if (activeTab === 'analysis' && !analysis && !analysisLoading && !analysisError) {
            const getAnalysis = async () => {
                setAnalysisLoading(true);
                try {
                    setAnalysisError(null);
                    const data = await fetchStockAnalysis(symbol);
                    if (data && data.error) {
                        setAnalysisError(data.error);
                    } else {
                        setAnalysis(data);
                    }
                } catch (err: any) {
                    console.error("Analysis fetch error:", err);
                    setAnalysisError(err.message || "Failed to load AI analysis.");
                } finally {
                    setAnalysisLoading(false);
                }
            };
            getAnalysis();
        }
    }, [activeTab, symbol, analysis, analysisLoading, analysisError]);

    useEffect(() => {
        if (isOpen && symbol) {
            loadData();
        }
    }, [isOpen, symbol]);

    const loadData = async () => {
        setLoading(true);
        setError(null);
        try {
            const [fundRes, finRes, ratioRes, ivRes] = await Promise.all([
                fetchFundamentals(symbol),
                fetchFinancials(symbol),
                fetchRatios(symbol),
                fetchIntrinsicValue(symbol)
            ]);
            setFundamentals(fundRes);
            setFinancials(finRes);
            setRatios(ratioRes);
            setIntrinsicValue(ivRes);
        } catch (err: any) {
            console.error(err);
            setError(err.message || "Failed to load stock details");
        } finally {
            setLoading(false);
        }
    };

    if (!isOpen) return null;

    const formatCurrency = (val: number | null | undefined) => {
        if (val === undefined || val === null) return '-';
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: fundamentals?.currency || 'USD',
            notation: val > 1000000 ? 'compact' : 'standard',
        }).format(val);
    };

    const formatPercent = (val: number | undefined) => {
        if (val === undefined || val === null) return '-';
        return formatPercentShared(val);
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

    const renderOverview = () => {
        if (!fundamentals) return null;

        const getUpsidePercentage = (iv?: number) => {
            if (!iv || !intrinsicValue?.current_price) return null;
            return (iv / intrinsicValue.current_price) - 1;
        };

        const formatUpside = (upside: number | null) => {
            if (upside === null) return null;
            const prefix = upside > 0 ? '+' : '';
            return `${prefix}${formatPercentShared(upside)}`;
        };

        const getUpsideColor = (upside: number | null) => {
            if (upside === null) return "";
            return upside > 0 ? "text-emerald-500 font-bold" : "text-rose-500 font-bold";
        };

        const dcfUpside = getUpsidePercentage(intrinsicValue?.models?.dcf?.intrinsic_value);
        const grahamUpside = getUpsidePercentage(intrinsicValue?.models?.graham?.intrinsic_value);

        return (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
                    {intrinsicValue?.models?.dcf?.intrinsic_value && (
                        <StatCard
                            label="DCF Intrinsic Value"
                            value={formatCurrency(intrinsicValue.models.dcf.intrinsic_value)}
                            subValue={formatUpside(dcfUpside)}
                            subValueColor={getUpsideColor(dcfUpside)}
                            rangeMin={formatCurrency(intrinsicValue.models.dcf.mc?.bear)}
                            rangeMax={formatCurrency(intrinsicValue.models.dcf.mc?.bull)}
                            icon={TrendingUp}
                            color="text-emerald-400"
                        />
                    )}
                    {intrinsicValue?.models?.graham?.intrinsic_value && (
                        <StatCard
                            label="Graham Intrinsic Value"
                            value={formatCurrency(intrinsicValue.models.graham.intrinsic_value)}
                            subValue={formatUpside(grahamUpside)}
                            subValueColor={getUpsideColor(grahamUpside)}
                            rangeMin={formatCurrency(intrinsicValue.models.graham.mc?.bear)}
                            rangeMax={formatCurrency(intrinsicValue.models.graham.mc?.bull)}
                            icon={Scale}
                            color="text-amber-400"
                        />
                    )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <StatCard label="Market Cap" value={formatCurrency(fundamentals.marketCap)} icon={Globe} color="text-cyan-400" />
                    <StatCard label="P/E Ratio (TTM)" value={fundamentals.trailingPE?.toFixed(2)} icon={TrendingUp} color="text-emerald-400" />
                    <StatCard label="Dividend Yield" value={formatPercent(fundamentals.dividendYield)} icon={DollarSign} color="text-amber-400" />
                    <StatCard label="52W High" value={formatCurrency(fundamentals.fiftyTwoWeekHigh)} icon={TrendingUp} color="text-blue-400" />
                    <StatCard label="52W Low" value={formatCurrency(fundamentals.fiftyTwoWeekLow)} icon={TrendingUp} color="text-pink-400" className="rotate-180" />
                    {!fundamentals.etf_data && (
                        <StatCard label="Beta" value={fundamentals.beta?.toFixed(2)} icon={Activity} color="text-purple-400" />
                    )}
                    {(fundamentals.expenseRatio || fundamentals.annualReportExpenseRatio || fundamentals.netExpenseRatio) && (
                        <StatCard
                            label="Expense Ratio"
                            value={formatPercent((fundamentals.expenseRatio || fundamentals.annualReportExpenseRatio || fundamentals.netExpenseRatio) / 100)}
                            icon={Receipt}
                            color="text-orange-400"
                        />
                    )}
                </div>

                <div className="bg-muted rounded-2xl p-6 border border-border shadow-md">
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

                <div className="overflow-x-auto rounded-2xl border border-border bg-muted">
                    <table className="w-full text-sm text-left">
                        <thead className="bg-secondary/50 font-semibold border-b border-border">
                            <tr>
                                <th className="px-6 py-3 font-semibold text-foreground sticky left-0 bg-card"></th>
                                <th className="px-6 py-3 font-semibold text-center text-muted-foreground">Trend</th>
                                {currentStatement.columns.map(col => (
                                    <th key={col} className="px-6 py-3 font-semibold text-center text-muted-foreground tabular-nums">{new Date(col).getFullYear()}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-border/50">
                            {currentStatement.index.map((item, idx) => (
                                <tr key={item} className="hover:bg-accent/5 transition-colors">
                                    <td className="px-6 py-3 font-medium text-foreground sticky left-0 bg-card min-w-[200px] border-r border-border/50 shadow-[4px_0_8px_-4px_rgba(0,0,0,0.1)]">{item}</td>
                                    <td className="px-6 py-3 text-center min-w-[100px]">
                                        <Sparkline data={currentStatement.data[idx] as number[]} />
                                    </td>
                                    {currentStatement.data[idx].map((val, vIdx) => (
                                        <td key={vIdx} className="px-6 py-3 text-foreground text-right font-medium tabular-nums">
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
                        dataKey="Return on Equity (ROE) (%)"
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

    const renderAnalysis = () => {
        if (analysisLoading) {
            return (
                <div className="space-y-6 animate-in fade-in duration-500">
                    <Skeleton className="h-32 w-full rounded-2xl" />
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <Skeleton className="h-48 rounded-2xl" />
                        <Skeleton className="h-48 rounded-2xl" />
                        <Skeleton className="h-48 rounded-2xl" />
                        <Skeleton className="h-48 rounded-2xl" />
                    </div>
                </div>
            );
        }

        if (analysisError) {
            const isRateLimit = analysisError.includes('429') || analysisError.toLowerCase().includes('too many requests');
            let displayError = analysisError.includes('Failed to resolve')
                ? 'Network connection issue. Please check your internet or DNS settings.'
                : analysisError.length > 200 ? analysisError.substring(0, 200) + '...' : analysisError;

            if (isRateLimit) {
                displayError = "Gemini API rate limit reached. The AI model is currently busy. Please wait a minute and try again.";
            }

            return (
                <div className="flex flex-col items-center justify-center py-20 text-center animate-in fade-in duration-500">
                    <div className="w-16 h-16 bg-destructive/10 rounded-full flex items-center justify-center mb-4">
                        <Info className="w-8 h-8 text-destructive" />
                    </div>
                    <h3 className="text-xl font-bold mb-2">{isRateLimit ? 'Rate Limit Reached' : 'Analysis Failed'}</h3>
                    <p className="text-muted-foreground max-w-md">{displayError}</p>
                    <button
                        onClick={() => {
                            setAnalysis(null);
                            setAnalysisError(null);
                        }}
                        className="mt-6 px-6 py-2 bg-secondary hover:bg-muted rounded-full transition-colors font-medium"
                    >
                        Try Again
                    </button>
                </div>
            );
        }

        if (!analysis) return (
            <div className="flex flex-col items-center justify-center py-20 text-center">
                <Sparkles className="w-12 h-12 text-purple-500/20 mb-4" />
                <p className="text-muted-foreground">No analysis data available.</p>
            </div>
        );

        const topics = [
            { id: 'moat', title: 'Moat & Edge', icon: Shield, color: 'text-blue-500', bg: 'bg-blue-500/10', content: analysis?.analysis?.moat, score: analysis?.scorecard?.moat },
            { id: 'strength', title: 'Financial Strength', icon: Zap, color: 'text-amber-500', bg: 'bg-amber-500/10', content: analysis?.analysis?.financial_strength, score: analysis?.scorecard?.financial_strength },
            { id: 'predictability', title: 'Predictability', icon: Target, color: 'text-emerald-500', bg: 'bg-emerald-500/10', content: analysis?.analysis?.predictability, score: analysis?.scorecard?.predictability },
            { id: 'growth', title: 'Growth Perspective', icon: LucideActivity, color: 'text-purple-500', bg: 'bg-purple-500/10', content: analysis?.analysis?.growth_perspective, score: analysis?.scorecard?.growth }
        ];

        return (
            <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
                {/* Scorecard Header */}
                <div className={cn(
                    "p-6 rounded-[2rem] border overflow-hidden relative",
                    isDarkMode ? "bg-slate-900/50 border-slate-800" : "bg-white border-slate-200 shadow-sm"
                )}>
                    <div className="flex items-center gap-4 relative z-10">
                        <div className="w-12 h-12 rounded-2xl bg-purple-500 flex items-center justify-center shadow-lg shadow-purple-500/20 shrink-0">
                            <Sparkles className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <h3 className="text-xl font-bold">AI Fundamental Review</h3>
                            <p className="text-sm text-muted-foreground leading-relaxed mt-1">{analysis.summary}</p>
                        </div>
                    </div>
                    {/* Decorative background element */}
                    <div className="absolute top-0 right-0 w-64 h-64 bg-purple-500/5 rounded-full blur-3xl -mr-32 -mt-32" />
                </div>

                {/* Score Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {topics.map(t => (
                        <div key={t.id} className={cn(
                            "p-4 rounded-3xl border flex flex-col items-center justify-center gap-2",
                            isDarkMode ? "bg-slate-900/30 border-slate-800" : "bg-white border-slate-200 shadow-sm"
                        )}>
                            <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">{t.id}</span>
                            <div className={cn("text-3xl font-black", t.color)}>{t.score}<span className="text-sm opacity-50 font-normal">/10</span></div>
                        </div>
                    ))}
                </div>

                {/* Narrative Details */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {topics.map(t => (
                        <div key={t.id} className={cn(
                            "p-6 rounded-[2rem] border transition-all hover:shadow-xl group",
                            isDarkMode ? "bg-slate-900/30 border-slate-800 hover:bg-slate-900/50" : "bg-white border-slate-200 hover:border-slate-300 shadow-sm"
                        )}>
                            <div className="flex items-center gap-3 mb-4">
                                <div className={cn("p-2.5 rounded-xl", t.bg)}>
                                    <t.icon className={cn("w-5 h-5", t.color)} />
                                </div>
                                <h4 className="font-bold text-lg">{t.title}</h4>
                            </div>
                            <p className="text-sm leading-relaxed text-muted-foreground group-hover:text-foreground transition-colors">
                                {t.content}
                            </p>
                        </div>
                    ))}
                </div>

                <div className="text-center pb-8">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-[0.2em] font-medium opacity-50">
                        Generated by Google Gemini 3 Flash
                    </p>
                </div>
            </div>
        );
    };

    const renderHoldings = () => {
        if (!fundamentals?.etf_data) return null;
        const { top_holdings, sector_weightings } = fundamentals.etf_data;

        // Prepare sector data for chart
        const sectorData = Object.entries(sector_weightings || {}).map(([name, value]) => ({
            name: name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
            value: value * 100
        })).sort((a, b) => b.value - a.value);

        const COLORS = ['#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899', '#f43f5e', '#f59e0b', '#10b981', '#6366f1'];

        return (
            <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* Top Holdings Table */}
                    <div className="bg-muted border border-border rounded-2xl p-6">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <List className="w-5 h-5 text-cyan-500" />
                            Top Holdings
                        </h3>
                        <div className="overflow-hidden rounded-xl border border-border/50">
                            <table className="w-full text-sm">
                                <thead className="bg-secondary/50">
                                    <tr>
                                        <th className="px-4 py-2 text-left font-medium text-muted-foreground">Symbol</th>
                                        <th className="px-4 py-2 text-right font-medium text-muted-foreground">% Assets</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-border/30">
                                    {top_holdings?.map((h, i) => (
                                        <tr key={i} className="hover:bg-accent/5">
                                            <td className="px-4 py-2 font-medium">{h.symbol}</td>
                                            <td className="px-4 py-2 text-right tabular-nums">{(h.percent * 100).toFixed(2)}%</td>
                                        </tr>
                                    ))}
                                    {(!top_holdings || top_holdings.length === 0) && (
                                        <tr>
                                            <td colSpan={2} className="px-4 py-8 text-center text-muted-foreground">No holdings data available</td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* Sector Allocation Chart */}
                    <div className="bg-muted border border-border rounded-2xl p-6">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <PieChartIcon className="w-5 h-5 text-cyan-500" />
                            Sector Allocation
                        </h3>
                        {sectorData.length > 0 ? (
                            <div className="h-[300px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie
                                            data={sectorData}
                                            cx="50%"
                                            cy="50%"
                                            innerRadius={60}
                                            outerRadius={80}
                                            paddingAngle={2}
                                            dataKey="value"
                                        >
                                            {sectorData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke="rgba(0,0,0,0.1)" />
                                            ))}
                                        </Pie>
                                        <Tooltip
                                            formatter={(value: any) => `${Number(value).toFixed(2)}%`}
                                            contentStyle={{ borderRadius: '12px', border: '1px solid var(--border)', backgroundColor: 'var(--menu-solid)' }}
                                            labelStyle={{ color: 'var(--foreground)' }}
                                        />
                                        <Legend
                                            layout="vertical"
                                            verticalAlign="middle"
                                            align="right"
                                            formatter={(value, entry: any) => <span className="text-xs text-muted-foreground ml-1">{value}</span>}
                                        />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>
                        ) : (
                            <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                                No sector data available
                            </div>
                        )}
                    </div>
                </div>
            </div>
        )
    };

    const VALUATION_INFO = {
        discount_rate: {
            description: "The rate used to discount future cash flows to their present value. High rate = lower valuation.",
            default: "Calculated WACC (~10%)"
        },
        growth_rate: {
            description: "Expected annual growth of cash flows during the projection years.",
            default: "Historical CAGR"
        },
        terminal_growth: {
            description: "Long-term growth rate after the projection period (stable stage).",
            default: "2.0%"
        },
        projection_years: {
            description: "Number of years to forecast explicit free cash flows.",
            default: "5 Years"
        },
        base_fcf: {
            description: "The starting free cash flow value for DCF projections.",
            default: "Latest TTM FCF"
        },
        eps: {
            description: "Earnings per share used as the base for the Graham Formula.",
            default: "TTM EPS"
        },
        graham_growth: {
            description: "Expected annual growth (g) used in Graham's Formula.",
            default: "Historical CAGR"
        },
        bond_yield: {
            description: "Current yield on high-quality bonds (proxy for risk-free rate).",
            default: "10Y Treasury (~4.5%)"
        }
    };

    const renderValuation = () => {
        if (!intrinsicValue) return null;
        const { models, average_intrinsic_value, margin_of_safety_pct, current_price } = intrinsicValue;

        return (
            <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                {/* Summary Header */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="bg-muted border border-border p-6 rounded-2xl flex flex-col items-center justify-center text-center">
                        <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider mb-2">Average Intrinsic Value</p>
                        <p className="text-3xl font-bold text-cyan-500">{formatCurrency(average_intrinsic_value)}</p>
                        {intrinsicValue.range && (
                            <p className="text-xs text-muted-foreground mt-2 font-medium">
                                Range: {formatCurrency(intrinsicValue.range.bear)} - {formatCurrency(intrinsicValue.range.bull)}
                            </p>
                        )}
                    </div>
                    <div className="bg-muted border border-border p-6 rounded-2xl flex flex-col items-center justify-center text-center">
                        <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider mb-2">Current Price</p>
                        <p className="text-3xl font-bold">{formatCurrency(current_price)}</p>
                    </div>
                    <div className={cn(
                        "border p-6 rounded-2xl flex flex-col items-center justify-center text-center transition-all",
                        (margin_of_safety_pct || 0) > 0
                            ? "bg-emerald-500/10 border-emerald-500/30 dark:bg-emerald-500/5 dark:border-emerald-500/20"
                            : "bg-rose-500/10 border-rose-500/30 dark:bg-rose-500/5 dark:border-rose-500/20"
                    )}>
                        <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider mb-2">Margin of Safety</p>
                        <p className={cn(
                            "text-3xl font-bold tracking-tight",
                            (margin_of_safety_pct || 0) > 0 ? "text-emerald-500" : "text-rose-500"
                        )}>
                            {margin_of_safety_pct?.toFixed(2)}%
                        </p>
                    </div>
                </div>

                {/* Models Detail */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* DCF Model */}
                    <div className="bg-muted border border-border rounded-2xl p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-lg font-semibold flex items-center gap-2">
                                <TrendingUp className="w-5 h-5 text-emerald-500" />
                                DCF Model
                            </h3>
                            {models.dcf.intrinsic_value && (
                                <Badge className="bg-emerald-500/20 text-emerald-500 border-none">
                                    {formatCurrency(models.dcf.intrinsic_value)}
                                </Badge>
                            )}
                        </div>
                        {models.dcf.error ? (
                            <p className="text-sm text-destructive bg-destructive/5 p-4 rounded-xl border border-destructive/20">{models.dcf.error}</p>
                        ) : (
                            <div className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <ParamItem
                                        label="Discount Rate (WACC)"
                                        value={formatPercentShared(models.dcf.parameters.discount_rate)}
                                        info={VALUATION_INFO.discount_rate}
                                    />
                                    <ParamItem
                                        label="Growth Rate"
                                        value={formatPercentShared(models.dcf.parameters.growth_rate)}
                                        info={VALUATION_INFO.growth_rate}
                                    />
                                    <ParamItem
                                        label="Terminal Growth"
                                        value={formatPercentShared(models.dcf.parameters.terminal_growth_rate)}
                                        info={VALUATION_INFO.terminal_growth}
                                    />
                                    <ParamItem
                                        label="Projection Years"
                                        value={models.dcf.parameters.projection_years}
                                        info={VALUATION_INFO.projection_years}
                                    />
                                </div>
                                <div className="pt-4 border-t border-border/50">
                                    <ParamItem
                                        label="Base Free Cash Flow"
                                        value={formatCurrency(models.dcf.parameters.base_fcf)}
                                        info={VALUATION_INFO.base_fcf}
                                    />
                                </div>
                                {models.dcf.mc && (
                                    <div className="pt-4 border-t border-border/50">
                                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-bold mb-3">Probabilistic Scenarios (Monte Carlo)</p>
                                        <div className="grid grid-cols-3 gap-2">
                                            <div
                                                className="bg-rose-500/5 border border-rose-500/10 p-2 rounded-lg text-center cursor-pointer hover:bg-rose-500/10 transition-colors group/mc"
                                                onClick={() => setViewingDistribution('dcf')}
                                            >
                                                <p className="text-[10px] text-rose-500 font-bold uppercase mb-1">Bear (10th)</p>
                                                <p className="text-sm font-bold">{formatCurrency(models.dcf.mc.bear)}</p>
                                            </div>
                                            <div
                                                className="bg-cyan-500/5 border border-cyan-500/10 p-2 rounded-lg text-center cursor-pointer hover:bg-cyan-500/10 transition-colors group/mc"
                                                onClick={() => setViewingDistribution('dcf')}
                                            >
                                                <p className="text-[10px] text-cyan-500 font-bold uppercase mb-1">Median (50th)</p>
                                                <p className="text-sm font-bold">{formatCurrency(models.dcf.mc.base)}</p>
                                            </div>
                                            <div
                                                className="bg-emerald-500/5 border border-emerald-500/10 p-2 rounded-lg text-center cursor-pointer hover:bg-emerald-500/10 transition-colors group/mc"
                                                onClick={() => setViewingDistribution('dcf')}
                                            >
                                                <p className="text-[10px] text-emerald-500 font-bold uppercase mb-1">Bull (90th)</p>
                                                <p className="text-sm font-bold">{formatCurrency(models.dcf.mc.bull)}</p>
                                            </div>
                                        </div>
                                        <p className="text-[9px] text-muted-foreground mt-2 text-center opacity-50 group-hover/mc:opacity-100 transition-opacity">Click to view distribution</p>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Graham Model */}
                    <div className="bg-muted border border-border rounded-2xl p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-lg font-semibold flex items-center gap-2">
                                <Scale className="w-5 h-5 text-amber-500" />
                                Graham's Formula
                            </h3>
                            {models.graham.intrinsic_value && (
                                <Badge className="bg-amber-500/20 text-amber-500 border-none">
                                    {formatCurrency(models.graham.intrinsic_value)}
                                </Badge>
                            )}
                        </div>
                        {models.graham.error ? (
                            <p className="text-sm text-destructive bg-destructive/5 p-4 rounded-xl border border-destructive/20">{models.graham.error}</p>
                        ) : (
                            <div className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <ParamItem
                                        label="Trailing EPS"
                                        value={models.graham.parameters.eps?.toFixed(2)}
                                        info={VALUATION_INFO.eps}
                                    />
                                    <ParamItem
                                        label="Growth Rate (g)"
                                        value={`${models.graham.parameters.growth_rate_pct?.toFixed(2)}%`}
                                        info={VALUATION_INFO.graham_growth}
                                    />
                                    <ParamItem
                                        label="Bond Yield (Y)"
                                        value={`${models.graham.parameters.bond_yield_proxy?.toFixed(2)}%`}
                                        info={VALUATION_INFO.bond_yield}
                                    />
                                </div>
                                <div className="mt-4 p-4 bg-secondary/5 rounded-xl flex flex-col items-center select-none border border-border/20 overflow-visible">
                                    <div className="flex items-center gap-2">
                                        <div className="flex items-baseline gap-1">
                                            <div className="group relative">
                                                <span className="text-lg font-bold text-foreground cursor-help decoration-dotted decoration-border/50 underline-offset-4 hover:text-cyan-500 transition-colors">V</span>
                                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-40 p-2 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[9px] rounded-lg shadow-2xl border border-slate-200 dark:border-white/10 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-tight text-center font-medium">
                                                    Intrinsic Value
                                                </div>
                                            </div>
                                            <span className="text-lg font-light opacity-30">=</span>
                                        </div>
                                        <div className="flex items-center gap-1.5 px-4">
                                            <div className="group relative">
                                                <span className="text-xs font-bold text-foreground cursor-help hover:text-cyan-500 transition-colors">EPS</span>
                                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-40 p-2 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[9px] rounded-lg shadow-2xl border border-slate-200 dark:border-white/10 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-tight text-center font-medium">
                                                    Trailing 12-Month Earnings Per Share
                                                </div>
                                            </div>
                                            <span className="text-[9px] opacity-40">×</span>
                                            <div className="group relative">
                                                <span className="px-1.5 py-0.5 bg-secondary/30 rounded-md border border-border/30 text-[10px] font-bold text-foreground cursor-help hover:border-cyan-500/50 transition-colors">
                                                    8.5 + 2G
                                                </span>
                                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2.5 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[9px] rounded-lg shadow-2xl border border-slate-200 dark:border-white/10 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-relaxed text-center font-medium">
                                                    <div className="mb-0.5"><span className="font-bold">8.5</span>: Base P/E for zero growth</div>
                                                    <div><span className="font-bold">G</span>: Expected long-term growth rate</div>
                                                </div>
                                            </div>
                                            <span className="text-[9px] opacity-40">×</span>
                                            <div className="group relative">
                                                <span className="text-xs font-bold text-foreground cursor-help hover:text-cyan-500 transition-colors">4.4</span>
                                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-40 p-2 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[9px] rounded-lg shadow-2xl border border-slate-200 dark:border-white/10 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-tight text-center font-medium">
                                                    Average yield of high-grade corporate bonds in 1962
                                                </div>
                                            </div>
                                            <span className="text-[10px] opacity-40 mx-0.5">/</span>
                                            <div className="group relative">
                                                <span className="text-xs font-bold text-foreground cursor-help hover:text-cyan-500 transition-colors">Y</span>
                                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-44 p-2 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[9px] rounded-lg shadow-2xl border border-slate-200 dark:border-white/10 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-tight text-center font-medium">
                                                    <div className="font-bold mb-1">Y = {models.graham.parameters?.bond_yield_proxy || '4.5'}%</div>
                                                    Current yield on AAA corporate bonds
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                {models.graham.mc && (
                                    <div className="mt-8 pt-4 border-t border-border/50">
                                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-bold mb-3">Probabilistic Scenarios (Monte Carlo)</p>
                                        <div className="grid grid-cols-3 gap-2">
                                            <div
                                                className="bg-rose-500/5 border border-rose-500/10 p-2 rounded-lg text-center cursor-pointer hover:bg-rose-500/10 transition-colors group/mc"
                                                onClick={() => setViewingDistribution('graham')}
                                            >
                                                <p className="text-[10px] text-rose-500 font-bold uppercase mb-1">Bear (10th)</p>
                                                <p className="text-sm font-bold">{formatCurrency(models.graham.mc.bear)}</p>
                                            </div>
                                            <div
                                                className="bg-amber-500/5 border border-amber-500/10 p-2 rounded-lg text-center cursor-pointer hover:bg-amber-500/10 transition-colors group/mc"
                                                onClick={() => setViewingDistribution('graham')}
                                            >
                                                <p className="text-[10px] text-amber-500 font-bold uppercase mb-1">Median (50th)</p>
                                                <p className="text-sm font-bold">{formatCurrency(models.graham.mc.base)}</p>
                                            </div>
                                            <div
                                                className="bg-emerald-500/5 border border-emerald-500/10 p-2 rounded-lg text-center cursor-pointer hover:bg-emerald-500/10 transition-colors group/mc"
                                                onClick={() => setViewingDistribution('graham')}
                                            >
                                                <p className="text-[10px] text-emerald-500 font-bold uppercase mb-1">Bull (90th)</p>
                                                <p className="text-sm font-bold">{formatCurrency(models.graham.mc.bull)}</p>
                                            </div>
                                        </div>
                                        <p className="text-[9px] text-muted-foreground mt-2 text-center opacity-50 group-hover/mc:opacity-100 transition-opacity">Click to view distribution</p>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>

                <div className="bg-secondary/20 rounded-2xl p-6 border border-border/50 italic text-sm text-muted-foreground text-center">
                    Note: Intrinsic value calculations are estimates based on various assumptions.
                    Actual stock performance may vary significantly.
                </div>
            </div>
        );
    };

    const ParamItem = ({ label, value, info }: { label: string, value: any, info?: { description: string, default: string } }) => (
        <div>
            <div className="flex items-center gap-1 mb-1">
                <p className="text-[10px] text-muted-foreground uppercase tracking-widest font-bold">{label}</p>
                {info && (
                    <div className="group relative">
                        <HelpCircle className="w-2.5 h-2.5 text-muted-foreground/50 hover:text-cyan-500 cursor-help" />
                        <div className="absolute bottom-full left-0 mb-2 w-48 p-3 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[10px] rounded-lg shadow-2xl border border-slate-200 dark:border-white/10 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100]">
                            {info.description}
                            <div className="mt-1 pt-1 border-t border-slate-100 dark:border-white/10 font-bold text-cyan-600 dark:text-cyan-400">Default: {info.default}</div>
                        </div>
                    </div>
                )}
            </div>
            <p className="text-sm font-semibold">{value ?? '-'}</p>
        </div>
    );

    return createPortal(
        <div className="fixed inset-0 z-[100] flex flex-col justify-end sm:justify-center items-center p-0 sm:p-4 isolate">
            <div className="absolute inset-0 bg-black/60" onClick={onClose} />

            <div
                style={{ backgroundColor: 'var(--menu-solid)' }}
                className="relative w-full max-w-5xl h-[94vh] sm:h-auto sm:max-h-[90vh] rounded-t-[2.5rem] sm:rounded-[2rem] flex flex-col shadow-2xl overflow-hidden animate-in slide-in-from-bottom sm:zoom-in-95 duration-300"
            >

                {/* Mobile Drag Handle */}
                <div className="sm:hidden w-full flex justify-center pt-3 pb-1 flex-shrink-0">
                    <div className="w-12 h-1.5 bg-border/50 rounded-full" />
                </div>

                {/* Sticky Header & Tabs Container */}
                <div className="sticky top-0 z-50 bg-card border-b border-border flex-shrink-0 shadow-sm">
                    {/* Header */}
                    <div className="p-5 sm:p-8 pb-3 sm:pb-4 flex justify-between items-start relative">
                        <div className="hidden sm:block absolute top-0 right-0 w-64 h-64 bg-cyan-500/10 rounded-full blur-[100px] -mr-32 -mt-32" />

                        <div className="flex items-center gap-4 sm:gap-6 relative z-10 text-foreground flex-1">
                            <div className="w-10 h-10 sm:w-16 sm:h-16 rounded-xl sm:rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-lg sm:text-3xl font-bold shadow-lg shadow-cyan-500/20 text-white overflow-hidden flex-shrink-0">
                                <StockIcon symbol={symbol} size="100%" className="w-full h-full p-2 bg-white" domain={domain} />
                            </div>
                            <div className="flex-1 min-w-0 pr-4">
                                <div className="flex items-center justify-between gap-2 mb-0.5 sm:mb-1">
                                    <div className="flex items-center gap-2 sm:gap-3 truncate">
                                        <h2 className="text-lg sm:text-3xl font-bold tracking-tight truncate max-w-[140px] sm:max-w-none">{fundamentals?.shortName || symbol}</h2>
                                        <Badge className="bg-secondary text-secondary-foreground border-none font-mono text-[9px] sm:text-xs">{symbol}</Badge>
                                    </div>
                                    {fundamentals?.regularMarketPrice && (
                                        <div className="flex items-baseline gap-1 text-cyan-600 dark:text-cyan-400">
                                            <span className="text-xl sm:text-4xl font-bold tracking-tight">
                                                {formatCurrency(fundamentals.regularMarketPrice)}
                                            </span>
                                        </div>
                                    )}
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
                            active={activeTab === 'analysis'}
                            onClick={() => setActiveTab('analysis')}
                            icon={Sparkles}
                            label="Analysis"
                        />
                        {!fundamentals?.etf_data && (
                            <>
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
                                <TabButton
                                    active={activeTab === 'valuation'}
                                    onClick={() => setActiveTab('valuation')}
                                    icon={DollarSign}
                                    label="Valuation"
                                />
                            </>
                        )}
                        {fundamentals?.etf_data && (
                            <TabButton
                                active={activeTab === 'holdings'}
                                onClick={() => setActiveTab('holdings')}
                                icon={PieChartIcon}
                                label="Holdings"
                            />
                        )}
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
                            {activeTab === 'analysis' && renderAnalysis()}
                            {activeTab === 'overview' && renderOverview()}
                            {activeTab === 'financials' && renderFinancials()}
                            {activeTab === 'ratios' && renderRatios()}
                            {activeTab === 'valuation' && renderValuation()}
                            {activeTab === 'holdings' && renderHoldings()}
                        </>
                    )}
                </div>

                {/* Distribution Modal */}
                {viewingDistribution && intrinsicValue && (
                    <div className={cn(
                        "fixed inset-0 z-[110] flex items-center justify-center p-4 backdrop-blur-sm animate-in fade-in duration-300",
                        isDarkMode ? "bg-black/60" : "bg-slate-500/20"
                    )}>
                        <div className={cn(
                            "w-full max-w-2xl rounded-3xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-300 border",
                            isDarkMode ? "bg-slate-900 border-slate-800 text-white" : "bg-white border-slate-200 text-slate-900"
                        )}>
                            <div className={cn(
                                "p-6 border-b flex items-center justify-between",
                                isDarkMode ? "bg-muted/30 border-slate-800" : "bg-slate-50/50 border-slate-100"
                            )}>
                                <div>
                                    <h3 className="text-xl font-bold flex items-center gap-2 text-inherit">
                                        <BarChart3 className="w-5 h-5 text-cyan-500" />
                                        {viewingDistribution === 'dcf' ? 'DCF' : 'Graham'} Probabilistic Distribution
                                    </h3>
                                    <p className={isDarkMode ? "text-slate-400 text-sm" : "text-slate-500 text-sm"}>Monte Carlo Simulation (10,000 iterations)</p>
                                </div>
                                <button
                                    onClick={() => setViewingDistribution(null)}
                                    className={cn(
                                        "p-2 rounded-full transition-colors",
                                        isDarkMode ? "hover:bg-slate-800" : "hover:bg-slate-100"
                                    )}
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>
                            <div className="p-6">
                                <div className="h-[300px] w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        {(() => {
                                            const histogram = intrinsicValue.models[viewingDistribution].mc?.histogram || [];
                                            const mc = intrinsicValue.models[viewingDistribution].mc;

                                            if (histogram.length === 0 || !mc) return null;

                                            const minPrice = histogram[0].price;
                                            const maxPrice = histogram[histogram.length - 1].price;
                                            const currentPrice = intrinsicValue.current_price || fundamentals?.regularMarketPrice || 0;

                                            // Ensure the domain includes the current price
                                            const domainMin = Math.min(minPrice, currentPrice > 0 ? currentPrice * 0.95 : minPrice);
                                            const domainMax = Math.max(maxPrice, currentPrice > 0 ? currentPrice * 1.05 : maxPrice);

                                            const range = maxPrice - minPrice;

                                            // Calculate percentage offsets for the gradient stops (relative to the histogram data range, not the visual domain)
                                            // Actually, for the gradient to match the bell curve shape, we should stick to using the data points.
                                            // But for the ReferenceLine to show, the visual XAxis needs the full domain.

                                            const bearOffset = Math.max(0, Math.min(100, ((mc.bear - minPrice) / range) * 100));
                                            const bullOffset = Math.max(0, Math.min(100, ((mc.bull - minPrice) / range) * 100));

                                            return (
                                                <AreaChart
                                                    data={histogram}
                                                    margin={{ top: 35, right: 20, left: 20, bottom: 0 }}
                                                >
                                                    <defs>
                                                        <linearGradient id="colorBell" x1="0" y1="0" x2="1" y2="0" gradientUnits="userSpaceOnUse">
                                                            {/* We use userSpaceOnUse to map exactly to the X-axis values, 
                                                                but Recharts often works better with objectBoundingBox for horizontal gradients.
                                                                Actually, simple 0-1 offsets work for x1=0, x2=1. 
                                                            */}
                                                        </linearGradient>

                                                        {/* Horizontal Gradient for the categories */}
                                                        <linearGradient id="colorBellFill" x1="0" y1="0" x2="1" y2="0">
                                                            <stop offset="0%" stopColor="#f43f5e" stopOpacity={0.4} />
                                                            <stop offset={`${bearOffset}%`} stopColor="#f43f5e" stopOpacity={0.4} />

                                                            <stop offset={`${bearOffset}%`} stopColor="#06b6d4" stopOpacity={0.4} />
                                                            <stop offset={`${bullOffset}%`} stopColor="#06b6d4" stopOpacity={0.4} />

                                                            <stop offset={`${bullOffset}%`} stopColor="#10b981" stopOpacity={0.4} />
                                                            <stop offset="100%" stopColor="#10b981" stopOpacity={0.4} />
                                                        </linearGradient>

                                                        {/* Vertical Gradient for the "fade out" effect */}
                                                        <linearGradient id="colorBellFade" x1="0" y1="0" x2="0" y2="1">
                                                            <stop offset="5%" stopColor="white" stopOpacity={1} />
                                                            <stop offset="95%" stopColor="white" stopOpacity={0} />
                                                        </linearGradient>

                                                        {/* Masking the categories with the fade gradient */}
                                                        <mask id="bellMask">
                                                            <rect x="0" y="0" width="100%" height="100%" fill="url(#colorBellFade)" />
                                                        </mask>
                                                    </defs>
                                                    <CartesianGrid
                                                        strokeDasharray="3 3"
                                                        vertical={false}
                                                        stroke={isDarkMode ? "#334155" : "#e2e8f0"}
                                                        opacity={isDarkMode ? 0.3 : 0.8}
                                                    />
                                                    <XAxis
                                                        dataKey="price"
                                                        type="number"
                                                        domain={[domainMin, domainMax]}
                                                        tickFormatter={(val) => formatCurrency(val)}
                                                        fontSize={10}
                                                        tickLine={false}
                                                        axisLine={false}
                                                        minTickGap={30}
                                                        stroke={isDarkMode ? "#94a3b8" : "#64748b"}
                                                    />
                                                    <YAxis hide />
                                                    <Tooltip
                                                        content={({ active, payload }) => {
                                                            if (active && payload && payload.length) {
                                                                const count = payload[0].value;
                                                                const probability = (count / 10000) * 100;
                                                                return (
                                                                    <div className={cn(
                                                                        "p-3 rounded-xl shadow-2xl outline-none border scale-105 transition-transform",
                                                                        isDarkMode
                                                                            ? "bg-slate-900 border-slate-700"
                                                                            : "bg-white border-slate-200"
                                                                    )}>
                                                                        <p className={cn(
                                                                            "text-[10px] uppercase font-bold mb-1 tracking-wider",
                                                                            isDarkMode ? "text-slate-500" : "text-slate-400"
                                                                        )}>Estimated Value</p>
                                                                        <p className={cn(
                                                                            "text-lg font-black",
                                                                            isDarkMode ? "text-white" : "text-slate-900"
                                                                        )}>{formatCurrency(payload[0].payload.price)}</p>

                                                                        <div className="flex flex-col gap-1 mt-3 pt-2 border-t border-border/50">
                                                                            <div className="flex items-center justify-between gap-4">
                                                                                <div className="flex items-center gap-1.5">
                                                                                    <div className="w-1.5 h-1.5 rounded-full bg-cyan-500" />
                                                                                    <span className={cn("text-[10px] font-bold uppercase", isDarkMode ? "text-slate-400" : "text-slate-500")}>Probability</span>
                                                                                </div>
                                                                                <span className="text-[10px] font-black text-cyan-500">{probability.toFixed(2)}%</span>
                                                                            </div>
                                                                            <div className="flex items-center justify-between gap-4">
                                                                                <div className="flex items-center gap-1.5">
                                                                                    <div className="w-1.5 h-1.5 rounded-full bg-slate-400" />
                                                                                    <span className={cn("text-[10px] font-bold uppercase", isDarkMode ? "text-slate-400" : "text-slate-500")}>Frequency</span>
                                                                                </div>
                                                                                <span className={cn("text-[10px] font-black", isDarkMode ? "text-slate-300" : "text-slate-700")}>{count.toLocaleString()} Iterations</span>
                                                                            </div>
                                                                        </div>
                                                                    </div>
                                                                );
                                                            }
                                                            return null;
                                                        }}
                                                    />
                                                    <Area
                                                        type="basis"
                                                        dataKey="count"
                                                        stroke="#06b6d4"
                                                        strokeWidth={4}
                                                        fill="url(#colorBellFill)"
                                                        mask="url(#bellMask)"
                                                        animationDuration={1500}
                                                    />
                                                    {/* Reference Lines for Bear, Base, Bull */}
                                                    {intrinsicValue.models[viewingDistribution].mc && (
                                                        <>
                                                            <ReferenceLine
                                                                x={intrinsicValue.models[viewingDistribution].mc!.bear}
                                                                stroke="#f43f5e"
                                                                strokeDasharray="4 4"
                                                                strokeWidth={2}
                                                                label={{ value: 'BEAR', position: 'top', fill: '#f43f5e', fontSize: 9, fontWeight: '900' }}
                                                            />
                                                            <ReferenceLine
                                                                x={intrinsicValue.models[viewingDistribution].mc!.base}
                                                                stroke="#06b6d4"
                                                                strokeDasharray="4 4"
                                                                strokeWidth={2}
                                                                label={{ value: 'MEDIAN', position: 'top', fill: '#06b6d4', fontSize: 9, fontWeight: '900' }}
                                                            />
                                                            <ReferenceLine
                                                                x={intrinsicValue.models[viewingDistribution].mc!.bull}
                                                                stroke="#10b981"
                                                                strokeDasharray="4 4"
                                                                strokeWidth={2}
                                                                label={{ value: 'BULL', position: 'top', fill: '#10b981', fontSize: 9, fontWeight: '900' }}
                                                            />
                                                        </>
                                                    )}
                                                    {/* Current Price Reference */}
                                                    {(intrinsicValue.current_price || fundamentals?.regularMarketPrice) && (
                                                        <ReferenceLine
                                                            x={intrinsicValue.current_price || fundamentals?.regularMarketPrice}
                                                            stroke={isDarkMode ? "#cbd5e1" : "#475569"}
                                                            strokeWidth={2}
                                                            strokeDasharray="3 3"
                                                            label={{
                                                                value: 'CURRENT',
                                                                position: 'top',
                                                                fill: isDarkMode ? "#cbd5e1" : "#475569",
                                                                fontSize: 9,
                                                                fontWeight: '900',
                                                                dy: -12
                                                            }}
                                                        />
                                                    )}
                                                </AreaChart>
                                            );
                                        })()}
                                    </ResponsiveContainer>
                                </div>
                                <div className={cn(
                                    "mt-6 flex items-center justify-between text-[10px] font-bold uppercase tracking-widest p-4 rounded-2xl",
                                    isDarkMode ? "bg-slate-800/50 text-slate-400" : "bg-slate-50 text-slate-500"
                                )}>
                                    <div className="flex items-center gap-2">
                                        <div className="w-2.5 h-2.5 rounded-full bg-rose-500 shadow-sm shadow-rose-500/20" /> <span className="hidden sm:inline">Bear: </span><span className={isDarkMode ? "text-white" : "text-slate-900"}>{formatCurrency(intrinsicValue.models[viewingDistribution].mc?.bear)}</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className="w-2.5 h-2.5 rounded-full bg-cyan-500 shadow-sm shadow-cyan-500/20" /> <span className="hidden sm:inline">Median: </span><span className={isDarkMode ? "text-white" : "text-slate-900"}>{formatCurrency(intrinsicValue.models[viewingDistribution].mc?.base)}</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/20" /> <span className="hidden sm:inline">Bull: </span><span className={isDarkMode ? "text-white" : "text-slate-900"}>{formatCurrency(intrinsicValue.models[viewingDistribution].mc?.bull)}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

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

function StatCard({ label, value, icon: Icon, color, className, rotate, subValue, subValueColor, rangeMin, rangeMax }: any) {
    return (
        <div className="bg-muted border border-border p-5 rounded-2xl flex items-center gap-4 transition-all hover:bg-muted/50 hover:border-accent group">
            <div className={cn("p-3 rounded-xl bg-card border border-border", color, rotate)}>
                <Icon className="w-5 h-5" />
            </div>
            <div className="flex-1 overflow-hidden">
                <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider mb-0.5 truncate">{label}</p>
                <div className="flex items-baseline gap-2">
                    <p className="text-lg font-bold text-foreground tracking-tight whitespace-nowrap">{value}</p>
                    {subValue && (
                        <span className={cn("text-xs whitespace-nowrap", subValueColor)}>
                            {subValue}
                        </span>
                    )}
                </div>
                {(rangeMin && rangeMax) && (
                    <p className="text-[10px] text-muted-foreground mt-0.5 font-medium">
                        Range: {rangeMin} - {rangeMax}
                    </p>
                )}
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
        <div className="bg-muted border border-border p-6 rounded-2xl">
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
                        ```
                        <Tooltip
                            content={({ active, payload, label }) => {
                                if (active && payload && payload.length) {
                                    return (
                                        <div
                                            style={{
                                                backgroundColor: 'var(--menu-solid)',
                                                borderRadius: '12px',
                                                border: '1px solid var(--border)',
                                                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                                            }}
                                            className="border border-border p-3 rounded-xl shadow-xl text-xs"
                                        >
                                            <p className="font-medium text-foreground mb-1">{label}</p>
                                            <div className="flex items-center gap-2">
                                                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                                                <span className="text-muted-foreground">{title}:</span>
                                                <span className="font-bold text-foreground">
                                                    {Number(payload[0].value).toFixed(2)}{suffix}
                                                </span>
                                            </div>
                                        </div>
                                    );
                                }
                                return null;
                            }}
                            cursor={{ stroke: 'var(--border)', strokeWidth: 1, strokeDasharray: '3 3' }}
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

function Sparkline({ data }: { data: number[] }) {
    const id = useId();
    if (!data || data.length < 2) return null;

    // Filter out null/undefined and reverse to chronological order (oldest to newest)
    const values = [...data].filter(v => v !== null && v !== undefined).reverse();
    if (values.length < 2) return null;

    const baseline = values[0];
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    const off = range <= 0 ? 0 : (max - baseline) / range;

    return (
        <div className="h-10 w-28 mx-auto">
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={values.map((v, i) => ({ value: v, index: i }))}>
                    <defs>
                        <linearGradient id={`splitFill-${id}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset={off} stopColor="#10b981" stopOpacity={0.15} />
                            <stop offset={off} stopColor="#ef4444" stopOpacity={0.15} />
                        </linearGradient>
                        <linearGradient id={`splitStroke-${id}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset={off} stopColor="#10b981" stopOpacity={1} />
                            <stop offset={off} stopColor="#ef4444" stopOpacity={1} />
                        </linearGradient>
                    </defs>
                    <YAxis hide domain={['dataMin', 'dataMax']} />
                    <ReferenceLine y={baseline} stroke="#71717a" strokeDasharray="2 2" strokeOpacity={0.3} />
                    <Area
                        type="monotone"
                        dataKey="value"
                        baseValue={baseline}
                        stroke={`url(#splitStroke-${id})`}
                        fill={`url(#splitFill-${id})`}
                        strokeWidth={1.5}
                        isAnimationActive={false}
                        dot={(props: any) => {
                            const { cx, cy, index } = props;
                            if (index === values.length - 1) {
                                const color = values[values.length - 1] >= baseline ? "#10b981" : "#ef4444";
                                return (
                                    <circle key="dot" cx={cx} cy={cy} r={2} fill={color} stroke="none" />
                                );
                            }
                            return <React.Fragment key={index} />;
                        }}
                    />
                </AreaChart>
            </ResponsiveContainer>
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
