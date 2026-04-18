'use client';

import React, { useState, useEffect, useId, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
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
    StockAnalysisResponse,
    fetchHoldings,
    Holding
} from '../lib/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
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
    Hash,
    Tag,
    LineChart as LineChartIcon,
    List,
    HelpCircle,
    Sparkles,
    Shield,
    Zap,
    Target,
    Activity as LucideActivity,
    CheckCircle2,
    RotateCcw,
    Loader2,
    AlertCircle
} from 'lucide-react';
import {
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
import StockPriceChart from './StockPriceChart';

interface StockDetailModalProps {
    symbol: string;
    isOpen: boolean;
    onClose: () => void;
    currency: string;
}

type TabType = 'overview' | 'chart' | 'financials' | 'ratios' | 'valuation' | 'holdings' | 'analysis';

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
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);
    const [analysis, setAnalysis] = useState<any>(null);
    const [analysisLoading, setAnalysisLoading] = useState(false);
    const [analysisError, setAnalysisError] = useState<string | null>(null);
    const { resolvedTheme } = useTheme();
    const isDarkMode = resolvedTheme === 'dark';

    // Get filters for holdings query (matching dashboard)
    const filters = useMemo(() => {
        if (typeof window === 'undefined') return { accounts: [], showClosed: false };
        try {
            const savedAccounts = localStorage.getItem('investa_selected_accounts');
            const savedShowClosed = localStorage.getItem('investa_show_closed');
            return {
                accounts: savedAccounts ? JSON.parse(savedAccounts) : [],
                showClosed: savedShowClosed === 'true'
            };
        } catch (e) {
            return { accounts: [], showClosed: false };
        }
    }, [isOpen]);

    // Fetch holdings to check if user has a position
    const { data: holdings = [] } = useQuery({
        queryKey: ['holdings', symbol, currency, filters.accounts, filters.showClosed],
        queryFn: () => fetchHoldings(currency, filters.accounts, filters.showClosed),
        enabled: isOpen && !!symbol,
        staleTime: 5 * 60 * 1000,
    });

    // Aggregate position across accounts
    const userPosition = useMemo(() => {
        if (!holdings.length) return null;

        const matchingHoldings = holdings.filter(h => h.Symbol === symbol);
        if (!matchingHoldings.length) return null;

        const aggregate = matchingHoldings.reduce((acc, curr) => {
            // Helper to get numeric value handling dynamic currency keys
            const getVal = (h: Holding, prefix: string) => {
                const exact = h[prefix];
                if (typeof exact === 'number') return exact;
                const withCurr = h[`${prefix} (${currency})`];
                if (typeof withCurr === 'number') return withCurr;
                // Search for key starting with prefix if needed
                const foundKey = Object.keys(h).find(k => k.startsWith(prefix));
                if (foundKey && typeof h[foundKey] === 'number') return h[foundKey] as number;
                return 0;
            };

            const qty = curr.Quantity || 0;
            const mktVal = getVal(curr, "Market Value");
            const costBasis = getVal(curr, "Cost Basis");
            const unrealizedGain = getVal(curr, "Unreal. Gain");
            const totalGain = getVal(curr, "Total Gain") || unrealizedGain; // Fallback
            const dividends = getVal(curr, "Dividends") || 0;

            return {
                Quantity: acc.Quantity + qty,
                "Market Value": acc["Market Value"] + mktVal,
                "Cost Basis": acc["Cost Basis"] + costBasis,
                "Unreal. Gain": acc["Unreal. Gain"] + unrealizedGain,
                "Total Gain": acc["Total Gain"] + totalGain,
                "Dividends": acc["Dividends"] + dividends,
                "Weighted IRR": (acc["Weighted IRR"] || 0) + ((curr["IRR (%)"] || 0) * mktVal),
                "fx_rate": (typeof curr.fx_rate === 'number' ? curr.fx_rate : 0) || acc.fx_rate || 1, // Store fx_rate
            };
        }, {
            Quantity: 0,
            "Market Value": 0,
            "Cost Basis": 0,
            "Unreal. Gain": 0,
            "Total Gain": 0,
            "Dividends": 0,
            "Weighted IRR": 0,
            "fx_rate": 1
        });

        const avgCost = aggregate.Quantity > 0 ? aggregate["Cost Basis"] / aggregate.Quantity : 0;
        const totalReturnPct = aggregate["Cost Basis"] > 0 ? (aggregate["Total Gain"] / aggregate["Cost Basis"]) * 100 : 0;
        const unrealizedGainPct = aggregate["Cost Basis"] > 0 ? (aggregate["Unreal. Gain"] / aggregate["Cost Basis"]) * 100 : 0;
        const aggregateIrr = aggregate["Market Value"] > 0 ? aggregate["Weighted IRR"] / aggregate["Market Value"] : 0;

        return {
            ...aggregate,
            "Avg Cost": avgCost,
            "Total Return %": totalReturnPct,
            "Unreal. Gain %": unrealizedGainPct,
            "IRR %": aggregateIrr
        };
    }, [holdings, symbol, currency]);

    const fxRate = useMemo(() => userPosition?.fx_rate ?? 1, [userPosition]);

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

    const loadData = async (force: boolean = false) => {
        setLoading(true);
        setError(null);

        try {
            // Priority 1: Fetch Fundamentals (Required for initial render)
            const fundRes = await fetchFundamentals(symbol, force);
            setFundamentals(fundRes);
            setLoading(false); // Unblock UI immediately after fundamentals

            // Priority 2: Fetch everything else in parallel
            // We don't await these for the main loading state
            Promise.allSettled([
                fetchFinancials(symbol, 'annual', force).then(setFinancials),
                fetchRatios(symbol, force).then(setRatios),
                fetchIntrinsicValue(symbol, force).then((data) => {
                    setIntrinsicValue(data);
                    // Dispatch event for live updates in ScreenerResults
                    if (data) {
                        window.dispatchEvent(new CustomEvent('stock-intrinsic-value-updated', {
                            detail: { symbol, data }
                        }));
                    }
                })
            ]).catch(err => {
                console.error("Background data fetch error:", err);
            });

        } catch (err: any) {
            console.error(err);
            setError(err.message || "Failed to load stock details");
            setLoading(false);
        }
    };

    const handleRegenerateAnalysis = async () => {
        setAnalysisLoading(true);
        setAnalysisError(null);
        try {
            const data = await fetchStockAnalysis(symbol, true);
            if (data && data.error) {
                setAnalysisError(data.error);
            } else {
                setAnalysis(data);
                // Dispatch event so screener can update live
                window.dispatchEvent(new CustomEvent('stock-analysis-updated', {
                    detail: { symbol, analysis: data }
                }));
            }
        } catch (err: any) {
            console.error("Analysis regeneration error:", err);
            setAnalysisError(err.message || "Failed to regenerate AI analysis.");
        } finally {
            setAnalysisLoading(false);
        }
    };

    if (!mounted || !isOpen) return null;

    const formatCurrency = (val: number | null | undefined, overrideCurrency?: string) => {
        if (val === undefined || val === null) return '-';
        const targetCurrency = overrideCurrency || fundamentals?.currency || 'USD';
        const formatted = new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: targetCurrency,
            notation: val > 1000000 ? 'compact' : 'standard',
            currencyDisplay: 'narrowSymbol',
        }).format(val);

        if (targetCurrency === 'THB') {
            return formatted.replace('THB', '฿').replace(/\s/g, '');
        }
        return formatted;
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
            return formatPercentShared(upside);
        };

        const getUpsideColor = (upside: number | null) => {
            if (upside === null) return "";
            return upside > 0 ? "text-emerald-500 font-bold" : "text-rose-500 font-bold";
        };

        const dcfUpside = getUpsidePercentage(intrinsicValue?.models?.dcf?.intrinsic_value);
        const grahamUpside = getUpsidePercentage(intrinsicValue?.models?.graham?.intrinsic_value);

        return (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                {userPosition && (
                    <div className="space-y-3">
                        <div className="flex items-center justify-between">
                            <h3 className="text-lg font-semibold flex items-center gap-2">
                                <Wallet className="w-5 h-5 text-indigo-500" />
                                Your Position
                            </h3>
                            <div className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider bg-secondary/50 px-2 py-1 rounded-md">
                                Aggregated
                            </div>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-3 gap-2.5">
                            <StatCard
                                label="Quantity"
                                value={userPosition.Quantity.toLocaleString()}
                                icon={Hash}
                                color="text-indigo-500"
                            />
                            <StatCard
                                label="Avg Cost"
                                value={formatCurrency(userPosition["Avg Cost"], currency)}
                                icon={Tag}
                                color="text-slate-500"
                            />
                            <StatCard
                                label="Market Value"
                                value={formatCurrency(userPosition["Market Value"], currency)}
                                icon={PieChartIcon}
                                color="text-indigo-500"
                            />
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
                            <StatCard
                                label="Unrealized G/L"
                                value={formatCurrency(userPosition["Unreal. Gain"], currency)}
                                subValue={`${(userPosition["Unreal. Gain %"] || 0).toFixed(2)}%`}
                                subValueColor={(userPosition["Unreal. Gain %"] || 0) >= 0 ? "text-emerald-500" : "text-rose-500"}
                                valueColor={(userPosition["Unreal. Gain"] || 0) >= 0 ? "text-emerald-500" : "text-rose-500"}
                                icon={LucideActivity}
                                color={(userPosition?.["Unreal. Gain"] ?? 0) >= 0 ? "bg-emerald-500/10 text-emerald-500" : "bg-rose-500/10 text-rose-500"}
                            />
                            <StatCard
                                label="Total Return"
                                value={formatCurrency(userPosition["Total Gain"], currency)}
                                subValue={`${(userPosition["Total Return %"] || 0).toFixed(2)}%`}
                                subValueColor={(userPosition["Total Return %"] || 0) >= 0 ? "text-emerald-500" : "text-rose-500"}
                                valueColor={(userPosition["Total Return %"] || 0) >= 0 ? "text-emerald-500" : "text-rose-500"}
                                icon={TrendingUp}
                                color={(userPosition["Total Return %"] || 0) >= 0 ? "bg-emerald-500/10 text-emerald-500" : "bg-rose-500/10 text-rose-500"}
                                extra={
                                    <p className="text-[11px] font-semibold text-amber-600 dark:text-amber-500/90 leading-tight">
                                        Divs: {formatCurrency(userPosition["Dividends"], currency)}
                                    </p>
                                }
                            />
                            <StatCard
                                label="IRR %"
                                value={`${(userPosition["IRR %"] || 0).toFixed(2)}%`}
                                icon={LineChartIcon}
                                valueColor={(userPosition["IRR %"] || 0) >= 0 ? "text-emerald-500" : "text-rose-500"}
                                color={(userPosition["IRR %"] || 0) >= 0 ? "bg-emerald-500/10 text-emerald-500" : "bg-rose-500/10 text-rose-500"}
                            />
                        </div>

                        <div className="h-px bg-border/40 w-full my-1" />
                    </div>
                )}

                <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                        <LayoutDashboard className="w-5 h-5 text-indigo-500" />
                        Market Overview
                    </h3>
                    <button
                        onClick={() => loadData(true)}
                        disabled={loading}
                        className="flex items-center gap-1.5 text-[10px] font-bold text-cyan-600 hover:text-cyan-700 dark:text-cyan-400 dark:hover:text-cyan-300 transition-colors uppercase tracking-wider"
                        title="Force Refresh Data"
                    >
                        <RotateCcw className="w-3 h-3" />
                        Refresh Data
                    </button>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 w-full">
                    {intrinsicValue?.models?.dcf?.intrinsic_value && (
                                <StatCard
                                    label="DCF Intrinsic Value"
                                    value={formatCurrency((intrinsicValue.models.dcf.intrinsic_value ?? 0) * fxRate, currency)}
                                    subValue={formatUpside(dcfUpside)}
                                    subValueColor={getUpsideColor(dcfUpside)}
                                    rangeMin={formatCurrency((intrinsicValue.models.dcf.mc?.bear ?? 0) * fxRate, currency)}
                                    rangeMax={formatCurrency((intrinsicValue.models.dcf.mc?.bull ?? 0) * fxRate, currency)}
                                    icon={TrendingUp}
                                    color="text-emerald-400"
                                />
                            )}
                            {intrinsicValue?.models?.graham?.intrinsic_value && (
                                <StatCard
                                    label="Graham Intrinsic Value"
                                    value={formatCurrency((intrinsicValue.models.graham.intrinsic_value ?? 0) * fxRate, currency)}
                                    subValue={formatUpside(grahamUpside)}
                                    subValueColor={getUpsideColor(grahamUpside)}
                                    rangeMin={formatCurrency((intrinsicValue.models.graham.mc?.bear ?? 0) * fxRate, currency)}
                                    rangeMax={formatCurrency((intrinsicValue.models.graham.mc?.bull ?? 0) * fxRate, currency)}
                                    icon={Scale}
                                    color="text-amber-400"
                                />
                            )}
                </div>

                <div className="grid grid-cols-2 md:grid-cols-3 gap-2.5">
                    <StatCard label="Market Cap" value={formatCurrency(fundamentals.marketCap)} icon={Globe} color="text-indigo-400" />
                    <StatCard label="P/E Ratio (TTM)" value={fundamentals.trailingPE?.toFixed(2)} icon={TrendingUp} color="text-emerald-400" />
                    <StatCard label="Dividend Yield" value={formatPercent(fundamentals.dividendYield)} icon={DollarSign} color="text-amber-400" />
                    <StatCard label="52W High" value={formatCurrency(fundamentals.fiftyTwoWeekHigh)} icon={TrendingUp} color="text-blue-400" />
                    <StatCard label="52W Low" value={formatCurrency(fundamentals.fiftyTwoWeekLow)} icon={TrendingUp} color="text-pink-400" className="rotate-180" />
                    {!fundamentals.etf_data && (
                        <StatCard label="Beta" value={fundamentals.beta?.toFixed(2)} icon={LucideActivity} color="text-purple-400" />
                    )}
                    {(fundamentals.expenseRatio || fundamentals.annualReportExpenseRatio || fundamentals.netExpenseRatio) && (
                        <StatCard
                            label="Expense Ratio"
                            value={formatPercent((fundamentals.expenseRatio || fundamentals.annualReportExpenseRatio || fundamentals.netExpenseRatio))}
                            icon={Receipt}
                            color="text-orange-400"
                        />
                    )}
                </div>

                <div className="bg-muted px-6 py-4">
                    <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                        <Building2 className="w-5 h-5 text-indigo-500" />
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
                                "flex items-center gap-2 px-3 sm:px-4 py-2 rounded-full text-[10px] sm:text-xs font-bold transition-all whitespace-nowrap flex-shrink-0",
                                finType === btn.id
                                    ? "bg-indigo-500 text-white"
                                    : "bg-muted/50 text-muted-foreground hover:bg-muted hover:text-foreground"
                            )}
                            title={btn.fullLabel}
                        >
                            <btn.icon className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                            <span className="hidden sm:inline">{btn.label}</span>
                        </button>
                    ))}
                </div>

                <div className="overflow-x-auto bg-muted">
                    <table className="w-full text-sm text-left">
                        <thead className="bg-secondary/50 font-semibold">
                            <tr>
                                <th className="px-6 py-3 font-semibold text-foreground sticky left-0 bg-card"></th>
                                <th className="px-6 py-3 font-semibold text-center text-muted-foreground">Trend</th>
                                {currentStatement.columns.map(col => (
                                    <th key={col} className="px-6 py-3 font-semibold text-center text-muted-foreground tabular-nums">{new Date(col).getFullYear()}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="">
                            {currentStatement.index.map((item, idx) => (
                                <tr key={item} className="hover:bg-accent/5 transition-colors">
                                    <td className="px-6 py-3 font-medium text-foreground sticky left-0 bg-card min-w-[200px]">{item}</td>
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
                    "p-6 rounded-[2rem] overflow-hidden relative",
                    isDarkMode ? "bg-slate-900/50" : "bg-white"
                )}>
                    <div className="flex items-center gap-4 relative z-10">
                        <div className="w-12 h-12 rounded-2xl bg-purple-500 flex items-center justify-center shrink-0">
                            <Sparkles className="w-6 h-6 text-white" />
                        </div>
                        <div className="flex flex-col">
                            <div className="flex items-center gap-3">
                                <h3 className="text-xl font-bold">AI Fundamental Review</h3>
                                <button
                                    onClick={handleRegenerateAnalysis}
                                    disabled={analysisLoading}
                                    className="flex items-center gap-1.5 text-[10px] font-bold text-purple-600 hover:text-purple-700 dark:text-purple-400 dark:hover:text-purple-300 transition-colors uppercase tracking-wider mt-0.5"
                                    title="Regenerate AI Analysis"
                                >
                                    {analysisLoading ? (
                                        <Loader2 className="w-3 h-3 animate-spin" />
                                    ) : (
                                        <RotateCcw className="w-3 h-3" />
                                    )}
                                    Regenerate
                                </button>
                            </div>
                            <div className="text-sm text-muted-foreground leading-relaxed mt-1 markdown-content bg-transparent p-0 border-none shadow-none">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>{analysis.summary || ""}</ReactMarkdown>
                            </div>
                        </div>
                    </div>
                    {/* Decorative background element */}
                    <div className="absolute top-0 right-0 w-64 h-64 bg-purple-500/5 rounded-full blur-3xl -mr-32 -mt-32" />
                </div>

                {/* Score Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {topics.map(t => (
                        <div key={t.id} className={cn(
                            "p-4 rounded-3xl flex flex-col items-center justify-center gap-2",
                            isDarkMode ? "bg-slate-900/10" : "bg-zinc-50/50"
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
                            "p-6 rounded-[2rem] transition-all group",
                            isDarkMode ? "bg-slate-900/30 hover:bg-slate-900/50" : "bg-white"
                        )}>
                            <div className="flex items-center gap-3 mb-4">
                                <div className={cn("p-2.5 rounded-xl", t.bg)}>
                                    <t.icon className={cn("w-5 h-5", t.color)} />
                                </div>
                                <h4 className="font-bold text-lg">{t.title}</h4>
                            </div>
                            <div className="text-sm leading-relaxed text-muted-foreground group-hover:text-foreground transition-colors markdown-content bg-transparent p-0 border-none shadow-none">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>{t.content || ""}</ReactMarkdown>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Market Sentiment & Catalysts */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Sentiment Gauge */}
                    <div className={cn(
                        "p-6 rounded-[2rem] transition-all",
                        isDarkMode ? "bg-slate-900/30" : "bg-white"
                    )}>
                        <div className="flex items-center justify-between mb-6">
                            <div className="flex items-center gap-3">
                                <div className="p-2.5 rounded-xl bg-indigo-500/10 text-indigo-500">
                                    <TrendingUp className="w-5 h-5" />
                                </div>
                                <h4 className="font-bold text-lg">Market Sentiment</h4>
                            </div>
                            {analysis.sentiment !== undefined && (
                                <Badge className={cn(
                                    "border-none px-3 py-1",
                                    analysis.sentiment >= 70 ? "bg-emerald-500/20 text-emerald-500" :
                                    analysis.sentiment >= 40 ? "bg-amber-500/20 text-amber-500" :
                                    "bg-rose-500/20 text-rose-500"
                                )}>
                                    {analysis.sentiment >= 70 ? 'Bullish' : analysis.sentiment >= 40 ? 'Neutral' : 'Bearish'}
                                </Badge>
                            )}
                        </div>
                        
                        {analysis.sentiment !== undefined ? (
                            <div className="flex flex-col items-center py-4">
                                <div className="relative w-full h-4 bg-muted rounded-full overflow-hidden mb-4">
                                    <div 
                                        className={cn(
                                            "h-full rounded-full transition-all duration-1000 ease-out",
                                            analysis.sentiment >= 70 ? "bg-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.5)]" :
                                            analysis.sentiment >= 40 ? "bg-amber-500 shadow-[0_0_15px_rgba(245,158,11,0.5)]" :
                                            "bg-rose-500 shadow-[0_0_15px_rgba(244,63,94,0.5)]"
                                        )}
                                        style={{ width: `${analysis.sentiment}%` }}
                                    />
                                </div>
                                <div className="flex justify-between w-full text-[10px] font-bold text-muted-foreground uppercase tracking-wider px-1">
                                    <span>Extreme Fear</span>
                                    <span className="text-foreground text-lg">{analysis.sentiment.toFixed(0)}%</span>
                                    <span>Extreme Greed</span>
                                </div>
                                <p className="text-xs text-muted-foreground text-center mt-6 leading-relaxed">
                                    Current market vibe based on news flow, analyst ratings, and social trends.
                                </p>
                            </div>
                        ) : (
                            <div className="flex flex-col items-center justify-center py-10 text-muted-foreground opacity-30 italic">
                                <div className="w-16 h-16 bg-muted rounded-2xl flex items-center justify-center text-muted-foreground mb-4">
                                    <LucideActivity className="w-8 h-8 mb-2" />
                                </div>
                                <p className="text-xs">Sentiment data pending...</p>
                            </div>
                        )}
                    </div>

                    {/* Catalyst Timeline */}
                    <div className={cn(
                        "p-6 rounded-[2rem] transition-all",
                        isDarkMode ? "bg-slate-900/30" : "bg-white"
                    )}>
                        <div className="flex items-center gap-3 mb-6">
                            <div className="p-2.5 rounded-xl bg-amber-500/10 text-amber-500">
                                <Calendar className="w-5 h-5" />
                            </div>
                            <h4 className="font-bold text-lg">Upcoming Catalysts</h4>
                        </div>
                        
                        <div className="space-y-4">
                            {analysis.catalysts && analysis.catalysts.length > 0 ? (
                                analysis.catalysts.map((c: any, i: number) => (
                                    <div key={i} className="flex gap-4 group">
                                        <div className="flex flex-col items-center">
                                            <div className={cn(
                                                "w-2 h-2 rounded-full mt-1.5 shrink-0",
                                                c.impact === 'High' ? "bg-rose-500" :
                                                c.impact === 'Medium' ? "bg-amber-500" : "bg-blue-500"
                                            )} />
                                            {i < analysis.catalysts.length - 1 && (
                                                <div className="w-px h-full bg-border/40 my-1" />
                                            )}
                                        </div>
                                        <div className="flex-1 pb-4">
                                            <div className="flex items-center justify-between mb-1">
                                                <p className="text-sm font-bold text-foreground transition-colors group-hover:text-indigo-500">{c.event}</p>
                                                <Badge variant="outline" className="text-[9px] font-bold uppercase border-muted-foreground/20 text-muted-foreground">
                                                    {c.impact}
                                                </Badge>
                                            </div>
                                            <p className="text-[11px] font-medium text-muted-foreground">{c.date}</p>
                                        </div>
                                    </div>
                                ))
                            ) : (
                                <div className="flex flex-col items-center justify-center py-10 text-muted-foreground opacity-30 italic">
                                    <Info className="w-8 h-8 mb-2" />
                                    <p className="text-xs">No catalysts detected in latest run.</p>
                                </div>
                            )}
                        </div>
                    </div>
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
                    <div className="bg-muted rounded-2xl p-6">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <List className="w-5 h-5 text-indigo-500" />
                            Top Holdings
                        </h3>
                        <div className="overflow-hidden rounded-xl">
                            <table className="w-full text-sm">
                                <thead className="bg-secondary/50">
                                    <tr>
                                        <th className="px-4 py-2 text-left font-medium text-muted-foreground">Symbol</th>
                                        <th className="px-4 py-2 text-right font-medium text-muted-foreground">% Assets</th>
                                    </tr>
                                </thead>
                                <tbody className="">
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
                    <div className="bg-muted rounded-2xl p-6">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <PieChartIcon className="w-5 h-5 text-indigo-500" />
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
                                            wrapperStyle={{ opacity: 1, zIndex: 1000 }}
                                            formatter={(value: any) => `${Number(value).toFixed(2)}%`}
                                            contentStyle={{ backgroundColor: 'transparent', border: 'none' }}
                                            content={({ active, payload }) => {
                                                if (active && payload && payload.length) {
                                                    return (
                                                        <div className="bg-background/95 backdrop-blur-xl p-3 rounded-xl border border-border/50 shadow-2xl">
                                                            <p className="font-medium text-foreground">{payload[0].name}</p>
                                                            <p className="text-sm text-muted-foreground">
                                                                {Number(payload[0].value).toFixed(2)}%
                                                            </p>
                                                        </div>
                                                    );
                                                }
                                                return null;
                                            }}
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

    const renderChart = () => (
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            <StockPriceChart
                symbol={symbol}
                currency={currency}
                avgCost={userPosition?.["Avg Cost"]}
                fxRate={userPosition?.["fx_rate"]}
                hidePrice={true}
            />
        </div>
    );

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
        fcf_margin: {
            description: "Free Cash Flow as a percentage of revenue, used to normalize future cash flow projections if current FCF is an outlier.",
            default: "5-Year Average"
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
                    <div className="bg-muted p-6 rounded-2xl flex flex-col items-center justify-center text-center">
                        <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider mb-2">Average Intrinsic Value</p>
                        <p className="text-3xl font-bold text-indigo-500">{formatCurrency((average_intrinsic_value ?? 0) * fxRate, currency)}</p>
                        {intrinsicValue?.range && (
                            <p className="text-xs text-muted-foreground mt-2 font-medium">
                                Range: {formatCurrency((intrinsicValue.range.bear ?? 0) * fxRate, currency)} - {formatCurrency((intrinsicValue.range.bull ?? 0) * fxRate, currency)}
                            </p>
                        )}
                    </div>
                    <div className="bg-muted p-6 rounded-2xl flex flex-col items-center justify-center text-center">
                        <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider mb-2">Current Price</p>
                        <p className="text-3xl font-bold">{formatCurrency((current_price ?? 0) * fxRate, currency)}</p>
                    </div>
                    <div className={cn(
                        "p-6 rounded-2xl flex flex-col items-center justify-center text-center transition-all",
                        (margin_of_safety_pct || 0) > 0
                            ? "bg-emerald-500/10 dark:bg-emerald-500/5"
                            : "bg-rose-500/10 dark:bg-rose-500/5"
                    )}>
                        <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider mb-2">Margin of Safety</p>
                        <p className={cn(
                            "text-3xl font-bold tracking-tight",
                            (margin_of_safety_pct || 0) > 0 ? "text-emerald-500" : "text-rose-500"
                        )}>
                            {(margin_of_safety_pct || 0).toFixed(2)}%
                        </p>
                    </div>
                </div>

                {intrinsicValue.valuation_note && (
                    <div className="bg-amber-500/10 p-4 rounded-2xl flex items-start gap-3 animate-in fade-in slide-in-from-top-2 duration-500">
                        <AlertCircle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
                        <div className="flex flex-col gap-1">
                            <p className="text-xs font-bold text-amber-600 dark:text-amber-400 uppercase tracking-wider">Model Discrepancy Note</p>
                            <p className="text-sm text-amber-700 dark:text-amber-500 leading-relaxed italic">{intrinsicValue.valuation_note}</p>
                        </div>
                    </div>
                )}

                {/* Models Detail */}
                {!models.dcf.parameters && !models.graham.parameters ? (
                    <div className="bg-muted rounded-2xl p-8 text-center animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <div className="w-16 h-16 bg-indigo-500/10 rounded-full flex items-center justify-center mx-auto mb-6">
                            <Info className="w-8 h-8 text-indigo-500" />
                        </div>
                        <h3 className="text-xl font-bold mb-3">Why standard models aren't shown?</h3>
                        <p className="text-muted-foreground text-sm leading-relaxed max-w-xl mx-auto mb-6">
                            Traditional valuation methods like <strong>Discounted Cash Flow (DCF)</strong> and <strong>Graham's Formula</strong> rely on free cash flow and earnings growth, which are company-specific metrics.
                            <br /><br />
                            For <strong>ETFs and Mutual Funds</strong>, the intrinsic value is best represented by the <strong>Net Asset Value (NAV)</strong>, which is the total value of the fund's assets minus its liabilities, divided by the number of outstanding shares.
                        </p>
                        <div className="inline-flex items-center gap-2 px-4 py-2 bg-background rounded-full text-xs font-medium text-foreground">
                            <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                            Using Industry Standard NAV Valuation
                        </div>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {/* DCF Model */}
                        <div className="bg-muted rounded-2xl p-6">
                            <div className="flex items-center justify-between mb-6">
                                <h3 className="text-lg font-semibold flex items-center gap-2">
                                    <TrendingUp className="w-5 h-5 text-emerald-500" />
                                    {models.dcf.model}
                                </h3>
                                {models.dcf.intrinsic_value !== undefined && (
                                    <div className="flex flex-col items-end">
                                        <Badge className="bg-emerald-500/20 text-emerald-500 border-none">
                                            {formatCurrency((models.dcf.intrinsic_value ?? 0) * fxRate, currency)}
                                        </Badge>
                                        {models.dcf.model !== 'DCF' && (
                                            <span className="text-[9px] text-muted-foreground mt-1">
                                                (Fallback Used)
                                            </span>
                                        )}
                                    </div>
                                )}
                            </div>
                            {models.dcf.error ? (
                                <p className="text-sm text-destructive bg-destructive/5 p-4 rounded-xl">{models.dcf.error}</p>
                            ) : !models.dcf.parameters ? (
                                <div className="flex flex-col items-center justify-center py-10 text-center opacity-50">
                                    <Info className="w-8 h-8 mb-2" />
                                    <p className="text-sm">Not applicable for this asset type.</p>
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    <div className="grid grid-cols-2 gap-4">
                                        <ParamItem
                                            label="Discount Rate (WACC)"
                                            value={formatPercentShared(models.dcf.parameters.discount_rate ?? 0)}
                                            info={VALUATION_INFO.discount_rate}
                                        />
                                        <ParamItem
                                            label="Growth Rate"
                                            value={formatPercentShared(models.dcf.parameters.growth_rate ?? 0)}
                                            info={VALUATION_INFO.growth_rate}
                                        />
                                        {models.dcf.parameters.applied_growth !== undefined && models.dcf.parameters.applied_growth !== models.dcf.parameters.growth_rate && (
                                            <ParamItem
                                                label="Applied Growth"
                                                value={formatPercentShared(models.dcf.parameters.applied_growth ?? 0)}
                                                className="text-cyan-500 font-bold"
                                                info={{ description: "The growth rate actually used in the projection after applying physical reality caps.", default: "100% Max" }}
                                            />
                                        )}
                                        <ParamItem
                                            label="Terminal Growth"
                                            value={formatPercentShared(models.dcf.parameters.terminal_growth_rate ?? 0)}
                                            info={VALUATION_INFO.terminal_growth}
                                        />
                                        <ParamItem
                                            label="Projection Years"
                                            value={models.dcf.parameters.projection_years}
                                            info={VALUATION_INFO.projection_years}
                                        />
                                        {models.dcf.parameters.note && (
                                            <div className="col-span-2 text-xs text-muted-foreground italic bg-secondary/30 p-2 rounded">
                                                Note: {models.dcf.parameters.note}
                                            </div>
                                        )}
                                    </div>
                                    <div className="pt-4">
                                        <ParamItem
                                            label="Base Free Cash Flow"
                                            value={formatCurrency((models.dcf.parameters.base_fcf ?? 0) * fxRate, currency)}
                                            info={VALUATION_INFO.base_fcf}
                                        />
                                        {models.dcf.parameters.fcf_margin && (
                                            <ParamItem
                                                label="Est. FCF Margin"
                                                value={formatPercentShared(models.dcf.parameters.fcf_margin ?? 0)}
                                                info={VALUATION_INFO.fcf_margin}
                                            />
                                        )}
                                    </div>
                                    {models.dcf.mc && (
                                        <div className="pt-4">
                                            <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-bold mb-3">Probabilistic Scenarios (Monte Carlo)</p>
                                            <div className="grid grid-cols-3 gap-2">
                                                <div
                                                    className="bg-rose-500/5 p-2 rounded-lg text-center cursor-pointer hover:bg-rose-500/10 transition-colors group/mc"
                                                    onClick={() => setViewingDistribution('dcf')}
                                                >
                                                    <p className="text-[10px] text-rose-500 font-bold uppercase mb-1">Bear (10th)</p>
                                                    <p className="text-sm font-bold">{formatCurrency((models.dcf.mc.bear ?? 0) * fxRate, currency)}</p>
                                                </div>
                                                <div
                                                    className="bg-indigo-500/5 p-2 rounded-lg text-center cursor-pointer hover:bg-indigo-500/10 transition-colors group/mc"
                                                    onClick={() => setViewingDistribution('dcf')}
                                                >
                                                    <p className="text-[10px] text-indigo-500 font-bold uppercase mb-1">Median (50th)</p>
                                                    <p className="text-sm font-bold">{formatCurrency((models.dcf.mc.base ?? 0) * fxRate, currency)}</p>
                                                </div>
                                                <div
                                                    className="bg-emerald-500/5 p-2 rounded-lg text-center cursor-pointer hover:bg-emerald-500/10 transition-colors group/mc"
                                                    onClick={() => setViewingDistribution('dcf')}
                                                >
                                                    <p className="text-[10px] text-emerald-500 font-bold uppercase mb-1">Bull (90th)</p>
                                                    <p className="text-sm font-bold">{formatCurrency((models.dcf.mc.bull ?? 0) * fxRate, currency)}</p>
                                                </div>
                                            </div>
                                            <p className="text-[9px] text-muted-foreground mt-2 text-center opacity-50 group-hover/mc:opacity-100 transition-opacity">Click to view distribution</p>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>

                        {/* Graham Model */}
                        <div className="bg-muted rounded-2xl p-6">
                            <div className="flex items-center justify-between mb-6">
                                <h3 className="text-lg font-semibold flex items-center gap-2">
                                    <Scale className="w-5 h-5 text-amber-500" />
                                    {models.graham.model}
                                </h3>
                                {models.graham.intrinsic_value !== undefined && (
                                    <div className="flex flex-col items-end">
                                        <Badge className="bg-amber-500/20 text-amber-500 border-none">
                                            {formatCurrency((models.graham.intrinsic_value ?? 0) * fxRate, currency)}
                                        </Badge>
                                        {models.graham.model !== "Graham's Revised Formula" && (
                                            <span className="text-[9px] text-muted-foreground mt-1">
                                                (Fallback Used)
                                            </span>
                                        )}
                                    </div>
                                )}
                            </div>
                            {models.graham.error ? (
                                <p className="text-sm text-destructive bg-destructive/5 p-4 rounded-xl">{models.graham.error}</p>
                            ) : !models.graham.parameters ? (
                                <div className="flex flex-col items-center justify-center py-10 text-center opacity-50">
                                    <Info className="w-8 h-8 mb-2" />
                                    <p className="text-sm">Not applicable for this asset type.</p>
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    <div className="grid grid-cols-2 gap-4">
                                        <ParamItem
                                            label="Trailing EPS"
                                            value={(models.graham.parameters.eps || 0).toFixed(2)}
                                            info={VALUATION_INFO.eps}
                                        />
                                        <ParamItem
                                            label="Growth Rate (g)"
                                            value={`${(models.graham.parameters.growth_rate_pct ?? 0).toFixed(2)}%`}
                                            info={VALUATION_INFO.graham_growth}
                                        />
                                        {models.graham.parameters.applied_growth_pct !== undefined && models.graham.parameters.applied_growth_pct !== models.graham.parameters.growth_rate_pct && (
                                            <ParamItem
                                                label="Applied Growth"
                                                value={`${(models.graham.parameters.applied_growth_pct || 0).toFixed(2)}%`}
                                                className="text-amber-500 font-bold"
                                                info={{ description: "The growth rate actually used in the formula after applying stability caps.", default: "30% Max" }}
                                            />
                                        )}
                                        <ParamItem
                                            label="Bond Yield (Y)"
                                            value={`${(models.graham.parameters.bond_yield_proxy || 0).toFixed(2)}%`}
                                            info={VALUATION_INFO.bond_yield}
                                        />
                                        {models.graham.parameters.note && (
                                            <div className="col-span-2 text-xs text-amber-600 dark:text-amber-400 bg-amber-500/10 p-3 rounded-xl flex items-start gap-2">
                                                <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                                                <span>{models.graham.parameters.note}</span>
                                            </div>
                                        )}
                                    </div>
                                    <div className="mt-4 p-4 bg-secondary/5 rounded-xl flex flex-col items-center select-none overflow-visible">
                                        <div className="flex items-center gap-2">
                                            <div className="flex items-baseline gap-1">
                                                <div className="group relative">
                                                    <span className="text-lg font-bold text-foreground cursor-help decoration-dotted underline-offset-4 hover:text-cyan-500 transition-colors">V</span>
                                                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-40 p-2 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[9px] rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-tight text-center font-medium">
                                                        Intrinsic Value
                                                    </div>
                                                </div>
                                                <span className="text-lg font-light opacity-30">=</span>
                                            </div>
                                            <div className="flex items-center gap-1.5 px-4">
                                                <div className="group relative">
                                                    <span className="text-xs font-bold text-foreground cursor-help hover:text-cyan-500 transition-colors">EPS</span>
                                                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-40 p-2 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[9px] rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-tight text-center font-medium">
                                                        Trailing 12-Month Earnings Per Share
                                                    </div>
                                                </div>
                                                <span className="text-[9px] opacity-40">×</span>
                                                <div className="group relative">
                                                    <span className="px-1.5 py-0.5 bg-secondary/30 rounded-md text-[10px] font-bold text-foreground cursor-help transition-colors">
                                                        8.5 + 2G
                                                    </span>
                                                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2.5 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[9px] rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-relaxed text-center font-medium">
                                                        <div className="mb-0.5"><span className="font-bold">8.5</span>: Base P/E for zero growth</div>
                                                        <div><span className="font-bold">G</span>: Expected long-term growth rate</div>
                                                    </div>
                                                </div>
                                                <span className="text-[9px] opacity-40">×</span>
                                                <div className="group relative">
                                                    <span className="text-xs font-bold text-foreground cursor-help hover:text-cyan-500 transition-colors">4.4</span>
                                                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-40 p-2 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[9px] rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-tight text-center font-medium">
                                                        Average yield of high-grade corporate bonds in 1962
                                                    </div>
                                                </div>
                                                <span className="text-[10px] opacity-40 mx-0.5">/</span>
                                                <div className="group relative">
                                                    <span className="text-xs font-bold text-foreground cursor-help hover:text-cyan-500 transition-colors">Y</span>
                                                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-44 p-2 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[9px] rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-tight text-center font-medium">
                                                        <div className="font-bold mb-1">Y = {models.graham.parameters?.bond_yield_proxy || '4.5'}%</div>
                                                        Current yield on AAA corporate bonds
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {models.graham.mc && (
                                        <div className="mt-8 pt-4">
                                            <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-bold mb-3">Probabilistic Scenarios (Monte Carlo)</p>
                                            <div className="grid grid-cols-3 gap-2">
                                                <div
                                                    className="bg-rose-500/5 relative overflow-hidden p-2 rounded-lg text-center cursor-pointer hover:bg-rose-500/10 transition-all group/mc"
                                                    onClick={() => setViewingDistribution('graham')}
                                                >
                                                    <div className="absolute -top-12 -right-12 w-24 h-24 blur-[30px] opacity-10 group-hover/mc:opacity-20 transition-opacity pointer-events-none bg-rose-500" />
                                                    <p className="text-[10px] text-rose-500 font-bold uppercase mb-1 relative z-10">Bear (10th)</p>
                                                    <p className="text-sm font-bold relative z-10">{formatCurrency(models.graham.mc.bear * fxRate, currency)}</p>
                                                </div>
                                                <div
                                                    className="bg-amber-500/5 relative overflow-hidden p-2 rounded-lg text-center cursor-pointer hover:bg-amber-500/10 transition-all group/mc shadow-none"
                                                    onClick={() => setViewingDistribution('graham')}
                                                >
                                                    <div className="absolute -top-12 -right-12 w-24 h-24 blur-[30px] opacity-10 group-hover/mc:opacity-20 transition-opacity pointer-events-none bg-amber-500" />
                                                    <p className="text-[10px] text-amber-500 font-bold uppercase mb-1 relative z-10">Median (50th)</p>
                                                    <p className="text-sm font-bold relative z-10">{formatCurrency(models.graham.mc.base * fxRate, currency)}</p>
                                                </div>
                                                <div
                                                    className="bg-emerald-500/5 relative overflow-hidden p-2 rounded-lg text-center cursor-pointer hover:bg-emerald-500/10 transition-all group/mc"
                                                    onClick={() => setViewingDistribution('graham')}
                                                >
                                                    <div className="absolute -top-12 -right-12 w-24 h-24 blur-[30px] opacity-10 group-hover/mc:opacity-20 transition-opacity pointer-events-none bg-emerald-500" />
                                                    <p className="text-[10px] text-emerald-500 font-bold uppercase mb-1 relative z-10">Bull (90th)</p>
                                                    <p className="text-sm font-bold relative z-10">{formatCurrency(models.graham.mc.bull * fxRate, currency)}</p>
                                                </div>
                                            </div>
                                            <p className="text-[9px] text-muted-foreground mt-2 text-center opacity-50 group-hover/mc:opacity-100 transition-opacity">Click to view distribution</p>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                )}

                <div className="bg-secondary/20 rounded-2xl p-6 italic text-sm text-muted-foreground text-center">
                    Note: Intrinsic value calculations are estimates based on various assumptions.
                    Actual stock performance may vary significantly.
                </div>
            </div>
        );
    };

    const ParamItem = ({ label, value, info, className }: { label: string, value: any, info?: { description: string, default: string }, className?: string }) => (
        <div>
            <div className="flex items-center gap-1 mb-1">
                <p className="text-[10px] text-muted-foreground uppercase tracking-widest font-bold">{label}</p>
                {info && (
                    <div className="group relative">
                        <HelpCircle className="w-2.5 h-2.5 text-muted-foreground/50 hover:text-indigo-500 cursor-help" />
                        <div className="absolute bottom-full left-0 mb-2 w-48 p-3 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[10px] rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100]">
                            {info.description}
                            <div className="mt-1 pt-1 font-bold text-indigo-600 dark:text-indigo-400">Default: {info.default}</div>
                        </div>
                    </div>
                )}
            </div>
            <p className={cn("text-sm font-semibold", className)}>{value ?? '-'}</p>
        </div>
    );

    return createPortal(
        <div className="fixed inset-0 z-[100] flex flex-col justify-end sm:justify-center items-center p-0 sm:p-4 isolate">
            <div className="absolute inset-0 bg-black/60" onClick={onClose} />

            <div
                className="relative w-full max-w-5xl h-[94vh] sm:h-auto sm:max-h-[90vh] rounded-t-[2.5rem] sm:rounded-[2rem] flex flex-col overflow-hidden animate-in slide-in-from-bottom sm:zoom-in-95 duration-300 bg-white dark:bg-zinc-950"
            >

                {/* Mobile Drag Handle */}
                <div className="sm:hidden w-full flex justify-center pt-3 pb-1 flex-shrink-0">
                    <div className="w-12 h-1.5 bg-secondary rounded-full" />
                </div>

                {/* Sticky Header & Tabs Container */}
                <div className="sticky top-0 z-50 bg-white/95 dark:bg-zinc-950/95 backdrop-blur-md flex-shrink-0">
                    {/* Header */}
                    <div className="p-4 sm:p-6 pb-2 sm:pb-3 flex justify-between items-start relative">
                        <div className="hidden sm:block absolute top-0 right-0 w-64 h-64 bg-indigo-500/10 rounded-full blur-[100px] -mr-32 -mt-32" />

                        <div className="flex items-center gap-4 sm:gap-6 relative z-10 text-foreground flex-1">
                            <div className="w-10 h-10 sm:w-16 sm:h-16 rounded-xl sm:rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-lg sm:text-3xl font-bold text-white overflow-hidden flex-shrink-0">
                                <StockIcon symbol={symbol} size="100%" className="w-full h-full p-2 bg-white" domain={domain} />
                            </div>
                            <div className="flex-1 min-w-0 pr-4">
                                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-1 sm:mb-2">
                                    <div className="flex items-center gap-2 sm:gap-3 min-w-0">
                                        <h2 className="text-lg sm:text-3xl font-black tracking-tight truncate shrink">{fundamentals?.shortName || symbol}</h2>
                                        <Badge className="bg-secondary text-secondary-foreground border-none font-mono text-[9px] sm:text-xs shrink-0">{symbol}</Badge>
                                    </div>
                                    {fundamentals?.regularMarketPrice && (
                                        <div className="flex items-baseline gap-1 text-indigo-600 dark:text-indigo-400">
                                            <span className="text-xl sm:text-3xl font-black tracking-tight tabular-nums">
                                                {formatCurrency(fundamentals.regularMarketPrice * fxRate, currency)}
                                            </span>
                                        </div>
                                    )}
                                </div>
                                <p className="text-muted-foreground flex items-center gap-1.5 sm:gap-2 text-[9px] sm:text-sm">
                                    <span className="font-semibold text-indigo-500">{fundamentals?.sector}</span>
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
                    <div className="px-4 sm:px-6 flex justify-around sm:justify-start gap-2 sm:gap-6 overflow-x-auto no-scrollbar">
                        <TabButton
                            active={activeTab === 'overview'}
                            onClick={() => setActiveTab('overview')}
                            icon={LayoutDashboard}
                            label="Overview"
                        />
                        <TabButton
                            active={activeTab === 'chart'}
                            onClick={() => setActiveTab('chart')}
                            icon={TrendingUp}
                            label="Chart"
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
                            </>
                        )}
                        <TabButton
                            active={activeTab === 'valuation'}
                            onClick={() => setActiveTab('valuation')}
                            icon={DollarSign}
                            label="Valuation"
                        />
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
                <div className="flex-1 overflow-y-auto p-4 sm:p-6 pt-2 sm:pt-4 custom-scrollbar">
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
                            <button onClick={() => loadData()} className="mt-6 px-6 py-2 bg-secondary hover:bg-muted rounded-full transition-colors">
                                Try Again
                            </button>
                        </div>
                    ) : (
                        <>
                            {activeTab === 'analysis' && renderAnalysis()}
                            {activeTab === 'overview' && renderOverview()}
                            {activeTab === 'chart' && renderChart()}
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
                            "w-full max-w-2xl rounded-3xl overflow-hidden animate-in zoom-in-95 duration-300",
                            isDarkMode ? "bg-slate-900 text-white" : "bg-white text-slate-900"
                        )}>
                            <div className={cn(
                                "p-6 flex items-center justify-between",
                                isDarkMode ? "bg-muted/30" : "bg-slate-50/50"
                            )}>
                                <div>
                                    <h3 className="text-xl font-bold flex items-center gap-2 text-inherit">
                                        <BarChart3 className="w-5 h-5 text-indigo-500" />
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
                                            const modelKey = viewingDistribution;
                                            if (!modelKey) return null;
                                            
                                            const model = intrinsicValue.models[modelKey];
                                            const mc = model?.mc;
                                            const rawHistogram = mc?.histogram || [];
                                            
                                            if (rawHistogram.length === 0 || !mc) return null;

                                            // Scale all values by fxRate
                                            const histogram = rawHistogram.map(h => ({ ...h, price: h.price * fxRate }));
                                            const scaledMc = {
                                                bear: (mc.bear ?? 0) * fxRate,
                                                base: (mc.base ?? 0) * fxRate,
                                                bull: (mc.bull ?? 0) * fxRate
                                            };

                                            const minPrice = histogram[0].price;
                                            const maxPrice = histogram[histogram.length - 1].price;
                                            const currentPrice = (intrinsicValue.current_price || fundamentals?.regularMarketPrice || 0) * fxRate;

                                            // Ensure the domain includes the current price
                                            const domainMin = Math.min(minPrice, currentPrice > 0 ? currentPrice * 0.95 : minPrice);
                                            const domainMax = Math.max(maxPrice, currentPrice > 0 ? currentPrice * 1.05 : maxPrice);

                                            const range = maxPrice - minPrice;

                                            const bearOffset = Math.max(0, Math.min(100, ((scaledMc.bear - minPrice) / range) * 100));
                                            const bullOffset = Math.max(0, Math.min(100, ((scaledMc.bull - minPrice) / range) * 100));

                                            return (
                                                <AreaChart
                                                    data={histogram}
                                                    margin={{ top: 35, right: 20, left: 20, bottom: 0 }}
                                                >
                                                    <defs>
                                                        <linearGradient id="colorBell" x1="0" y1="0" x2="1" y2="0" gradientUnits="userSpaceOnUse">
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
                                                        tickFormatter={(val) => formatCurrency(val, currency)}
                                                        fontSize={10}
                                                        tickLine={false}
                                                        axisLine={false}
                                                        minTickGap={30}
                                                        stroke={isDarkMode ? "#94a3b8" : "#64748b"}
                                                    />
                                                    <YAxis hide />
                                                    <Tooltip
                                                        wrapperStyle={{ opacity: 1, zIndex: 1000 }}
                                                        content={({ active, payload }) => {
                                                            if (active && payload && payload.length) {
                                                                const count = Number(payload[0].value);
                                                                const probability = (count / 10000) * 100;
                                                                return (
                                                                    <div className="p-3 rounded-xl outline-none scale-105 transition-transform bg-white/95 dark:bg-slate-950/95 backdrop-blur-md">
                                                                        <p className={cn(
                                                                            "text-[10px] uppercase font-bold mb-1 tracking-wider",
                                                                            isDarkMode ? "text-slate-500" : "text-slate-400"
                                                                        )}>Estimated Value</p>
                                                                        <p className={cn(
                                                                            "text-lg font-black",
                                                                            isDarkMode ? "text-white" : "text-slate-900"
                                                                        )}>{formatCurrency(payload[0].payload.price, currency)}</p>

                                                                        <div className="flex flex-col gap-1 mt-3 pt-2">
                                                                            <div className="flex items-center justify-between gap-4">
                                                                                <div className="flex items-center gap-1.5">
                                                                                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                                                                                    <span className={cn("text-[10px] font-bold uppercase", isDarkMode ? "text-slate-400" : "text-slate-500")}>Probability</span>
                                                                                </div>
                                                                                <span className="text-[10px] font-black text-indigo-500">{probability.toFixed(2)}%</span>
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
                                                    {scaledMc && (
                                                        <>
                                                            <ReferenceLine
                                                                x={scaledMc.bear}
                                                                stroke="#f43f5e"
                                                                strokeDasharray="4 4"
                                                                strokeWidth={2}
                                                                label={{ value: 'BEAR', position: 'top', fill: '#f43f5e', fontSize: 9, fontWeight: '900' }}
                                                            />
                                                            <ReferenceLine
                                                                x={scaledMc.base}
                                                                stroke="#06b6d4"
                                                                strokeDasharray="4 4"
                                                                strokeWidth={2}
                                                                label={{ value: 'MEDIAN', position: 'top', fill: '#06b6d4', fontSize: 9, fontWeight: '900' }}
                                                            />
                                                            <ReferenceLine
                                                                x={scaledMc.bull}
                                                                stroke="#10b981"
                                                                strokeDasharray="4 4"
                                                                strokeWidth={2}
                                                                label={{ value: 'BULL', position: 'top', fill: '#10b981', fontSize: 9, fontWeight: '900' }}
                                                            />
                                                        </>
                                                    )}
                                                    {/* Current Price Reference */}
                                                    {currentPrice > 0 && (
                                                        <ReferenceLine
                                                            x={currentPrice}
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
                                    {(() => {
                                        const modelKey = viewingDistribution;
                                        if (!modelKey) return null;
                                        const mc = intrinsicValue.models[modelKey]?.mc;
                                        if (!mc) return null;

                                        return (
                                            <>
                                                <div className="flex items-center gap-2">
                                                    <div className="w-2.5 h-2.5 rounded-full bg-rose-500" /> <span className="hidden sm:inline">Bear: </span><span className={isDarkMode ? "text-white" : "text-slate-900"}>{formatCurrency((mc.bear ?? 0) * fxRate, currency)}</span>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <div className="w-2.5 h-2.5 rounded-full bg-indigo-500" /> <span className="hidden sm:inline">Median: </span><span className={isDarkMode ? "text-white" : "text-slate-900"}>{formatCurrency((mc.base ?? 0) * fxRate, currency)}</span>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> <span className="hidden sm:inline">Bull: </span><span className={isDarkMode ? "text-white" : "text-slate-900"}>{formatCurrency((mc.bull ?? 0) * fxRate, currency)}</span>
                                                </div>
                                            </>
                                        );
                                    })()}
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Footer */}
                <div className="px-4 sm:px-8 py-3 sm:py-4 bg-muted/30 flex flex-col sm:flex-row justify-between items-center gap-2 text-[9px] sm:text-[10px] text-muted-foreground uppercase tracking-widest font-bold pb-[calc(0.75rem+env(safe-area-inset-bottom))] sm:pb-4">
                    <div className="flex gap-4">
                        <span>Exchange: {fundamentals?.exchange || 'Unknown'}</span>
                        <span>Currency: {fundamentals?.currency || 'USD'}</span>
                    </div>
                    <span className="text-center sm:text-right">Data by Yahoo Finance & Stock Alchemist</span>
                </div>
            </div>
        </div >,
        document.body
    );
}
function StatCard({ icon: Icon, label, value, subValue, color, valueColor, subValueColor, extra, rangeMin, rangeMax, rotate }: any) {
    return (
        <div className="bg-muted py-1.5 px-3 rounded-xl flex items-center gap-3 transition-all hover:bg-muted/50 group relative overflow-hidden">
            {/* Soft background glow */}
            <div className={cn(
                "absolute -top-8 -right-8 w-20 h-20 blur-[25px] opacity-10 transition-opacity duration-500 group-hover:opacity-20 pointer-events-none rounded-full",
                color?.includes('emerald') ? 'bg-emerald-500' :
                    color?.includes('rose') ? 'bg-rose-500' :
                        color?.includes('indigo') ? 'bg-indigo-500' :
                            color?.includes('amber') ? 'bg-amber-500' :
                                color?.includes('purple') ? 'bg-purple-500' :
                                    'bg-slate-500'
            )} />

            <div className={cn("p-2 rounded-lg bg-card relative z-10", color, rotate)}>
                <Icon className="w-4 h-4" />
            </div>
            <div className="flex-1 overflow-hidden relative z-10">
                <p className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider truncate">{label}</p>
                <div className="flex items-baseline gap-1.5">
                    <p className={cn("text-base font-bold tracking-tight whitespace-nowrap", valueColor || "text-foreground")}>{value}</p>
                    {subValue && (
                        <span className={cn("text-xs whitespace-nowrap", subValueColor)}>
                            {subValue}
                        </span>
                    )}
                </div>
                {(rangeMin && rangeMax) ? (
                    <p className="text-[10px] text-muted-foreground font-medium grayscale opacity-70">
                        Range: {rangeMin} - {rangeMax}
                    </p>
                ) : extra ? (
                    <div>
                        {extra}
                    </div>
                ) : null}
            </div>
        </div>
    );
}

function TabButton({ active, onClick, icon: Icon, label }: any) {
    return (
        <button
            onClick={onClick}
            className={cn(
                "py-4 px-4 flex items-center gap-2 text-sm font-medium transition-all relative border-b-2 outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/20",
                active ? "text-indigo-600 dark:text-indigo-400 border-indigo-500" : "text-muted-foreground hover:text-foreground border-transparent"
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
        <div className="bg-muted p-6 rounded-2xl">
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
                            wrapperStyle={{ opacity: 1, zIndex: 1000 }}
                            content={({ active, payload, label }) => {
                                if (active && payload && payload.length) {
                                    return (
                                        <div className="bg-white/95 dark:bg-slate-950/95 backdrop-blur-md p-3 rounded-xl text-xs">
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
