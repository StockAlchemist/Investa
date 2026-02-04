"use client";

import React, { useState, useMemo, useEffect } from 'react';
import { BrainCircuit, Loader2, Sparkles, ChevronRight, BarChart3, TrendingUp, TrendingDown, Target, ChevronUp, RotateCcw, Search, SlidersHorizontal, X, Filter } from 'lucide-react';
import WatchlistStar from './WatchlistStar';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { formatCurrency, formatCompactNumber, formatPercent, cn } from "@/lib/utils";

import { useStockModal } from '@/context/StockModalContext';

interface ScreenResult {
    symbol: string;
    name: string;
    price: number;
    intrinsic_value: number | null;
    margin_of_safety: number | null;
    pe_ratio: number | null;
    market_cap: number | null;
    sector: string | null;
    has_ai_review: boolean;
    ai_score: number | null;
}

interface ScreenerResultsProps {
    results: ScreenResult[];
    onReview: (symbol: string, force?: boolean) => Promise<any>;
    reviewingSymbol: string | null;
    currency: string;
}

type SortKey = 'symbol' | 'price' | 'intrinsic_value' | 'margin_of_safety' | 'pe_ratio' | 'ai_score';
type SortDirection = 'asc' | 'desc';

interface SortConfig {
    key: SortKey;
    direction: SortDirection;
}

const ScreenerResults: React.FC<ScreenerResultsProps> = ({ results, onReview, reviewingSymbol, currency }) => {
    const [expandedReview, setExpandedReview] = useState<string | null>(null);
    const [reviews, setReviews] = useState<Record<string, any>>({});
    const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'margin_of_safety', direction: 'desc' });
    const { openStockDetail } = useStockModal();

    // Filter States
    const [searchQuery, setSearchQuery] = useState("");
    const [minMOS, setMinMOS] = useState<number | "">("");
    const [maxPE, setMaxPE] = useState<number | "">("");
    const [marketCapCategory, setMarketCapCategory] = useState<string>("all");
    const [onlyAI, setOnlyAI] = useState(false);
    const [showFilters, setShowFilters] = useState(false);

    const [localResults, setLocalResults] = useState<ScreenResult[]>(results);

    // Sync local state with props when results change (e.g. new scan)
    useEffect(() => {
        setLocalResults(results);
    }, [results]);

    // Listen for live updates from StockDetailModal
    useEffect(() => {
        const handleIntrinsicUpdate = (event: CustomEvent) => {
            const { symbol, data } = event.detail;
            setLocalResults(prev => prev.map(item => {
                if (item.symbol === symbol) {
                    return {
                        ...item,
                        intrinsic_value: data.average_intrinsic_value,
                        margin_of_safety: data.margin_of_safety_pct,
                        // Update PE if it comes back (it might not be in IV response directly, but usually we just want IV/MOS)
                    };
                }
                return item;
            }));
        };

        window.addEventListener('stock-intrinsic-value-updated' as any, handleIntrinsicUpdate as any);
        return () => {
            window.removeEventListener('stock-intrinsic-value-updated' as any, handleIntrinsicUpdate as any);
        };
    }, []);

    const filteredResults = useMemo(() => {
        return localResults.filter(item => {
            // Search Filter
            if (searchQuery) {
                const query = searchQuery.toLowerCase();
                if (!item.symbol.toLowerCase().includes(query) && !item.name.toLowerCase().includes(query)) {
                    return false;
                }
            }

            // MOS Filter
            if (minMOS !== "" && (item.margin_of_safety === null || item.margin_of_safety < minMOS)) {
                return false;
            }

            // PE Filter
            if (maxPE !== "" && (item.pe_ratio === null || item.pe_ratio > maxPE)) {
                return false;
            }

            // AI Filter
            if (onlyAI && !item.has_ai_review) {
                return false;
            }

            // Market Cap Filter
            if (marketCapCategory !== "all") {
                const cap = item.market_cap || 0;
                if (marketCapCategory === "micro" && cap >= 300000000) return false;
                if (marketCapCategory === "small" && (cap < 300000000 || cap >= 2000000000)) return false;
                if (marketCapCategory === "mid" && (cap < 2000000000 || cap >= 10000000000)) return false;
                if (marketCapCategory === "large" && (cap < 10000000000 || cap >= 200000000000)) return false;
                if (marketCapCategory === "mega" && cap < 200000000000) return false;
            }

            return true;
        });
    }, [localResults, searchQuery, minMOS, maxPE, onlyAI, marketCapCategory]);

    const sortedResults = useMemo(() => {
        const sorted = [...filteredResults];
        sorted.sort((a, b) => {
            const aVal = a[sortConfig.key];
            const bVal = b[sortConfig.key];

            if (aVal === bVal) return 0;
            if (aVal === null || aVal === undefined) return 1;
            if (bVal === null || bVal === undefined) return -1;

            if (sortConfig.direction === 'asc') {
                return aVal < bVal ? -1 : 1;
            } else {
                return aVal > bVal ? -1 : 1;
            }
        });
        return sorted;
    }, [filteredResults, sortConfig]);

    const handleSort = (key: SortKey) => {
        setSortConfig(prev => ({
            key,
            direction: prev.key === key && prev.direction === 'desc' ? 'asc' : 'desc'
        }));
    };

    const handleReviewClick = async (symbol: string) => {
        if (reviews[symbol]) {
            setExpandedReview(expandedReview === symbol ? null : symbol);
            return;
        }

        const data = await onReview(symbol);
        if (data) {
            setReviews(prev => ({ ...prev, [symbol]: data }));
            setExpandedReview(symbol);
        }
    };

    const handleRegenerate = async (symbol: string) => {
        const data = await onReview(symbol, true);
        if (data) {
            setReviews(prev => ({ ...prev, [symbol]: data }));
        }
    };

    const SortIndicator = ({ column }: { column: SortKey }) => {
        if (sortConfig.key !== column) return <ChevronUp className="w-3 h-3 opacity-0 group-hover:opacity-30 ml-1" />;
        return sortConfig.direction === 'asc'
            ? <ChevronUp className="w-3 h-3 ml-1 text-primary" />
            : <ChevronUp className="w-3 h-3 ml-1 text-primary rotate-180 transition-transform" />;
    };

    // No longer need click outside here, handled by WatchlistStar
    /*
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setOpenWatchlistDropdown(null);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, []);
    */

    if (!results || results.length === 0) return null;

    return (
        <Card className="bg-card shadow-sm overflow-hidden">
            <CardHeader className="pb-2">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                    <div className="flex items-center gap-2">
                        <Target className="w-5 h-5 text-purple-500" />
                        <CardTitle className="text-xl font-bold text-foreground">Scan Results</CardTitle>
                        <Badge variant="secondary" className="ml-2 bg-secondary/50 text-muted-foreground border-none">
                            {filteredResults.length} of {localResults.length}
                        </Badge>
                    </div>
                    <div className="flex items-center gap-2 w-full sm:w-auto">
                        <div className="relative group flex-1 sm:flex-initial">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground group-focus-within:text-cyan-500 transition-colors" />
                            <Input
                                placeholder="Search symbol or name..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="pl-9 h-9 w-full sm:w-[240px] bg-secondary/50 border-border focus:ring-cyan-500/20 text-foreground"
                            />
                            {searchQuery && (
                                <button
                                    onClick={() => setSearchQuery("")}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                                >
                                    <X className="w-3.5 h-3.5" />
                                </button>
                            )}
                        </div>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setShowFilters(!showFilters)}
                            className={cn(
                                "h-9 gap-2 border-border shrink-0",
                                showFilters && "bg-secondary border-cyan-500/50 text-cyan-500"
                            )}
                        >
                            <SlidersHorizontal className="w-4 h-4" />
                            <span className="inline">Filters</span>
                        </Button>
                    </div>
                </div>

                {showFilters && (
                    <div className="mt-4 p-4 rounded-xl bg-secondary/20 border-border/50 animate-in fade-in slide-in-from-top-2 duration-200">
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                            {/* MOS Filter */}
                            <div className="space-y-2">
                                <label className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Min Margin of Safety (%)</label>
                                <Input
                                    type="number"
                                    placeholder="e.g. 15"
                                    value={minMOS}
                                    onChange={(e) => setMinMOS(e.target.value === "" ? "" : Number(e.target.value))}
                                    className="h-9 bg-background/50 border-border text-foreground"
                                />
                            </div>

                            {/* PE Filter */}
                            <div className="space-y-2">
                                <label className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Max P/E Ratio</label>
                                <Input
                                    type="number"
                                    placeholder="e.g. 25"
                                    value={maxPE}
                                    onChange={(e) => setMaxPE(e.target.value === "" ? "" : Number(e.target.value))}
                                    className="h-9 bg-background/50 border-border text-foreground"
                                />
                            </div>

                            {/* Market Cap Filter */}
                            <div className="space-y-2">
                                <label className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Market Cap</label>
                                <select
                                    value={marketCapCategory}
                                    onChange={(e) => setMarketCapCategory(e.target.value)}
                                    className="w-full h-9 px-3 bg-background/50 border border-border rounded-md text-sm font-medium focus:outline-none focus:ring-1 focus:ring-cyan-500/50 text-foreground"
                                >
                                    <option value="all">All Sizes</option>
                                    <option value="micro">Micro Cap (&lt; $300M)</option>
                                    <option value="small">Small Cap ($300M - $2B)</option>
                                    <option value="mid">Mid Cap ($2B - $10B)</option>
                                    <option value="large">Large Cap ($10B - $200B)</option>
                                    <option value="mega">Mega Cap (&gt; $200B)</option>
                                </select>
                            </div>

                            {/* AI Filter & Reset */}
                            <div className="flex items-end gap-3">
                                <button
                                    onClick={() => setOnlyAI(!onlyAI)}
                                    className={cn(
                                        "flex-1 h-9 px-3 rounded-md border text-xs font-bold uppercase tracking-wider transition-all flex items-center justify-center gap-2",
                                        onlyAI
                                            ? "bg-purple-500/10 border-purple-500/30 text-purple-500"
                                            : "bg-background/50 border-border text-muted-foreground hover:bg-muted"
                                    )}
                                >
                                    <BrainCircuit className="w-3.5 h-3.5" />
                                    AI Reviewed
                                </button>
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => {
                                        setSearchQuery("");
                                        setMinMOS("");
                                        setMaxPE("");
                                        setMarketCapCategory("all");
                                        setOnlyAI(false);
                                    }}
                                    className="h-9 text-muted-foreground hover:text-red-500"
                                >
                                    Reset
                                </Button>
                            </div>
                        </div>
                    </div>
                )}
            </CardHeader>
            <CardContent className="p-0">
                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="bg-secondary/30 border-y border-border">
                                <th
                                    className="px-6 py-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider cursor-pointer group hover:bg-secondary/50 transition-colors"
                                    onClick={() => handleSort('symbol')}
                                >
                                    <div className="flex items-center">
                                        Asset <SortIndicator column="symbol" />
                                    </div>
                                </th>
                                <th
                                    className="px-6 py-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right cursor-pointer group hover:bg-secondary/50 transition-colors"
                                    onClick={() => handleSort('price')}
                                >
                                    <div className="flex items-center justify-end">
                                        Price <SortIndicator column="price" />
                                    </div>
                                </th>
                                <th
                                    className="px-6 py-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right cursor-pointer group hover:bg-secondary/50 transition-colors"
                                    onClick={() => handleSort('intrinsic_value')}
                                >
                                    <div className="flex items-center justify-end">
                                        Intrinsic <SortIndicator column="intrinsic_value" />
                                    </div>
                                </th>
                                <th
                                    className="px-6 py-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right cursor-pointer group hover:bg-secondary/50 transition-colors"
                                    onClick={() => handleSort('margin_of_safety')}
                                >
                                    <div className="flex items-center justify-end">
                                        MOS <SortIndicator column="margin_of_safety" />
                                    </div>
                                </th>
                                <th
                                    className="px-6 py-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right cursor-pointer group hover:bg-secondary/50 transition-colors"
                                    onClick={() => handleSort('pe_ratio')}
                                >
                                    <div className="flex items-center justify-end">
                                        P/E <SortIndicator column="pe_ratio" />
                                    </div>
                                </th>
                                <th
                                    className="px-6 py-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right cursor-pointer group hover:bg-secondary/50 transition-colors"
                                    onClick={() => handleSort('ai_score')}
                                >
                                    <div className="flex items-center justify-end">
                                        AI Score <SortIndicator column="ai_score" />
                                    </div>
                                </th>
                                <th className="px-6 py-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right">AI Audit</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-border/50">
                            {sortedResults.map((row) => (
                                <React.Fragment key={row.symbol}>
                                    <tr className="hover:bg-secondary/20 transition-all group border-b border-border/10">
                                        <td className="px-6 py-4">
                                            <div className="flex items-center gap-3">
                                                <WatchlistStar symbol={row.symbol} />
                                                <button
                                                    onClick={() => openStockDetail(row.symbol, currency)}
                                                    className="flex flex-col items-start hover:text-cyan-500 transition-colors text-left group"
                                                >
                                                    <div className="font-bold text-foreground text-sm tracking-tight group-hover:text-cyan-500">{row.symbol}</div>
                                                    <div className="text-[11px] font-medium text-muted-foreground truncate max-w-[160px] group-hover:text-cyan-500/80">{row.name}</div>
                                                </button>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-right font-mono font-medium text-sm text-foreground tabular-nums">
                                            {formatCurrency(row.price, 'USD')}
                                        </td>
                                        <td className="px-6 py-4 text-right tabular-nums text-sm font-medium text-muted-foreground">
                                            {typeof row.intrinsic_value === 'number' ? formatCurrency(row.intrinsic_value, 'USD') : '-'}
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            {row.margin_of_safety !== null ? (
                                                <div className={cn(
                                                    "flex items-center justify-end font-mono font-bold text-sm",
                                                    row.margin_of_safety > 15 ? 'text-emerald-600 dark:text-emerald-400' :
                                                        row.margin_of_safety > 0 ? 'text-cyan-600 dark:text-cyan-400' : 'text-red-600 dark:text-red-500'
                                                )}>
                                                    {row.margin_of_safety > 0 ? <TrendingUp className="h-3 w-3 mr-1" /> : <TrendingDown className="h-3 w-3 mr-1" />}
                                                    {typeof row.margin_of_safety === 'number' ? formatPercent(row.margin_of_safety / 100) : '-'}
                                                </div>
                                            ) : (
                                                <div className="text-right">
                                                    <span className="text-muted-foreground/30 text-xs font-medium">N/A</span>
                                                </div>
                                            )}
                                        </td>
                                        <td className="px-6 py-4 text-right font-mono text-sm text-muted-foreground tabular-nums">
                                            {typeof row.pe_ratio === 'number' ? row.pe_ratio.toFixed(1) : '-'}
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            {row.ai_score !== null && row.ai_score !== undefined ? (
                                                <div className={cn(
                                                    "flex items-center justify-end font-mono font-bold text-sm",
                                                    row.ai_score >= 8 ? 'text-emerald-500' :
                                                        row.ai_score >= 6 ? 'text-cyan-500' :
                                                            row.ai_score >= 4 ? 'text-amber-500' : 'text-red-500'
                                                )}>
                                                    {typeof row.ai_score === 'number' ? `${row.ai_score.toFixed(1)}/10` : 'N/A'}
                                                </div>
                                            ) : (
                                                <div className="text-right">
                                                    <span className="text-muted-foreground/30 text-xs font-medium">N/A</span>
                                                </div>
                                            )}
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            <button
                                                onClick={() => handleReviewClick(row.symbol)}
                                                disabled={reviewingSymbol === row.symbol}
                                                className={cn(
                                                    "text-[11px] font-bold uppercase tracking-wider px-3 py-1.5 rounded-md transition-all border inline-flex items-center gap-1.5",
                                                    row.has_ai_review
                                                        ? 'bg-purple-500/10 border-purple-500/20 text-purple-600 hover:bg-purple-500/20 dark:text-purple-400'
                                                        : 'bg-muted border-border text-muted-foreground hover:bg-muted/80'
                                                )}
                                            >
                                                {reviewingSymbol === row.symbol ? (
                                                    <>
                                                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                                        <span>Analyzing</span>
                                                    </>
                                                ) : row.has_ai_review ? (
                                                    <>
                                                        <Sparkles className="w-3.5 h-3.5" />
                                                        <span>Review</span>
                                                    </>
                                                ) : (
                                                    <>
                                                        <BrainCircuit className="w-3.5 h-3.5" />
                                                        <span>Analyze</span>
                                                    </>
                                                )}
                                            </button>
                                        </td>
                                    </tr>
                                    {expandedReview === row.symbol && reviews[row.symbol] && (
                                        <tr className="bg-secondary/5">
                                            <td colSpan={6} className="p-0">
                                                <div className="p-6 border-t border-border animate-in fade-in slide-in-from-top-2 duration-300">
                                                    <div className="bg-card rounded-2xl p-6 shadow-sm space-y-6">
                                                        <div className="flex flex-col lg:flex-row justify-between items-start gap-4">
                                                            <div className="flex items-center gap-3">
                                                                <div className="p-2 rounded-lg bg-purple-500/10 text-purple-500">
                                                                    <Sparkles className="w-5 h-5" />
                                                                </div>
                                                                <div className="flex flex-col">
                                                                    <h3 className="text-lg font-bold text-foreground">AI Technical & Fundamental Audit</h3>
                                                                    <button
                                                                        onClick={() => handleRegenerate(row.symbol)}
                                                                        disabled={reviewingSymbol === row.symbol}
                                                                        className="flex items-center gap-1.5 text-[10px] font-bold text-purple-600 hover:text-purple-700 dark:text-purple-400 dark:hover:text-purple-300 transition-colors uppercase tracking-wider mt-0.5"
                                                                    >
                                                                        {reviewingSymbol === row.symbol ? (
                                                                            <Loader2 className="w-3 h-3 animate-spin" />
                                                                        ) : (
                                                                            <RotateCcw className="w-3 h-3" />
                                                                        )}
                                                                        Regenerate Analysis
                                                                    </button>
                                                                </div>
                                                            </div>
                                                            <div className="flex flex-wrap gap-3">
                                                                {Object.entries(reviews[row.symbol].scorecard || {}).map(([k, v]: [string, any]) => (
                                                                    <div key={k} className="px-3 py-1.5 rounded-lg bg-secondary/50 border border-border flex items-center gap-2">
                                                                        <span className="text-[10px] uppercase text-muted-foreground font-bold tracking-wider">{k.replace('_', ' ')}</span>
                                                                        <span className={cn(
                                                                            "text-xs font-bold",
                                                                            v >= 8 ? 'text-emerald-500' : v >= 6 ? 'text-cyan-500' : 'text-amber-500'
                                                                        )}>{v}/10</span>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>

                                                        <div className="p-4 rounded-xl bg-muted/30 border-l-4 border-purple-500/50">
                                                            <p className="text-sm text-foreground/90 leading-relaxed font-semibold italic">
                                                                "{reviews[row.symbol].summary}"
                                                            </p>
                                                        </div>

                                                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                                            {Object.entries(reviews[row.symbol].analysis || {}).map(([k, v]: [string, any]) => (
                                                                <div key={k} className="space-y-1.5">
                                                                    <h4 className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                                                                        <ChevronRight className="w-3 h-3 text-purple-500" />
                                                                        {k.replace('_', ' ')}
                                                                    </h4>
                                                                    <p className="text-foreground/80 text-xs font-medium leading-relaxed">{v}</p>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div>
                                                </div>
                                            </td>
                                        </tr>
                                    )}
                                </React.Fragment>
                            ))}
                        </tbody>
                    </table>
                    {results.length === 0 && (
                        <div className="p-16 text-center flex flex-col items-center gap-3">
                            <BarChart3 className="w-10 h-10 text-muted-foreground/20" />
                            <div>
                                <p className="text-foreground font-bold text-base tracking-tight">No results yet</p>
                                <p className="text-muted-foreground text-xs font-medium">Configure parameters and execute scan to discover opportunities.</p>
                            </div>
                        </div>
                    )}
                </div>
            </CardContent>
        </Card>
    );
};

export default ScreenerResults;
