"use client";

import React, { useState, useEffect } from 'react';
import ScreenerInput from './ScreenerInput';
import ScreenerResults from './ScreenerResults';
import { Telescope } from 'lucide-react';
import { runScreener, fetchScreenerReview, fetchWatchlist, addToWatchlist, removeFromWatchlist, getWatchlists } from '@/lib/api';
import { useQuery, useMutation, useQueryClient, useQueries } from '@tanstack/react-query';

interface ScreenerViewProps {
    currency: string;
}

const ScreenerView: React.FC<ScreenerViewProps> = ({ currency }) => {
    const [results, setResults] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [reviewingSymbol, setReviewingSymbol] = useState<string | null>(null);
    const [isInitialized, setIsInitialized] = useState(false);
    const queryClient = useQueryClient();

    // Fetch all watchlists metadata
    const { data: watchlists = [] } = useQuery({
        queryKey: ['watchlists'],
        queryFn: ({ signal }) => getWatchlists(signal),
    });

    // Fetch all items from all watchlists to build a symbol mapping
    const watchlistQueries = useQueries({
        queries: watchlists.map(wl => ({
            queryKey: ['watchlist', currency, wl.id],
            queryFn: ({ signal }: { signal?: AbortSignal }) => fetchWatchlist(currency, wl.id, signal),
            staleTime: 1000 * 60 * 5, // 5 minutes
        }))
    });

    const symbolWatchlistMap = React.useMemo(() => {
        const map: Record<string, Set<number>> = {};
        watchlistQueries.forEach((query, index) => {
            if (query.data) {
                const watchlistId = watchlists[index].id;
                query.data.forEach(item => {
                    if (!map[item.Symbol]) map[item.Symbol] = new Set();
                    map[item.Symbol].add(watchlistId);
                });
            }
        });
        return map;
    }, [watchlistQueries, watchlists]);

    const starredSymbols = React.useMemo(() => {
        return new Set(Object.keys(symbolWatchlistMap));
    }, [symbolWatchlistMap]);

    const addMutation = useMutation({
        mutationFn: ({ symbol, watchlistId }: { symbol: string, watchlistId: number }) => addToWatchlist(symbol, "", watchlistId),
        onSuccess: (_, variables) => {
            queryClient.invalidateQueries({ queryKey: ['watchlist', currency, variables.watchlistId] });
        },
    });

    const removeMutation = useMutation({
        mutationFn: ({ symbol, watchlistId }: { symbol: string, watchlistId: number }) => removeFromWatchlist(symbol, watchlistId),
        onSuccess: (_, variables) => {
            queryClient.invalidateQueries({ queryKey: ['watchlist', currency, variables.watchlistId] });
        },
    });

    const handleToggleWatchlist = (symbol: string, watchlistId: number) => {
        if (symbolWatchlistMap[symbol]?.has(watchlistId)) {
            removeMutation.mutate({ symbol, watchlistId });
        } else {
            addMutation.mutate({ symbol, watchlistId });
        }
    };

    useEffect(() => {
        const handleUpdate = (e: any) => {
            const { symbol, analysis } = e.detail;

            // Calculate average AI score live
            let aiScore = null;
            if (analysis && analysis.scorecard) {
                const vals = Object.values(analysis.scorecard).filter(v => typeof v === 'number') as number[];
                if (vals.length > 0) {
                    aiScore = vals.reduce((a, b) => a + b, 0) / vals.length;
                }
            }

            // If intrinsic value data was returned (from backend manual analysis), update it too
            const ivData = analysis?.intrinsic_value_data;

            setResults(prev => prev.map(item => {
                if (item.symbol === symbol) {
                    return {
                        ...item,
                        has_ai_review: true,
                        ai_score: aiScore,
                        ...(ivData ? {
                            intrinsic_value: ivData.average_intrinsic_value,
                            margin_of_safety: ivData.margin_of_safety_pct
                        } : {})
                    };
                }
                return item;
            }));
        };

        window.addEventListener('stock-analysis-updated', handleUpdate as EventListener);
        return () => window.removeEventListener('stock-analysis-updated', handleUpdate as EventListener);
    }, []);

    const [isRefreshing, setIsRefreshing] = useState(false);

    // ... (existing code: fetchWatchlist, etc.)

    const handleRunScreener = async (universeType: string, universeId: string | null, manualSymbols: string[]) => {
        setIsLoading(true);
        setResults([]); // Clear previous results
        try {
            // PHASE 1: Fast Load (Cache Only)
            const fastData = await runScreener({
                universe_type: universeType,
                universe_id: universeId,
                manual_symbols: manualSymbols,
                fast_mode: true
            });

            if (fastData && fastData.length > 0) {
                // Deduplicate by symbol to prevent key errors
                const uniqueFastData = Array.from(new Map(fastData.map((item: any) => [item.symbol, item])).values());
                setResults(uniqueFastData);
                setIsLoading(false); // Stop main loading spinner, show content
                setIsRefreshing(true); // Start background refresh indicator
            }

            // PHASE 2: Fresh Load (Live Data)
            // Even if we showed usage cache, we fetch fresh to ensure accuracy
            const freshData = await runScreener({
                universe_type: universeType,
                universe_id: universeId,
                manual_symbols: manualSymbols,
                fast_mode: false
            });

            // Deduplicate by symbol
            const uniqueFreshData = Array.from(new Map(freshData.map((item: any) => [item.symbol, item])).values());
            setResults(uniqueFreshData);
        } catch (e) {
            console.error("Screening error", e);
        } finally {
            setIsLoading(false);
            setIsRefreshing(false);
        }
    };

    const handleReview = async (symbol: string, force: boolean = false) => {
        setReviewingSymbol(symbol);
        try {
            const data = await fetchScreenerReview(symbol, force);
            if (data) {
                // Dispatch event so all components (and our own listener) update live
                window.dispatchEvent(new CustomEvent('stock-analysis-updated', {
                    detail: { symbol, analysis: data }
                }));
            }
            return data;
        } catch (e) {
            console.error("Review error", e);
        } finally {
            setReviewingSymbol(null);
        }
        return null;
    };

    return (
        <div className="space-y-6 animate-in fade-in duration-500">
            <header className="space-y-2">
                <h2 className="text-2xl font-bold leading-none tracking-tight bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent w-fit">
                    Market Screener
                </h2>
                <p className="text-muted-foreground text-sm font-medium max-w-2xl leading-relaxed">
                    Identify high-probability investment opportunities using quantitative <span className="text-cyan-500/80">intrinsic value models</span> and <span className="text-blue-500/80">AI-powered fundamental audits</span>.
                </p>
            </header>

            <div className="space-y-6">
                <ScreenerInput onRunScreener={handleRunScreener} isLoading={isLoading} />

                {isRefreshing && (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground bg-muted/50 px-3 py-1.5 rounded-md w-fit animate-pulse">
                        <Telescope className="h-3 w-3" />
                        <span>Updating live prices...</span>
                    </div>
                )}

                <ScreenerResults
                    results={results}
                    onReview={handleReview}
                    reviewingSymbol={reviewingSymbol}
                    currency={currency}
                />
            </div>
        </div>
    );
};

export default ScreenerView;
