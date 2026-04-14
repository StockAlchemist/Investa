"use client";

import React, { useState, useEffect } from 'react';
import { Filter, RefreshCw, Loader2, Info, ChevronDown, Sparkles } from 'lucide-react';
import { getWatchlists, WatchlistMeta } from '@/lib/api';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";


interface ScreenerInputProps {
    onRunScreener: (universeType: string, universeId: string | null, manualSymbols: string[], narrativePrompt?: string) => void;
    isLoading: boolean;
}

const ScreenerInput: React.FC<ScreenerInputProps> = ({ onRunScreener, isLoading }) => {
    const [universeType, setUniverseType] = useState<string>("watchlist");
    const [watchlists, setWatchlists] = useState<WatchlistMeta[]>([]);
    const [selectedWatchlistId, setSelectedWatchlistId] = useState<string>("");
    const [manualSymbols, setManualSymbols] = useState<string>("");
    const [narrativePrompt, setNarrativePrompt] = useState<string>("");
    const [isFetchingLists, setIsFetchingLists] = useState(true);

    useEffect(() => {
        const fetchWatchlistsData = async () => {
            setIsFetchingLists(true);
            try {
                const data = await getWatchlists();
                setWatchlists(data);
                if (data && data.length > 0) {
                    setSelectedWatchlistId(data[0].id.toString());
                }
            } catch (e) {
                console.warn("Failed to fetch watchlists", e);
            } finally {
                setIsFetchingLists(false);
            }
        };
        fetchWatchlistsData();
    }, []);

    const handleRun = (e: React.FormEvent) => {
        e.preventDefault();
        const symbols = manualSymbols
            .split(/[\n,]+/)
            .map(s => s.trim())
            .filter(s => s.length > 0);

        onRunScreener(universeType, selectedWatchlistId || null, symbols, narrativePrompt);
    };

    return (
        <div className="metric-card card-shine p-6 relative overflow-hidden">
            {/* Accent bar - cyan */}
            <div className="absolute top-0 left-4 right-4 h-[2px] rounded-full bg-cyan-500 opacity-40" />
            <div className="pb-4">
                <div className="flex items-center gap-2">
                    <Filter className="w-4 h-4 text-[#0097b2]" />
                    <h3 className="section-label">Initial Parameters</h3>
                </div>
            </div>
            <div>
                <form onSubmit={handleRun} className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 items-end">

                        {/* Universe Strategy */}
                        <div className="space-y-2">
                            <label className="text-sm font-semibold text-muted-foreground whitespace-nowrap">Universe</label>
                            <div className="relative">
                                <select
                                    value={universeType}
                                    onChange={(e) => setUniverseType(e.target.value)}
                                    className="w-full h-10 px-3 bg-secondary rounded-md text-foreground font-medium appearance-none focus:outline-none focus:ring-1 focus:ring-cyan-500/50 cursor-pointer"
                                >
                                    <option value="watchlist" className="bg-card text-foreground">Watchlist</option>
                                    <option value="narrative" className="bg-card text-foreground text-cyan-400 font-bold">Narrative Search (AI) ✨</option>
                                    <option value="holdings" className="bg-card text-foreground">Holdings</option>
                                    <option value="sp500" className="bg-card text-foreground">S&P 500 (Large Cap)</option>
                                    <option value="sp400" className="bg-card text-foreground">S&P 400 (Mid Cap)</option>
                                    <option value="russell2000" className="bg-card text-foreground">Russell 2000 (Small Cap)</option>
                                    <option value="all" className="bg-card text-foreground">All Database Stocks</option>
                                    <option value="manual" className="bg-card text-foreground">Custom List</option>
                                </select>
                                <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
                            </div>
                        </div>

                        {/* Watchlist Selection */}
                        {universeType === "watchlist" && (
                            <div className="space-y-2">
                                <label className="text-sm font-semibold text-muted-foreground whitespace-nowrap">Target Portfolio</label>
                                <div className="relative">
                                    <select
                                        value={selectedWatchlistId}
                                        onChange={(e) => setSelectedWatchlistId(e.target.value)}
                                        className="w-full h-10 px-3 bg-secondary rounded-md text-foreground font-medium appearance-none focus:outline-none focus:ring-1 focus:ring-cyan-500/50 cursor-pointer"
                                    >
                                        {isFetchingLists ? (
                                            <option value="" disabled>Loading watchlists...</option>
                                        ) : watchlists.length > 0 ? (
                                            watchlists.map(w => (
                                                <option key={w.id} value={w.id.toString()} className="bg-card text-foreground">
                                                    {w.name}
                                                </option>
                                            ))
                                        ) : (
                                            <option value="" disabled>No watchlists found</option>
                                        )}
                                    </select>
                                    <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
                                </div>
                            </div>
                        )}

                        {/* Narrative Search Prompt */}
                        {universeType === "narrative" && (
                            <div className="space-y-2 lg:col-span-2">
                                <label className="text-sm font-semibold text-cyan-500/80 flex items-center gap-1.5">
                                    <Sparkles className="w-3 h-3" />
                                    AI Search Prompt
                                </label>
                                <Input
                                    value={narrativePrompt}
                                    onChange={(e) => setNarrativePrompt(e.target.value)}
                                    placeholder="e.g. Find high-growth tech stocks with margin of safety > 20%"
                                    className="bg-secondary/50 border-cyan-500/20 text-foreground font-medium h-10 placeholder:text-muted-foreground/50 focus-visible:ring-cyan-500/30"
                                />
                            </div>
                        )}

                        {/* Manual Symbols */}
                        {universeType === "manual" && (
                            <div className="space-y-2 lg:col-span-2">
                                <label className="text-sm font-semibold text-muted-foreground whitespace-nowrap">Manual Symbols</label>
                                <Input
                                    value={manualSymbols}
                                    onChange={(e) => setManualSymbols(e.target.value)}
                                    placeholder="e.g. AAPL, MSFT, NVIDIA"
                                    className="bg-secondary text-foreground font-medium h-10 placeholder:text-muted-foreground/50 border-none"
                                />
                            </div>
                        )}

                        {/* Execute Button */}
                        <div className={universeType === 'watchlist' ? '' : 'lg:col-span-1'}>
                            <Button
                                type="submit"
                                disabled={isLoading || (universeType === 'watchlist' && !selectedWatchlistId && !isFetchingLists)}
                                className="w-full h-10 bg-[#0097b2] hover:bg-[#00869e] text-white font-bold tracking-tight rounded-md transition-all flex items-center justify-center gap-2"
                            >
                                {isLoading ? (
                                    <>
                                        <Loader2 className="w-4 h-4 animate-spin text-white" />
                                        <span>AI is analyzing...</span>
                                    </>
                                ) : (
                                    <>
                                        {universeType === 'narrative' ? <Sparkles className="w-4 h-4" /> : <RefreshCw className="w-4 h-4" />}
                                        <span>{universeType === 'narrative' ? 'Search with AI' : 'Execute Screen'}</span>
                                    </>
                                )}
                            </Button>
                        </div>
                    </div>

                    {/* Information Note */}
                    <div className="flex items-start gap-2 p-3 rounded-lg bg-secondary/30">
                        {universeType === 'narrative' ? (
                            <>
                                <Sparkles className="w-4 h-4 text-cyan-400 mt-0.5 flex-shrink-0" />
                                <p className="text-[11px] font-medium text-muted-foreground leading-relaxed">
                                    <span className="text-cyan-400 font-bold">Narrative Search</span> uses Gemini to translate your natural language into a database query. It works best on stocks already in your local cache.
                                </p>
                            </>
                        ) : (
                            <>
                                <Info className="w-4 h-4 text-cyan-500 mt-0.5 flex-shrink-0" />
                                <p className="text-[11px] font-medium text-muted-foreground leading-relaxed">
                                    Screening large universes may take 1-5 minutes on the first run to build the local metadata cache (S&P 400 ~4m, Russell 2000 ~20m). Subsequent runs are instant.
                                </p>
                            </>
                        )}
                    </div>
                </form>
            </div>
        </div>
    );
};

export default ScreenerInput;
