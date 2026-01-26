"use client";

import React, { useState, useEffect } from 'react';
import { Filter, RefreshCw, Loader2, Info, ChevronDown } from 'lucide-react';
import { getWatchlists, WatchlistMeta } from '@/lib/api';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface ScreenerInputProps {
    onRunScreener: (universeType: string, universeId: string | null, manualSymbols: string[]) => void;
    isLoading: boolean;
}

const ScreenerInput: React.FC<ScreenerInputProps> = ({ onRunScreener, isLoading }) => {
    const [universeType, setUniverseType] = useState<string>("watchlist");
    const [watchlists, setWatchlists] = useState<WatchlistMeta[]>([]);
    const [selectedWatchlistId, setSelectedWatchlistId] = useState<string>("");
    const [manualSymbols, setManualSymbols] = useState<string>("");
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

        onRunScreener(universeType, selectedWatchlistId || null, symbols);
    };

    return (
        <Card className="bg-card border-border shadow-sm">
            <CardHeader className="pb-4">
                <div className="flex items-center gap-2">
                    <Filter className="w-5 h-5 text-[#0097b2]" />
                    <CardTitle className="text-xl font-bold text-foreground">Initial Parameters</CardTitle>
                </div>
            </CardHeader>
            <CardContent>
                <form onSubmit={handleRun} className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 items-end">

                        {/* Universe Strategy */}
                        <div className="space-y-2">
                            <label className="text-sm font-semibold text-muted-foreground whitespace-nowrap">Universe Strategy</label>
                            <div className="relative">
                                <select
                                    value={universeType}
                                    onChange={(e) => setUniverseType(e.target.value)}
                                    className="w-full h-10 px-3 bg-secondary border border-border rounded-md text-foreground font-medium appearance-none focus:outline-none focus:ring-1 focus:ring-cyan-500/50 cursor-pointer"
                                >
                                    <option value="watchlist" className="bg-card text-foreground">My Watchlist</option>
                                    <option value="holdings" className="bg-card text-foreground">My Current Holdings</option>
                                    <option value="sp500" className="bg-card text-foreground">S&P 500 Benchmarks</option>
                                    <option value="manual" className="bg-card text-foreground">Custom Ticker List</option>
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
                                        className="w-full h-10 px-3 bg-secondary border border-border rounded-md text-foreground font-medium appearance-none focus:outline-none focus:ring-1 focus:ring-cyan-500/50 cursor-pointer"
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

                        {/* Manual Symbols */}
                        {universeType === "manual" && (
                            <div className="space-y-2 lg:col-span-2">
                                <label className="text-sm font-semibold text-muted-foreground whitespace-nowrap">Manual Symbols</label>
                                <Input
                                    value={manualSymbols}
                                    onChange={(e) => setManualSymbols(e.target.value)}
                                    placeholder="e.g. AAPL, MSFT, NVIDIA"
                                    className="bg-secondary border-border text-foreground font-medium h-10 placeholder:text-muted-foreground/50"
                                />
                            </div>
                        )}

                        {/* Execute Button */}
                        <div className={universeType === 'watchlist' ? '' : 'lg:col-span-1'}>
                            <Button
                                type="submit"
                                disabled={isLoading || (universeType === 'watchlist' && !selectedWatchlistId && !isFetchingLists)}
                                className="w-full h-10 bg-[#0097b2] hover:bg-[#00869e] text-white font-bold tracking-tight rounded-md shadow-sm transition-all flex items-center justify-center gap-2"
                            >
                                {isLoading ? (
                                    <>
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        <span>Scanning Market...</span>
                                    </>
                                ) : (
                                    <>
                                        <RefreshCw className="w-4 h-4" />
                                        <span>Execute Screen</span>
                                    </>
                                )}
                            </Button>
                        </div>
                    </div>

                    {/* Information Note */}
                    <div className="flex items-start gap-2 p-3 rounded-lg bg-secondary/30 border border-border/50">
                        <Info className="w-4 h-4 text-cyan-500 mt-0.5 flex-shrink-0" />
                        <p className="text-[11px] font-medium text-muted-foreground leading-relaxed">
                            Screening large universes like the S&P 500 may take up to 60 seconds on the first run to build the local metadata cache. Results are sorted by Margin of Safety by default.
                        </p>
                    </div>
                </form>
            </CardContent>
        </Card>
    );
};

export default ScreenerInput;
