"use client";

import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchWatchlist, addToWatchlist, removeFromWatchlist, WatchlistItem } from '@/lib/api';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Plus, Trash2, TrendingUp, TrendingDown, RefreshCw } from "lucide-react";
import { LineChart, Line, YAxis, ResponsiveContainer } from 'recharts';
import { Skeleton } from "@/components/ui/skeleton";
import { formatCurrency, formatPercent } from "@/lib/utils";
import StockTicker from './StockTicker';

interface WatchlistProps {
    currency: string;
}

export default function Watchlist({ currency }: WatchlistProps) {
    const queryClient = useQueryClient();
    const [newSymbol, setNewSymbol] = useState('');
    const [newNote, setNewNote] = useState('');

    const { data: watchlist, isLoading, isError, refetch } = useQuery({
        queryKey: ['watchlist', currency],
        queryFn: () => fetchWatchlist(currency),
    });

    const addMutation = useMutation({
        mutationFn: ({ symbol, note }: { symbol: string, note: string }) => addToWatchlist(symbol, note),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['watchlist'] });
            setNewSymbol('');
            setNewNote('');
        },
        onError: (error) => {
            alert(`Error adding to watchlist: ${error.message}`);
        }
    });

    const removeMutation = useMutation({
        mutationFn: (symbol: string) => removeFromWatchlist(symbol),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['watchlist'] });
        },
    });

    const handleAdd = (e: React.FormEvent) => {
        e.preventDefault();
        if (newSymbol.trim()) {
            addMutation.mutate({ symbol: newSymbol.trim().toUpperCase(), note: newNote });
        }
    };

    if (isLoading) {
        return (
            <div className="space-y-4">
                <Card className="bg-card border-border">
                    <CardHeader>
                        <Skeleton className="h-8 w-48" />
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-2">
                            {[1, 2, 3].map(i => <Skeleton key={i} className="h-12 w-full" />)}
                        </div>
                    </CardContent>
                </Card>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <Card className="bg-card border-border">
                <CardHeader className="flex flex-row items-center justify-between">
                    <div>
                        <CardTitle className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                            My Watchlist
                        </CardTitle>
                        <p className="text-sm text-muted-foreground mt-1">Track assets you're interested in</p>
                    </div>
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => refetch()}
                        className="text-muted-foreground hover:text-foreground"
                    >
                        <RefreshCw className="h-4 w-4" />
                    </Button>
                </CardHeader>
                <CardContent>
                    <form onSubmit={handleAdd} className="flex flex-col md:flex-row gap-3 mb-6">
                        <Input
                            placeholder="Symbol (e.g. AAPL, BTC-USD)"
                            value={newSymbol}
                            onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                            className="bg-secondary border-border text-foreground md:w-48"
                        />
                        <Input
                            placeholder="Note (optional)"
                            value={newNote}
                            onChange={(e) => setNewNote(e.target.value)}
                            className="bg-secondary border-border text-foreground flex-1"
                        />
                        <Button
                            type="submit"
                            disabled={addMutation.isPending || !newSymbol.trim()}
                            className="bg-cyan-600 hover:bg-cyan-500 text-white font-medium"
                        >
                            <Plus className="h-4 w-4 mr-2" />
                            Add
                        </Button>
                    </form>

                    <div className="overflow-x-auto rounded-lg border border-border">
                        <table className="min-w-full divide-y divide-border/50">
                            <thead className="bg-secondary/50 font-semibold border-b border-border">
                                <tr>
                                    <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Symbol</th>
                                    <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Name</th>
                                    <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground">Price</th>
                                    <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground">Day Change</th>
                                    <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">7D Trend</th>
                                    <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Note</th>
                                    <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-border/50 bg-transparent">
                                {watchlist?.length === 0 ? (
                                    <tr>
                                        <td colSpan={7} className="text-center py-12 text-muted-foreground text-sm">
                                            No symbols in your watchlist yet.
                                        </td>
                                    </tr>
                                ) : (
                                    watchlist?.map((item) => (
                                        <tr key={item.Symbol} className="hover:bg-accent/5 transition-colors">
                                            <td className="px-4 py-3 whitespace-nowrap text-sm">
                                                <StockTicker symbol={item.Symbol} currency={currency} />
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap text-foreground text-sm max-w-[200px] truncate">{item.Name || '-'}</td>
                                            <td className="px-4 py-3 whitespace-nowrap text-right font-mono font-medium text-sm text-foreground tabular-nums">
                                                {item.Price ? formatCurrency(item.Price, item.Currency || 'USD') : '-'}
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap text-right text-sm">
                                                <div className={`flex items-center justify-end font-mono tabular-nums ${(item["Day Change"] || 0) >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'
                                                    }`}>
                                                    {(item["Day Change"] || 0) >= 0 ? <TrendingUp className="h-3 w-3 mr-1" /> : <TrendingDown className="h-3 w-3 mr-1" />}
                                                    {item["Day Change %"] ? formatPercent(item["Day Change %"] / 100) : '-'}
                                                </div>
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap w-28">
                                                <div className="h-8 w-24">
                                                    {item.Sparkline && item.Sparkline.length > 1 ? (
                                                        <ResponsiveContainer width="100%" height="100%">
                                                            <LineChart data={item.Sparkline.map(v => ({ value: v }))}>
                                                                <YAxis hide domain={['dataMin', 'dataMax']} />
                                                                <Line
                                                                    type="monotone"
                                                                    dataKey="value"
                                                                    stroke={item.Sparkline[item.Sparkline.length - 1] >= item.Sparkline[0] ? "#10b981" : "#f43f5e"}
                                                                    strokeWidth={2}
                                                                    dot={false}
                                                                    isAnimationActive={false}
                                                                />
                                                            </LineChart>
                                                        </ResponsiveContainer>
                                                    ) : (
                                                        <span className="text-[10px] text-muted-foreground text-center block">no trend</span>
                                                    )}
                                                </div>
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap text-muted-foreground italic text-xs max-w-[150px] truncate">
                                                {item.Note}
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap text-right">
                                                <Button
                                                    variant="ghost"
                                                    size="icon"
                                                    onClick={() => removeMutation.mutate(item.Symbol)}
                                                    className="h-8 w-8 text-muted-foreground hover:text-rose-500 hover:bg-rose-500/10"
                                                >
                                                    <Trash2 className="h-4 w-4" />
                                                </Button>
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
