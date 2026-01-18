"use client";

import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
    fetchWatchlist,
    addToWatchlist,
    removeFromWatchlist,
    WatchlistItem,
    getWatchlists,
    createWatchlist,
    deleteWatchlist,
    renameWatchlist,
    WatchlistMeta
} from '@/lib/api';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Plus, Trash2, TrendingUp, TrendingDown, RefreshCw, Pencil, Check, X, ListPlus } from "lucide-react";
import { AreaChart, Area, YAxis, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Skeleton } from "@/components/ui/skeleton";
import { formatCurrency, formatPercent, formatCompactNumber, cn } from "@/lib/utils";
import StockTicker from './StockTicker';

interface WatchlistProps {
    currency: string;
}

export default function Watchlist({ currency }: WatchlistProps) {
    const queryClient = useQueryClient();
    const [activeWatchlistId, setActiveWatchlistId] = useState<number>(1);
    const [isCreating, setIsCreating] = useState(false);
    const [newListName, setNewListName] = useState("");

    // Rename State
    const [isRenaming, setIsRenaming] = useState(false);
    const [renameName, setRenameName] = useState("");

    // Create Item States
    const [newSymbol, setNewSymbol] = useState('');
    const [newNote, setNewNote] = useState('');

    // Fetch Watchlists Metadata
    const { data: watchlists = [], isLoading: isLoadingLists } = useQuery({
        queryKey: ['watchlists'],
        queryFn: ({ signal }) => getWatchlists(signal),
    });

    // Fetch Active Watchlist Items
    const { data: watchlist, isLoading: isLoadingItems, isError, refetch } = useQuery({
        queryKey: ['watchlist', currency, activeWatchlistId],
        queryFn: ({ signal }) => fetchWatchlist(currency, activeWatchlistId, signal),
    });

    // Find current watchlist object for display name usually
    const currentList = watchlists.find(w => w.id === activeWatchlistId) || { name: 'Watchlist', id: 1 };

    // Mutations
    const createListMutation = useMutation({
        mutationFn: createWatchlist,
        onSuccess: (newItem) => {
            queryClient.invalidateQueries({ queryKey: ['watchlists'] });
            setIsCreating(false);
            setNewListName("");
            setActiveWatchlistId(newItem.id);
        },
        onError: (err: any) => alert(`Failed to create list: ${err.message}`)
    });

    const renameListMutation = useMutation({
        mutationFn: ({ id, name }: { id: number, name: string }) => renameWatchlist(id, name),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['watchlists'] });
            setIsRenaming(false);
            setRenameName("");
        },
        onError: (err: any) => alert(`Failed to rename list: ${err.message}`)
    });

    const deleteListMutation = useMutation({
        mutationFn: deleteWatchlist,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['watchlists'] });
            setActiveWatchlistId(1); // Switch to default
        },
        onError: (err: any) => alert(`Failed to delete list: ${err.message}`)
    });

    const addMutation = useMutation({
        mutationFn: ({ symbol, note }: { symbol: string, note: string }) => addToWatchlist(symbol, note, activeWatchlistId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['watchlist', currency, activeWatchlistId] });
            setNewSymbol('');
            setNewNote('');
        },
        onError: (error) => {
            alert(`Error adding to watchlist: ${error.message}`);
        }
    });

    const removeMutation = useMutation({
        mutationFn: (symbol: string) => removeFromWatchlist(symbol, activeWatchlistId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['watchlist', currency, activeWatchlistId] });
        },
    });

    const [editingSymbol, setEditingSymbol] = useState<string | null>(null);
    const [editNote, setEditNote] = useState('');

    const startEditing = (item: WatchlistItem) => {
        setEditingSymbol(item.Symbol);
        setEditNote(item.Note || '');
    };

    const cancelEditing = () => {
        setEditingSymbol(null);
        setEditNote('');
    };

    const saveEdit = (symbol: string) => {
        addMutation.mutate({ symbol, note: editNote });
        setEditingSymbol(null);
        setEditNote('');
    };


    const handleAdd = (e: React.FormEvent) => {
        e.preventDefault();
        if (newSymbol.trim()) {
            addMutation.mutate({ symbol: newSymbol.trim().toUpperCase(), note: newNote });
        }
    };

    const handleCreateList = (e: React.FormEvent) => {
        e.preventDefault();
        if (newListName.trim()) {
            createListMutation.mutate(newListName.trim());
        }
    }

    const handleRenameList = (e: React.FormEvent) => {
        e.preventDefault();
        if (renameName.trim()) {
            renameListMutation.mutate({ id: activeWatchlistId, name: renameName.trim() });
        }
    }

    const startRenaming = () => {
        setRenameName(currentList.name || "");
        setIsRenaming(true);
    }

    const handleDeleteList = () => {
        if (watchlists.length <= 1) {
            alert("At least one watchlist must exist.");
            return;
        }
        if (confirm(`Delete watchlist "${currentList.name}"? This action cannot be undone.`)) {
            // If deleting active list, switch to another one first (optimistically) or let query error handle it?
            // Better to switch to first available that is NOT this one.
            const nextList = watchlists.find(w => w.id !== activeWatchlistId) || { id: 1 };
            setActiveWatchlistId(nextList.id);
            deleteListMutation.mutate(activeWatchlistId);
        }
    }

    const isLoading = isLoadingItems || isLoadingLists;

    if (isLoading && !watchlists.length) {
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
            {/* Watchlist Selector Tabs */}
            <div className="flex flex-wrap items-center gap-2 pb-2 overflow-x-auto no-scrollbar mask-grad-right">
                {watchlists.map((wl) => (
                    <button
                        key={wl.id}
                        onClick={() => setActiveWatchlistId(wl.id)}
                        className={cn(
                            "px-3 py-1.5 text-sm font-medium border rounded-md shadow-sm transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500",
                            activeWatchlistId === wl.id
                                ? "bg-[#0097b2] text-white border-transparent shadow-sm"
                                : "text-foreground bg-secondary border-border hover:bg-accent/10"
                        )}
                    >
                        {wl.name}
                    </button>
                ))}

                {isCreating ? (
                    <form onSubmit={handleCreateList} className="flex items-center gap-1 animate-in fade-in slide-in-from-left-2">
                        <Input
                            value={newListName}
                            onChange={(e) => setNewListName(e.target.value)}
                            placeholder="List Name"
                            className="h-8 w-32 text-xs bg-background text-foreground border-border focus-visible:ring-1"
                            autoFocus
                        />
                        <Button type="submit" size="icon" variant="ghost" className="h-8 w-8 hover:bg-green-500/10 hover:text-green-500">
                            <Check className="h-4 w-4" />
                        </Button>
                        <Button type="button" onClick={() => setIsCreating(false)} size="icon" variant="ghost" className="h-8 w-8 hover:bg-red-500/10 hover:text-red-500">
                            <X className="h-4 w-4" />
                        </Button>
                    </form>
                ) : (
                    <button
                        onClick={() => setIsCreating(true)}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-foreground bg-secondary border border-border rounded-md shadow-sm hover:bg-accent/10 transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500"
                    >
                        <Plus className="h-3.5 w-3.5" />
                        <span>New List</span>
                    </button>
                )}
            </div>

            <Card className="bg-card border-border">
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                    <div>
                        <div className="flex items-center gap-3">
                            {isRenaming ? (
                                <form onSubmit={handleRenameList} className="flex items-center gap-2">
                                    <Input
                                        key={activeWatchlistId} // Force remount on list switch
                                        value={renameName}
                                        onChange={(e) => setRenameName(e.target.value)}
                                        className="h-9 w-48 text-lg font-bold bg-background text-foreground border-border focus-visible:ring-1"
                                        autoFocus
                                        onKeyDown={(e) => {
                                            if (e.key === 'Escape') setIsRenaming(false);
                                        }}
                                    />
                                    <Button type="submit" size="icon" variant="ghost" className="h-8 w-8 hover:bg-green-500/10 hover:text-green-500">
                                        <Check className="h-4 w-4" />
                                    </Button>
                                    <Button type="button" onClick={() => setIsRenaming(false)} size="icon" variant="ghost" className="h-8 w-8 hover:bg-red-500/10 hover:text-red-500">
                                        <X className="h-4 w-4" />
                                    </Button>
                                </form>
                            ) : (
                                <>
                                    <CardTitle className="text-xl font-bold text-foreground">
                                        {currentList.name}
                                    </CardTitle>

                                    <div className="flex items-center gap-0.5">
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            className="h-8 w-8 text-muted-foreground hover:text-primary hover:bg-primary/10 ml-1"
                                            onClick={startRenaming}
                                            title="Rename Watchlist"
                                        >
                                            <Pencil className="h-4 w-4" />
                                        </Button>

                                        {(watchlists.length > 1) && ( // Allow delete if more than 1 list
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                className="h-8 w-8 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                                                onClick={handleDeleteList}
                                                title="Delete Watchlist"
                                            >
                                                <Trash2 className="h-4 w-4" />
                                            </Button>
                                        )}
                                    </div>
                                </>
                            )}
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">
                            {watchlist?.length || 0} items tracked
                        </p>
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
                                    <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground whitespace-nowrap">Mkt Cap</th>
                                    <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground whitespace-nowrap">PE</th>
                                    <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground whitespace-nowrap">Div Yield</th>
                                    <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">7D Trend</th>
                                    <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Note</th>
                                    <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-border/50 bg-transparent">
                                {watchlist?.length === 0 ? (
                                    <tr>
                                        <td colSpan={10} className="text-center py-12 text-muted-foreground text-sm">
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
                                                <div className={`flex items-center justify-end font-mono tabular-nums ${(item["Day Change"] || 0) >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500'
                                                    }`}>
                                                    {(item["Day Change"] || 0) >= 0 ? <TrendingUp className="h-3 w-3 mr-1" /> : <TrendingDown className="h-3 w-3 mr-1" />}
                                                    {item["Day Change %"] ? formatPercent(item["Day Change %"] / 100) : '-'}
                                                </div>
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap text-right text-sm tabular-nums">
                                                {item["Market Cap"] ? formatCompactNumber(item["Market Cap"], item.Currency || 'USD') : '-'}
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap text-right text-sm tabular-nums">
                                                {item["PE Ratio"] ? item["PE Ratio"].toFixed(2) : '-'}
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap text-right text-sm tabular-nums">
                                                {(() => {
                                                    let y = item["Dividend Yield"];
                                                    if (y && y > 1) y = y / 100;
                                                    return y ? formatPercent(y) : '-';
                                                })()}
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap w-28">
                                                <div className="h-8 w-24">
                                                    {item.Sparkline && item.Sparkline.length > 1 ? (
                                                        <ResponsiveContainer width="100%" height="100%">
                                                            <AreaChart data={item.Sparkline.map(v => ({ value: v }))}>
                                                                <defs>
                                                                    {(() => {
                                                                        const val = item.Sparkline!;
                                                                        const baseline = val[0];
                                                                        const min = Math.min(...val);
                                                                        const max = Math.max(...val);
                                                                        const range = max - min;
                                                                        const off = range <= 0 ? 0 : (max - baseline) / range;

                                                                        return (
                                                                            <>
                                                                                <linearGradient id={`splitFill-wl-${item.Symbol}`} x1="0" y1="0" x2="0" y2="1">
                                                                                    <stop offset={off} stopColor="#10b981" stopOpacity={0.15} />
                                                                                    <stop offset={off} stopColor="#ef4444" stopOpacity={0.15} />
                                                                                </linearGradient>
                                                                                <linearGradient id={`splitStroke-wl-${item.Symbol}`} x1="0" y1="0" x2="0" y2="1">
                                                                                    <stop offset={off} stopColor="#10b981" stopOpacity={1} />
                                                                                    <stop offset={off} stopColor="#ef4444" stopOpacity={1} />
                                                                                </linearGradient>
                                                                            </>
                                                                        );
                                                                    })()}
                                                                </defs>
                                                                <YAxis hide domain={['dataMin', 'dataMax']} />
                                                                <ReferenceLine y={item.Sparkline[0]} stroke="#71717a" strokeDasharray="2 2" strokeOpacity={0.3} />
                                                                <Area
                                                                    type="monotone"
                                                                    dataKey="value"
                                                                    baseValue={item.Sparkline[0]}
                                                                    stroke={`url(#splitStroke-wl-${item.Symbol})`}
                                                                    fill={`url(#splitFill-wl-${item.Symbol})`}
                                                                    strokeWidth={1.5}
                                                                    isAnimationActive={false}
                                                                    dot={false}
                                                                />
                                                            </AreaChart>
                                                        </ResponsiveContainer>
                                                    ) : (
                                                        <span className="text-[10px] text-muted-foreground text-center block">no trend</span>
                                                    )}
                                                </div>
                                            </td>
                                            <td className={`px-4 py-3 whitespace-nowrap text-xs max-w-[150px] ${editingSymbol === item.Symbol ? '' : 'truncate text-muted-foreground italic'}`}>
                                                {editingSymbol === item.Symbol ? (
                                                    <Input
                                                        value={editNote}
                                                        onChange={(e) => setEditNote(e.target.value)}
                                                        className="h-7 text-xs text-foreground bg-background border-input w-full min-w-[120px]"
                                                        autoFocus
                                                        onKeyDown={(e) => {
                                                            if (e.key === 'Enter') saveEdit(item.Symbol);
                                                            if (e.key === 'Escape') cancelEditing();
                                                        }}
                                                    />
                                                ) : (
                                                    item.Note
                                                )}
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap text-right">
                                                <div className="flex items-center justify-end gap-1">
                                                    {editingSymbol === item.Symbol ? (
                                                        <>
                                                            <Button
                                                                variant="ghost"
                                                                size="icon"
                                                                onClick={() => saveEdit(item.Symbol)}
                                                                className="h-8 w-8 text-emerald-500 hover:text-emerald-400 hover:bg-emerald-500/10"
                                                            >
                                                                <Check className="h-4 w-4" />
                                                            </Button>
                                                            <Button
                                                                variant="ghost"
                                                                size="icon"
                                                                onClick={cancelEditing}
                                                                className="h-8 w-8 text-muted-foreground hover:text-foreground hover:bg-secondary"
                                                            >
                                                                <X className="h-4 w-4" />
                                                            </Button>
                                                        </>
                                                    ) : (
                                                        <>
                                                            <Button
                                                                variant="ghost"
                                                                size="icon"
                                                                onClick={() => startEditing(item)}
                                                                className="h-8 w-8 text-cyan-500 hover:text-cyan-400 hover:bg-cyan-500/10"
                                                            >
                                                                <Pencil className="h-4 w-4" />
                                                            </Button>
                                                            <Button
                                                                variant="ghost"
                                                                size="icon"
                                                                onClick={() => removeMutation.mutate(item.Symbol)}
                                                                className="h-8 w-8 text-red-500 hover:text-red-400 hover:bg-red-500/10"
                                                            >
                                                                <Trash2 className="h-4 w-4" />
                                                            </Button>
                                                        </>
                                                    )}
                                                </div>
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>
                </CardContent>
            </Card>
        </div >
    );
}
