"use client";

import React, { useState, useEffect, useId } from 'react';
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
import { Button } from "@/components/ui/button";
import { CardHeader, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Plus, Trash2, TrendingUp, TrendingDown, RefreshCw, Pencil, Check, X, ListPlus, ArrowUpDown, ArrowUp, ArrowDown, HelpCircle } from "lucide-react";
import { AreaChart, Area, YAxis, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Skeleton } from "@/components/ui/skeleton";
import { formatCurrency, formatPercent, formatCompactNumber, cn, getHeatmapClass } from "@/lib/utils";
import StockTicker from './StockTicker';
import { TrendSparkline } from './ui/TrendSparkline';

interface WatchlistProps {
    currency: string;
}

type SortDirection = 'asc' | 'desc';

interface SortConfig {
    key: string;
    direction: SortDirection;
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

    // Sorting State
    const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'Symbol', direction: 'asc' });

    const handleSort = (key: string) => {
        setSortConfig(current => ({
            key,
            direction: current.key === key && current.direction === 'asc' ? 'desc' : 'asc',
        }));
    };

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
        onMutate: async (newListName) => {
            const temporaryId = Date.now();
            await queryClient.cancelQueries({ queryKey: ['watchlists'] });
            const previousWatchlists = queryClient.getQueryData<WatchlistMeta[]>(['watchlists']);
            queryClient.setQueryData<WatchlistMeta[]>(['watchlists'], (old) => {
                const newList = { id: temporaryId, name: newListName, created_at: new Date().toISOString() };
                return old ? [...old, newList] : [newList];
            });
            setIsCreating(false);
            setNewListName("");
            setActiveWatchlistId(temporaryId);
            return { previousWatchlists, temporaryId };
        },
        onSuccess: (newItem, variables, context) => {
            queryClient.setQueryData<WatchlistMeta[]>(['watchlists'], (old) => {
                if (!old) return [newItem];
                return old.map(list => list.id === context?.temporaryId ? newItem : list);
            });
            if (activeWatchlistId === context?.temporaryId) {
                setActiveWatchlistId(newItem.id);
            }
        },
        onError: (err: any, variables, context) => {
            if (context?.previousWatchlists) {
                queryClient.setQueryData(['watchlists'], context.previousWatchlists);
            }
            alert(`Failed to create list: ${err.message}`)
        },
        onSettled: () => {
            queryClient.invalidateQueries({ queryKey: ['watchlists'] });
        }
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
        onMutate: async ({ symbol, note }) => {
            await queryClient.cancelQueries({ queryKey: ['watchlist', currency, activeWatchlistId] });
            const previousWatchlist = queryClient.getQueryData<WatchlistItem[]>(['watchlist', currency, activeWatchlistId]);

            queryClient.setQueryData<WatchlistItem[]>(['watchlist', currency, activeWatchlistId], (old) => {
                const newItem: WatchlistItem = {
                    Symbol: symbol,
                    Note: note,
                    AddedOn: new Date().toISOString(),
                    Name: "Loading...",
                    Price: null,
                    "Day Change": null,
                    "Day Change %": null,
                    Currency: currency,
                    Sparkline: [],
                    "Market Cap": null,
                    "PE Ratio": null,
                    "Dividend Yield": null,
                    ai_score: null,
                    intrinsic_value: null,
                    margin_of_safety: null,
                    has_ai_review: false
                };
                return old ? [newItem, ...old] : [newItem];
            });
            setNewSymbol('');
            setNewNote('');
            return { previousWatchlist };
        },
        onError: (err, variables, context) => {
            if (context?.previousWatchlist) {
                queryClient.setQueryData(['watchlist', currency, activeWatchlistId], context.previousWatchlist);
            }
            alert(`Error adding to watchlist: ${err.message}`);
        },
        onSettled: () => {
            queryClient.invalidateQueries({ queryKey: ['watchlist', currency, activeWatchlistId] });
        }
    });

    const removeMutation = useMutation({
        mutationFn: (symbol: string) => removeFromWatchlist(symbol, activeWatchlistId),
        onMutate: async (symbol) => {
            await queryClient.cancelQueries({ queryKey: ['watchlist', currency, activeWatchlistId] });
            const previousWatchlist = queryClient.getQueryData<WatchlistItem[]>(['watchlist', currency, activeWatchlistId]);

            queryClient.setQueryData<WatchlistItem[]>(['watchlist', currency, activeWatchlistId], (old) => {
                return old ? old.filter(item => item.Symbol !== symbol) : [];
            });
            return { previousWatchlist };
        },
        onError: (err, variables, context) => {
            if (context?.previousWatchlist) {
                queryClient.setQueryData(['watchlist', currency, activeWatchlistId], context.previousWatchlist);
            }
        },
        onSettled: () => {
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
            const nextList = watchlists.find(w => w.id !== activeWatchlistId) || { id: 1 };
            setActiveWatchlistId(nextList.id);
            deleteListMutation.mutate(activeWatchlistId);
        }
    }

    const isLoading = isLoadingItems || isLoadingLists;

    // --- Sorting Logic ---
    const sortedWatchlist = React.useMemo(() => {
        if (!watchlist) return [];
        return [...watchlist].sort((a, b) => {
            const getValue = (item: WatchlistItem, key: string) => {
                switch (key) {
                    case 'Symbol': return item.Symbol;
                    case 'Name': return item.Name || '';
                    case 'Price': return item.Price || 0;
                    case 'Day Change': return item["Day Change"] || 0;
                    case 'Mkt Cap': return item["Market Cap"] || 0;
                    case 'PE': return item["PE Ratio"] || 0;
                    case 'Div Yield': return item["Dividend Yield"] || 0;
                    case 'Intrinsic Value': return item.intrinsic_value || 0;
                    case 'AI Score': return item.ai_score || 0;
                    case 'Sentiment': return item.ai_sentiment || 0;
                    case 'Note': return item.Note || '';
                    default: return 0;
                }
            };

            const valA = getValue(a, sortConfig.key);
            const valB = getValue(b, sortConfig.key);

            if (typeof valA === 'string' && typeof valB === 'string') {
                return sortConfig.direction === 'asc'
                    ? valA.localeCompare(valB)
                    : valB.localeCompare(valA);
            }
            return sortConfig.direction === 'asc'
                ? (valA as number) - (valB as number)
                : (valB as number) - (valA as number);
        });
    }, [watchlist, sortConfig]);

    if (isLoading && !watchlists.length) {
        return (
            <div className="space-y-4">
                <div className="metric-card card-shine p-6 relative overflow-hidden">
                    <div className="absolute top-0 left-4 right-4 h-[2px] rounded-full bg-indigo-500 opacity-50" />
                    <CardHeader className="p-0 mb-4">
                        <Skeleton className="h-8 w-48 opacity-50 rounded-lg" />
                    </CardHeader>
                    <CardContent className="p-0">
                        <div className="space-y-3">
                            {[1, 2, 3].map(i => <Skeleton key={i} className="h-14 w-full opacity-40 rounded-2xl" />)}
                        </div>
                    </CardContent>
                </div>
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
                            "px-3 py-1.5 text-sm font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500",
                            activeWatchlistId === wl.id
                                ? "bg-indigo-600 text-white"
                                : "text-indigo-500 hover:bg-accent/10"
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
                        className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-foreground bg-secondary rounded-md hover:bg-accent/10 transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    >
                        <Plus className="h-3.5 w-3.5" />
                        <span>New List</span>
                    </button>
                )}
            </div>

            <div className="metric-card card-shine relative overflow-hidden">
                <div className="absolute top-0 left-0 right-0 h-[2px] bg-sky-500 opacity-80" />

                {/* Header */}
                <div className="flex flex-row items-center justify-between p-5 pb-2">
                    <div>
                        <div className="flex items-center gap-3">
                            {isRenaming ? (
                                <form onSubmit={handleRenameList} className="flex items-center gap-2">
                                    <Input
                                        key={activeWatchlistId}
                                        value={renameName}
                                        onChange={(e) => setRenameName(e.target.value)}
                                        className="h-9 w-48 text-lg font-bold bg-background text-foreground border-none focus-visible:ring-1"
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
                                    <h2 className="section-label text-base">
                                        {currentList.name}
                                    </h2>

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

                                        {(watchlists.length > 1) && (
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
                        <p className="text-xs text-muted-foreground/60 mt-0.5">
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
                </div>

                <div className="px-5 pb-5">
                    <form onSubmit={handleAdd} className="flex flex-col md:flex-row items-end gap-3 mb-6 bg-muted/30 dark:bg-white/[0.03] backdrop-blur-md p-4 rounded-2xl border border-border/40 dark:border-white/[0.05]">
                        <div className="flex flex-col gap-1.5 md:w-56 w-full">
                            <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider px-1">Symbol</label>
                            <Input
                                placeholder="e.g. AAPL, BTC-USD"
                                value={newSymbol}
                                onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                                className="bg-background/50 backdrop-blur-sm border-border/40 hover:border-indigo-500/50 focus-visible:ring-indigo-500 rounded-2xl transition-all text-foreground placeholder:text-muted-foreground/50"
                            />
                        </div>
                        <div className="flex flex-col gap-1.5 flex-1 w-full">
                            <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider px-1">Note (optional)</label>
                            <Input
                                placeholder="Add a description..."
                                value={newNote}
                                onChange={(e) => setNewNote(e.target.value)}
                                className="bg-background/50 backdrop-blur-sm border-border/40 hover:border-indigo-500/50 focus-visible:ring-indigo-500 rounded-2xl transition-all text-foreground placeholder:text-muted-foreground/50"
                            />
                        </div>
                        <Button
                            type="submit"
                            disabled={addMutation.isPending || !newSymbol.trim()}
                            className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold h-10 px-6 shadow-lg shadow-indigo-500/20 active:scale-95 transition-all"
                        >
                            <Plus className="h-4 w-4 mr-2" />
                            Add to List
                        </Button>
                    </form>

                    <div className="overflow-x-auto rounded-lg">
                        <table className="min-w-full">
                            <thead className="bg-secondary sticky top-0 z-10 font-semibold">
                                <tr>
                                    {[
                                        { key: 'Symbol', label: 'Symbol', align: 'left' },
                                        { key: 'Name', label: 'Name', align: 'left' },
                                        { key: 'Price', label: 'Price', align: 'right' },
                                        { key: 'Day Change', label: 'Day Change', align: 'right' },
                                        { key: 'Mkt Cap', label: 'Mkt Cap', align: 'right' },
                                        { key: 'PE', label: 'PE', align: 'right' },
                                        { key: 'Div Yield', label: 'Div Yield', align: 'right' },
                                        { key: 'AI Score', label: 'AI Score', align: 'center' },
                                        { key: 'Intrinsic Value', label: 'Intrinsic Value', align: 'right' },
                                        { key: 'Sentiment', label: 'Sentiment', align: 'center' },
                                        { key: 'Catalyst', label: 'Catalyst', align: 'center', disableSort: true },
                                        { key: '7D Trend', label: '7D Trend', align: 'left', disableSort: true },
                                        { key: 'Note', label: 'Note', align: 'left' },
                                    ].map((col) => (
                                        <th
                                            key={col.key}
                                            className={`px-4 py-3 text-xs font-semibold text-muted-foreground whitespace-nowrap ${col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : 'text-left'} ${!col.disableSort ? 'cursor-pointer hover:text-foreground hover:bg-muted/50 transition-colors select-none' : ''} ${col.key === 'Symbol' ? 'sticky left-0 z-20 bg-secondary/95 backdrop-blur-md shadow-[2px_0_5px_-2px_rgba(0,0,0,0.3)]' : ''}`}
                                            onClick={() => !col.disableSort && handleSort(col.key)}
                                        >
                                            <div className={`flex items-center gap-1 ${col.align === 'right' ? 'justify-end' : col.align === 'center' ? 'justify-center' : 'justify-start'}`}>
                                                {col.label}
                                                {!col.disableSort && (
                                                    <span className="text-muted-foreground/50">
                                                        {sortConfig.key === col.key ? (
                                                            sortConfig.direction === 'asc' ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />
                                                        ) : (
                                                            <ArrowUpDown className="h-3 w-3 opacity-0 group-hover:opacity-50" />
                                                        )}
                                                    </span>
                                                )}
                                            </div>
                                        </th>

                                    ))}
                                    <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="bg-transparent divide-y-none">
                                {watchlist?.length === 0 ? (
                                    <tr>
                                        <td colSpan={11} className="text-center py-12 text-muted-foreground text-sm">
                                            No symbols in your watchlist yet.
                                        </td>
                                    </tr>
                                ) : (
                                    sortedWatchlist?.map((item, idx) => (
                                        <tr key={item.Symbol} className="hover:bg-accent/5 transition-colors">
                                            <td className="px-4 py-3 whitespace-nowrap text-sm sticky left-0 z-10 bg-background/90 backdrop-blur-md shadow-[2px_0_5px_-2px_rgba(0,0,0,0.1)]">
                                                <StockTicker symbol={item.Symbol} currency={currency} />
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap text-foreground text-sm max-w-[200px] truncate">{item.Name || '-'}</td>
                                            <td className="px-4 py-3 whitespace-nowrap text-right font-mono font-medium text-sm text-foreground tabular-nums">
                                                {item.Price ? formatCurrency(item.Price, item.Currency || 'USD') : '-'}
                                            </td>
                                            <td className={cn(
                                                "px-4 py-3 whitespace-nowrap text-right text-sm transition-colors",
                                                getHeatmapClass(item["Day Change %"])
                                            )}>
                                                <div className={`flex items-center justify-end font-mono tabular-nums ${(item["Day Change"] || 0) >= 0 ? 'text-emerald-600 dark:text-emerald-400 font-bold' : 'text-red-600 dark:text-red-500 font-bold'
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
                                            <td className="px-4 py-3 whitespace-nowrap text-center text-sm">
                                                {item.ai_score ? (
                                                    <div className={cn(
                                                        "inline-flex items-center justify-center w-9 h-9 rounded-full font-bold text-xs shadow-sm border mx-auto",
                                                        item.ai_score >= 7.5 ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/20" :
                                                        item.ai_score >= 5.0 ? "bg-amber-500/10 text-amber-600 border-amber-500/20" :
                                                        "bg-red-500/10 text-red-600 border-red-500/20"
                                                    )}>
                                                        {item.ai_score.toFixed(1)}
                                                    </div>
                                                ) : (
                                                    <span className="text-muted-foreground">-</span>
                                                )}
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap text-right text-sm tabular-nums">
                                                {item.intrinsic_value ? (
                                                    <div className="flex flex-col items-end">
                                                        <span className="font-mono font-medium">
                                                            {formatCurrency(item.intrinsic_value, item.Currency || 'USD')}
                                                        </span>
                                                        {item.margin_of_safety !== undefined && (
                                                            <span className={cn(
                                                                "text-[10px] font-bold",
                                                                item.margin_of_safety > 0 ? "text-emerald-500" : "text-rose-500"
                                                            )}>
                                                                {item.margin_of_safety > 0 ? '+' : ''}{item.margin_of_safety.toFixed(1)}% MOS
                                                            </span>
                                                        )}
                                                    </div>
                                                ) : (
                                                    <span className="text-muted-foreground">-</span>
                                                )}
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap text-center text-sm">
                                                {item.ai_sentiment !== null && item.ai_sentiment !== undefined ? (
                                                    <div className="flex flex-col items-center gap-1">
                                                        <div className="w-16 h-1.5 bg-muted rounded-full overflow-hidden">
                                                            <div 
                                                                className={cn(
                                                                    "h-full rounded-full transition-all duration-500",
                                                                    item.ai_sentiment >= 70 ? "bg-emerald-500" :
                                                                    item.ai_sentiment >= 40 ? "bg-amber-500" :
                                                                    "bg-rose-500"
                                                                )}
                                                                style={{ width: `${item.ai_sentiment}%` }}
                                                            />
                                                        </div>
                                                        <span className="text-[10px] font-bold opacity-70">
                                                            {item.ai_sentiment.toFixed(0)}%
                                                        </span>
                                                    </div>
                                                ) : (
                                                    <span className="text-muted-foreground">-</span>
                                                )}
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap text-center text-sm">
                                                {item.ai_catalysts && item.ai_catalysts.length > 0 ? (
                                                    <div className="relative group cursor-help inline-block">
                                                        <div className="p-1.5 rounded-full bg-amber-500/10 text-amber-500 animate-pulse-subtle">
                                                            <HelpCircle className="h-4 w-4" />
                                                        </div>
                                                        <div className={cn(
                                                            "absolute left-1/2 -translate-x-1/2 w-max max-w-[350px] p-4 bg-popover border border-border rounded-xl shadow-2xl opacity-0 group-hover:opacity-100 transition-all duration-200 z-50 pointer-events-none backdrop-blur-md whitespace-normal",
                                                            idx < 3 ? "top-full mt-2" : "bottom-full mb-2"
                                                        )}>
                                                            <p className="text-[10px] font-extrabold uppercase text-primary mb-3 tracking-widest border-b border-border pb-1">Upcoming Catalysts</p>
                                                            <div className="space-y-3">
                                                                {item.ai_catalysts.map((c, idx) => (
                                                                    <div key={idx} className="text-left border-l-2 border-amber-500 pl-3 py-0.5">
                                                                        <p className="text-[12px] font-bold text-foreground leading-snug">{c.event}</p>
                                                                        <div className="flex items-center gap-2 mt-1">
                                                                            <span className="text-[10px] text-muted-foreground font-medium">{c.date}</span>
                                                                            <span className={`text-[9px] px-1.5 py-0.5 rounded-full font-bold uppercase ${
                                                                                c.impact === 'High' ? 'bg-red-500/10 text-red-500' : 
                                                                                c.impact === 'Medium' ? 'bg-amber-500/10 text-amber-500' : 
                                                                                'bg-blue-500/10 text-blue-500'
                                                                            }`}>
                                                                                {c.impact} Impact
                                                                            </span>
                                                                        </div>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    </div>
                                                ) : (
                                                    <span className="text-muted-foreground">-</span>
                                                )}
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap w-28">
                                                <div className="h-8 w-24">
                                                    <TrendSparkline data={item.Sparkline || []} />
                                                </div>
                                            </td>
                                            <td className={`px-4 py-3 whitespace-nowrap text-xs max-w-[150px] ${editingSymbol === item.Symbol ? '' : 'truncate text-muted-foreground italic'}`}>
                                                {editingSymbol === item.Symbol ? (
                                                    <Input
                                                        value={editNote}
                                                        onChange={(e) => setEditNote(e.target.value)}
                                                        className="h-7 text-xs text-foreground bg-background border-none w-full min-w-[120px]"
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
                                                                className="h-8 w-8 text-indigo-500 hover:text-indigo-400 hover:bg-indigo-500/10"
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
                </div>
            </div>
        </div>
    );
}
