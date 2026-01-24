'use client';

import React, { createContext, useContext, useMemo, ReactNode } from 'react';
import { useQuery, useQueries, useMutation, useQueryClient } from '@tanstack/react-query';
import { getWatchlists, fetchWatchlist, addToWatchlist, removeFromWatchlist } from '@/lib/api';

interface WatchlistContextType {
    watchlists: { id: number; name: string }[];
    symbolWatchlistMap: Record<string, Set<number>>;
    starredSymbols: Set<string>;
    toggleWatchlist: (symbol: string, watchlistId: number) => void;
    isLoading: boolean;
}

const WatchlistContext = createContext<WatchlistContextType | undefined>(undefined);

export function WatchlistProvider({ children }: { children: ReactNode }) {
    const queryClient = useQueryClient();

    // Fetch all watchlists metadata
    const { data: watchlists = [], isLoading: isLoadingLists } = useQuery({
        queryKey: ['watchlists'],
        queryFn: ({ signal }) => getWatchlists(signal),
    });

    // Fetch all items from all watchlists to build a symbol mapping
    // We use a fixed currency 'USD' for the mapping since the symbols don't change
    const watchlistQueries = useQueries({
        queries: watchlists.map(wl => ({
            queryKey: ['watchlist', 'USD', wl.id],
            queryFn: ({ signal }: { signal?: AbortSignal }) => fetchWatchlist('USD', wl.id, signal),
            staleTime: 1000 * 60 * 5, // 5 minutes
        }))
    });

    const isLoadingItems = watchlistQueries.some(q => q.isLoading);

    const symbolWatchlistMap = useMemo(() => {
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

    const starredSymbols = useMemo(() => {
        return new Set(Object.keys(symbolWatchlistMap));
    }, [symbolWatchlistMap]);

    const addMutation = useMutation({
        mutationFn: ({ symbol, watchlistId }: { symbol: string, watchlistId: number }) => addToWatchlist(symbol, "", watchlistId),
        onSuccess: (_, variables) => {
            // Invalidate BOTH the common USD query and any potential localized queries
            queryClient.invalidateQueries({ queryKey: ['watchlist'] });
        },
    });

    const removeMutation = useMutation({
        mutationFn: ({ symbol, watchlistId }: { symbol: string, watchlistId: number }) => removeFromWatchlist(symbol, watchlistId),
        onSuccess: (_, variables) => {
            queryClient.invalidateQueries({ queryKey: ['watchlist'] });
        },
    });

    const toggleWatchlist = (symbol: string, watchlistId: number) => {
        if (symbolWatchlistMap[symbol]?.has(watchlistId)) {
            removeMutation.mutate({ symbol, watchlistId });
        } else {
            addMutation.mutate({ symbol, watchlistId });
        }
    };

    return (
        <WatchlistContext.Provider value={{
            watchlists,
            symbolWatchlistMap,
            starredSymbols,
            toggleWatchlist,
            isLoading: isLoadingLists || isLoadingItems
        }}>
            {children}
        </WatchlistContext.Provider>
    );
}

export function useWatchlist() {
    const context = useContext(WatchlistContext);
    if (context === undefined) {
        throw new Error('useWatchlist must be used within a WatchlistProvider');
    }
    return context;
}
