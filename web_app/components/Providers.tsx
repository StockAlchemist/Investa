'use client';

import React, { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PersistQueryClientProvider } from '@tanstack/react-query-persist-client';
import { createSyncStoragePersister } from '@tanstack/query-sync-storage-persister';

import { ThemeProvider } from 'next-themes';
import { StockModalProvider } from '@/context/StockModalContext';
import { WatchlistProvider } from '@/context/WatchlistContext';

export default function Providers({ children }: { children: React.ReactNode }) {
    const [queryClient] = useState(() => new QueryClient({
        defaultOptions: {
            queries: {
                // Garbage collect time needs to be higher than stale time for persistence to be effective
                gcTime: 1000 * 60 * 60 * 24, // 24 hours
                staleTime: 1000 * 60 * 5, // 5 minutes
                refetchOnWindowFocus: false,
            },
        },
    }));

    const [persister] = useState(() => {
        if (typeof window !== 'undefined') {
            return createSyncStoragePersister({
                storage: window.localStorage,
                key: 'INVESTA_QUERY_CACHE', // Unique key for this app
                throttleTime: 1000,
            });
        }
        return undefined;
    });

    if (typeof window === 'undefined' || !persister) {
        // Fallback for SSR or if window is undefined, though this is a client component
        return (
            <QueryClientProvider client={queryClient}>
                <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
                    <StockModalProvider>
                        <WatchlistProvider>
                            {children}
                        </WatchlistProvider>
                    </StockModalProvider>
                </ThemeProvider>
            </QueryClientProvider>
        );
    }

    return (
        <PersistQueryClientProvider
            client={queryClient}
            persistOptions={{
                persister,
                // localStorage holds ~5MB; if the serialized cache tips over,
                // the write fails silently and instant-restore dies for every
                // tab. Skip bulky datasets that load fast anyway.
                dehydrateOptions: {
                    shouldDehydrateQuery: (query) =>
                        query.state.status === 'success' &&
                        query.queryKey[0] !== 'transactions',
                },
            }}
        >
            <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
                <StockModalProvider>
                    <WatchlistProvider>
                        {children}
                    </WatchlistProvider>
                </StockModalProvider>
            </ThemeProvider>
            {/* <ReactQueryDevtools initialIsOpen={false} /> */}
        </PersistQueryClientProvider>
    );
}
