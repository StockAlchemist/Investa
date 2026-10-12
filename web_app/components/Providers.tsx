'use client';

import React, { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PersistQueryClientProvider } from '@tanstack/react-query-persist-client';
import { queryPersister } from '@/lib/query_persistence';

import { ThemeProvider } from 'next-themes';
import { StockModalProvider } from '@/context/StockModalContext';
import { WatchlistProvider } from '@/context/WatchlistContext';
import { useImportServersFromHash } from '@/lib/servers';

export default function Providers({ children }: { children: React.ReactNode }) {
    const [queryClient] = useState(() => new QueryClient({
        defaultOptions: {
            queries: {
                // Garbage collect time needs to be higher than stale time for persistence to be effective
                gcTime: 1000 * 60 * 60, // Match the one-hour persistence window
                staleTime: 1000 * 60 * 5, // 5 minutes
                refetchOnWindowFocus: false,
            },
        },
    }));

    // A server switch carries the saved list in the URL fragment; take it in
    // here, at the root, before a signed-out redirect to /login drops it.
    useImportServersFromHash();

    if (typeof window === 'undefined') {
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
                persister: queryPersister,
                buster: 'portfolio-contract-v2',
                maxAge: 60 * 60 * 1000,
                // Persist compact dashboard data; large tables remain on demand.
                dehydrateOptions: {
                    shouldDehydrateQuery: (query) =>
                        query.state.status === 'success' &&
                        ['summary', 'headline', 'holdings', 'history', 'settings', 'indices', 'marketStatus'].includes(String(query.queryKey[0])) &&
                        (!Array.isArray(query.state.data) || query.state.data.length <= 5000),
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
