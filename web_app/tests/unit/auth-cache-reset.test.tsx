import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, waitFor, act } from '@testing-library/react';
import React, { useEffect } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider, useAuth } from '@/context/AuthContext';
import { CHAT_HISTORY_STORAGE_KEY, QUERY_CACHE_STORAGE_KEY } from '@/lib/user_storage';
import * as api from '@/lib/api';

vi.mock('next/navigation', () => ({
    useRouter: () => ({ push: vi.fn() }),
}));

vi.mock('@/lib/api', async (importOriginal) => {
    const actual = await importOriginal<typeof api>();
    return {
        ...actual,
        fetchCurrentUser: vi.fn(),
        logoutRequest: vi.fn().mockResolvedValue(undefined),
    };
});

const ALICE = { id: 1, username: 'alice', created_at: '2026-01-01' } as api.User;
const BOB = { id: 2, username: 'bob', created_at: '2026-01-01' } as api.User;

function seedAlice(client: QueryClient) {
    // A user-specific key with no username in it: the kind that used to leak.
    client.setQueryData(['holdings', 'AAPL', 'USD', undefined, false], [{ Symbol: 'AAPL' }]);
    localStorage.setItem(QUERY_CACHE_STORAGE_KEY, '{"clientState":{"queries":["alice"]}}');
    localStorage.setItem(CHAT_HISTORY_STORAGE_KEY, '[{"role":"user","text":"alice\'s holdings"}]');
}

type Auth = ReturnType<typeof useAuth>;

function Probe({ onAuth }: { onAuth: (auth: Auth) => void }) {
    const auth = useAuth();
    useEffect(() => { onAuth(auth); });
    return null;
}

function renderAuth(client: QueryClient) {
    const latest: { auth?: Auth } = {};
    render(
        <QueryClientProvider client={client}>
            <AuthProvider><Probe onAuth={(a) => { latest.auth = a; }} /></AuthProvider>
        </QueryClientProvider>,
    );
    return () => latest.auth!;
}

describe('AuthProvider cache reset', () => {
    beforeEach(() => {
        localStorage.clear();
        vi.mocked(api.fetchCurrentUser).mockReset();
    });

    it('drops the cached portfolio on logout', async () => {
        localStorage.setItem('investa_user', JSON.stringify(ALICE));
        vi.mocked(api.fetchCurrentUser).mockResolvedValue(ALICE);
        const client = new QueryClient();
        const auth = renderAuth(client);
        await waitFor(() => expect(api.fetchCurrentUser).toHaveBeenCalled());
        seedAlice(client);

        act(() => auth().logout());

        expect(client.getQueryCache().getAll()).toHaveLength(0);
        expect(localStorage.getItem(QUERY_CACHE_STORAGE_KEY)).toBeNull();
        expect(localStorage.getItem(CHAT_HISTORY_STORAGE_KEY)).toBeNull();
    });

    it("drops the previous user's cache when the session belongs to someone else", async () => {
        localStorage.setItem('investa_user', JSON.stringify(ALICE));
        const client = new QueryClient();
        seedAlice(client);
        vi.mocked(api.fetchCurrentUser).mockResolvedValue(BOB);

        renderAuth(client);

        await waitFor(() => expect(client.getQueryCache().getAll()).toHaveLength(0));
        expect(localStorage.getItem(QUERY_CACHE_STORAGE_KEY)).toBeNull();
        expect(localStorage.getItem(CHAT_HISTORY_STORAGE_KEY)).toBeNull();
    });

    it('keeps the cache when the same user comes back', async () => {
        localStorage.setItem('investa_user', JSON.stringify(ALICE));
        const client = new QueryClient();
        seedAlice(client);
        vi.mocked(api.fetchCurrentUser).mockResolvedValue(ALICE);

        renderAuth(client);

        await waitFor(() => expect(api.fetchCurrentUser).toHaveBeenCalled());
        expect(client.getQueryCache().getAll()).toHaveLength(1);
    });
});
