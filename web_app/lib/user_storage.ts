/**
 * localStorage entries that hold one user's data under a key every user of the
 * browser shares. AuthContext removes them whenever the signed-in user changes,
 * so one user's portfolio or AI conversation never opens in another's session.
 */

/** The persisted React Query cache (see Providers). */
export const QUERY_CACHE_STORAGE_KEY = 'INVESTA_QUERY_CACHE';

/** The AI assistant's conversation, which discusses the user's holdings. */
export const CHAT_HISTORY_STORAGE_KEY = 'investa_chat_history';

export function forgetStoredUserData(): void {
    for (const key of [QUERY_CACHE_STORAGE_KEY, CHAT_HISTORY_STORAGE_KEY]) {
        try { localStorage.removeItem(key); } catch {}
    }
}
