/** Keep the same receipt key after an ambiguous failure, including a page reload. */
const pending = new Map<string, string>();

export async function submissionKey(endpoint: string, payload: unknown): Promise<{ key: string; forget: () => void }> {
    const user = typeof window !== 'undefined' ? localStorage.getItem('investa_user') ?? '' : '';
    const bytes = new TextEncoder().encode(JSON.stringify([endpoint, user, payload], (_key, value) => value && typeof value === 'object' && !Array.isArray(value)
        ? Object.fromEntries(Object.keys(value).sort().map(key => [key, value[key]])) : value));
    // Insecure LAN origins expose getRandomValues but not SubtleCrypto.
    const digest = crypto.subtle
        ? Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('')
        : [2166136261, 2654435761].map(seed => Array.from(bytes).reduce((hash, byte) => Math.imul(hash ^ byte, 16777619) >>> 0, seed).toString(16)).join('-');

    const storageKey = `investa_submission_${digest}`;
    let key = pending.get(storageKey);
    try { key ??= sessionStorage.getItem(storageKey) ?? undefined; } catch {}
    key ??= Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, '0')).join('');
    pending.set(storageKey, key);
    try { sessionStorage.setItem(storageKey, key); } catch {}
    return { key, forget: () => {
        pending.delete(storageKey);
        try { sessionStorage.removeItem(storageKey); } catch {}
    } };
}
