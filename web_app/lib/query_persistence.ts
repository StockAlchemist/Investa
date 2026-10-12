import type { Persister, PersistedClient } from '@tanstack/react-query-persist-client';

const DATABASE = 'investa-query-cache';
let generation = 0;
let pendingWrite: ReturnType<typeof setTimeout> | undefined;

function database(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DATABASE, 1);
        request.onupgradeneeded = () => request.result.createObjectStore('cache');
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

async function write(value?: PersistedClient, expected = generation): Promise<void> {
    if (typeof indexedDB === 'undefined') return;
    try {
        const db = await database();
        if (expected !== generation) { db.close(); return; }
        await new Promise<void>((resolve, reject) => {
            const tx = db.transaction('cache', 'readwrite');
            const store = tx.objectStore('cache');
            if (value) store.put(value, 'client');
            else store.delete('client');
            tx.oncomplete = () => resolve();
            tx.onerror = tx.onabort = () => reject(tx.error);
        });
        db.close();
    } catch { /* Unavailable/quota-limited storage must not prevent using the app. */ }
}

export async function clearPersistedQueries(): Promise<void> {
    generation += 1;
    clearTimeout(pendingWrite);
    await write(undefined, generation);
}

export const queryPersister: Persister = {
    persistClient: (client) => {
        const expected = generation;
        clearTimeout(pendingWrite);
        pendingWrite = setTimeout(() => { void write(client, expected); }, 1000);
    },
    removeClient: clearPersistedQueries,
    restoreClient: async () => {
        const expected = generation;
        if (typeof indexedDB === 'undefined') return undefined;
        try {
            const db = await database();
            const client = await new Promise<PersistedClient | undefined>((resolve, reject) => {
                const request = db.transaction('cache').objectStore('cache').get('client');
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });
            db.close();
            return expected === generation ? client : undefined;
        } catch { return undefined; }
    },
};
