'use client';

import { useEffect, useSyncExternalStore } from 'react';

/**
 * Saved Investa servers — the web twin of the native `ServerStore`.
 *
 * The native apps save a *backend* URL and point their requests at it. The web
 * app can't: its session is an httpOnly SameSite=Lax cookie, which a browser
 * refuses to set from, or send to, another host. So a web "server" is another
 * machine's Investa **web app** (`http://100.127.10.38:3000`), and switching
 * opens it in this tab, where it has its own sign-in.
 *
 * The list lives in localStorage, which is per origin, so a switch carries it
 * along in the URL fragment (never sent to a server) and the destination merges
 * it in — otherwise the machine you land on wouldn't know the way back.
 */
export interface SavedServer {
    id: string;
    name: string;
    /** The web app's origin: scheme, host and port, no path. */
    url: string;
}

const STORAGE_KEY = 'investa_saved_servers';
const CHANGE_EVENT = 'investa:servers';
const HASH_KEY = 'investa-servers';
const MAX_NAME = 60;

/**
 * `100.127.10.38:3000` → `http://100.127.10.38:3000`. Returns null for anything
 * that isn't an http(s) address. Paths are dropped: a server is an origin.
 */
export function normalizeServerUrl(input: string): string | null {
    const raw = input.trim();
    if (!raw) return null;
    try {
        const url = new URL(/^[a-z][a-z0-9+.-]*:\/\//i.test(raw) ? raw : `http://${raw}`);
        if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
        return url.origin;
    } catch {
        return null;
    }
}

/**
 * The backend's address (`…:8000/api`) is what the native apps save, so it is
 * the one people reach for — but it serves JSON, not the app.
 */
export function looksLikeApiUrl(input: string): boolean {
    try {
        const raw = input.trim();
        const url = new URL(/^[a-z][a-z0-9+.-]*:\/\//i.test(raw) ? raw : `http://${raw}`);
        return url.pathname.replace(/\/+$/, '') === '/api' || url.pathname.startsWith('/api/');
    } catch {
        return false;
    }
}

/** `http://100.127.10.38:3000` → `100.127.10.38:3000`. */
export function defaultServerName(origin: string): string {
    try {
        return new URL(origin).host;
    } catch {
        return origin;
    }
}

export function currentOrigin(): string {
    return typeof window === 'undefined' ? '' : window.location.origin;
}

function sanitize(list: unknown): SavedServer[] {
    if (!Array.isArray(list)) return [];
    const seen = new Set<string>();
    const out: SavedServer[] = [];
    for (const item of list) {
        if (!item || typeof item !== 'object') continue;
        const { id, name, url } = item as Record<string, unknown>;
        const origin = typeof url === 'string' ? normalizeServerUrl(url) : null;
        if (!origin || seen.has(origin)) continue;
        seen.add(origin);
        out.push({
            id: typeof id === 'string' && id ? id : newId(),
            name: (typeof name === 'string' && name.trim() ? name.trim() : defaultServerName(origin)).slice(0, MAX_NAME),
            url: origin,
        });
    }
    return out;
}

function newId(): string {
    return typeof crypto !== 'undefined' && 'randomUUID' in crypto
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

// useSyncExternalStore needs the same array back until the stored value
// changes, so the parse is cached against the raw string.
let cachedRaw: string | null | undefined;
let cachedList: SavedServer[] = [];
const EMPTY: SavedServer[] = [];

function readServers(): SavedServer[] {
    let raw: string | null = null;
    try {
        raw = window.localStorage.getItem(STORAGE_KEY);
    } catch {
        return cachedList;
    }
    if (raw !== cachedRaw) {
        cachedRaw = raw;
        try {
            cachedList = raw ? sanitize(JSON.parse(raw)) : [];
        } catch {
            cachedList = [];
        }
    }
    return cachedList;
}

function writeServers(list: SavedServer[]) {
    try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
    } catch {
        // Private mode or a full quota: the list just won't persist.
    }
    window.dispatchEvent(new Event(CHANGE_EVENT));
}

function subscribe(onChange: () => void) {
    const onStorage = (e: StorageEvent) => { if (e.key === STORAGE_KEY) onChange(); };
    window.addEventListener(CHANGE_EVENT, onChange);
    window.addEventListener('storage', onStorage);
    return () => {
        window.removeEventListener(CHANGE_EVENT, onChange);
        window.removeEventListener('storage', onStorage);
    };
}

/**
 * Saves `url` under `name` (or its host). An address already in the list is
 * renamed rather than listed twice. Returns the saved entry, or null when the
 * address isn't usable.
 */
export function addServer(name: string, url: string): SavedServer | null {
    const origin = normalizeServerUrl(url);
    if (!origin) return null;
    const label = (name.trim() || defaultServerName(origin)).slice(0, MAX_NAME);
    const list = readServers();
    const existing = list.find(s => s.url === origin);
    const saved = existing ? { ...existing, name: label } : { id: newId(), name: label, url: origin };
    writeServers(existing ? list.map(s => (s.id === existing.id ? saved : s)) : [...list, saved]);
    return saved;
}

export function removeServer(id: string) {
    writeServers(readServers().filter(s => s.id !== id));
}

/**
 * Opens `origin`'s Investa in this tab, carrying the saved list — plus the
 * server being left, so the destination can always switch back to it.
 */
export function switchToServer(origin: string) {
    const here = currentOrigin();
    const list = readServers();
    const carried = list.some(s => s.url === here)
        ? list
        : [...list, { id: newId(), name: defaultServerName(here), url: here }];
    const payload = encodeURIComponent(JSON.stringify(carried.map(({ name, url }) => ({ name, url }))));
    window.location.assign(`${origin}/#${HASH_KEY}=${payload}`);
}

/**
 * Merges a list carried in by `switchToServer`, then strips it from the
 * address bar. Names already saved here win: the carried list only adds.
 */
export function importServersFromHash() {
    if (typeof window === 'undefined') return;
    const hash = window.location.hash.slice(1);
    if (!hash.startsWith(`${HASH_KEY}=`)) return;
    try {
        const incoming = sanitize(JSON.parse(decodeURIComponent(hash.slice(HASH_KEY.length + 1))));
        const list = readServers();
        const known = new Set(list.map(s => s.url));
        const added = incoming.filter(s => !known.has(s.url));
        if (added.length > 0) writeServers([...list, ...added]);
    } catch {
        // A mangled fragment only costs the carried list.
    }
    window.history.replaceState(null, '', window.location.pathname + window.location.search);
}

/** The saved list, kept in step across components and tabs. */
export function useSavedServers(): SavedServer[] {
    return useSyncExternalStore(subscribe, readServers, () => EMPTY);
}

const noSubscribe = () => () => {};

/** This page's origin; '' during the server render, where there is none. */
export function useCurrentOrigin(): string {
    return useSyncExternalStore(noSubscribe, currentOrigin, () => '');
}

/**
 * What a quick switcher offers: this server first (saved or not), then the
 * rest in saved order. Null until there is somewhere else to go, so a
 * single-server setup draws no switcher at all.
 */
export function useServerChoices(): { here: string; currentName: string; rows: SavedServer[] } | null {
    const servers = useSavedServers();
    const here = useCurrentOrigin();
    if (!here || !servers.some(s => s.url !== here)) return null;
    const current = servers.find(s => s.url === here);
    const currentName = current?.name ?? defaultServerName(here);
    return {
        here,
        currentName,
        rows: [{ id: current?.id ?? 'here', name: currentName, url: here }, ...servers.filter(s => s.url !== here)],
    };
}

/** Mount once near the root, before any redirect can drop the fragment. */
export function useImportServersFromHash() {
    useEffect(() => { importServersFromHash(); }, []);
}
