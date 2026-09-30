import { beforeEach, describe, expect, it } from 'vitest';
import {
    addServer,
    defaultServerName,
    importServersFromHash,
    looksLikeApiUrl,
    normalizeServerUrl,
    removeServer,
} from '../../lib/servers';

const stored = () => JSON.parse(window.localStorage.getItem('investa_saved_servers') ?? '[]');

describe('saved servers', () => {
    beforeEach(() => {
        window.localStorage.clear();
        window.history.replaceState(null, '', '/');
    });

    it('normalizes an address to its origin', () => {
        expect(normalizeServerUrl('100.127.10.38:3000')).toBe('http://100.127.10.38:3000');
        expect(normalizeServerUrl(' https://mac.tail1234.ts.net/some/path ')).toBe('https://mac.tail1234.ts.net');
        expect(normalizeServerUrl('javascript:alert(1)')).toBeNull();
        expect(normalizeServerUrl('')).toBeNull();
    });

    it('recognises the backend API address', () => {
        expect(looksLikeApiUrl('http://100.127.10.38:8000/api')).toBe(true);
        expect(looksLikeApiUrl('100.127.10.38:8000/api/')).toBe(true);
        expect(looksLikeApiUrl('http://100.127.10.38:3000')).toBe(false);
    });

    it('names an unnamed server by host and port', () => {
        expect(defaultServerName('http://100.127.10.38:3000')).toBe('100.127.10.38:3000');
    });

    it('renames rather than duplicates an address already saved', () => {
        addServer('', 'http://100.127.10.38:3000/');
        addServer('Home Mac', '100.127.10.38:3000');
        expect(stored()).toHaveLength(1);
        expect(stored()[0]).toMatchObject({ name: 'Home Mac', url: 'http://100.127.10.38:3000' });
        removeServer(stored()[0].id);
        expect(stored()).toHaveLength(0);
    });

    it('merges a carried list, keeps local names, and clears the fragment', () => {
        addServer('Home Mac', 'http://100.127.10.38:3000');
        const carried = [
            { name: 'Renamed elsewhere', url: 'http://100.127.10.38:3000' },
            { name: 'Laptop', url: 'http://100.66.59.98:3000' },
            { name: 'Bad', url: 'javascript:alert(1)' },
        ];
        window.history.replaceState(null, '', `/#investa-servers=${encodeURIComponent(JSON.stringify(carried))}`);
        importServersFromHash();
        expect(stored().map((s: { name: string }) => s.name)).toEqual(['Home Mac', 'Laptop']);
        expect(window.location.hash).toBe('');
    });
});
