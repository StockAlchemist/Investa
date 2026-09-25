'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Check, ChevronDown, Heart, ListPlus, Plus } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useWatchlist } from '@/context/WatchlistContext';

/**
 * Favourite and watchlist controls for the stock window.
 *
 * The heart is the one-tap path: it toggles the built-in Favorites list, which
 * the server creates the first time it is used. The menu beside it covers every
 * other list — a check marks the ones already holding the stock — and can make
 * a new list with the stock already on it.
 */
export function StockWatchlistActions({ symbol }: { symbol: string }) {
    const { watchlists, symbolWatchlistMap, toggleWatchlist, isFavorite, toggleFavorite, createListWithSymbol } =
        useWatchlist();
    const [open, setOpen] = useState(false);
    const [creating, setCreating] = useState(false);
    const [newName, setNewName] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);
    const [coords, setCoords] = useState<{ top: number; right: number }>({ top: 0, right: 0 });
    const triggerRef = useRef<HTMLButtonElement>(null);
    const menuRef = useRef<HTMLDivElement>(null);

    const favorite = isFavorite(symbol);
    const memberships = symbolWatchlistMap[symbol] ?? new Set<number>();
    const listCount = memberships.size;

    const close = useCallback(() => {
        setOpen(false);
        setCreating(false);
        setNewName('');
        setError(null);
    }, []);

    const place = useCallback(() => {
        const rect = triggerRef.current?.getBoundingClientRect();
        if (rect) setCoords({ top: rect.bottom + 8, right: window.innerWidth - rect.right });
    }, []);

    useEffect(() => {
        if (!open) return;
        place();
        const onPointer = (event: MouseEvent) => {
            const target = event.target as Node;
            if (!menuRef.current?.contains(target) && !triggerRef.current?.contains(target)) close();
        };
        const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') close(); };
        document.addEventListener('mousedown', onPointer);
        document.addEventListener('keydown', onKey);
        window.addEventListener('resize', place);
        window.addEventListener('scroll', place, true);
        return () => {
            document.removeEventListener('mousedown', onPointer);
            document.removeEventListener('keydown', onKey);
            window.removeEventListener('resize', place);
            window.removeEventListener('scroll', place, true);
        };
    }, [open, close, place]);

    const onFavorite = async () => {
        setBusy(true);
        try {
            await toggleFavorite(symbol);
        } catch {
            // The heart simply stays as it was; the list screen is the fallback.
        } finally {
            setBusy(false);
        }
    };

    const onCreate = async (event: React.FormEvent) => {
        event.preventDefault();
        if (!newName.trim()) return;
        setBusy(true);
        setError(null);
        try {
            await createListWithSymbol(newName, symbol);
            setCreating(false);
            setNewName('');
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Could not create the list');
        } finally {
            setBusy(false);
        }
    };

    return (
        <div className="flex items-center gap-2">
            <button
                type="button"
                onClick={onFavorite}
                disabled={busy}
                aria-pressed={favorite}
                aria-label={favorite ? `Remove ${symbol} from Favorites` : `Add ${symbol} to Favorites`}
                className={cn(
                    'inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border text-xs sm:text-sm font-semibold transition-colors whitespace-nowrap',
                    favorite
                        ? 'border-primary/40 bg-primary/10 text-primary'
                        : 'border-border/60 bg-card text-foreground hover:bg-accent'
                )}
            >
                <Heart className={cn('w-4 h-4', favorite && 'fill-current')} aria-hidden />
                <span className="hidden sm:inline">{favorite ? 'Favorited' : 'Favorite'}</span>
            </button>

            <button
                ref={triggerRef}
                type="button"
                onClick={() => (open ? close() : setOpen(true))}
                aria-haspopup="true"
                aria-expanded={open}
                aria-label={`Watchlists for ${symbol}`}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-border/60 bg-card text-foreground hover:bg-accent text-xs sm:text-sm font-semibold transition-colors whitespace-nowrap"
            >
                <ListPlus className="w-4 h-4" aria-hidden />
                <span className="hidden sm:inline">Watchlists</span>
                {listCount > 0 && (
                    <span className="rounded-full bg-muted px-1.5 text-[11px] tabular-nums text-ink-2">{listCount}</span>
                )}
                <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" aria-hidden />
            </button>

            {open && typeof document !== 'undefined' && createPortal(
                <div
                    ref={menuRef}
                    role="group"
                    aria-label={`Watchlists for ${symbol}`}
                    style={{ position: 'fixed', top: coords.top, right: coords.right, backgroundColor: 'var(--menu-solid)' }}
                    className="menu-panel z-[9999] w-64 animate-in fade-in zoom-in-95 duration-150"
                >
                    <div className="menu-heading">Add {symbol} to</div>
                    <div className="max-h-[50vh] overflow-y-auto">
                        {watchlists.length === 0 && (
                            <p className="px-2.5 py-2 text-xs text-muted-foreground">No watchlists yet.</p>
                        )}
                        {watchlists.map(wl => {
                            const member = memberships.has(wl.id);
                            return (
                                <button
                                    key={wl.id}
                                    type="button"
                                    aria-checked={member}
                                    role="menuitemcheckbox"
                                    onClick={() => toggleWatchlist(symbol, wl.id)}
                                    className="menu-item justify-between"
                                >
                                    <span className="flex min-w-0 items-center gap-2">
                                        {wl.is_favorites && <Heart className="w-3.5 h-3.5 shrink-0 text-muted-foreground" aria-hidden />}
                                        <span className="truncate">{wl.name}</span>
                                    </span>
                                    {member && <Check className="w-4 h-4 shrink-0 text-primary" aria-hidden />}
                                </button>
                            );
                        })}
                    </div>
                    <div className="my-1 h-px bg-border" />
                    {creating ? (
                        <form onSubmit={onCreate} className="space-y-1.5 p-1.5">
                            <input
                                autoFocus
                                value={newName}
                                onChange={e => { setNewName(e.target.value); setError(null); }}
                                placeholder="List name"
                                aria-label="New watchlist name"
                                className="h-8 w-full rounded-md border border-border bg-card px-2 text-sm outline-none focus:border-primary"
                            />
                            {error && <p className="text-xs text-down">{error}</p>}
                            <div className="flex justify-end gap-1.5">
                                <button type="button" onClick={() => { setCreating(false); setError(null); }}
                                        className="rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-muted">
                                    Cancel
                                </button>
                                <button type="submit" disabled={busy || !newName.trim()}
                                        className="rounded-md bg-primary px-2.5 py-1 text-xs font-semibold text-primary-foreground disabled:opacity-50">
                                    Create &amp; add
                                </button>
                            </div>
                        </form>
                    ) : (
                        <button type="button" onClick={() => setCreating(true)} className="menu-item">
                            <Plus className="w-4 h-4" aria-hidden />
                            <span>New list…</span>
                        </button>
                    )}
                </div>,
                document.body
            )}
        </div>
    );
}
