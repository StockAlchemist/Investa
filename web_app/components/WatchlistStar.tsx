'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Star } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useWatchlist } from '@/context/WatchlistContext';
import StockIcon from './StockIcon';
import { createPortal } from 'react-dom';

interface WatchlistStarProps {
    symbol: string;
    size?: "sm" | "md" | "lg" | number;
    className?: string;
    iconClassName?: string;
    showDropdown?: boolean;
}

export default function WatchlistStar({ symbol, size = "md", className, iconClassName, showDropdown = true }: WatchlistStarProps) {
    const { watchlists, symbolWatchlistMap, toggleWatchlist } = useWatchlist();
    const [openWatchlistDropdown, setOpenWatchlistDropdown] = useState(false);
    const [dropdownPosition, setDropdownPosition] = useState({ top: 0, left: 0, right: 0, width: 0 });
    const dropdownRef = useRef<HTMLDivElement>(null);
    const starButtonRef = useRef<HTMLButtonElement>(null);

    const isStarred = Array.from(symbolWatchlistMap[symbol] || []).length > 0;

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node) && 
                starButtonRef.current && !starButtonRef.current.contains(event.target as Node)) {
                setOpenWatchlistDropdown(false);
            }
        };

        if (openWatchlistDropdown) {
            document.addEventListener('mousedown', handleClickOutside);
            // Calculate position
            if (starButtonRef.current) {
                const rect = starButtonRef.current.getBoundingClientRect();
                
                // Estimate height (approx 40px per list item + 32px header + padding)
                const estimatedHeight = 32 + (watchlists.length * 40) + 16;
                const windowHeight = window.innerHeight;
                const spaceBelow = windowHeight - rect.bottom;
                const shouldShowAbove = spaceBelow < estimatedHeight && rect.top > estimatedHeight;

                setDropdownPosition({
                    top: shouldShowAbove ? rect.top - estimatedHeight : rect.bottom, 
                    left: rect.left,
                    right: rect.right,
                    width: rect.width
                });
            }
        }
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, [openWatchlistDropdown, watchlists.length]);

    const getIconSize = () => {
        if (typeof size === 'number') return size;
        switch (size) {
            case "sm": return 24;
            case "md": return 40;
            case "lg": return 64;
            default: return 40;
        }
    };

    const effectiveSize = getIconSize();

    return (
        <div className={cn("relative group/star flex-shrink-0", className)}>
            <StockIcon
                symbol={symbol}
                size={effectiveSize}
                className={cn(
                    "rounded-lg bg-white dark:bg-gray-800 shadow-sm p-1 border border-border",
                    iconClassName
                )}
            />
            <button
                ref={starButtonRef}
                onClick={(e) => {
                    e.stopPropagation();
                    if (showDropdown && watchlists.length > 0) {
                        setOpenWatchlistDropdown(!openWatchlistDropdown);
                    } else {
                        toggleWatchlist(symbol, 1);
                    }
                }}
                className={cn(
                    "absolute -top-1.5 -right-1.5 p-1 rounded-full border bg-white dark:bg-zinc-900 shadow-sm transition-all z-10",
                    isStarred
                        ? "text-yellow-500 border-yellow-200 dark:border-yellow-900 bg-yellow-50 dark:bg-yellow-950/30 scale-100"
                        : "text-muted-foreground/30 opacity-0 group-hover/star:opacity-100 hover:text-yellow-500 hover:scale-110",
                    openWatchlistDropdown && "opacity-100 ring-1 ring-indigo-500"
                )}
            >
                <Star className={cn("h-3 w-3", isStarred && "fill-current")} />
            </button>

            {openWatchlistDropdown && showDropdown && watchlists.length > 0 && typeof document !== 'undefined' && createPortal(
                <div
                    ref={dropdownRef}
                    className="fixed w-48 bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-xl shadow-2xl z-[9999] overflow-hidden py-1 animate-in fade-in zoom-in-95 duration-200"
                    style={{
                        // eslint-disable-next-line react-hooks/refs -- reads the trigger button's rect during render to position the portal dropdown
                        top: dropdownPosition.top + (dropdownPosition.top < (starButtonRef.current?.getBoundingClientRect().top ?? 0) ? -8 : 8),
                        left: (dropdownPosition.right - 192 < 10) 
                            ? dropdownPosition.left 
                            : dropdownPosition.right - 192,
                    }}
                >
                    <div className="px-3 py-2 text-[10px] font-bold text-muted-foreground uppercase tracking-wider border-b border-zinc-100 dark:border-zinc-800/50">
                        Add to Watchlist
                    </div>
                    {watchlists.map((wl) => {
                        const isInThisList = symbolWatchlistMap[symbol]?.has(wl.id);
                        return (
                            <button
                                key={wl.id}
                                onClick={(e) => {
                                    e.stopPropagation();
                                    toggleWatchlist(symbol, wl.id);
                                }}
                                className="w-full px-3 py-2 text-xs font-medium flex items-center justify-between hover:bg-zinc-100 dark:hover:bg-zinc-800/50 transition-colors text-foreground"
                            >
                                <span>{wl.name}</span>
                                {isInThisList && (
                                    <div className="w-2 h-2 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.5)]" />
                                )}
                            </button>
                        );
                    })}
                </div>,
                document.body
            )}
        </div>
    );
}
