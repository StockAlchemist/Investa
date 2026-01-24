'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Star, Check } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useWatchlist } from '@/context/WatchlistContext';
import StockIcon from './StockIcon';

interface WatchlistStarProps {
    symbol: string;
    size?: "sm" | "md" | "lg" | number;
    className?: string;
    iconClassName?: string;
    showDropdown?: boolean;
}

export default function WatchlistStar({ symbol, size = "md", className, iconClassName, showDropdown = true }: WatchlistStarProps) {
    const { watchlists, symbolWatchlistMap, starredSymbols, toggleWatchlist } = useWatchlist();
    const [openWatchlistDropdown, setOpenWatchlistDropdown] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    const isStarred = starredSymbols.has(symbol);

    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setOpenWatchlistDropdown(false);
            }
        }
        if (openWatchlistDropdown) {
            document.addEventListener("mousedown", handleClickOutside);
        }
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [openWatchlistDropdown]);

    const getIconSize = () => {
        if (typeof size === 'number') return size;
        switch (size) {
            case "sm": return 24;
            case "lg": return 48;
            default: return 36;
        }
    };

    const effectiveSize = getIconSize();

    return (
        <div className={cn("relative group/star flex-shrink-0", className)} ref={dropdownRef}>
            <StockIcon
                symbol={symbol}
                size={effectiveSize}
                className={cn(
                    "rounded-lg bg-white dark:bg-gray-800 shadow-sm p-1 border border-border",
                    iconClassName
                )}
            />
            <button
                onClick={(e) => {
                    e.stopPropagation();
                    if (showDropdown && watchlists.length > 0) {
                        setOpenWatchlistDropdown(!openWatchlistDropdown);
                    } else {
                        // Toggle default watchlist (id 1) if no dropdown wanted or only 1 list
                        toggleWatchlist(symbol, 1);
                    }
                }}
                className={cn(
                    "absolute -top-1.5 -right-1.5 p-1 rounded-full border bg-background shadow-sm transition-all z-10",
                    isStarred
                        ? "text-yellow-500 border-yellow-200 dark:border-yellow-900 bg-yellow-50 dark:bg-yellow-950/30 scale-100"
                        : "text-muted-foreground/30 opacity-0 group-hover/star:opacity-100 hover:text-yellow-500 hover:scale-110"
                )}
            >
                <Star className={cn("h-3 w-3", isStarred && "fill-current")} />
            </button>

            {openWatchlistDropdown && showDropdown && watchlists.length > 0 && (
                <div
                    className="absolute left-0 top-full mt-2 w-48 bg-card border border-border rounded-xl shadow-xl z-50 overflow-hidden py-1 animate-in fade-in zoom-in-95 duration-200"
                    style={{ backgroundColor: 'var(--menu-solid)' }}
                >
                    <div className="px-3 py-2 text-[10px] font-bold text-muted-foreground uppercase tracking-wider border-b border-border/50">
                        Add to Watchlist
                    </div>
                    {watchlists.map((wl) => {
                        const isInList = symbolWatchlistMap[symbol]?.has(wl.id);
                        return (
                            <button
                                key={wl.id}
                                onClick={(e) => {
                                    e.stopPropagation();
                                    toggleWatchlist(symbol, wl.id);
                                }}
                                className="w-full px-3 py-2 text-xs font-medium flex items-center justify-between hover:bg-accent/5 transition-colors text-foreground"
                            >
                                <span>{wl.name}</span>
                                {isInList && <Check className="h-3 w-3 text-cyan-500" />}
                            </button>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
