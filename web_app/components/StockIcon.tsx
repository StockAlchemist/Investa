'use client';

import React, { useState, useEffect } from 'react';
import Image from 'next/image';
import { cn } from '@/lib/utils';

interface StockIconProps {
    symbol: string;
    size?: number | string;
    className?: string;
    domain?: string | null;
}

export default function StockIcon({ symbol, size = 24, className, domain }: StockIconProps) {
    const [error, setError] = useState(false);
    const [sourceIndex, setSourceIndex] = useState(0);

    // Reset error state when symbol changes
    useEffect(() => {
        setError(false);
        setSourceIndex(0);
    }, [symbol]);

    if (!symbol) return null;

    if (symbol === 'SET') {
        const effectiveSize = typeof size === 'number' ? size : 24; // Default to 24 if size is string for Image component
        return (
            <div
                className={cn(
                    "overflow-hidden rounded-full bg-white flex items-center justify-center",
                    className
                )}
                style={{ width: effectiveSize, height: effectiveSize }}
            >
                <Image
                    src="/set-logo-v2.png"
                    alt="SET"
                    width={effectiveSize}
                    height={effectiveSize}
                    className="object-contain p-[2px]"
                />
            </div>
        );
    }

    if (symbol.includes('IBKR')) {
        const effectiveSize = typeof size === 'number' ? size : 24;
        return (
            <div
                className={cn(
                    "overflow-hidden rounded-full bg-[#D62930] flex items-center justify-center", // IBKR red background
                    className
                )}
                style={{ width: effectiveSize, height: effectiveSize }}
            >
                <Image
                    src="/ibkr-logo.png"
                    alt="IBKR"
                    width={effectiveSize}
                    height={effectiveSize}
                    className="object-contain" // The logo itself might act as the full fill
                />
            </div>
        );
    }

    if (symbol.includes('WeBull')) {
        const effectiveSize = typeof size === 'number' ? size : 24;
        return (
            <div
                className={cn(
                    "overflow-hidden rounded-full bg-white flex items-center justify-center",
                    className
                )}
                style={{ width: effectiveSize, height: effectiveSize }}
            >
                <Image
                    src="/webull-logo.png"
                    alt="WeBull"
                    width={effectiveSize}
                    height={effectiveSize}
                    className="object-contain p-[2px]"
                />
            </div>
        );
    }

    if (symbol.toLowerCase().startsWith('scb')) {
        const effectiveSize = typeof size === 'number' ? size : 24;
        return (
            <div
                className={cn(
                    "overflow-hidden rounded-full bg-[#4E2A84] flex items-center justify-center",
                    className
                )}
                style={{ width: effectiveSize, height: effectiveSize }}
            >
                <Image
                    src="/scb-logo.png"
                    alt="SCB"
                    width={effectiveSize}
                    height={effectiveSize}
                    className="object-cover"
                />
            </div>
        );
    }
    if (symbol.toLowerCase().startsWith('es')) {
        const effectiveSize = typeof size === 'number' ? size : 24;
        return (
            <div
                className={cn(
                    "overflow-hidden rounded-full bg-[#ED1C24] flex items-center justify-center",
                    className
                )}
                style={{ width: effectiveSize, height: effectiveSize }}
            >
                <Image
                    src="/es-logo.png"
                    alt="ES"
                    width={effectiveSize}
                    height={effectiveSize}
                    className="object-contain"
                />
            </div>
        );
    }
    if (symbol === '$CASH' || symbol.includes('Cash ($)') || symbol === 'Cash') {
        const effectiveSize = typeof size === 'number' ? size : 24;
        return (
            <div
                className={cn(
                    "overflow-hidden rounded-full bg-white flex items-center justify-center",
                    className
                )}
                style={{ width: effectiveSize, height: effectiveSize }}
            >
                <Image
                    src="/cash-logo.png"
                    alt="CASH"
                    width={effectiveSize}
                    height={effectiveSize}
                    className="object-contain p-[2px]"
                />
            </div>
        );
    }
    // Brand mappings for improved domain lookup (same as in StockDetailModal originally)
    const brandMappings: Record<string, string> = {
        'GOOG': 'google.com',
        'GOOGL': 'google.com',
        'META': 'facebook.com',
        'BRK.B': 'berkshirehathaway.com',
        'BRK.A': 'berkshirehathaway.com',
        'PLTR': 'palantir.com',
    };

    const effectiveDomain = brandMappings[symbol] || domain;

    // Manual overrides for specific symbols to use local images
    const localOverrides: Record<string, string> = {
        'PLTR': '/pltr.png',
        'ASML': '/asml.png',
        'UNH': '/unh.png',
        'AAPL': '/aapl.png',
    };

    const sources = [
        localOverrides[symbol],
        `https://financialmodelingprep.com/image-stock/${symbol}.png`,
        effectiveDomain ? `https://logo.clearbit.com/${effectiveDomain}` : null,
        effectiveDomain ? `https://www.google.com/s2/favicons?domain=${effectiveDomain}&sz=128` : null,
    ].filter(Boolean) as string[];

    const handleError = () => {
        if (sourceIndex < sources.length - 1) {
            setSourceIndex(prev => prev + 1);
        } else {
            setError(true);
        }
    };

    const currentSource = sources[sourceIndex];

    const sizeStyle = typeof size === 'number' ? { width: size, height: size } : { width: size, height: size };

    // Generate a consistent color based on symbol
    const getInitialsColor = (str: string) => {
        const colors = [
            'bg-red-500', 'bg-orange-500', 'bg-amber-500', 'bg-yellow-500',
            'bg-lime-500', 'bg-green-500', 'bg-emerald-500', 'bg-teal-500',
            'bg-cyan-500', 'bg-sky-500', 'bg-blue-500', 'bg-indigo-500',
            'bg-violet-500', 'bg-purple-500', 'bg-fuchsia-500', 'bg-pink-500',
            'bg-rose-500'
        ];
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = str.charCodeAt(i) + ((hash << 5) - hash);
        }
        return colors[Math.abs(hash) % colors.length];
    };

    if (error || !currentSource) {
        return (
            <div
                className={cn(
                    "rounded-md flex items-center justify-center text-white font-bold uppercase shrink-0 overflow-hidden",
                    getInitialsColor(symbol),
                    className
                )}
                style={sizeStyle}
            >
                <span style={{ fontSize: typeof size === 'number' ? size * 0.5 : '0.7em' }}>
                    {symbol.slice(0, 1)}
                </span>
            </div>
        );
    }

    return (
        <img
            src={currentSource}
            alt={symbol}
            className={cn("rounded-md object-contain bg-white shrink-0", className)}
            style={sizeStyle}
            onError={handleError}
            loading="lazy"
        />
    );
}
