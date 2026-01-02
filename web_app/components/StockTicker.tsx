'use client';

import React from 'react';
import { useStockModal } from '@/context/StockModalContext';
import { cn } from '@/lib/utils';

interface StockTickerProps {
    symbol: string;
    currency?: string;
    className?: string;
    children?: React.ReactNode;
}

export default function StockTicker({ symbol, currency, className, children }: StockTickerProps) {
    const { openStockDetail } = useStockModal();

    return (
        <button
            onClick={(e) => {
                e.stopPropagation();
                openStockDetail(symbol, currency);
            }}
            className={cn(
                "inline-flex items-center font-bold text-cyan-600 dark:text-cyan-400 hover:text-cyan-500 transition-colors bg-cyan-500/0 hover:bg-cyan-500/5 px-1.5 py-0.5 rounded-md -mx-1.5",
                className
            )}
        >
            {children || symbol}
        </button>
    );
}
