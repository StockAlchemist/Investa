'use client';

import React from 'react';
import { useStockModal } from '@/context/StockModalContext';
import { cn } from '@/lib/utils';
import StockIcon from './StockIcon';
import WatchlistStar from './WatchlistStar';

interface StockTickerProps {
    symbol: string;
    currency?: string;
    className?: string;
    children?: React.ReactNode;
    showIcon?: boolean;
    showStar?: boolean;
}

export default function StockTicker({ symbol, currency, className, children, showIcon = true, showStar = true }: StockTickerProps) {
    const { openStockDetail } = useStockModal();

    return (
        <div className={cn("inline-flex items-center gap-1.5", className)}>
            {showIcon && (showStar ? <WatchlistStar symbol={symbol} size="md" showDropdown={false} className="" /> : <StockIcon symbol={symbol} size={36} />)}
            <button
                onClick={(e) => {
                    e.stopPropagation();
                    openStockDetail(symbol, currency);
                }}
                className="font-bold text-cyan-600 dark:text-cyan-400 hover:text-cyan-500 transition-colors bg-cyan-500/0 hover:bg-cyan-500/5 px-1.5 py-0.5 rounded-md -mx-1.5"
            >
                {children || symbol}
            </button>
        </div>
    );
}
