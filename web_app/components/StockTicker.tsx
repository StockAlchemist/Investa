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

    const handleClick = (e: React.MouseEvent) => {
        e.stopPropagation();
        openStockDetail(symbol, currency);
    };

    return (
        <div className={cn("inline-flex items-center gap-1.5", className)}>
            {showIcon && (
                showStar ? (
                    <WatchlistStar
                        symbol={symbol}
                        size="md"
                        showDropdown={false}
                        onIconClick={() => openStockDetail(symbol, currency)}
                    />
                ) : (
                    <button
                        type="button"
                        onClick={handleClick}
                        className="cursor-pointer hover:opacity-80 transition-opacity focus:outline-none"
                    >
                        <StockIcon symbol={symbol} size={36} />
                    </button>
                )
            )}
            <button
                type="button"
                onClick={handleClick}
                className="font-bold text-indigo-600 dark:text-indigo-400 hover:text-indigo-500 transition-colors bg-indigo-500/0 hover:bg-indigo-500/5 px-1.5 py-0.5 rounded-md -mx-1.5 cursor-pointer"
            >
                {children || symbol}
            </button>
        </div>
    );
}
