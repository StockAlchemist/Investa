'use client';

import React from 'react';
import { useStockModal } from '@/context/StockModalContext';
import { cn } from '@/lib/utils';
import StockIcon from './StockIcon';

interface StockTickerProps {
    symbol: string;
    currency?: string;
    className?: string;
    children?: React.ReactNode;
    showIcon?: boolean;
}

export default function StockTicker({ symbol, currency, className, children, showIcon = true }: StockTickerProps) {
    const { openStockDetail } = useStockModal();

    return (
        <button
            onClick={(e) => {
                e.stopPropagation();
                openStockDetail(symbol, currency);
            }}
            className={cn(
                "inline-flex items-center font-bold text-cyan-600 dark:text-cyan-400 hover:text-cyan-500 transition-colors bg-cyan-500/0 hover:bg-cyan-500/5 px-1.5 py-0.5 rounded-md -mx-1.5 gap-1.5",
                className
            )}
        >
            {showIcon && <StockIcon symbol={symbol} size={16} />}
            {children || symbol}
        </button>
    );
}
