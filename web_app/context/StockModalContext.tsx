'use client';

import React, { createContext, useContext, useState, ReactNode } from 'react';
import StockDetailModal from '@/components/StockDetailModal';

interface StockModalContextType {
    openStockDetail: (symbol: string, currency?: string) => void;
    closeStockDetail: () => void;
}

const StockModalContext = createContext<StockModalContextType | undefined>(undefined);

export function StockModalProvider({ children, defaultCurrency = 'USD' }: { children: ReactNode; defaultCurrency?: string }) {
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
    const [modalCurrency, setModalCurrency] = useState(defaultCurrency);

    const openStockDetail = (symbol: string, currency?: string) => {
        if (currency) setModalCurrency(currency);
        setSelectedSymbol(symbol);
    };

    const closeStockDetail = () => {
        setSelectedSymbol(null);
    };

    return (
        <StockModalContext.Provider value={{ openStockDetail, closeStockDetail }}>
            {children}
            {selectedSymbol && (
                <StockDetailModal
                    symbol={selectedSymbol}
                    isOpen={!!selectedSymbol}
                    onClose={closeStockDetail}
                    currency={modalCurrency}
                />
            )}
        </StockModalContext.Provider>
    );
}

export function useStockModal() {
    const context = useContext(StockModalContext);
    if (context === undefined) {
        throw new Error('useStockModal must be used within a StockModalProvider');
    }
    return context;
}
