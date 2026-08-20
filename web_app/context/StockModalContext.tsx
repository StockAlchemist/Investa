'use client';

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';

interface StockModalContextType {
    selectedSymbol: string | null;
    symbolHistory: string[];
    modalCurrency: string;
    openStockDetail: (symbol: string, currency?: string) => void;
    closeStockDetail: () => void;
    goBack: () => void;
    canGoBack: boolean;
}

const StockModalContext = createContext<StockModalContextType | undefined>(undefined);

export function StockModalProvider({ children, defaultCurrency = 'USD' }: { children: ReactNode; defaultCurrency?: string }) {
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
    const [symbolHistory, setSymbolHistory] = useState<string[]>([]);
    const [modalCurrency, setModalCurrency] = useState(defaultCurrency);

    const openStockDetail = useCallback((symbol: string, currency?: string) => {
        if (!symbol) return;
        const cleanSymbol = symbol.trim().toUpperCase();
        if (currency) setModalCurrency(currency);
        
        setSelectedSymbol(prev => {
            if (prev && prev !== cleanSymbol) {
                setSymbolHistory(history => [...history, prev]);
            }
            return cleanSymbol;
        });
    }, []);

    const goBack = useCallback(() => {
        setSymbolHistory(history => {
            if (history.length > 0) {
                const nextHistory = [...history];
                const prevSymbol = nextHistory.pop()!;
                setSelectedSymbol(prevSymbol);
                return nextHistory;
            } else {
                setSelectedSymbol(null);
                return [];
            }
        });
    }, []);

    const closeStockDetail = useCallback(() => {
        setSelectedSymbol(null);
        setSymbolHistory([]);
    }, []);

    return (
        <StockModalContext.Provider
            value={{
                selectedSymbol,
                symbolHistory,
                modalCurrency,
                openStockDetail,
                closeStockDetail,
                goBack,
                canGoBack: symbolHistory.length > 0,
            }}
        >
            {children}
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

