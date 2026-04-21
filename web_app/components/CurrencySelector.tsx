import React, { useState, useRef, useEffect } from 'react';
import { cn, CURRENCY_SYMBOLS } from '@/lib/utils';
import { ChevronDown, Globe } from 'lucide-react';

const AVAILABLE_CURRENCIES = ['USD', 'THB', 'EUR', 'GBP', 'JPY', 'CNY'];

interface CurrencySelectorProps {
    currentCurrency: string;
    onChange: (currency: string) => void;
    fxRate?: number;
    side?: 'right' | 'bottom';
    availableCurrencies?: string[];
    align?: 'left' | 'right';
}

export default function CurrencySelector({ currentCurrency, onChange, fxRate, side = 'bottom', availableCurrencies, align = 'right' }: CurrencySelectorProps) {
    const currencies = availableCurrencies || AVAILABLE_CURRENCIES;

    const [isOpen, setIsOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, []);

    return (
        <div className="relative inline-block text-left" ref={dropdownRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={cn(
                    "flex flex-col items-center gap-1.5 p-3 rounded-2xl transition-all duration-300 group",
                    side === 'bottom' ? "bg-transparent" : "hover:bg-accent/10",
                    "font-semibold tracking-tight w-[60px]",
                    isOpen ? "ring-2 ring-cyan-500/20" : "text-cyan-500",
                    side === 'bottom' && "flex-row py-2 px-2 h-[44px] justify-center w-auto min-w-0"
                )}
                title={`Currency: ${currentCurrency}`}
            >
                <div className={cn(
                    "p-2 rounded-xl transition-all duration-300 hidden sm:block",
                    isOpen ? "bg-cyan-500 text-white" : "text-cyan-500 group-hover:scale-110",
                    side === 'bottom' && (isOpen ? "p-1" : "p-1 bg-secondary")
                )}>
                    <Globe className={cn(side === 'bottom' ? "w-3.5 h-3.5" : "w-5 h-5")} />
                </div>
                <div className="flex flex-col items-center leading-none gap-0">
                    <div className={cn(
                        "flex items-center gap-1 font-bold uppercase text-[14px]",
                        side === 'right' ? "hidden lg:block" : "block"
                    )}>
                        <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                            {currentCurrency}
                        </span>
                    </div>
                    {currentCurrency !== 'USD' && fxRate && (
                        <div className="text-[12px] text-cyan-600 dark:text-cyan-400 font-black font-mono mt-0.5">
                            {fxRate.toFixed(2)}
                        </div>
                    )}
                </div>
            </button>

            {isOpen && (
                <div className={cn(
                    "absolute overflow-hidden z-[100] transition-all animate-in fade-in zoom-in duration-200",
                    side === 'right'
                        ? "left-full top-0 ml-4 slide-in-from-left-2"
                        : cn("top-full mt-2 slide-in-from-top-2", align === 'left' ? "left-0" : "right-0"),
                    "w-32 rounded-2xl"
                )} style={{ backgroundColor: 'var(--menu-solid)' }}>
                    <div className="p-2 grid gap-1">
                        <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-widest text-cyan-500 mb-1">
                            Currencies
                        </div>
                        {currencies.map(currency => (
                            <button
                                key={currency}
                                onClick={() => {
                                    onChange(currency);
                                    setIsOpen(false);
                                }}
                                className={cn(
                                    "flex items-center w-full px-3 py-2 text-sm font-medium rounded-xl transition-all duration-200 group text-left",
                                    currentCurrency === currency
                                        ? "bg-[#0097b2] text-white"
                                        : "text-popover-foreground hover:bg-black/5 dark:hover:bg-white/5"
                                )}
                            >
                                <span className="flex-1 flex items-center">
                                    <span className="w-14 font-bold uppercase tracking-wider">{currency}</span>
                                    <span className={cn(
                                        "text-lg font-normal leading-none",
                                        currentCurrency === currency ? "text-cyan-100" : "text-cyan-500"
                                    )}>
                                        {CURRENCY_SYMBOLS[currency] || ''}
                                    </span>
                                </span>
                                {currentCurrency === currency && (
                                    <div className="w-1.5 h-1.5 rounded-full bg-white" />
                                )}
                            </button>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

