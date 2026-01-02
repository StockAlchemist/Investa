import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';
import { ChevronDown, Globe } from 'lucide-react';

interface CurrencySelectorProps {
    currentCurrency: string;
    onChange: (currency: string) => void;
    fxRate?: number;
    side?: 'right' | 'bottom';
}

const AVAILABLE_CURRENCIES = ['USD', 'THB', 'EUR', 'GBP', 'JPY', 'CNY'];

export default function CurrencySelector({ currentCurrency, onChange, fxRate, side = 'bottom' }: CurrencySelectorProps) {
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
                    "bg-white/5 hover:bg-white/10 dark:bg-black/20 dark:hover:bg-black/30",
                    "border border-white/10 dark:border-white/5 backdrop-blur-xl shadow-lg shadow-black/5",
                    "font-semibold tracking-tight min-w-[80px]",
                    isOpen ? "border-cyan-500/50 ring-2 ring-cyan-500/20" : "text-cyan-500",
                    side === 'bottom' && "flex-row py-2 px-4 h-[44px]"
                )}
                title={`Currency: ${currentCurrency}`}
            >
                <div className={cn(
                    "p-2 rounded-xl transition-all duration-300 hidden sm:block",
                    isOpen ? "bg-cyan-500 text-white shadow-lg shadow-cyan-500/30" : "bg-white/5 text-cyan-500 group-hover:scale-110",
                    side === 'bottom' && "p-1"
                )}>
                    <Globe className={cn(side === 'bottom' ? "w-3.5 h-3.5" : "w-5 h-5")} />
                </div>
                <div className="flex flex-col items-start leading-none gap-0">
                    <span className={cn(
                        "bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent truncate font-bold uppercase text-[14px]",
                        side === 'right' ? "hidden lg:block max-w-[40px]" : "block max-w-[60px]"
                    )}>
                        {currentCurrency}
                    </span>
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
                        : "right-0 top-full mt-2 slide-in-from-top-2",
                    "w-48 rounded-2xl bg-popover border border-border shadow-2xl shadow-black/40"
                )}>
                    <div className="p-2 grid gap-1">
                        <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-widest text-muted-foreground/50 border-b border-border mb-1">
                            Currencies
                        </div>
                        {AVAILABLE_CURRENCIES.map(currency => (
                            <button
                                key={currency}
                                onClick={() => {
                                    onChange(currency);
                                    setIsOpen(false);
                                }}
                                className={cn(
                                    "flex items-center w-full px-3 py-2 text-sm font-medium rounded-xl transition-all duration-200 group text-left",
                                    currentCurrency === currency
                                        ? "bg-brand text-brand-foreground"
                                        : "text-popover-foreground hover:bg-black/5 dark:hover:bg-white/5"
                                )}
                            >
                                <span className="flex-1">{currency}</span>
                                {currentCurrency === currency && (
                                    <div className="w-1.5 h-1.5 rounded-full bg-brand-foreground" />
                                )}
                            </button>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

