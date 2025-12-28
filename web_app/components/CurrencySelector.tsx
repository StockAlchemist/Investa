import React, { useState, useRef, useEffect } from 'react';

interface CurrencySelectorProps {
    currentCurrency: string;
    onChange: (currency: string) => void;
    fxRate?: number;
}

const AVAILABLE_CURRENCIES = ['USD', 'THB', 'EUR', 'GBP', 'JPY', 'CNY'];

export default function CurrencySelector({ currentCurrency, onChange, fxRate }: CurrencySelectorProps) {
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
        <div className="relative" ref={dropdownRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="text-xs font-medium px-2 py-1 bg-black/5 dark:bg-white/5 border border-black/5 dark:border-white/10 rounded text-foreground hover:bg-black/10 dark:hover:bg-white/10 transition-colors flex items-center gap-2 backdrop-blur-md"
            >
                {currentCurrency !== 'USD' && fxRate && (
                    <span className="text-muted-foreground mr-1 border-r border-border pr-2">
                        {fxRate.toFixed(2)}
                    </span>
                )}
                <span>{currentCurrency}</span>
                <svg className={`w-3 h-3 transition-transform ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
            </button>

            {isOpen && (
                <div className="absolute right-0 top-full mt-1 bg-popover border border-border rounded-lg shadow-xl overflow-hidden z-50 min-w-[80px]">
                    {AVAILABLE_CURRENCIES.map(currency => (
                        <button
                            key={currency}
                            onClick={() => {
                                onChange(currency);
                                setIsOpen(false);
                            }}
                            className={`w-full text-left px-3 py-2 text-sm hover:bg-black/5 dark:hover:bg-white/10 transition-colors ${currentCurrency === currency
                                ? 'text-cyan-600 dark:text-cyan-400 font-medium bg-cyan-500/10'
                                : 'text-popover-foreground'
                                }`}
                        >
                            {currency}
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}
