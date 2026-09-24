import React, { useState, useRef, useEffect } from 'react';
import { cn, CURRENCY_SYMBOLS } from '@/lib/utils';
import { Check, ChevronDown } from 'lucide-react';

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
                type="button"
                onClick={() => setIsOpen(!isOpen)}
                aria-haspopup="true"
                aria-expanded={isOpen}
                aria-label={`Display currency: ${currentCurrency}`}
                className="select-trigger"
                title={`Currency: ${currentCurrency}${currentCurrency !== 'USD' && fxRate ? ` (1 USD = ${fxRate.toFixed(2)} ${currentCurrency})` : ''}`}
            >
                <span className="tabular-nums">{currentCurrency}</span>
                {currentCurrency !== 'USD' && fxRate && (
                    <span className={cn('text-xs text-muted-foreground tabular-nums', side === 'right' && 'hidden lg:inline')}>
                        {fxRate.toFixed(2)}
                    </span>
                )}
                <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" aria-hidden="true" />
            </button>

            {isOpen && (
                <div
                    role="group"
                    aria-label="Currencies"
                    className={cn(
                        "menu-panel absolute z-[100] w-40 animate-in fade-in zoom-in-95 duration-150",
                        side === 'right'
                            ? "left-full top-0 ml-2"
                            : cn("top-full mt-1.5", align === 'left' ? "left-0" : "right-0"),
                    )}
                >
                    <div className="menu-heading">Currency</div>
                    {currencies.map(currency => {
                        const selected = currentCurrency === currency;
                        return (
                            <button
                                key={currency}
                                type="button"
                                aria-pressed={selected}
                                onClick={() => {
                                    onChange(currency);
                                    setIsOpen(false);
                                }}
                                className="menu-item"
                            >
                                <span className="w-10 tabular-nums">{currency}</span>
                                <span className="flex-1 text-muted-foreground font-normal">{CURRENCY_SYMBOLS[currency] || ''}</span>
                                {selected && <Check className="w-4 h-4 text-primary" aria-hidden="true" />}
                            </button>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
