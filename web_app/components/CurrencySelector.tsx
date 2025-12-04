import React, { useState, useRef, useEffect } from 'react';

interface CurrencySelectorProps {
    currentCurrency: string;
    onChange: (currency: string) => void;
}

const AVAILABLE_CURRENCIES = ['USD', 'THB', 'EUR', 'GBP', 'JPY', 'CNY'];

export default function CurrencySelector({ currentCurrency, onChange }: CurrencySelectorProps) {
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
                className="text-xs font-medium px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors flex items-center gap-1"
            >
                {currentCurrency}
                <svg className={`w-3 h-3 transition-transform ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
            </button>

            {isOpen && (
                <div className="absolute right-0 top-full mt-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg overflow-hidden z-50 min-w-[80px]">
                    {AVAILABLE_CURRENCIES.map(currency => (
                        <button
                            key={currency}
                            onClick={() => {
                                onChange(currency);
                                setIsOpen(false);
                            }}
                            className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-50 dark:hover:bg-gray-700 ${currentCurrency === currency
                                    ? 'text-blue-600 dark:text-blue-400 font-medium bg-blue-50 dark:bg-blue-900/20'
                                    : 'text-gray-700 dark:text-gray-300'
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
