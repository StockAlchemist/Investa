import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';

interface AccountSelectorProps {
    availableAccounts: string[];
    selectedAccounts: string[];
    onChange: (accounts: string[]) => void;
}

export default function AccountSelector({ availableAccounts, selectedAccounts, onChange }: AccountSelectorProps) {
    const [isOpen, setIsOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    // Close dropdown when clicking outside
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

    const toggleAccount = (account: string) => {
        if (selectedAccounts.includes(account)) {
            onChange(selectedAccounts.filter(a => a !== account));
        } else {
            onChange([...selectedAccounts, account]);
        }
    };

    const handleSelectAll = () => {
        onChange([]); // Empty means all
    };

    const isAllSelected = selectedAccounts.length === 0;

    const getLabel = () => {
        if (isAllSelected) return (
            <>
                <span className="hidden sm:inline">All Accounts</span>
                <span className="sm:hidden">All</span>
            </>
        );
        if (selectedAccounts.length === 1) return selectedAccounts[0];
        return (
            <>
                <span className="hidden sm:inline">{selectedAccounts.length} Accounts Selected</span>
                <span className="sm:hidden">{selectedAccounts.length} Accs</span>
            </>
        );
    };

    return (
        <div className="relative" ref={dropdownRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={cn(
                    "flex flex-col items-center justify-center gap-1.5 p-3 rounded-2xl transition-all duration-300 group",
                    "bg-white/5 hover:bg-white/10 dark:bg-black/20 dark:hover:bg-black/30",
                    "border border-white/10 dark:border-white/5 backdrop-blur-xl shadow-lg shadow-black/5",
                    "font-semibold tracking-tight min-w-[80px]",
                    isOpen ? "border-cyan-500/50 ring-2 ring-cyan-500/20" : "text-cyan-500",
                    // Use a flex-row layout on mobile or if aligned in a row context if needed, 
                    // but for now relying on the base 'currency' style which is flex-col usually, 
                    // but the user's screenshot suggests a simple centered text box.
                    // CurrencySelector has: side === 'bottom' && "flex-row py-2 px-4 h-[44px] justify-center"
                    // We'll adopt a similar height/padding for consistency.
                    "flex-row py-2 px-4 h-[44px]"
                )}
            >
                <div className="flex flex-col items-center leading-none gap-0">
                    <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent truncate font-bold uppercase text-[14px]">
                        {getLabel()}
                    </span>
                </div>
            </button>

            {isOpen && (
                <div className="absolute right-0 top-full mt-2 min-w-full w-max origin-top-right bg-popover border border-border rounded-xl shadow-xl outline-none z-50 overflow-hidden">
                    <div className="py-1 max-h-60 overflow-y-auto">
                        <button
                            onClick={handleSelectAll}
                            className={`group flex items-center justify-between w-full px-4 py-3 text-sm font-medium transition-colors ${isAllSelected
                                ? 'bg-brand text-brand-foreground'
                                : 'text-popover-foreground hover:bg-black/5 dark:hover:bg-white/5'
                                }`}
                        >
                            <span className="whitespace-nowrap">All Accounts</span>
                            {isAllSelected && (
                                <svg className="w-4 h-4 text-brand-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                </svg>
                            )}
                        </button>

                        {availableAccounts.map((account) => {
                            const isSelected = selectedAccounts.includes(account);
                            return (
                                <button
                                    key={account}
                                    onClick={() => toggleAccount(account)}
                                    className={`group flex items-center justify-between w-full px-4 py-3 text-sm font-medium transition-colors ${isSelected
                                        ? 'bg-brand text-brand-foreground'
                                        : 'text-popover-foreground hover:bg-black/5 dark:hover:bg-white/5'
                                        } last:border-0`}
                                >
                                    <span className="whitespace-nowrap">{account}</span>
                                    {isSelected && (
                                        <svg className="w-4 h-4 text-brand-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                        </svg>
                                    )}
                                </button>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}
