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
                    "flex items-center gap-2 px-4 py-2 text-xs font-medium transition-all duration-300",
                    "bg-white/5 hover:bg-white/10 dark:bg-black/20 dark:hover:bg-black/30",
                    "border border-white/10 dark:border-white/5 backdrop-blur-xl shadow-lg shadow-black/5 rounded-2xl",
                    isOpen ? "border-cyan-500/50 ring-2 ring-cyan-500/20" : "text-cyan-500"
                )}
            >
                <span className="font-semibold text-muted-foreground/80 uppercase tracking-tighter mr-1 border-r border-border pr-2 hidden sm:inline">Accounts</span>
                <span className="font-medium text-cyan-500">{getLabel()}</span>
                <svg
                    className={`w-3.5 h-3.5 text-cyan-500/60 transition-transform ${isOpen ? 'transform rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
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
