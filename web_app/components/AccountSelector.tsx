import React, { useState, useRef, useEffect } from 'react';

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
        if (isAllSelected) return "All Accounts";
        if (selectedAccounts.length === 1) return selectedAccounts[0];
        return `${selectedAccounts.length} Accounts Selected`;
    };

    return (
        <div className="relative mb-4" ref={dropdownRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex justify-between w-full px-4 py-2 text-sm font-medium text-foreground bg-card border border-border rounded-lg shadow-sm hover:bg-accent/10 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 backdrop-blur-xl transition-all"
            >
                <span className="font-medium">{getLabel()}</span>
                <svg
                    className={`w-5 h-5 text-muted-foreground transition-transform ${isOpen ? 'transform rotate-180' : ''}`}
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
                            className={`group flex items-center justify-between w-full px-4 py-3 text-sm font-medium ${isAllSelected
                                ? 'bg-cyan-500/10 text-cyan-700 dark:text-cyan-400'
                                : 'text-popover-foreground'
                                } hover:bg-accent/10 transition-colors border-b border-border`}
                        >
                            <span className="whitespace-nowrap">All Accounts</span>
                            {isAllSelected && (
                                <svg className="w-4 h-4 text-cyan-600 dark:text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
                                    className={`group flex items-center justify-between w-full px-4 py-3 text-sm font-medium ${isSelected
                                        ? 'bg-cyan-500/10 text-cyan-700 dark:text-cyan-400'
                                        : 'text-popover-foreground'
                                        } hover:bg-accent/10 transition-colors border-b border-border last:border-0`}
                                >
                                    <span className="whitespace-nowrap">{account}</span>
                                    {isSelected && (
                                        <svg className="w-4 h-4 text-cyan-600 dark:text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
