import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';

interface AccountSelectorProps {
    availableAccounts: string[];
    selectedAccounts: string[];
    onChange: (accounts: string[]) => void;
    accountGroups?: Record<string, string[]>;
}

export default function AccountSelector({ availableAccounts, selectedAccounts, onChange, accountGroups = {} }: AccountSelectorProps) {
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

    const handleSelectGroup = (groupName: string, accounts: string[]) => {
        // Filter accounts to only include those that are actually available
        const validAccounts = accounts.filter(acc => availableAccounts.includes(acc));
        onChange(validAccounts);
        setIsOpen(false); // Optional: close on group selection? Or keep open? Let's keep open for consistency if it acts like a preset.
        // Actually, usually presets replace the selection, so closing might feel nicer.
        // But if we treat it as "Select these", user might want to tweak. 
        // Let's decide based on UX. The desktop app behavior is unknown, but usually groups act as quick-filters.
        // Let's keep it open to allow refinement, or close it if it feels like a "Navigation" action.
        // Given it's a multi-select, replacing current selection seems appropriate.
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

    const hasGroups = Object.keys(accountGroups).length > 0;

    return (
        <div className="relative" ref={dropdownRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={cn(
                    "flex flex-col items-center justify-center gap-1.5 p-3 rounded-2xl transition-all duration-300 group",
                    "bg-card hover:bg-accent/10",
                    "border border-border shadow-sm",
                    "font-semibold tracking-tight min-w-[80px]",
                    isOpen ? "border-cyan-500/50 ring-2 ring-cyan-500/20" : "text-cyan-500",
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
                <div
                    style={{ backgroundColor: 'var(--menu-solid)' }}
                    className="absolute right-0 top-full mt-2 min-w-[200px] w-max origin-top-right border border-border rounded-xl shadow-xl outline-none z-50 overflow-hidden"
                >
                    <div className="py-1 max-h-[80vh] overflow-y-auto">
                        {/* All Accounts Option */}
                        <button
                            onClick={handleSelectAll}
                            className={`group flex items-center justify-between w-full px-4 py-3 text-sm font-medium transition-colors ${isAllSelected
                                ? 'bg-[#0097b2] text-white shadow-sm'
                                : 'text-popover-foreground hover:bg-black/5 dark:hover:bg-white/5'
                                }`}
                        >
                            <span className="whitespace-nowrap">All Accounts</span>
                            {isAllSelected && (
                                <div className="w-1.5 h-1.5 rounded-full bg-white" />
                            )}
                        </button>

                        {/* Account Groups Section */}
                        {hasGroups && (
                            <>
                                <div className="px-4 py-2 text-xs font-bold text-cyan-600 dark:text-cyan-400 uppercase tracking-wider bg-cyan-50/50 dark:bg-cyan-950/20 border-y border-border/50">
                                    Groups
                                </div>
                                {Object.entries(accountGroups).map(([groupName, groupAccounts]) => {
                                    // Check if this group is currently exactly selected (ignoring order)
                                    const isGroupSelected = !isAllSelected &&
                                        selectedAccounts.length === groupAccounts.length &&
                                        groupAccounts.every(acc => selectedAccounts.includes(acc));

                                    return (
                                        <button
                                            key={groupName}
                                            onClick={() => handleSelectGroup(groupName, groupAccounts)}
                                            className={`group flex items-center justify-between w-full px-4 py-2 text-sm font-medium transition-colors ${isGroupSelected
                                                ? 'bg-[#0097b2] text-white shadow-sm'
                                                : 'text-popover-foreground hover:bg-black/5 dark:hover:bg-white/5'
                                                }`}
                                        >
                                            <span className="whitespace-nowrap">{groupName}</span>
                                            {isGroupSelected && (
                                                <div className="w-1.5 h-1.5 rounded-full bg-white" />
                                            )}
                                        </button>
                                    );
                                })}
                            </>
                        )}


                        {/* Individual Accounts Section */}
                        {hasGroups && (
                            <div className="px-4 py-2 text-xs font-bold text-cyan-600 dark:text-cyan-400 uppercase tracking-wider bg-cyan-50/50 dark:bg-cyan-950/20 border-y border-border/50">
                                Individual
                            </div>
                        )}

                        {availableAccounts
                            .filter(account => account !== 'All Accounts')
                            .map((account) => {
                                const isSelected = selectedAccounts.includes(account);
                                return (
                                    <button
                                        key={account}
                                        onClick={() => toggleAccount(account)}
                                        className={`group flex items-center justify-between w-full px-4 py-3 text-sm font-medium transition-colors ${isSelected
                                            ? 'bg-[#0097b2] text-white shadow-sm'
                                            : 'text-popover-foreground hover:bg-black/5 dark:hover:bg-white/5'
                                            } last:border-0`}
                                    >
                                        <span className="whitespace-nowrap">{account}</span>
                                        {isSelected && (
                                            <div className="w-1.5 h-1.5 rounded-full bg-white" />
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
