import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { cn } from '@/lib/utils';

interface AccountSelectorProps {
    availableAccounts: string[];
    selectedAccounts: string[];
    onChange: (accounts: string[]) => void;
    accountGroups?: Record<string, string[]>;
    closedAccounts?: string[];
    variant?: 'default' | 'ghost';
    align?: 'left' | 'right';
}

export default function AccountSelector({ availableAccounts, selectedAccounts, onChange, accountGroups = {}, closedAccounts = [], variant = 'default', align = 'right' }: AccountSelectorProps) {
    const closedSet = new Set(closedAccounts);
    const [isOpen, setIsOpen] = useState(false);
    const triggerRef = useRef<HTMLDivElement>(null);
    const menuRef = useRef<HTMLDivElement>(null);
    // The menu is rendered in a portal (fixed-positioned) so it can't be clipped
    // or occluded by sibling cards when the selector is embedded mid-page.
    const [coords, setCoords] = useState<{ top: number; left?: number; right?: number }>({ top: 0 });

    const updateCoords = useCallback(() => {
        const el = triggerRef.current;
        if (!el) return;
        const r = el.getBoundingClientRect();
        if (align === 'left') {
            setCoords({ top: r.bottom + 8, left: r.left });
        } else {
            setCoords({ top: r.bottom + 8, right: window.innerWidth - r.right });
        }
    }, [align]);

    // Reposition while open (the page can scroll/resize under the fixed menu).
    useEffect(() => {
        if (!isOpen) return;
        updateCoords();
        window.addEventListener("scroll", updateCoords, true);
        window.addEventListener("resize", updateCoords);
        return () => {
            window.removeEventListener("scroll", updateCoords, true);
            window.removeEventListener("resize", updateCoords);
        };
    }, [isOpen, updateCoords]);

    // Close dropdown when clicking outside (trigger and portal menu are separate trees).
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            const target = event.target as Node;
            if (triggerRef.current?.contains(target)) return;
            if (menuRef.current?.contains(target)) return;
            setIsOpen(false);
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
        <div className="relative" ref={triggerRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={cn(
                    "flex flex-col items-center justify-center gap-1.5 p-3 rounded-2xl transition-all duration-300 group",
                    variant === 'ghost' ? "bg-transparent border-none shadow-none" : "bg-card border-none shadow-none",
                    "font-semibold tracking-tight min-w-[80px]",
                    isOpen ? "" : "text-cyan-500",
                    "flex-row py-2 px-2 h-[44px]"
                )}
            >
                <div className="flex flex-col items-center leading-none gap-0">
                    <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent truncate font-bold uppercase text-[14px]">
                        {getLabel()}
                    </span>
                </div>
            </button>

            {isOpen && typeof document !== 'undefined' && createPortal(
                <div
                    ref={menuRef}
                    style={{
                        backgroundColor: 'var(--menu-solid)',
                        position: 'fixed',
                        top: coords.top,
                        left: coords.left,
                        right: coords.right,
                    }}
                    className={cn(
                        "min-w-[200px] w-max border border-border rounded-xl shadow-xl outline-none z-[100] overflow-hidden",
                        align === 'left' ? "origin-top-left" : "origin-top-right"
                    )}
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
                                const isClosed = closedSet.has(account);
                                return (
                                    <button
                                        key={account}
                                        onClick={() => toggleAccount(account)}
                                        className={`group flex items-center justify-between gap-2 w-full px-4 py-3 text-sm font-medium transition-colors ${isSelected
                                            ? 'bg-[#0097b2] text-white shadow-sm'
                                            : 'text-popover-foreground hover:bg-black/5 dark:hover:bg-white/5'
                                            } last:border-0`}
                                    >
                                        <span className={cn("whitespace-nowrap", !isSelected && isClosed && "text-muted-foreground")}>{account}</span>
                                        <span className="flex items-center gap-2">
                                            {isClosed && (
                                                <span
                                                    title="This account is closed"
                                                    className={cn(
                                                        "text-[10px] font-bold uppercase tracking-wide px-1.5 py-0.5 rounded-full",
                                                        isSelected
                                                            ? "bg-white/20 text-white"
                                                            : "bg-amber-500/15 text-amber-600 dark:text-amber-400"
                                                    )}
                                                >
                                                    Closed
                                                </span>
                                            )}
                                            {isSelected && (
                                                <div className="w-1.5 h-1.5 rounded-full bg-white" />
                                            )}
                                        </span>
                                    </button>
                                );
                            })}
                    </div>
                </div>,
                document.body
            )}
        </div>
    );
}
