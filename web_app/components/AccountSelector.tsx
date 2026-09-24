import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { cn } from '@/lib/utils';
import { Check, ChevronDown } from 'lucide-react';

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
                type="button"
                onClick={() => setIsOpen(!isOpen)}
                aria-haspopup="true"
                aria-expanded={isOpen}
                className={cn("select-trigger", variant === 'ghost' && "border-transparent bg-transparent")}
            >
                <span>{getLabel()}</span>
                <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" aria-hidden="true" />
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
                    role="group"
                    aria-label="Accounts"
                    className={cn(
                        "menu-panel min-w-[220px] w-max outline-none z-[100]",
                        align === 'left' ? "origin-top-left" : "origin-top-right"
                    )}
                >
                    <div className="max-h-[80vh] overflow-y-auto">
                        {/* All Accounts Option */}
                        <button
                            type="button"
                            aria-pressed={isAllSelected}
                            onClick={handleSelectAll}
                            className="menu-item justify-between"
                        >
                            <span className="whitespace-nowrap">All Accounts</span>
                            {isAllSelected && <Check className="w-4 h-4 text-primary" aria-hidden="true" />}
                        </button>

                        {/* Account Groups Section */}
                        {hasGroups && (
                            <>
                                <div className="menu-divider" />
                                <div className="menu-heading">Groups</div>
                                {Object.entries(accountGroups).map(([groupName, groupAccounts]) => {
                                    // Check if this group is currently exactly selected (ignoring order)
                                    const isGroupSelected = !isAllSelected &&
                                        selectedAccounts.length === groupAccounts.length &&
                                        groupAccounts.every(acc => selectedAccounts.includes(acc));

                                    return (
                                        <button
                                            key={groupName}
                                            type="button"
                                            aria-pressed={isGroupSelected}
                                            onClick={() => handleSelectGroup(groupName, groupAccounts)}
                                            className="menu-item justify-between"
                                        >
                                            <span className="whitespace-nowrap">{groupName}</span>
                                            {isGroupSelected && <Check className="w-4 h-4 text-primary" aria-hidden="true" />}
                                        </button>
                                    );
                                })}
                            </>
                        )}


                        {/* Individual Accounts Section */}
                        {hasGroups && (
                            <>
                                <div className="menu-divider" />
                                <div className="menu-heading">Individual</div>
                            </>
                        )}

                        {availableAccounts
                            .filter(account => account !== 'All Accounts')
                            .sort((a, b) => {
                                const aClosed = closedSet.has(a) ? 1 : 0;
                                const bClosed = closedSet.has(b) ? 1 : 0;
                                if (aClosed !== bClosed) return aClosed - bClosed;
                                return a.localeCompare(b);
                            })
                            .map((account) => {
                                const isSelected = selectedAccounts.includes(account);
                                const isClosed = closedSet.has(account);
                                return (
                                    <button
                                        key={account}
                                        type="button"
                                        aria-pressed={isSelected}
                                        onClick={() => toggleAccount(account)}
                                        className="menu-item justify-between"
                                    >
                                        <span className={cn("whitespace-nowrap", !isSelected && isClosed && "text-muted-foreground")}>{account}</span>
                                        <span className="flex items-center gap-2">
                                            {isClosed && (
                                                <span
                                                    title="This account is closed"
                                                    className="text-xs font-medium px-1.5 py-0.5 rounded-md bg-warn-tint text-warn-ink"
                                                >
                                                    Closed
                                                </span>
                                            )}
                                            {isSelected && <Check className="w-4 h-4 text-primary" aria-hidden="true" />}
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
