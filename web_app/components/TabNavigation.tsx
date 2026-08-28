import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';
import { Menu, LogOut } from 'lucide-react';
import { PRIMARY_NAV, SECONDARY_NAV } from '@/lib/navigation';

// Same destinations, same words as the sidebar — see lib/navigation.
const TABS = [...PRIMARY_NAV, ...SECONDARY_NAV];

interface TabNavigationProps {
    activeTab: string;
    onTabChange: (tab: string) => void;
    onLogout?: () => void;
    side?: 'right' | 'bottom';
    align?: 'left' | 'right';
}

export default function TabNavigation({ activeTab, onTabChange, onLogout, side = 'bottom', align = 'right' }: TabNavigationProps) {
    const [isOpen, setIsOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    const activeTabObj = TABS.find((t) => t.id === activeTab);
    const ActiveIcon = activeTabObj?.icon || Menu;

    // Close dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };

        if (isOpen) {
            document.addEventListener('mousedown', handleClickOutside);
        }

        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, [isOpen]);

    return (
        <div className="relative inline-block text-left" ref={dropdownRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={cn(
                    "flex flex-col items-center gap-1.5 p-3 rounded-2xl transition-all duration-300 group",
                    side === 'bottom' ? "bg-transparent" : "hover:bg-accent/10",
                    "text-xs font-semibold tracking-tight w-[60px]",
                    isOpen ? "ring-2 ring-ring/25" : "text-foreground/80 hover:text-foreground",
                    side === 'bottom' && "flex-row py-2 px-4 justify-center w-auto min-w-0 h-[44px]"
                )}
                title={activeTabObj?.label}
            >
                <div className={cn(
                    "p-2 rounded-xl transition-all duration-300",
                    isOpen
                        ? 'bg-primary text-primary-foreground'
                        : 'text-primary group-hover:scale-110',
                    side === 'bottom' && "p-1.5",
                    side === 'bottom' && !isOpen && "bg-secondary"
                )}>
                    <ActiveIcon className={cn(side === 'bottom' ? "w-4 h-4" : "w-5 h-5")} />
                </div>
            </button>

            {isOpen && (
                <div className={cn(
                    "absolute overflow-hidden z-[100] transition-all animate-in fade-in zoom-in duration-200",
                    side === 'right'
                        ? "left-full top-0 ml-4 slide-in-from-left-2"
                        : `${align === 'left' ? 'left-0 origin-top-left' : 'right-0 origin-top-right'} top-full mt-2 slide-in-from-top-2`,
                    "w-56 rounded-inset border border-border shadow-lg"
                )} style={{ backgroundColor: 'var(--menu-solid)' }}>
                    <div className="p-2 grid gap-1">
                        <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-widest text-muted-foreground/50 mb-1">
                            Navigation
                        </div>
                        {TABS.map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => {
                                    onTabChange(tab.id);
                                    setIsOpen(false);
                                }}
                                className={cn(
                                    "flex items-center gap-3 w-full px-3 py-2 text-sm font-medium rounded-xl transition-all duration-200 group text-left",
                                    activeTab === tab.id
                                        ? 'bg-primary text-primary-foreground'
                                        : "text-popover-foreground hover:bg-muted"
                                )}
                            >
                                <tab.icon className={cn(
                                    "w-4 h-4 transition-transform duration-300",
                                    activeTab === tab.id
                                        ? "text-primary-foreground"
                                        : "text-muted-foreground group-hover:scale-110"
                                )} />
                                <span className="flex-1">{tab.label}</span>
                                {activeTab === tab.id && (
                                    <div className="w-1.5 h-1.5 rounded-full bg-primary-foreground" />
                                )}
                            </button>
                        ))}
                        {onLogout && (
                            <>
                                <div className="my-1" />
                                <button
                                    onClick={() => {
                                        if (confirm('Are you sure you want to log out?')) {
                                            onLogout();
                                            setIsOpen(false);
                                        }
                                    }}
                                    className="flex items-center gap-3 w-full px-3 py-2 text-sm font-medium rounded-xl transition-all duration-200 group text-left text-rose-500 hover:bg-rose-500/10"
                                >
                                    <LogOut className="w-4 h-4 transition-transform duration-300 group-hover:scale-110" />
                                    <span className="flex-1">Log Out</span>
                                </button>
                            </>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
