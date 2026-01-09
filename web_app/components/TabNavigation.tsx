import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';
import {
    ChevronDown,
    LayoutDashboard,
    Eye,
    History,
    PieChart,
    TrendingUp,
    DollarSign,
    Activity,
    Coins,
    Settings as SettingsIcon,
    Menu
} from 'lucide-react';

interface TabNavigationProps {
    activeTab: string;
    onTabChange: (tab: string) => void;
    side?: 'right' | 'bottom';
}

const TABS = [
    { id: 'performance', label: 'Performance', icon: LayoutDashboard },
    { id: 'watchlist', label: 'Watchlist', icon: Eye },
    { id: 'transactions', label: 'Transactions', icon: History },
    { id: 'allocation', label: 'Allocation', icon: PieChart },
    { id: 'asset_change', label: 'Asset Change', icon: TrendingUp },
    { id: 'capital_gains', label: 'Cap. Gains', icon: DollarSign },
    { id: 'analytics', label: 'Analytics', icon: Activity },
    { id: 'dividend', label: 'Dividend', icon: Coins },
    { id: 'settings', label: 'Settings', icon: SettingsIcon },
];

export default function TabNavigation({ activeTab, onTabChange, side = 'bottom' }: TabNavigationProps) {
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

    const handleTabSelect = (tabId: string) => {
        onTabChange(tabId);
        setIsOpen(false);
    };

    return (
        <div className="relative inline-block text-left" ref={dropdownRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={cn(
                    "flex flex-col items-center gap-1.5 p-3 rounded-2xl transition-all duration-300 group",
                    "bg-card hover:bg-accent/10",
                    "border border-border shadow-sm",
                    "text-xs font-semibold tracking-tight w-[60px]",
                    isOpen ? "border-cyan-500/50 ring-2 ring-cyan-500/20" : "text-foreground/80 hover:text-foreground",
                    side === 'bottom' && "flex-row py-2 px-4 justify-center w-auto min-w-0"
                )}
                title={activeTabObj?.label}
            >
                <div className={cn(
                    "p-2 rounded-xl transition-all duration-300",
                    isOpen ? "bg-cyan-500 text-white shadow-lg shadow-cyan-500/30" : "bg-secondary text-cyan-500 group-hover:scale-110",
                    side === 'bottom' && "p-1.5"
                )}>
                    <ActiveIcon className={cn(side === 'bottom' ? "w-4 h-4" : "w-5 h-5")} />
                </div>
            </button>

            {isOpen && (
                <div className={cn(
                    "absolute overflow-hidden z-[100] transition-all animate-in fade-in zoom-in duration-200",
                    side === 'right'
                        ? "left-full top-0 ml-4 slide-in-from-left-2"
                        : "right-0 top-full mt-2 slide-in-from-top-2 origin-top-right",
                    "w-56 rounded-2xl bg-white dark:bg-zinc-950 border border-border shadow-2xl shadow-black/40"
                )}>
                    <div className="p-2 grid gap-1">
                        <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-widest text-muted-foreground/50 border-b border-border mb-1">
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
                                        ? "bg-[#0097b2] text-white shadow-sm"
                                        : "text-popover-foreground hover:bg-black/5 dark:hover:bg-white/5"
                                )}
                            >
                                <tab.icon className={cn(
                                    "w-4 h-4 transition-transform duration-300",
                                    activeTab === tab.id ? "text-white" : "text-cyan-500 group-hover:scale-110"
                                )} />
                                <span className="flex-1">{tab.label}</span>
                                {activeTab === tab.id && (
                                    <div className="w-1.5 h-1.5 rounded-full bg-white" />
                                )}
                            </button>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
