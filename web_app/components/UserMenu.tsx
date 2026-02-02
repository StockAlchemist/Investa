import React, { useState, useRef, useEffect } from 'react';
import { UserCircle, LogOut, User, ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

interface UserMenuProps {
    user?: any;
    onLogout?: () => void;
    onUserClick: () => void;
    align?: 'left' | 'right';
}

export default function UserMenu({ user, onLogout, onUserClick, align = 'right' }: UserMenuProps) {
    const [isOpen, setIsOpen] = useState(false);
    const menuRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, []);

    return (
        <div className="relative inline-block text-left" ref={menuRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={cn(
                    "p-2 rounded-xl transition-all duration-300 group",
                    isOpen ? "bg-accent/10 text-cyan-500" : "text-cyan-500 hover:bg-accent/10"
                )}
                title="Account"
            >
                <UserCircle className="w-5 h-5 transition-transform duration-300 group-hover:scale-110" />
            </button>

            {isOpen && (
                <div
                    className={cn(
                        "absolute top-full mt-2 w-48 rounded-2xl border border-border shadow-2xl shadow-black/40 overflow-hidden z-[100] transition-all animate-in fade-in zoom-in duration-200 slide-in-from-top-2",
                        align === 'right' ? "right-0" : "left-0"
                    )}
                    style={{ backgroundColor: 'var(--menu-solid)' }}
                >
                    <div className="p-2 grid gap-1">
                        <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-widest text-muted-foreground/50 border-b border-border mb-1">
                            {user?.username || 'Account'}
                        </div>
                        <button
                            onClick={() => {
                                onUserClick();
                                setIsOpen(false);
                            }}
                            className="flex items-center gap-3 w-full px-3 py-2 text-sm font-medium rounded-xl transition-all duration-200 text-popover-foreground hover:bg-black/5 dark:hover:bg-white/5 text-left"
                        >
                            <User className="w-4 h-4 text-cyan-500" />
                            <span>User Settings</span>
                        </button>
                        <button
                            onClick={() => {
                                setIsOpen(false);
                                onLogout?.();
                            }}
                            className="flex items-center gap-3 w-full px-3 py-2 text-sm font-medium rounded-xl transition-all duration-200 text-red-500 hover:bg-red-500/10 text-left"
                        >
                            <LogOut className="w-4 h-4" />
                            <span>Log Out</span>
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
