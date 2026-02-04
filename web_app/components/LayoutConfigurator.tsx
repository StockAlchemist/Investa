import { useState, useRef, useEffect } from 'react';
import { SlidersHorizontal } from 'lucide-react';
import { cn } from '@/lib/utils';
import { DEFAULT_ITEMS } from '@/lib/dashboard_constants';

interface LayoutConfiguratorProps {
    visibleItems: string[];
    onVisibleItemsChange: (items: string[]) => void;
    className?: string;
    variant?: 'default' | 'ghost';
    align?: 'left' | 'right';
}

export default function LayoutConfigurator({
    visibleItems,
    onVisibleItemsChange,
    className,
    variant = 'default',
    align = 'right'
}: LayoutConfiguratorProps) {
    const [isOpen, setIsOpen] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    // Close dropdown when clicking outside
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, []);

    const toggleItem = (id: string) => {
        if (visibleItems.includes(id)) {
            // Don't allow hiding all items - keep at least 1
            if (visibleItems.length > 1) {
                onVisibleItemsChange(visibleItems.filter(i => i !== id));
            }
        } else {
            // Add back and resort
            const newVisible = DEFAULT_ITEMS
                .filter(item => item.id === id || visibleItems.includes(item.id))
                .map(item => item.id);

            onVisibleItemsChange(newVisible);
        }
    };

    return (
        <div className={cn("relative", className)} ref={containerRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={cn(
                    "flex items-center justify-center gap-2 p-2 rounded-xl transition-all duration-300 group",
                    variant === 'ghost' ? "bg-transparent border-none shadow-none" : "bg-card border-none shadow-sm",
                    isOpen ? "border-cyan-500/50 ring-2 ring-cyan-500/20" : "text-cyan-500",
                    "h-[44px] px-3"
                )}
                title="Configure Dashboard"
                aria-label="Configure Dashboard Layout"
                aria-expanded={isOpen}
            >
                <SlidersHorizontal className="w-4 h-4 text-cyan-500" />
                <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent font-bold uppercase text-[12px] hidden lg:block">
                    Layout
                </span>
            </button>

            {isOpen && (
                <div
                    style={{ backgroundColor: 'var(--menu-solid)' }}
                    className={cn(
                        "absolute top-full mt-2 min-w-[240px] w-max origin-top border border-border rounded-xl shadow-xl outline-none z-50 overflow-hidden",
                        align === 'left' ? "left-0 origin-top-left" : "right-0 origin-top-right"
                    )}
                >
                    <div className="py-1 max-h-[80vh] overflow-y-auto">
                        <div className="px-4 py-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider bg-muted/30">
                            Dashboard Elements
                        </div>
                        {DEFAULT_ITEMS.map((item) => {
                            const isVisible = visibleItems.includes(item.id);
                            return (
                                <button
                                    key={item.id}
                                    onClick={() => toggleItem(item.id)}
                                    className={cn(
                                        "group flex items-center justify-between w-full px-4 py-3 text-sm font-medium transition-colors last:border-0",
                                        isVisible
                                            ? 'bg-[#0097b2] text-white shadow-sm'
                                            : 'text-popover-foreground hover:bg-black/5 dark:hover:bg-white/5'
                                    )}
                                >
                                    <span>{item.title}</span>
                                    {isVisible && (
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
