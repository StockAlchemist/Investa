import React from 'react';
import { cn, getColorForString } from '@/lib/utils';

interface SemanticBadgeProps {
    text: string;
    className?: string;
    onClick?: () => void;
}

export function SemanticBadge({ text, className, onClick }: SemanticBadgeProps) {
    const colorClass = getColorForString(text);

    return (
        <span
            onClick={onClick}
            className={cn(
                "inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-widest leading-none",
                colorClass,
                onClick && "cursor-pointer hover:opacity-80 transition-all active:scale-95",
                className
            )}
        >
            {text}
        </span>
    );
}
