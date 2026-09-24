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
                "inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium leading-4 whitespace-nowrap",
                colorClass,
                onClick && "cursor-pointer hover:opacity-80 transition-all active:scale-95",
                className
            )}
        >
            {text}
        </span>
    );
}
