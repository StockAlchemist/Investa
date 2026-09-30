'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Check, ChevronDown, Server } from 'lucide-react';
import { cn } from '@/lib/utils';
import { defaultServerName, switchToServer, useServerChoices } from '@/lib/servers';

interface ServerSelectorProps {
    align?: 'left' | 'right';
    /** Opens upward — for the login page, where it sits at the bottom. */
    placement?: 'bottom' | 'top';
    className?: string;
}

/**
 * Quick switch between saved Investa servers (Settings › Advanced › Servers
 * keeps the list). Renders nothing until there is somewhere else to go, so a
 * single-server setup keeps the header it had.
 */
export default function ServerSelector({ align = 'right', placement = 'bottom', className }: ServerSelectorProps) {
    const choices = useServerChoices();
    const [isOpen, setIsOpen] = useState(false);
    const ref = useRef<HTMLDivElement>(null);

    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (ref.current && !ref.current.contains(event.target as Node)) setIsOpen(false);
        }
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    if (!choices) return null;
    const { here, currentName, rows } = choices;

    return (
        <div className={cn('relative inline-block text-left', className)} ref={ref}>
            <button
                type="button"
                onClick={() => setIsOpen(!isOpen)}
                aria-haspopup="true"
                aria-expanded={isOpen}
                aria-label={`Server: ${currentName}`}
                title={`Server: ${currentName}`}
                className="select-trigger"
            >
                <Server className="w-3.5 h-3.5 text-muted-foreground" aria-hidden="true" />
                <span className="hidden md:inline whitespace-nowrap">{currentName}</span>
                <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" aria-hidden="true" />
            </button>

            {isOpen && (
                <div
                    role="group"
                    aria-label="Servers"
                    className={cn(
                        'menu-panel absolute z-[100] w-72 animate-in fade-in zoom-in-95 duration-150',
                        placement === 'top' ? 'bottom-full mb-1.5' : 'top-full mt-1.5',
                        align === 'left' ? 'left-0' : 'right-0',
                    )}
                >
                    <div className="menu-heading">Server</div>
                    {rows.map(server => {
                        const selected = server.url === here;
                        return (
                            <button
                                key={server.id}
                                type="button"
                                aria-pressed={selected}
                                onClick={() => {
                                    setIsOpen(false);
                                    if (!selected) switchToServer(server.url);
                                }}
                                className="menu-item"
                            >
                                <span className="flex-1 min-w-0 text-left whitespace-nowrap">{server.name}</span>
                                <span className="text-xs text-muted-foreground font-normal tabular-nums whitespace-nowrap">
                                    {defaultServerName(server.url)}
                                </span>
                                {selected && <Check className="w-4 h-4 text-primary shrink-0" aria-hidden="true" />}
                            </button>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
