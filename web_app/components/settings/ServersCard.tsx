'use client';

import React, { useState } from 'react';
import { Check, Trash2, ExternalLink, Plus } from 'lucide-react';
import { cn } from '../../lib/utils';
import {
    addServer,
    defaultServerName,
    looksLikeApiUrl,
    normalizeServerUrl,
    removeServer,
    switchToServer,
    useCurrentOrigin,
    useSavedServers,
} from '../../lib/servers';
import {
    cardClassName,
    cardHeadClassName,
    sectionTitleClassName,
    countBadgeClassName,
    labelClassName,
    inputClassName,
    primaryButtonClassName,
    secondaryButtonClassName,
} from './constants';

/**
 * Saved Investa servers and the switch between them — the web twin of the
 * native System & Server card. See lib/servers.ts for why a web server is
 * another machine's web app rather than a backend URL.
 */
export function ServersCard() {
    const servers = useSavedServers();
    const here = useCurrentOrigin();

    const [name, setName] = useState('');
    const [address, setAddress] = useState('');
    const [status, setStatus] = useState<{ tone: 'up' | 'down'; text: string } | null>(null);

    const hereSaved = servers.some(s => s.url === here);
    const target = normalizeServerUrl(address);

    const validate = (): string | null => {
        if (looksLikeApiUrl(address)) {
            setStatus({ tone: 'down', text: 'That is the backend API address. Enter the Investa web address instead, e.g. http://100.127.10.38:3000' });
            return null;
        }
        if (!target) {
            setStatus({ tone: 'down', text: 'Enter an http:// or https:// address.' });
            return null;
        }
        return target;
    };

    const handleSave = () => {
        if (!validate()) return;
        const saved = addServer(name, address);
        if (!saved) return;
        setName('');
        setAddress('');
        setStatus({ tone: 'up', text: `Saved ${saved.name}` });
    };

    const handleOpen = () => {
        const origin = validate();
        if (!origin) return;
        if (origin === here) {
            setStatus({ tone: 'up', text: 'Already on this server.' });
            return;
        }
        switchToServer(origin);
    };

    return (
        <div className={cardClassName}>
            <div className={cardHeadClassName}>
                <h3 className={sectionTitleClassName}>Servers</h3>
                {servers.length > 0 && <span className={countBadgeClassName}>{servers.length}</span>}
            </div>
            <p className="text-xs text-muted-foreground mb-4 leading-relaxed">
                Other machines running Investa — home LAN, Tailscale. Switching opens that machine&apos;s Investa in this tab,
                with its own sign-in, and brings this list along. Switch from here or from the header (inside the accounts menu on a phone).
            </p>

            {servers.length > 0 && (
                <ul className="card-inset p-0 mb-5 divide-y divide-border">
                    {servers.map(server => {
                        const isHere = server.url === here;
                        return (
                            <li key={server.id} className="flex items-center gap-3 px-4 py-2.5">
                                <span className={cn('w-4 shrink-0', isHere ? 'text-primary' : 'text-transparent')} aria-hidden="true">
                                    <Check className="w-4 h-4" />
                                </span>
                                <span className="flex-1 min-w-0">
                                    <span className="block text-sm font-semibold text-foreground whitespace-nowrap">
                                        {server.name}
                                        {isHere && <span className="ml-2 text-xs font-normal text-muted-foreground">This server</span>}
                                    </span>
                                    <span className="block text-xs text-muted-foreground tabular-nums whitespace-nowrap">
                                        {server.url}
                                    </span>
                                </span>
                                {!isHere && (
                                    <button
                                        type="button"
                                        onClick={() => switchToServer(server.url)}
                                        className={cn(secondaryButtonClassName, 'h-8 px-2.5')}
                                    >
                                        <ExternalLink className="w-3.5 h-3.5" />
                                        Open
                                    </button>
                                )}
                                <button
                                    type="button"
                                    onClick={() => removeServer(server.id)}
                                    aria-label={`Remove ${server.name}`}
                                    title={`Remove ${server.name}`}
                                    className="h-8 w-8 inline-flex items-center justify-center rounded-control text-muted-foreground hover:text-down hover:bg-down/5 transition-colors cursor-pointer"
                                >
                                    <Trash2 className="w-4 h-4" />
                                </button>
                            </li>
                        );
                    })}
                </ul>
            )}

            <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_minmax(0,2fr)] gap-4 mb-4">
                <div className="space-y-1.5">
                    <label htmlFor="server-name" className={labelClassName}>Name</label>
                    <input
                        id="server-name"
                        type="text"
                        placeholder="e.g. Home Mac"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        className={inputClassName}
                    />
                </div>
                <div className="space-y-1.5">
                    <label htmlFor="server-address" className={labelClassName}>Web address</label>
                    <input
                        id="server-address"
                        type="url"
                        inputMode="url"
                        autoCapitalize="none"
                        autoCorrect="off"
                        spellCheck={false}
                        placeholder="http://100.127.10.38:3000"
                        value={address}
                        onChange={(e) => { setAddress(e.target.value); setStatus(null); }}
                        onKeyDown={(e) => { if (e.key === 'Enter') handleSave(); }}
                        className={inputClassName}
                    />
                </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
                <button type="button" onClick={handleSave} disabled={!address.trim()} className={secondaryButtonClassName}>
                    <Plus className="w-4 h-4" />
                    Save to list
                </button>
                <button type="button" onClick={handleOpen} disabled={!address.trim()} className={primaryButtonClassName}>
                    Open
                </button>
                {here && !hereSaved && (
                    <button
                        type="button"
                        onClick={() => {
                            const saved = addServer(name, here);
                            if (saved) { setName(''); setStatus({ tone: 'up', text: `Saved ${saved.name}` }); }
                        }}
                        className="text-sm font-medium text-primary hover:underline cursor-pointer"
                    >
                        Save this server ({defaultServerName(here)})
                    </button>
                )}
                {status && (
                    <p className={cn('text-sm font-medium animate-in fade-in', status.tone === 'up' ? 'text-up' : 'text-down')}>
                        {status.text}
                    </p>
                )}
            </div>
        </div>
    );
}
