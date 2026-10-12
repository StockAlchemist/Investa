'use client';

import { useRef, useState } from 'react';
import { Share2, X } from 'lucide-react';
import { useDialogFocus } from './hooks/useDialogFocus';

export default function ShareViewButton() {
    const [link, setLink] = useState<string | null>(null);
    const [copied, setCopied] = useState(false);
    const dialog = useRef<HTMLDivElement>(null);
    useDialogFocus(dialog, link !== null, () => setLink(null));

    const share = async () => {
        const url = window.location.href;
        if (navigator.share) {
            try { await navigator.share({ title: 'Investa', url }); return; }
            catch (error) { if (error instanceof DOMException && error.name === 'AbortError') return; }
        }
        try {
            await navigator.clipboard.writeText(url);
            setCopied(true);
        } catch { setLink(url); }
    };

    return <>
        <button type="button" onClick={() => void share()} aria-label="Share current view" title={copied ? 'Link copied' : 'Share current view'}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-control text-muted-foreground hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring">
            <Share2 className="h-4 w-4" />
        </button>
        {copied && <span className="sr-only" role="status">View link copied</span>}
        {link && <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/40 p-4">
            <div ref={dialog} role="dialog" aria-modal="true" aria-label="Share current view" className="w-full max-w-md rounded-xl border border-border bg-card p-5 shadow-xl">
                <div className="flex items-center justify-between"><h2 className="font-semibold">Share current view</h2>
                    <button type="button" aria-label="Close share dialog" onClick={() => setLink(null)}><X className="h-4 w-4" /></button></div>
                <p className="my-3 text-sm text-muted-foreground">Copy this link to open the same view.</p>
                <input aria-label="View link" readOnly value={link} onFocus={event => event.currentTarget.select()} className="w-full rounded-control border border-border bg-background p-2 text-sm" />
            </div>
        </div>}
    </>;
}
