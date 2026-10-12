import { useEffect, useRef, type RefObject } from 'react';

export function useDialogFocus(ref: RefObject<HTMLElement | null>, open: boolean, close: () => void) {
    const closeRef = useRef(close);
    useEffect(() => { closeRef.current = close; }, [close]);
    useEffect(() => {
        if (!open || !ref.current) return;
        const previous = document.activeElement as HTMLElement | null;
        const dialog = ref.current;
        const controls = () => Array.from(dialog.querySelectorAll<HTMLElement>(
            'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex="0"]',
        )).filter(element => !element.hidden && element.getAttribute('aria-hidden') !== 'true');
        controls()[0]?.focus();
        const handle = (event: KeyboardEvent) => {
            if (event.key === 'Escape') { event.preventDefault(); closeRef.current(); }
            if (event.key !== 'Tab') return;
            const items = controls();
            if (!items.length) { event.preventDefault(); return; }
            const first = items[0], last = items[items.length - 1];
            if (event.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) {
                event.preventDefault(); last.focus();
            } else if (!event.shiftKey && (document.activeElement === last || !dialog.contains(document.activeElement))) {
                event.preventDefault(); first.focus();
            }
        };
        document.addEventListener('keydown', handle);
        return () => { document.removeEventListener('keydown', handle); previous?.focus(); };
    }, [open, ref]);
}
