'use client';

/**
 * The phone shell's single dropdown primitive — the web twin of
 * `Views/PopoverMenu.swift`.
 *
 * Every control in the iPhone control bar (accounts, currency, layout) opens
 * one of these, so they share one look and one width behaviour, exactly as the
 * native bar's controls all share `PopoverMenu`. Build the body from
 * `MenuToggleRow` / `MenuRow` / `MenuSectionHeader` / `MenuDivider`.
 */

import { createContext, useCallback, useContext, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Check } from 'lucide-react';
import { cn } from '@/lib/utils';

/** Lets a row dismiss the menu it sits in without threading a callback down. */
const DismissContext = createContext<() => void>(() => {});

/** A popover sizes itself to its content, and content wider than the phone is
 *  simply clipped — the native menu caps at 320pt for the same reason. */
const WIDTH_CAP = 320;

interface MobileMenuProps {
  label: React.ReactNode;
  /** Accessible name for the trigger — the label is usually an icon alone. */
  ariaLabel: string;
  minWidth?: number;
  maxHeight?: number;
  /** Which edge of the trigger the menu hangs from. */
  align?: 'left' | 'right';
  className?: string;
  children: React.ReactNode;
}

export function MobileMenu({
  label, ariaLabel, minWidth = 220, maxHeight = 440, align = 'left', className, children,
}: MobileMenuProps) {
  const [open, setOpen] = useState(false);
  const [coords, setCoords] = useState<{ top: number; left: number }>({ top: 0, left: 0 });
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  const close = useCallback(() => setOpen(false), []);

  // Anchor under the trigger, then clamp inside the viewport. The phone is
  // narrow enough that a menu hung off a right-hand control would otherwise
  // run past the screen edge.
  useLayoutEffect(() => {
    if (!open) return;
    const place = () => {
      const t = triggerRef.current?.getBoundingClientRect();
      if (!t) return;
      const width = Math.min(Math.max(minWidth, menuRef.current?.offsetWidth ?? minWidth), WIDTH_CAP);
      const raw = align === 'right' ? t.right - width : t.left;
      const left = Math.min(Math.max(8, raw), window.innerWidth - width - 8);
      setCoords({ top: t.bottom + 6, left });
    };
    place();
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    return () => {
      window.removeEventListener('resize', place);
      window.removeEventListener('scroll', place, true);
    };
  }, [open, align, minWidth]);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent | TouchEvent) => {
      const target = e.target as Node;
      if (menuRef.current?.contains(target) || triggerRef.current?.contains(target)) return;
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('touchstart', onPointerDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('touchstart', onPointerDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        onClick={() => setOpen(o => !o)}
        aria-label={ariaLabel}
        aria-haspopup="menu"
        aria-expanded={open}
        className={cn(
          'flex items-center justify-center shrink-0 rounded-control text-foreground transition-colors',
          'active:bg-foreground/10',
          className,
        )}
      >
        {label}
      </button>

      {open && typeof document !== 'undefined' && createPortal(
        <div
          ref={menuRef}
          role="menu"
          style={{
            position: 'fixed',
            top: coords.top,
            left: coords.left,
            minWidth,
            maxWidth: Math.max(minWidth, WIDTH_CAP),
            maxHeight,
          }}
          className="z-[100] overflow-y-auto overscroll-contain rounded-inset border border-border bg-popover py-1.5 shadow-2xl"
        >
          <DismissContext.Provider value={close}>{children}</DismissContext.Provider>
        </div>,
        document.body,
      )}
    </>
  );
}

/** A tappable menu row. Runs `onSelect`, then dismisses. */
export function MenuRow({ title, icon: Icon, trailing, destructive, onSelect }: {
  title: string;
  icon?: React.ComponentType<{ className?: string }>;
  trailing?: string;
  destructive?: boolean;
  onSelect: () => void;
}) {
  const dismiss = useContext(DismissContext);
  return (
    <button
      type="button"
      role="menuitem"
      onClick={() => { onSelect(); dismiss(); }}
      className={cn(
        'flex w-full items-center gap-2.5 px-3.5 py-2.5 text-left text-sm transition-colors active:bg-foreground/10',
        destructive ? 'text-down' : 'text-popover-foreground',
      )}
    >
      {Icon && <Icon className="w-5 h-5 shrink-0" />}
      <span className="flex-1">{title}</span>
      {trailing && <span className="text-xs text-muted-foreground shrink-0">{trailing}</span>}
    </button>
  );
}

/**
 * A row with a trailing checkmark for on/off state. Multi-select rows keep the
 * menu open (`dismissOnTap` false); single-select rows set it true — the same
 * split the native `MenuToggleRow` makes.
 */
export function MenuToggleRow({ title, isOn, trailing, dismissOnTap = false, onSelect }: {
  title: string;
  isOn: boolean;
  trailing?: string;
  dismissOnTap?: boolean;
  onSelect: () => void;
}) {
  const dismiss = useContext(DismissContext);
  return (
    <button
      type="button"
      role="menuitemcheckbox"
      aria-checked={isOn}
      onClick={() => { onSelect(); if (dismissOnTap) dismiss(); }}
      className="flex w-full items-center gap-2.5 px-3.5 py-2.5 text-left text-sm text-popover-foreground transition-colors active:bg-foreground/10"
    >
      <span className="flex-1">{title}</span>
      {trailing && <span className="text-[11px] font-medium text-muted-foreground shrink-0">{trailing}</span>}
      <Check className={cn('w-3.5 h-3.5 shrink-0 text-primary', !isOn && 'opacity-0')} strokeWidth={3} />
    </button>
  );
}

/** A small uppercased section header inside a menu. */
export function MenuSectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <p className="px-3.5 pt-2 pb-0.5 text-[11px] font-semibold uppercase text-muted-foreground">
      {children}
    </p>
  );
}

/** A divider sized to the menu's padding. */
export function MenuDivider() {
  return <div className="my-1 border-t border-border" />;
}
