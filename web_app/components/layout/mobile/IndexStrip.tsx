'use client';

/**
 * Market indices in the phone's top bar — the web twin of `IndexStrip` in
 * `App/MainView.swift`, which sits in the same place (the navigation bar's
 * trailing edge) on the native iPhone app.
 *
 * The native strip is a `ViewThatFits` that falls back from name + price +
 * change + percent to name + percent when the bar runs out of room. A web
 * layout can't measure itself the same way, so the fallback is pinned to a
 * width instead: the full strip needs roughly 90px per index, which no iPhone
 * affords beside the app icon, so the short form carries the phone and the full
 * form appears on the wider end of the compact range.
 *
 * `ViewThatFits` renders the one form it chose, and so does this — the two are
 * not both in the DOM with one hidden. A hidden twin would read every figure
 * twice to a screen reader and put a second, invisible "+180.20" in the page
 * for anything searching it by text.
 *
 * It starts on the *narrow* form and widens only once measured, the same rule
 * `prefersStackedLayout` follows natively: the server has no width to render
 * against, so assuming the wide one would flash the wrong strip on every load.
 */

import { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';

interface IndexData {
  name?: string;
  price?: number | null;
  change?: number | null;
  changesPercentage?: number | null;
}

/** DOW / S&P / NAS / RUT — the native `shortName(_:)`, verbatim. */
function shortName(name?: string): string {
  if (!name) return 'IDX';
  const upper = name.toUpperCase();
  if (upper.includes('DOW')) return 'DOW';
  if (upper.includes('S&P')) return 'S&P';
  if (upper.includes('NASDAQ') || upper.includes('NAS')) return 'NAS';
  if (upper.includes('RUSSELL')) return 'RUT';
  return upper.slice(0, 3);
}

const number = (v?: number | null) =>
  v == null ? '—' : v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

function Triangle({ up }: { up: boolean }) {
  // The native strip draws `arrowtriangle.up.fill`; a glyph keeps the baseline
  // aligned with the digits beside it, which an SVG icon does not.
  return <span className="text-[8px] leading-none">{up ? '▲' : '▼'}</span>;
}

/** True once the shell is wide enough for the full strip. False until measured. */
function useWideBar(): boolean {
  const [wide, setWide] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia('(min-width: 600px)');
    const sync = () => setWide(mq.matches);
    sync();
    mq.addEventListener('change', sync);
    return () => mq.removeEventListener('change', sync);
  }, []);
  return wide;
}

export function IndexStrip({ indices, onClick }: {
  indices: Record<string, IndexData>;
  onClick?: () => void;
}) {
  const wide = useWideBar();
  const list = Object.values(indices ?? {});
  if (list.length === 0) return null;

  return (
    <button
      type="button"
      onClick={onClick}
      aria-label="Market indices"
      className="flex items-center overflow-hidden whitespace-nowrap rounded-control px-1 py-1 text-foreground"
    >
      {wide ? (
        /* Full strip — name, price, change, percent. */
        <span className="flex items-center gap-3">
          {list.map((index, i) => {
            const up = (index.change ?? 0) >= 0;
            return (
              <span key={index.name ?? i} className="flex items-center gap-0.5">
                <span className="text-[11px] font-bold">{shortName(index.name)}</span>
                <span className="pl-0.5 text-[11px] tabular-nums text-muted-foreground">{number(index.price)}</span>
                {index.change != null && (
                  <span className={cn('pl-0.5 text-[11px] font-medium tabular-nums', up ? 'text-up' : 'text-down')}>
                    {up ? '+' : ''}{number(index.change)}
                  </span>
                )}
                <span className={cn('flex items-center pl-0.5 text-[11px] tabular-nums', up ? 'text-up' : 'text-down')}>
                  (<Triangle up={up} />{Math.abs(index.changesPercentage ?? 0).toFixed(2)}%)
                </span>
              </span>
            );
          })}
        </span>
      ) : (
        /* Short strip — name and percent only. */
        <span className="flex items-center gap-2.5">
          {list.map((index, i) => {
            const up = (index.change ?? 0) >= 0;
            return (
              <span key={index.name ?? i} className="flex items-center gap-0.5">
                <span className="text-[11px] font-bold">{shortName(index.name)}</span>
                <span className={cn('flex items-center pl-0.5 text-[11px] tabular-nums', up ? 'text-up' : 'text-down')}>
                  <Triangle up={up} />{Math.abs(index.changesPercentage ?? 0).toFixed(2)}%
                </span>
              </span>
            );
          })}
        </span>
      )}
    </button>
  );
}
