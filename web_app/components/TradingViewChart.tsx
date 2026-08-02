'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Maximize2, Minimize2 } from 'lucide-react';
import { useTheme } from 'next-themes';
import { toTradingViewSymbol } from '../lib/tradingview';

interface TradingViewChartProps {
    /** Investa (Yahoo) symbol — mapped to TradingView's `EXCHANGE:TICKER`. */
    symbol: string;
    /** yfinance `info.exchange` code, used to pin US listings to a venue. */
    exchange?: string;
    /** Chart height in px. */
    height?: number;
}

const SCRIPT_SRC = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';

/**
 * How the chart is currently presented.
 *
 * `native` uses the Fullscreen API, which lifts the element into the browser's
 * top layer — the only way out of the `backdrop-filter` on the surrounding
 * `.glass-card`, which otherwise makes that card the containing block for any
 * `position: fixed` descendant and clips a CSS-only overlay back into it.
 *
 * `overlay` is the fallback for browsers that refuse (iOS Safari allows
 * fullscreen on `<video>` only): a fixed panel portalled to `document.body`,
 * clear of that containing block. The portal moves the widget in the DOM, so
 * that path — and only that path — reloads the chart when it opens or closes.
 */
type DisplayMode = 'inline' | 'native' | 'overlay';

/**
 * TradingView's Advanced Chart, embedded as their free widget.
 *
 * The widget is not a React component: its loader script reads a JSON config
 * from its own `innerHTML` and replaces the surrounding container with an
 * iframe. So every config change (symbol, theme) tears the container down and
 * re-injects the script rather than updating in place.
 *
 * Everything the chart draws comes from TradingView, not from Investa's
 * backend — prices, splits and dividends are theirs, and can differ slightly
 * from the Yahoo-sourced series the Price/Return views draw.
 */
export default function TradingViewChart({ symbol, exchange, height = 560 }: TradingViewChartProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const wrapperRef = useRef<HTMLDivElement>(null);
    const [failed, setFailed] = useState(false);
    const [ready, setReady] = useState(false);
    const [mode, setMode] = useState<DisplayMode>('inline');
    const { resolvedTheme } = useTheme();

    // Rendered server-side `resolvedTheme` is undefined; wait for the client so
    // the widget isn't built in the wrong palette and then rebuilt.
    const [mounted, setMounted] = useState(false);
    useEffect(() => {
        // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time mount flag for the SSR hydration guard
        setMounted(true);
    }, []);

    const tvSymbol = toTradingViewSymbol(symbol, exchange);
    const theme = resolvedTheme === 'dark' ? 'dark' : 'light';

    useEffect(() => {
        const container = containerRef.current;
        if (!container || !tvSymbol || !mounted) return;

        // eslint-disable-next-line react-hooks/set-state-in-effect -- load state belongs to the widget being (re)mounted just below
        setFailed(false);
        setReady(false);
        container.innerHTML = '';

        const widget = document.createElement('div');
        widget.className = 'tradingview-widget-container__widget';
        widget.style.height = '100%';
        widget.style.width = '100%';
        container.appendChild(widget);

        const script = document.createElement('script');
        script.src = SCRIPT_SRC;
        script.async = true;
        script.type = 'text/javascript';
        script.innerHTML = JSON.stringify({
            symbol: tvSymbol,
            autosize: true,
            interval: 'D',
            // Market-local, per Investa's time convention — never the viewer's
            // device zone. TradingView resolves "exchange" to the listing venue.
            timezone: 'exchange',
            theme,
            style: '1',            // candlesticks
            locale: 'en',
            withdateranges: true,
            hide_side_toolbar: false,
            allow_symbol_change: false,
            details: false,
            calendar: false,
            support_host: 'https://www.tradingview.com',
        });
        script.onload = () => setReady(true);
        script.onerror = () => setFailed(true);
        container.appendChild(script);

        return () => {
            container.innerHTML = '';
        };
    }, [tvSymbol, theme, mounted]);

    const exitFullscreen = useCallback(() => {
        if (document.fullscreenElement === wrapperRef.current) void document.exitFullscreen();
        setMode('inline');
    }, []);

    const enterFullscreen = useCallback(() => {
        const el = wrapperRef.current;
        // Ask for real fullscreen first; fall back to the portalled overlay both
        // when the API is missing and when the browser rejects the request.
        if (el?.requestFullscreen && document.fullscreenEnabled) {
            el.requestFullscreen()
                .then(() => setMode('native'))
                .catch(() => setMode('overlay'));
        } else {
            setMode('overlay');
        }
    }, []);

    // Esc, F11 and the browser's own exit affordance all leave fullscreen
    // without going through our button — follow the document back to inline.
    useEffect(() => {
        if (mode !== 'native') return;
        const onChange = () => {
            if (document.fullscreenElement !== wrapperRef.current) setMode('inline');
        };
        document.addEventListener('fullscreenchange', onChange);
        return () => document.removeEventListener('fullscreenchange', onChange);
    }, [mode]);

    // The overlay is ours, so it owns the two behaviours the browser would
    // otherwise provide for free: Esc to leave, and no scrolling behind it.
    useEffect(() => {
        if (mode !== 'overlay') return;
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key !== 'Escape') return;
            e.stopPropagation();   // Esc closes the chart, not whatever is behind it
            setMode('inline');
        };
        document.addEventListener('keydown', onKeyDown, true);
        const previousOverflow = document.body.style.overflow;
        document.body.style.overflow = 'hidden';
        return () => {
            document.removeEventListener('keydown', onKeyDown, true);
            document.body.style.overflow = previousOverflow;
        };
    }, [mode]);

    if (!tvSymbol) {
        return (
            <div
                className="flex items-center justify-center rounded-xl border border-border bg-secondary/40 text-sm text-muted-foreground text-center px-6"
                style={{ height }}
            >
                {symbol} has no TradingView listing we can resolve — use the Price or Return&nbsp;% view.
            </div>
        );
    }

    const fullscreen = mode !== 'inline';

    const chart = (
        <div
            ref={wrapperRef}
            // No `relative` here: while natively fullscreen it would override the
            // UA's `position: fixed` on the top-layer element and collapse the
            // chart. The inner wrapper below is what the overlays position against.
            className={
                'flex flex-col ' + (
                    mode === 'native' ? 'w-full h-full bg-background p-2'
                        : mode === 'overlay' ? 'fixed inset-0 z-[200] bg-background p-2'
                            : 'w-full'
                )
            }
            style={fullscreen ? undefined : { height }}
        >
            {/* The toggle gets a row to itself. Floated over the chart it would
                sit on top of TradingView's own top-right toolbar button. */}
            <div className="flex items-center justify-end shrink-0 pb-1.5">
                <button
                    type="button"
                    onClick={fullscreen ? exitFullscreen : enterFullscreen}
                    title={fullscreen ? 'Exit full screen (Esc)' : 'Full screen'}
                    aria-label={fullscreen ? 'Exit full screen' : 'Full screen'}
                    className="flex items-center gap-1.5 px-2 py-1 rounded-md border border-border bg-secondary text-[10px] font-bold uppercase tracking-wider text-muted-foreground transition-colors hover:text-foreground hover:bg-accent/10"
                >
                    {fullscreen ? <Minimize2 className="h-3 w-3" /> : <Maximize2 className="h-3 w-3" />}
                    {fullscreen ? 'Exit Full Screen' : 'Full Screen'}
                </button>
            </div>

            <div className="relative flex-1 min-h-0">
                {!ready && !failed && (
                    <div className="absolute inset-0 flex items-center justify-center z-10">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
                    </div>
                )}
                {failed && (
                    <div className="absolute inset-0 flex items-center justify-center z-10 text-sm text-muted-foreground text-center px-6">
                        TradingView couldn&apos;t be reached. Check your connection, or use the Price
                        or Return&nbsp;% view.
                    </div>
                )}
                <div
                    ref={containerRef}
                    className="tradingview-widget-container h-full w-full overflow-hidden rounded-xl"
                />
            </div>
        </div>
    );

    // Native fullscreen keeps the widget where it is — no remount, so the chart
    // survives the transition with whatever interval and drawings it had.
    return mode === 'overlay' ? createPortal(chart, document.body) : chart;
}
