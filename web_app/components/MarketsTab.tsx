'use client';

import React, { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { AreaChart, Area, YAxis, ResponsiveContainer } from 'recharts';
import { ExternalLink, Newspaper, TrendingUp, TrendingDown, Search, X, BarChart3 } from 'lucide-react';
import { cn, formatCurrency } from '@/lib/utils';
import { fetchMarketNews, fetchStockNews, type MarketNewsItem, type Holding } from '@/lib/api';
import { useStockModal } from '@/context/StockModalContext';

interface MarketIndex {
    name: string;
    price: number;
    change: number;
    changesPercentage: number;
    sparkline?: number[];
}

interface MarketsTabProps {
    indices: Record<string, MarketIndex>;
    onIndexClick: () => void;
    holdings?: Holding[];
    currency?: string;
    portfolioSymbols?: string[];
    watchlistSymbols?: string[];
}

function getIndexStyle(name: string): { stroke: string; borderClass: string } {
    const n = name.toLowerCase();
    if (n.includes('nasdaq'))                            return { stroke: '#8b5cf6', borderClass: 'border-l-violet-500' };
    if (n.includes('s&p') || n.includes('500'))          return { stroke: '#06b6d4', borderClass: 'border-l-cyan-500'   };
    if (n.includes('dow') || n.includes('jones'))        return { stroke: '#f59e0b', borderClass: 'border-l-amber-500'  };
    if (n.includes('russell'))                           return { stroke: '#f97316', borderClass: 'border-l-orange-500' };
    if (n.includes('ftse'))                              return { stroke: '#3b82f6', borderClass: 'border-l-blue-500'   };
    if (n.includes('nikkei') || n.includes('japan'))     return { stroke: '#ec4899', borderClass: 'border-l-pink-500'   };
    if (n.includes('dax') || n.includes('germany'))      return { stroke: '#14b8a6', borderClass: 'border-l-teal-500'   };
    return                                                      { stroke: '#10b981', borderClass: 'border-l-emerald-500' };
}

function IndexCard({ index, onClick }: { index: MarketIndex; onClick: () => void }) {
    const isUp = index.change >= 0;
    const { stroke, borderClass } = getIndexStyle(index.name);
    const gradientId = `ig-${index.name.replace(/[^a-zA-Z0-9]/g, '')}`;

    return (
        <div
            onClick={onClick}
            className={cn(
                'rounded-2xl border border-border/60 bg-card overflow-hidden cursor-pointer group',
                'hover:shadow-lg hover:-translate-y-0.5 transition-all duration-200',
                'border-l-4', borderClass,
            )}
        >
            {/* Header */}
            <div className="px-5 pt-5 pb-3">
                <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                        <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground truncate">
                            {index.name}
                        </p>
                        <p className="text-3xl font-bold tabular-nums text-foreground mt-1 leading-none">
                            {index.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </p>
                    </div>
                    <span className={cn(
                        'flex items-center gap-1 px-2.5 py-1 rounded-lg text-sm font-bold tabular-nums shrink-0 mt-1',
                        isUp
                            ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
                            : 'bg-rose-500/10 text-rose-600 dark:text-rose-400',
                    )}>
                        {isUp
                            ? <TrendingUp className="w-3.5 h-3.5" />
                            : <TrendingDown className="w-3.5 h-3.5" />}
                        {isUp ? '+' : ''}{index.changesPercentage.toFixed(2)}%
                    </span>
                </div>
                <p className={cn(
                    'text-sm font-semibold tabular-nums mt-1.5',
                    isUp ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400',
                )}>
                    {isUp ? '+' : ''}{index.change.toFixed(2)} pts
                </p>
            </div>

            {/* Edge-to-edge chart */}
            {index.sparkline && index.sparkline.length > 1 ? (
                <div className="h-24 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart
                            data={index.sparkline.map((v: number) => ({ value: v }))}
                            margin={{ top: 4, right: 0, bottom: 0, left: 0 }}
                        >
                            <defs>
                                <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%"  stopColor={stroke} stopOpacity={0.3} />
                                    <stop offset="95%" stopColor={stroke} stopOpacity={0}   />
                                </linearGradient>
                            </defs>
                            <YAxis
                                hide
                                domain={[
                                    (d: number) => d * 0.9995,
                                    (d: number) => d * 1.0005,
                                ]}
                            />
                            <Area
                                type="monotone"
                                dataKey="value"
                                stroke={stroke}
                                fill={`url(#${gradientId})`}
                                strokeWidth={2.5}
                                dot={false}
                                isAnimationActive={false}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            ) : (
                <div className="h-8" />
            )}

            {/* Footer label */}
            <p className="px-5 pb-3 text-[10px] text-muted-foreground/60 font-medium">7D Trend</p>
        </div>
    );
}

function timeAgo(isoDate: string): string {
    const diff = Date.now() - new Date(isoDate).getTime();
    const m = Math.floor(diff / 60000);
    if (m < 1) return 'just now';
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ago`;
    return `${Math.floor(h / 24)}d ago`;
}

function NewsCard({ item }: { item: MarketNewsItem }) {
    return (
        <a
            href={item.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex gap-3 p-3 rounded-xl border border-border/60 bg-card hover:bg-muted/40 transition-colors group"
        >
            {item.thumbnail && (
                <img
                    src={item.thumbnail}
                    alt=""
                    className="w-16 h-16 rounded-lg object-cover shrink-0 bg-muted"
                    onError={e => { (e.target as HTMLImageElement).style.display = 'none'; }}
                />
            )}
            <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-foreground leading-snug line-clamp-2 group-hover:text-primary transition-colors">
                    {item.title}
                </p>
                <div className="flex items-center gap-2 mt-1.5 flex-wrap">
                    {item.symbol && (
                        <span className="text-[10px] font-bold text-indigo-600 dark:text-indigo-400 bg-indigo-500/10 px-1.5 py-0.5 rounded shrink-0">
                            {item.symbol}
                        </span>
                    )}
                    <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wide truncate">
                        {item.provider}
                    </span>
                    {item.pub_date && (
                        <>
                            <span className="text-muted-foreground/40 text-[10px]">·</span>
                            <span className="text-[10px] text-muted-foreground shrink-0">
                                {timeAgo(item.pub_date)}
                            </span>
                        </>
                    )}
                    <ExternalLink className="w-3 h-3 text-muted-foreground/40 ml-auto shrink-0 group-hover:text-primary transition-colors" />
                </div>
            </div>
        </a>
    );
}

function NewsSection({ title, news, isLoading }: { title: string; news: MarketNewsItem[]; isLoading: boolean }) {
    return (
        <div>
            <div className="flex items-center gap-2 mb-4">
                <Newspaper className="w-5 h-5 text-muted-foreground" />
                <h2 className="text-2xl font-bold tracking-tight text-foreground">{title}</h2>
            </div>
            {isLoading ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {Array.from({ length: 6 }).map((_, i) => (
                        <div key={i} className="flex gap-3 p-3 rounded-xl border border-border/60 animate-pulse">
                            <div className="w-16 h-16 rounded-lg bg-muted shrink-0" />
                            <div className="flex-1 space-y-2 py-1">
                                <div className="h-3 bg-muted rounded w-full" />
                                <div className="h-3 bg-muted rounded w-4/5" />
                                <div className="h-2 bg-muted rounded w-1/3 mt-2" />
                            </div>
                        </div>
                    ))}
                </div>
            ) : news.length === 0 ? (
                <p className="text-sm text-muted-foreground">No news available.</p>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {news.map((item, i) => <NewsCard key={i} item={item} />)}
                </div>
            )}
        </div>
    );
}

function SummaryTile({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: 'pos' | 'neg' | 'neutral' }) {
    return (
        <div className="flex-1 min-w-0 px-2 sm:px-4 py-2.5 first:pl-0 last:pr-0">
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground/80 font-semibold mb-1.5">{label}</div>
            <div className={cn(
                'text-lg sm:text-xl font-bold tabular-nums leading-none truncate',
                tone === 'pos' ? 'text-emerald-600 dark:text-emerald-400'
                : tone === 'neg' ? 'text-red-600 dark:text-red-400'
                : 'text-foreground',
            )}>
                {value}
            </div>
            {sub && <div className="text-[10px] text-muted-foreground/70 mt-1 leading-none truncate">{sub}</div>}
        </div>
    );
}

function MarketsSummaryBar({ indices }: { indices: Record<string, MarketIndex> }) {
    const list = Object.values(indices);
    if (list.length === 0) return null;

    const up = list.filter(i => i.changesPercentage >= 0).length;
    const down = list.length - up;
    const best = list.reduce((a, b) => (b.changesPercentage > a.changesPercentage ? b : a));
    const worst = list.reduce((a, b) => (b.changesPercentage < a.changesPercentage ? b : a));

    return (
        <div className="metric-card p-3 sm:p-4">
            <div className="flex divide-x divide-border/60">
                <SummaryTile label="Breadth" value={`${up} ▲ / ${down} ▼`} sub={`${list.length} indices`} tone={up >= down ? 'pos' : 'neg'} />
                <SummaryTile label="Best" value={`${best.changesPercentage >= 0 ? '+' : ''}${best.changesPercentage.toFixed(2)}%`} sub={best.name} tone="pos" />
                <SummaryTile label="Worst" value={`${worst.changesPercentage >= 0 ? '+' : ''}${worst.changesPercentage.toFixed(2)}%`} sub={worst.name} tone="neg" />
            </div>
        </div>
    );
}

interface MoverRow {
    symbol: string;
    pct: number;
    price: number | null;
}

function MoversColumn({ title, rows, currency, positive, onPick }: {
    title: string;
    rows: MoverRow[];
    currency: string;
    positive: boolean;
    onPick: (symbol: string) => void;
}) {
    const Icon = positive ? TrendingUp : TrendingDown;
    const tone = positive ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400';
    return (
        <div>
            <div className="flex items-center gap-1.5 mb-2">
                <Icon className={cn('w-3.5 h-3.5', tone)} />
                <h4 className="text-[11px] uppercase tracking-wider font-semibold text-muted-foreground">{title}</h4>
            </div>
            {rows.length === 0 ? (
                <p className="text-xs text-muted-foreground/60 py-2">No movers.</p>
            ) : (
                <div className="space-y-1">
                    {rows.map(r => (
                        <button
                            key={r.symbol}
                            type="button"
                            onClick={() => onPick(r.symbol)}
                            className="w-full flex items-center justify-between gap-3 px-2 py-1.5 rounded-md hover:bg-muted/50 transition-colors text-left"
                        >
                            <span className="text-xs font-bold text-foreground truncate">{r.symbol}</span>
                            <span className="flex items-center gap-3 shrink-0 tabular-nums">
                                {r.price != null && (
                                    <span className="text-[11px] text-muted-foreground/70">{formatCurrency(r.price, currency)}</span>
                                )}
                                <span className={cn('text-xs font-bold', tone)}>
                                    {r.pct >= 0 ? '+' : ''}{r.pct.toFixed(2)}%
                                </span>
                            </span>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}

function YourMovers({ holdings, currency, onPick }: { holdings: Holding[]; currency: string; onPick: (symbol: string) => void }) {
    const { gainers, losers } = useMemo(() => {
        const isCash = (s: string) => {
            const u = (s || '').toUpperCase();
            return u === '$CASH' || u === 'CASH' || u.startsWith('CASH (');
        };
        const priceKey = `Price (${currency})`;
        // Deduplicate by symbol (holdings across multiple accounts share the same ticker)
        const bySymbol = new Map<string, MoverRow>();
        holdings
            .filter(h => !isCash(h.Symbol) && typeof h['Day Change %'] === 'number')
            .forEach(h => {
                if (!bySymbol.has(h.Symbol)) {
                    bySymbol.set(h.Symbol, {
                        symbol: h.Symbol,
                        pct: h['Day Change %'] as number,
                        price: typeof h[priceKey] === 'number' ? (h[priceKey] as number) : null,
                    });
                }
            });
        const sorted = [...bySymbol.values()].sort((a, b) => b.pct - a.pct);
        return {
            gainers: sorted.filter(r => r.pct > 0).slice(0, 5),
            losers: sorted.filter(r => r.pct < 0).slice(-5).reverse(),
        };
    }, [holdings, currency]);

    if (gainers.length === 0 && losers.length === 0) return null;

    return (
        <div>
            <div className="flex items-center gap-2 mb-4">
                <BarChart3 className="w-5 h-5 text-muted-foreground" />
                <h2 className="text-2xl font-bold tracking-tight text-foreground">Your Movers Today</h2>
            </div>
            <div className="metric-card p-5 grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-4">
                <MoversColumn title="Top Gainers" rows={gainers} currency={currency} positive onPick={onPick} />
                <MoversColumn title="Top Losers" rows={losers} currency={currency} positive={false} onPick={onPick} />
            </div>
        </div>
    );
}

export default function MarketsTab({ indices, onIndexClick, holdings = [], currency = 'USD', portfolioSymbols = [], watchlistSymbols = [] }: MarketsTabProps) {
    const { openStockDetail } = useStockModal();
    const [newsQuery, setNewsQuery] = useState('');
    const { data: news = [], isLoading: newsLoading } = useQuery({
        queryKey: ['market-news'],
        queryFn: () => fetchMarketNews(20),
        staleTime: 5 * 60 * 1000,
    });

    const allStockSymbols = Array.from(new Set([...portfolioSymbols, ...watchlistSymbols]));

    const { data: stockNews = [], isLoading: stockNewsLoading } = useQuery({
        queryKey: ['stock-news', allStockSymbols.join(',')],
        queryFn: () => fetchStockNews(allStockSymbols, 30),
        staleTime: 5 * 60 * 1000,
        enabled: allStockSymbols.length > 0,
    });

    const filterNews = (items: MarketNewsItem[]) => {
        const q = newsQuery.trim().toLowerCase();
        if (!q) return items;
        return items.filter(n =>
            n.title?.toLowerCase().includes(q)
            || n.provider?.toLowerCase().includes(q)
            || (n.symbol || '').toLowerCase().includes(q),
        );
    };
    const filteredStockNews = filterNews(stockNews);
    const filteredMarketNews = filterNews(news);

    return (
        <div className="space-y-8">
            {/* Market breadth summary */}
            <MarketsSummaryBar indices={indices} />

            {/* Indices */}
            <div>
                <h2 className="text-2xl font-bold tracking-tight text-foreground mb-4">Market Indices</h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.values(indices).map((index: MarketIndex) => (
                        <IndexCard key={index.name} index={index} onClick={onIndexClick} />
                    ))}
                </div>
            </div>

            {/* Your Movers Today */}
            <YourMovers holdings={holdings} currency={currency} onPick={(s) => openStockDetail(s, currency)} />

            {/* News search */}
            <div className="relative max-w-md">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground/60 pointer-events-none" />
                <input
                    type="text"
                    placeholder="Search news by headline, ticker, or source..."
                    value={newsQuery}
                    onChange={(e) => setNewsQuery(e.target.value)}
                    className="bg-card border border-border/60 text-foreground rounded-lg pl-9 pr-8 py-2 text-sm w-full focus:ring-primary focus:border-primary"
                />
                {newsQuery && (
                    <button
                        onClick={() => setNewsQuery('')}
                        className="absolute right-2 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
                        title="Clear search"
                    >
                        <X className="w-3.5 h-3.5" />
                    </button>
                )}
            </div>

            {/* Your Holdings & Watchlist News */}
            {allStockSymbols.length > 0 && (
                <NewsSection
                    title="Your Holdings & Watchlist"
                    news={filteredStockNews}
                    isLoading={stockNewsLoading}
                />
            )}

            {/* General Market News */}
            <NewsSection title="Market News" news={filteredMarketNews} isLoading={newsLoading} />
        </div>
    );
}
