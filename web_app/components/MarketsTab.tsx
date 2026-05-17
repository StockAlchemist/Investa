'use client';

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { AreaChart, Area, YAxis, ResponsiveContainer } from 'recharts';
import { ExternalLink, Newspaper, TrendingUp, TrendingDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import { fetchMarketNews, fetchStockNews, type MarketNewsItem } from '@/lib/api';

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

export default function MarketsTab({ indices, onIndexClick, portfolioSymbols = [], watchlistSymbols = [] }: MarketsTabProps) {
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

    return (
        <div className="space-y-8">
            {/* Indices */}
            <div>
                <h2 className="text-2xl font-bold tracking-tight text-foreground mb-4">Market Indices</h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.values(indices).map((index: MarketIndex) => (
                        <IndexCard key={index.name} index={index} onClick={onIndexClick} />
                    ))}
                </div>
            </div>

            {/* Your Holdings & Watchlist News */}
            {allStockSymbols.length > 0 && (
                <NewsSection
                    title="Your Holdings & Watchlist"
                    news={stockNews}
                    isLoading={stockNewsLoading}
                />
            )}

            {/* General Market News */}
            <NewsSection title="Market News" news={news} isLoading={newsLoading} />
        </div>
    );
}
