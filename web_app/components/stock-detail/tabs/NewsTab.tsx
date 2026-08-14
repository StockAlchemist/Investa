/* eslint-disable @next/next/no-img-element -- renders a dynamic external news thumbnail */

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Newspaper, ExternalLink } from 'lucide-react';
import { fetchStockNews } from '../../../lib/api';

function timeAgo(iso: string): string {
    const m = Math.floor((Date.now() - new Date(iso).getTime()) / 60000);
    if (m < 1) return 'just now';
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ago`;
    return `${Math.floor(h / 24)}d ago`;
}

interface NewsTabProps {
    symbol: string;
    isOpen: boolean;
}

export const NewsTab: React.FC<NewsTabProps> = ({ symbol, isOpen }) => {
    const { data: newsItems = [], isLoading: newsLoading } = useQuery({
        queryKey: ['stock-news-modal', symbol],
        queryFn: () => fetchStockNews([symbol], 20),
        staleTime: 5 * 60 * 1000,
        enabled: isOpen && !!symbol,
    });

    if (newsLoading) {
        return (
            <div className="space-y-3 animate-in fade-in duration-300">
                {Array.from({ length: 5 }).map((_, i) => (
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
        );
    }

    if (!newsItems.length) {
        return (
            <div className="flex flex-col items-center justify-center py-16 text-center animate-in fade-in duration-300">
                <Newspaper className="w-10 h-10 text-muted-foreground/30 mb-3" />
                <p className="text-sm text-muted-foreground">No recent news found for {symbol}.</p>
            </div>
        );
    }

    return (
        <div className="space-y-3 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {newsItems.map((item, i) => (
                <a
                    key={i}
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
            ))}
        </div>
    );
};
