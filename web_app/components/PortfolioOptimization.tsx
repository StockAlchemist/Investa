'use client';

import React from 'react';
import { 
    Zap, 
    ArrowRightLeft, 
    ShieldAlert, 
    TrendingDown, 
    CheckCircle2, 
    ArrowUpRight,
    Info
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useStockModal } from '@/context/StockModalContext';

interface Optimization {
    type: 'tax_loss_harvesting' | 'rebalancing' | 'diversification';
    title: string;
    description: string;
    symbol: string;
    action: 'Sell' | 'Buy' | 'Swap' | 'Hold';
    priority: 'High' | 'Medium' | 'Low';
}

interface PortfolioOptimizationProps {
    optimizations: Optimization[];
}

export default function PortfolioOptimization({ optimizations }: PortfolioOptimizationProps) {
    const { openStockDetail } = useStockModal();
    if (!optimizations || optimizations.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center p-8 text-center bg-card/30 rounded-[2rem] border border-dashed border-white/10 mt-6">
                <CheckCircle2 className="w-8 h-8 text-emerald-500/50 mb-3" />
                <h3 className="text-sm font-bold uppercase tracking-widest text-muted-foreground">Portfolio Optimized</h3>
                <p className="text-xs text-muted-foreground/60 max-w-[200px] mt-2">
                    No urgent rebalancing or tax saving opportunities detected.
                </p>
            </div>
        );
    }

    const getTypeIcon = (type: string) => {
        switch (type) {
            case 'tax_loss_harvesting': return <TrendingDown className="w-5 h-5 text-rose-500" />;
            case 'rebalancing': return <ArrowRightLeft className="w-5 h-5 text-indigo-500" />;
            case 'diversification': return <ShieldAlert className="w-5 h-5 text-amber-500" />;
            default: return <Zap className="w-5 h-5 text-purple-500" />;
        }
    };

    const getPriorityColor = (priority: string) => {
        switch (priority) {
            case 'High': return "bg-rose-500/20 text-rose-500 border-rose-500/20";
            case 'Medium': return "bg-amber-500/20 text-amber-500 border-amber-500/20";
            case 'Low': return "bg-indigo-500/20 text-indigo-500 border-indigo-500/20";
            default: return "bg-slate-500/20 text-slate-500 border-slate-500/20";
        }
    };

    return (
        <div className="space-y-4 pt-4">
            <div className="flex items-center gap-3 px-1 mb-2">
                <div className="p-2 rounded-xl bg-orange-500/10 text-orange-500">
                    <Zap className="w-4 h-4" />
                </div>
                <div>
                    <h3 className="text-sm font-black uppercase tracking-[0.15em]">AI Optimization Hub</h3>
                    <p className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">Suggested Actions</p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {optimizations.map((opt, idx) => (
                    <div 
                        key={idx}
                        className={cn(
                            "group relative p-5 rounded-[2rem] transition-all duration-300",
                            "bg-white dark:bg-slate-900/40 border border-black/5 dark:border-white/5",
                            "hover:shadow-2xl hover:shadow-indigo-500/10 hover:-translate-y-1"
                        )}
                    >
                        <div className="flex items-start justify-between mb-4">
                            <div className="flex items-center gap-3">
                                <div className="p-2.5 rounded-2xl bg-muted/50 group-hover:bg-muted transition-colors">
                                    {getTypeIcon(opt.type)}
                                </div>
                                <div>
                                    <h4 className="font-bold text-sm leading-tight">{opt.title}</h4>
                                    <span className="text-[10px] font-bold text-muted-foreground/60 uppercase tracking-widest">{opt.symbol}</span>
                                </div>
                            </div>
                            <Badge className={cn("text-[8px] font-black uppercase px-2 py-0.5 border-none", getPriorityColor(opt.priority))}>
                                {opt.priority} Priority
                            </Badge>
                        </div>

                        <p className="text-xs leading-relaxed text-muted-foreground group-hover:text-foreground transition-colors line-clamp-3">
                            {opt.description}
                        </p>

                        <div className="mt-6 flex items-center justify-between">
                             <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-500/5 text-[10px] font-black uppercase tracking-widest text-muted-foreground">
                                <ArrowUpRight className="w-3 h-3" />
                                {opt.action} Recommended
                            </div>
                            <Button 
                                variant="ghost" 
                                size="sm" 
                                className="h-8 rounded-full text-[10px] font-black uppercase tracking-widest group-hover:bg-indigo-500 group-hover:text-white transition-all"
                                onClick={() => opt.symbol !== 'N/A' && openStockDetail(opt.symbol)}
                            >
                                Review Lot Details
                            </Button>
                        </div>

                        {/* Rationale Indicator */}
                        <div className="absolute top-4 right-4 opacity-10 group-hover:opacity-100 transition-opacity">
                            <div className="tooltip-trigger">
                                <Info className="w-3.5 h-3.5" />
                            </div>
                        </div>
                    </div>
                ))}
            </div>
            
            <p className="text-[9px] text-center text-muted-foreground/40 mt-4 uppercase tracking-[0.1em] px-10">
                Optimizations based on modern portfolio theory and current tax legislation. 
                Always verify with your individual tax bracket and investment horizon.
            </p>
        </div>
    );
}
