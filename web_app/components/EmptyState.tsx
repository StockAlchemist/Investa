'use client';

import { cn } from '@/lib/utils';
import {
  UploadCloud, ArrowRight, Wallet, TrendingUp,
  PlusCircle, FileSpreadsheet,
} from 'lucide-react';

interface EmptyStateProps {
  onNavigate?: (tab: string) => void;
  className?: string;
}

export function EmptyState({ onNavigate, className }: EmptyStateProps) {
  return (
    <div className={cn('flex flex-col items-center justify-center py-16 px-6 text-center', className)}>
      {/* Illustration */}
      <div className="relative mb-6">
        <div className="w-20 h-20 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
          <Wallet className="w-9 h-9 text-primary/70" />
        </div>
        <div className="absolute -top-2 -right-2 w-7 h-7 rounded-full bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center">
          <TrendingUp className="w-3.5 h-3.5 text-emerald-500" />
        </div>
      </div>

      <h2 className="text-xl font-bold text-foreground mb-2">No portfolio data yet</h2>
      <p className="text-sm text-muted-foreground max-w-sm mb-8 leading-relaxed">
        Get started by importing your transactions or adding holdings manually. Investa will calculate your returns, allocations, and more.
      </p>

      {/* Action cards */}
      <div className="grid sm:grid-cols-2 gap-3 w-full max-w-md">
        <button
          onClick={() => onNavigate?.('transactions')}
          className="group flex items-start gap-3 p-4 rounded-xl border border-border bg-card hover:border-primary/40 hover:bg-primary/5 transition-all duration-200 text-left"
        >
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0 mt-0.5 group-hover:bg-primary/20 transition-colors">
            <PlusCircle className="w-4 h-4 text-primary" />
          </div>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-foreground">Add Transaction</p>
            <p className="text-xs text-muted-foreground mt-0.5">Record a buy, sell, or transfer</p>
          </div>
          <ArrowRight className="w-4 h-4 text-muted-foreground/40 shrink-0 self-center ml-auto group-hover:text-primary/60 group-hover:translate-x-0.5 transition-all" />
        </button>

        <button
          onClick={() => onNavigate?.('settings')}
          className="group flex items-start gap-3 p-4 rounded-xl border border-border bg-card hover:border-primary/40 hover:bg-primary/5 transition-all duration-200 text-left"
        >
          <div className="w-8 h-8 rounded-lg bg-violet-500/10 flex items-center justify-center shrink-0 mt-0.5 group-hover:bg-violet-500/20 transition-colors">
            <FileSpreadsheet className="w-4 h-4 text-violet-500" />
          </div>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-foreground">Import CSV</p>
            <p className="text-xs text-muted-foreground mt-0.5">Bulk import from a spreadsheet</p>
          </div>
          <ArrowRight className="w-4 h-4 text-muted-foreground/40 shrink-0 self-center ml-auto group-hover:text-primary/60 group-hover:translate-x-0.5 transition-all" />
        </button>
      </div>

      {/* Tips */}
      <div className="mt-8 flex items-start gap-2 p-3 rounded-lg bg-muted/40 border border-border/50 max-w-md w-full text-left">
        <UploadCloud className="w-4 h-4 text-muted-foreground shrink-0 mt-0.5" />
        <p className="text-xs text-muted-foreground leading-relaxed">
          <span className="font-semibold text-foreground">Tip:</span> Investa supports IBKR sync and CSV imports from most major brokers. Check Settings → Import for supported formats.
        </p>
      </div>
    </div>
  );
}
