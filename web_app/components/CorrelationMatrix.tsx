import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { fetchCorrelationMatrix, CorrelationData } from '../lib/api'; // fetchCorrelationMatrix type imported but not used directly
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface CorrelationMatrixProps {
    data: CorrelationData | null;
    isLoading: boolean;
    period: string;
    onPeriodChange: (p: string) => void;
}

const PERIOD_OPTIONS = [
    { value: '3m', label: '3 Months' },
    { value: '6m', label: '6 Months' },
    { value: '1y', label: '1 Year' },
    { value: '3y', label: '3 Years' },
    { value: '5y', label: '5 Years' },
];

export function CorrelationMatrix({ data, isLoading, period, onPeriodChange }: CorrelationMatrixProps) {
    // Removed local state and useEffect
    // const [period, setPeriod] = useState<string>('1y'); -> now prop

    // const [error, setError] = useState<string | null>(null); // We can handle error if data is null and not loading, or just fail gracefully
    // const [loading, setLoading] = useState<boolean>(true); -> now prop

    const [hoveredCell, setHoveredCell] = useState<{ x: string, y: string, value: number } | null>(null);

    // Removed useEffect for fetching data

    const getColor = (value: number) => {
        // -1 (red) -> 0 (white/grey) -> 1 (green)
        // Or for correlation: 
        // 1 (high correlation) -> red/orange? Usually diversification is goal, so high correlation is "bad" (risk).
        // -1 (negative correlation) -> green? (hedging).
        // 0 (uncorrelated) -> neutral.

        // Let's use:
        // 1.0 -> #ef4444 (Red - High Risk/Overlap)
        // 0.5 -> #f97316 (Orange)
        // 0.0 -> #e5e7eb (Gray - Neutral)
        // -0.5 -> #22c55e (Green - Diversifier)
        // -1.0 -> #15803d (Dark Green)

        // Tailwind colors roughly:
        // Red-500: #ef4444
        // Gray-200: #e5e7eb (Light mode) / Gray-800 (Dark mode)
        // Green-500: #22c55e

        // Simple interpolation logic for style
        if (value > 0) {
            // White to Red
            // Opacity approach is easiest
            return `rgba(239, 68, 68, ${value})`; // Red with alpha
        } else {
            // White to Green
            return `rgba(34, 197, 94, ${Math.abs(value)})`; // Green with alpha
        }
    };

    // Better color scale for dark mode
    const getCellColor = (value: number) => {
        // Updated Logic: Positive = Green (Standard Math), Negative = Red
        // Strong positive (1) = Green
        // Neutral (0) = Gray/Transparent
        // Strong negative (-1) = Red

        if (value >= 0.8) return "bg-emerald-500/90 text-white";
        if (value >= 0.5) return "bg-emerald-500/60 text-white";
        if (value >= 0.2) return "bg-emerald-500/30 text-emerald-900 dark:text-emerald-100";

        if (value >= -0.2) return "bg-muted/50 text-muted-foreground"; // Neutral

        // Negative (Red)
        if (value >= -0.5) return "bg-red-500/30 text-red-900 dark:text-red-100";
        return "bg-red-500/80 text-white";
    };

    /* Error handling handled by parent or empty state */

    // Prepare matrix grid
    const assets = data?.assets || [];
    const matrix = data?.correlation || [];

    // Create lookup map
    const matrixMap: Record<string, number> = {};
    matrix.forEach(item => {
        matrixMap[`${item.x}:${item.y}`] = item.value;
    });

    return (
        <Card className="h-full border-border/50 bg-card transition-all duration-300 hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-xl font-bold bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
                    Correlation Matrix
                </CardTitle>
                <div className="relative">
                    <select
                        value={period}
                        onChange={(e) => onPeriodChange(e.target.value)}
                        className="w-[120px] h-8 text-xs bg-background/50 border border-input/50 rounded-md px-2 focus:ring-primary/20 focus:outline-none appearance-none cursor-pointer"
                    >
                        {PERIOD_OPTIONS.map(opt => (
                            <option key={opt.value} value={opt.value}>
                                {opt.label}
                            </option>
                        ))}
                    </select>
                    <ChevronDown className="absolute right-2 top-2 h-4 w-4 text-muted-foreground pointer-events-none" />
                </div>
            </CardHeader>
            <CardContent>
                {isLoading ? (
                    <div className="space-y-4">
                        <Skeleton className="h-[300px] w-full rounded-xl bg-muted/20" />
                    </div>
                ) : assets.length === 0 ? (
                    <div className="h-[300px] flex items-center justify-center text-muted-foreground text-sm">
                        Not enough data for correlation matrix.
                    </div>
                ) : (
                    <div className="w-full overflow-x-auto">
                        <div
                            className="bg-card rounded-lg p-4 inline-block min-w-full"
                        >
                            {/* Grid Container */}
                            <div
                                style={{
                                    display: 'grid',
                                    gridTemplateColumns: `auto repeat(${assets.length}, minmax(40px, 1fr))`,
                                    gap: '4px'
                                }}
                                className="text-xs"
                            >
                                {/* Header Row */}
                                <div className="font-bold text-muted-foreground"></div> {/* Corner */}
                                {assets.map(asset => (
                                    <div key={asset} className="flex items-center justify-center font-semibold text-muted-foreground -rotate-45 h-12 self-end">
                                        <span className="translate-y-1">{asset}</span>
                                    </div>
                                ))}

                                {/* Data Rows */}
                                {assets.map(rowAsset => (
                                    <React.Fragment key={rowAsset}>
                                        {/* Row Label */}
                                        <div className="flex items-center justify-end font-semibold text-muted-foreground pr-2">
                                            {rowAsset}
                                        </div>

                                        {/* Cells */}
                                        {assets.map(colAsset => {
                                            const val = matrixMap[`${rowAsset}:${colAsset}`];
                                            const isSelf = rowAsset === colAsset;

                                            // Don't show redundant upper triangle? 
                                            // Actually matrix is full. Showing full is fine.
                                            // Maybe dim diagonal.

                                            return (
                                                <div
                                                    key={`${rowAsset}-${colAsset}`}
                                                    className={cn(
                                                        "aspect-square rounded flex items-center justify-center cursor-pointer transition-all duration-200 border border-transparent hover:border-foreground/20 hover:scale-110 hover:z-10 relative group",
                                                        isSelf ? "bg-muted/10 text-transparent" : getCellColor(val)
                                                    )}
                                                    onMouseEnter={() => setHoveredCell({ x: rowAsset, y: colAsset, value: val })}
                                                    onMouseLeave={() => setHoveredCell(null)}
                                                >
                                                    {!isSelf && (
                                                        <span className="opacity-0 group-hover:opacity-100 font-bold text-[10px] select-none pointer-events-none">
                                                            {val > 0 ? '+' : ''}{val.toFixed(2)}
                                                        </span>
                                                    )}
                                                </div>
                                            );
                                        })}
                                    </React.Fragment>
                                ))}
                            </div>

                            {/* Simple Legend */}
                            <div className="mt-6 flex items-center justify-center gap-6 text-xs text-muted-foreground">
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded bg-red-500/80"></div>
                                    <span>Negative Correlation</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded bg-muted/50 border border-border"></div>
                                    <span>Neutral</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded bg-emerald-500/90"></div>
                                    <span>Positive Correlation</span>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
