import { Skeleton } from "@/components/ui/skeleton";
import ChartSkeleton from "./ChartSkeleton";
import TableSkeleton from "./TableSkeleton";

interface TabContentSkeletonProps {
    type?: 'full' | 'chart-only' | 'table-only';
}

export default function TabContentSkeleton({ type = 'full' }: TabContentSkeletonProps) {
    return (
        <div className="space-y-6">
            {/* Metric Cards */}
            {type === 'full' && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {[...Array(3)].map((_, i) => (
                        <div key={i} className="metric-card card-shine p-4 rounded-2xl space-y-2 border border-border/40">
                            <Skeleton className="h-4 w-24 opacity-50" />
                            <Skeleton className="h-8 w-32 opacity-40" />
                        </div>
                    ))}
                </div>
            )}

            {(type === 'full' || type === 'chart-only') && (
                <ChartSkeleton />
            )}

            {(type === 'full' || type === 'table-only') && (
                <div className="metric-card rounded-2xl overflow-hidden p-0 border border-border/40">
                    <div className="p-4 border-b border-border/30 flex justify-between">
                        <Skeleton className="h-6 w-48 opacity-50" />
                        <Skeleton className="h-4 w-24 opacity-40" />
                    </div>
                    <div className="p-4 space-y-4">
                        {[...Array(5)].map((_, i) => (
                            <div key={i} className="flex justify-between items-center">
                                <div className="flex gap-4">
                                    <Skeleton className="h-4 w-24 opacity-50" />
                                    <Skeleton className="h-4 w-32 opacity-40" />
                                </div>
                                <Skeleton className="h-4 w-24 opacity-50" />
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
