import { Skeleton } from "@/components/ui/skeleton";

export default function ChartSkeleton() {
    return (
        <div className="bg-card p-6 rounded-xl shadow-sm border border-border">
            <div className="flex justify-between items-center mb-6">
                <Skeleton className="h-6 w-48" />
                <Skeleton className="h-8 w-24" />
            </div>
            <div className="h-[300px] w-full flex items-end gap-2">
                {[...Array(12)].map((_, i) => (
                    <Skeleton
                        key={i}
                        className="w-full rounded-t-sm"
                        style={{
                            height: `${(i * 17) % 60 + 20}%`,
                            opacity: 0.5 + ((i * 11) % 50 / 100)
                        }}
                    />
                ))}
            </div>
        </div>
    );
}
