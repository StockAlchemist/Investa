import { Skeleton } from "@/components/ui/skeleton";

export default function AppShellSkeleton() {
    return (
        <div className="min-h-screen bg-background pb-20">
            <div className="fixed inset-0 z-[-1] bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-indigo-900/20 via-background to-background pointer-events-none" />

            {/* Sidebar - Desktop Skeleton */}
            <aside className="fixed left-0 top-0 bottom-0 w-[72px] flex flex-col items-center py-6 border-r border-border bg-background/40 backdrop-blur-2xl z-[60] hidden md:flex">
                <div className="flex-1 flex flex-col items-center gap-6">
                    {/* Navigation Items */}
                    {[1, 2, 3, 4].map((i) => (
                        <Skeleton key={i} className="w-10 h-10 rounded-xl" />
                    ))}

                    <div className="mt-8">
                        <Skeleton className="w-10 h-10 rounded-xl" />
                    </div>
                </div>

                <div className="mt-auto flex flex-col items-center gap-4 pb-4">
                    <Skeleton className="w-10 h-10 rounded-xl" />
                    <Skeleton className="w-10 h-10 rounded-full" />
                </div>
            </aside>

            {/* Header Skeleton */}
            <header className="sticky top-0 z-50 w-full border-b border-border bg-background/60 backdrop-blur-xl">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 md:pl-[72px] py-3 sm:py-4 flex justify-between items-center gap-4 sm:gap-8">
                    <div className="flex items-center gap-4">
                        {/* Logo Skeleton */}
                        <Skeleton className="w-12 h-12 rounded-xl" />
                        <div className="hidden sm:flex flex-col gap-2">
                            <Skeleton className="h-6 w-32" />
                            <Skeleton className="h-3 w-24" />
                        </div>
                    </div>

                    <div className="flex items-center gap-4">
                        {/* Market Indices Skeletons */}
                        {[1, 2, 3].map((i) => (
                            <Skeleton key={i} className="hidden lg:block h-14 w-32 rounded-xl" />
                        ))}

                        <div className="hidden md:block">
                            <Skeleton className="h-10 w-40 rounded-xl" />
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Content Skeleton */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6 md:pl-[72px]">
                {/* Dashboard Grid Skeleton */}
                <div className="mb-10 space-y-6">
                    {/* Scalar Metrics Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-2 lg:grid-cols-4 gap-3 md:gap-6">
                        {[1, 2, 3, 4].map((i) => (
                            <div key={i} className="col-span-1 h-32 rounded-xl border border-border bg-card/50 p-6">
                                <Skeleton className="h-4 w-24 mb-4" />
                                <Skeleton className="h-8 w-32" />
                            </div>
                        ))}
                    </div>

                    {/* Complex Metrics Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        <div className="lg:col-span-2 h-[400px] rounded-xl border border-border bg-card/50 p-6">
                            <Skeleton className="h-full w-full" />
                        </div>
                        <div className="lg:col-span-2 h-[400px] rounded-xl border border-border bg-card/50 p-6">
                            <Skeleton className="h-full w-full" />
                        </div>
                    </div>

                    {/* Graph Skeleton */}
                    <div className="h-[400px] rounded-xl border border-border bg-card/50 p-6">
                        <Skeleton className="h-full w-full" />
                    </div>
                </div>
            </div>

            {/* Bottom Nav Skeleton (Mobile) */}
            <div className="fixed bottom-0 left-0 right-0 border-t border-border px-4 py-3 flex justify-between items-center md:hidden bg-background">
                {[1, 2, 3, 4].map((i) => (
                    <Skeleton key={i} className="w-8 h-8 rounded-full" />
                ))}
            </div>
        </div>
    );
}
