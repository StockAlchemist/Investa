import React from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface TransactionsPaginationProps {
    currentPage: number;
    totalPages: number;
    pageSize: number;
    setPageSize: (size: number) => void;
    setCurrentPage: (page: number | ((prev: number) => number)) => void;
    totalCount: number;
}

export const TransactionsPagination: React.FC<TransactionsPaginationProps> = ({
    currentPage,
    totalPages,
    pageSize,
    setPageSize,
    setCurrentPage,
    totalCount,
}) => {
    if (totalCount === 0) return null;

    const startIdx = (currentPage - 1) * pageSize + 1;
    const endIdx = Math.min(currentPage * pageSize, totalCount);

    return (
        <div className="flex flex-wrap items-center justify-between gap-3 pt-2 text-xs text-muted-foreground">
            <div className="flex items-center gap-2">
                <span>Show</span>
                <select
                    value={pageSize}
                    onChange={e => {
                        setPageSize(Number(e.target.value));
                        setCurrentPage(1);
                    }}
                    className="px-2 py-1 bg-background border border-border/60 rounded-lg text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-cyan-500"
                >
                    <option value={10}>10</option>
                    <option value={25}>25</option>
                    <option value={50}>50</option>
                    <option value={100}>100</option>
                </select>
                <span>per page</span>
                <span className="hidden sm:inline text-muted-foreground/60">·</span>
                <span className="hidden sm:inline">
                    Showing <span className="font-semibold text-foreground">{startIdx}</span> to <span className="font-semibold text-foreground">{endIdx}</span> of <span className="font-semibold text-foreground">{totalCount}</span>
                </span>
            </div>

            <div className="flex items-center gap-1 ml-auto">
                <button
                    onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                    disabled={currentPage <= 1}
                    className="p-1.5 rounded-lg border border-border/60 bg-background text-foreground hover:bg-secondary disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                    title="Previous page"
                >
                    <ChevronLeft className="w-3.5 h-3.5" />
                </button>
                <span className="px-2 py-1 font-medium text-foreground">
                    {currentPage} / {totalPages}
                </span>
                <button
                    onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                    disabled={currentPage >= totalPages}
                    className="p-1.5 rounded-lg border border-border/60 bg-background text-foreground hover:bg-secondary disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                    title="Next page"
                >
                    <ChevronRight className="w-3.5 h-3.5" />
                </button>
            </div>
        </div>
    );
};
