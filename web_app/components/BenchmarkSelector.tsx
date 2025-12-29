import React, { useState, useRef, useEffect } from 'react';

interface BenchmarkSelectorProps {
    selectedBenchmarks: string[];
    onBenchmarkChange: (benchmarks: string[]) => void;
}

const BENCHMARKS = [
    "S&P 500",
    "Dow Jones",
    "NASDAQ",
    "Russell 2000",
    "SPY (S&P 500 ETF)",
    "QQQ (Nasdaq 100 ETF)",
    "DIA (Dow Jones ETF)",
    "S&P 500 Total Return",
];

export default function BenchmarkSelector({ selectedBenchmarks, onBenchmarkChange }: BenchmarkSelectorProps) {
    const [isOpen, setIsOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    const handleToggle = (benchmark: string) => {
        if (selectedBenchmarks.includes(benchmark)) {
            onBenchmarkChange(selectedBenchmarks.filter(b => b !== benchmark));
        } else {
            onBenchmarkChange([...selectedBenchmarks, benchmark]);
        }
    };

    // Close on click outside
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, []);

    return (
        <div className="relative" ref={dropdownRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center space-x-2 px-3 py-1 text-sm font-medium text-foreground bg-secondary border border-border rounded-md hover:bg-accent/10 transition-colors"
            >
                <span>Index</span>
                <span className="bg-accent/20 text-xs px-1.5 py-0.5 rounded-full text-foreground">
                    {selectedBenchmarks.length}
                </span>
            </button>

            {isOpen && (
                <div className="absolute left-0 md:right-0 mt-2 w-56 bg-popover backdrop-blur-xl rounded-md shadow-2xl border border-border z-50">
                    <div className="p-2 space-y-1">
                        {BENCHMARKS.map((benchmark) => (
                            <label key={benchmark} className="flex items-center space-x-2 px-2 py-1.5 hover:bg-accent/10 rounded cursor-pointer transition-colors">
                                <input
                                    type="checkbox"
                                    checked={selectedBenchmarks.includes(benchmark)}
                                    onChange={() => handleToggle(benchmark)}
                                    className="rounded border-border bg-secondary text-cyan-500 focus:ring-cyan-500"
                                />
                                <span className="text-sm text-foreground">{benchmark}</span>
                            </label>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
