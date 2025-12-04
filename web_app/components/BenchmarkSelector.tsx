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
                className="flex items-center space-x-2 px-3 py-1 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700"
            >
                <span>Benchmarks</span>
                <span className="bg-gray-200 dark:bg-gray-600 text-xs px-1.5 py-0.5 rounded-full">
                    {selectedBenchmarks.length}
                </span>
            </button>

            {isOpen && (
                <div className="absolute right-0 mt-2 w-56 bg-white dark:bg-gray-800 rounded-md shadow-lg border border-gray-200 dark:border-gray-700 z-50">
                    <div className="p-2 space-y-1">
                        {BENCHMARKS.map((benchmark) => (
                            <label key={benchmark} className="flex items-center space-x-2 px-2 py-1.5 hover:bg-gray-100 dark:hover:bg-gray-700 rounded cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={selectedBenchmarks.includes(benchmark)}
                                    onChange={() => handleToggle(benchmark)}
                                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                />
                                <span className="text-sm text-gray-700 dark:text-gray-300">{benchmark}</span>
                            </label>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
