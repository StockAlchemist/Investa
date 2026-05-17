import React, { useState, useRef, useEffect } from 'react';
import { X, Plus } from 'lucide-react';

interface BenchmarkSelectorProps {
    selectedBenchmarks: string[];
    onBenchmarkChange: (benchmarks: string[]) => void;
}

const PRESET_BENCHMARKS = [
    "S&P 500",
    "Dow Jones",
    "NASDAQ",
    "Russell 2000",
    "SPY (S&P 500 ETF)",
    "QQQ (Nasdaq 100 ETF)",
    "DIA (Dow Jones ETF)",
    "S&P 500 Total Return",
];

const isCustomTicker = (b: string) => !PRESET_BENCHMARKS.includes(b);

export default function BenchmarkSelector({ selectedBenchmarks, onBenchmarkChange }: BenchmarkSelectorProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [customInput, setCustomInput] = useState('');
    const dropdownRef = useRef<HTMLDivElement>(null);

    const handleToggle = (benchmark: string) => {
        if (selectedBenchmarks.includes(benchmark)) {
            onBenchmarkChange(selectedBenchmarks.filter(b => b !== benchmark));
        } else {
            onBenchmarkChange([...selectedBenchmarks, benchmark]);
        }
    };

    const handleAddCustom = () => {
        const ticker = customInput.trim().toUpperCase();
        if (!ticker) return;
        if (selectedBenchmarks.includes(ticker)) {
            setCustomInput('');
            return;
        }
        onBenchmarkChange([...selectedBenchmarks, ticker]);
        setCustomInput('');
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

    const customSelected = selectedBenchmarks.filter(isCustomTicker);

    return (
        <div className="relative" ref={dropdownRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center space-x-2 px-3 py-1 text-sm font-medium text-foreground bg-secondary border-none rounded-md hover:bg-accent/10 transition-colors"
            >
                <span>Index</span>
                <span className="bg-accent/20 text-xs px-1.5 py-0.5 rounded-full text-foreground">
                    {selectedBenchmarks.length}
                </span>
            </button>

            {isOpen && (
                <div className="absolute left-0 md:right-0 mt-2 w-64 bg-white dark:bg-zinc-950 rounded-md z-50 border border-border shadow-lg">
                    {/* Preset list */}
                    <div className="p-2 space-y-1 max-h-64 overflow-y-auto">
                        {PRESET_BENCHMARKS.map((benchmark) => (
                            <label
                                key={benchmark}
                                className={`flex items-center space-x-2 px-2 py-1.5 rounded cursor-pointer transition-colors ${selectedBenchmarks.includes(benchmark)
                                    ? 'bg-[#0097b2] text-white underline-none'
                                    : 'hover:bg-accent/10 text-foreground'
                                    }`}
                            >
                                <input
                                    type="checkbox"
                                    checked={selectedBenchmarks.includes(benchmark)}
                                    onChange={() => handleToggle(benchmark)}
                                    className={`rounded border-none bg-secondary focus:ring-cyan-500 ${selectedBenchmarks.includes(benchmark) ? 'text-white' : 'text-cyan-500'}`}
                                />
                                <span className={`text-sm ${selectedBenchmarks.includes(benchmark) ? 'text-white' : 'text-foreground'}`}>{benchmark}</span>
                            </label>
                        ))}

                        {/* Custom ticker chips */}
                        {customSelected.length > 0 && (
                            <div className="pt-2 mt-2 border-t border-border space-y-1">
                                <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/60 px-2">
                                    Custom
                                </p>
                                {customSelected.map(ticker => (
                                    <div
                                        key={ticker}
                                        className="flex items-center justify-between px-2 py-1.5 rounded bg-[#0097b2]/15 text-foreground"
                                    >
                                        <span className="text-sm font-mono font-semibold">{ticker}</span>
                                        <button
                                            onClick={() => handleToggle(ticker)}
                                            className="p-0.5 rounded hover:bg-background/60 transition-colors"
                                            aria-label={`Remove ${ticker}`}
                                        >
                                            <X className="w-3 h-3" />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Custom ticker input */}
                    <div className="border-t border-border p-2">
                        <div className="flex items-center gap-1.5">
                            <input
                                type="text"
                                value={customInput}
                                onChange={e => setCustomInput(e.target.value)}
                                onKeyDown={e => {
                                    if (e.key === 'Enter') {
                                        e.preventDefault();
                                        handleAddCustom();
                                    }
                                }}
                                placeholder="Custom ticker (e.g., VTI)"
                                className="flex-1 text-xs px-2 py-1.5 bg-secondary text-foreground rounded border border-transparent focus:border-[#0097b2] focus:outline-none placeholder:text-muted-foreground/60 font-mono uppercase"
                                spellCheck={false}
                                autoComplete="off"
                            />
                            <button
                                onClick={handleAddCustom}
                                disabled={!customInput.trim()}
                                className="p-1.5 rounded bg-[#0097b2] text-white hover:bg-[#0097b2]/85 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                                aria-label="Add custom ticker"
                            >
                                <Plus className="w-3.5 h-3.5" />
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
