"use client";

import React, { useState, useEffect } from 'react';
import { Trash2, Info, HelpCircle, Loader2, Edit2 } from 'lucide-react';
import { updateSettings, Settings as SettingsType, fetchIntrinsicValue, IntrinsicValueResponse } from '../lib/api';
import { useQueryClient } from '@tanstack/react-query';
import { cn } from '../lib/utils';

interface ManualValuationSettingsProps {
    settings: SettingsType;
}

const PARAM_INFO = {
    dcf_discount_rate: {
        label: "Discount Rate (DCF)",
        description: "The rate used to discount future cash flows to their present value. High rate = lower valuation.",
        default: "Calculated WACC (~10%)",
        isPercent: true
    },
    dcf_growth_rate: {
        label: "Growth Rate (DCF)",
        description: "Expected annual growth of cash flows during the projection years.",
        default: "Historical CAGR",
        isPercent: true
    },
    dcf_terminal_growth: {
        label: "Terminal Growth (DCF)",
        description: "Long-term growth rate after the projection period (stable stage).",
        default: "2.0%",
        isPercent: true
    },
    dcf_projection_years: {
        label: "Projection Years (DCF)",
        description: "Number of years to forecast explicit free cash flows.",
        default: "5 Years",
        isPercent: false
    },
    dcf_fcf: {
        label: "Base Free Cash Flow",
        description: "The starting free cash flow value for DCF projections.",
        default: "Latest TTM FCF",
        isPercent: false
    },
    graham_eps: {
        label: "Graham EPS",
        description: "Earnings per share used as the base for the Graham Formula.",
        default: "TTM EPS",
        isPercent: false
    },
    graham_growth_rate: {
        label: "Graham Growth Rate",
        description: "Expected annual growth (g) used in Graham's Formula.",
        default: "Historical CAGR",
        isPercent: false
    },
    graham_bond_yield: {
        label: "Graham Bond Yield (Y)",
        description: "Current yield on high-quality bonds (proxy for risk-free rate).",
        default: "10Y Treasury (~4.5%)",
        isPercent: false
    }
};

export default function ManualValuationSettings({ settings }: ManualValuationSettingsProps) {
    const queryClient = useQueryClient();
    const [symbol, setSymbol] = useState('');
    const [formData, setFormData] = useState<Record<string, string>>({});
    const [liveDefaults, setLiveDefaults] = useState<Record<string, any>>({});
    const [isLoadingDefaults, setIsLoadingDefaults] = useState(false);

    useEffect(() => {
        if (!symbol || symbol.length < 1) {
            setLiveDefaults({});
            return;
        }

        const timer = setTimeout(async () => {
            setIsLoadingDefaults(true);
            try {
                const results = await fetchIntrinsicValue(symbol);
                if (results && results.models) {
                    const defaults: Record<string, any> = {};
                    const dcf = results.models.dcf?.parameters;
                    const graham = results.models.graham?.parameters;

                    if (dcf) {
                        defaults.dcf_discount_rate = dcf.discount_rate;
                        defaults.dcf_growth_rate = dcf.growth_rate;
                        defaults.dcf_terminal_growth = dcf.terminal_growth_rate;
                        defaults.dcf_projection_years = dcf.projection_years;
                        defaults.dcf_fcf = dcf.base_fcf;
                    }

                    if (graham) {
                        defaults.graham_eps = graham.eps;
                        defaults.graham_growth_rate = graham.growth_rate_pct;
                        defaults.graham_bond_yield = graham.bond_yield_proxy;
                    }
                    setLiveDefaults(defaults);
                }
            } catch (err) {
                console.error("Failed to fetch live defaults:", err);
            } finally {
                setIsLoadingDefaults(false);
            }
        }, 800);

        return () => clearTimeout(timer);
    }, [symbol]);

    const valuationOverrides = settings.valuation_overrides || {};

    // Load existing overrides when symbol matches
    useEffect(() => {
        const symUpper = symbol.toUpperCase();
        if (valuationOverrides[symUpper]) {
            const existing = valuationOverrides[symUpper];
            const newFormData: Record<string, string> = {};
            Object.entries(existing).forEach(([key, val]: [string, any]) => {
                const info = PARAM_INFO[key as keyof typeof PARAM_INFO];
                if (info) {
                    newFormData[key] = info.isPercent ? (val * 100).toFixed(2) : val.toString();
                }
            });
            setFormData(newFormData);
        } else {
            setFormData({});
        }
    }, [symbol, valuationOverrides]);

    const handleFillDefaults = () => {
        if (!liveDefaults) return;
        const newFormData: Record<string, string> = { ...formData };
        Object.entries(liveDefaults).forEach(([key, val]) => {
            const info = PARAM_INFO[key as keyof typeof PARAM_INFO];
            if (info) {
                newFormData[key] = info.isPercent ? (val * 100).toFixed(2) : val.toString();
            }
        });
        setFormData(newFormData);
    };

    const handleInputChange = (key: string, value: string) => {
        setFormData(prev => ({ ...prev, [key]: value }));
    };

    const handleAddOverride = async () => {
        if (!symbol) return;

        const currentOverrides = { ...valuationOverrides };
        const symUpper = symbol.toUpperCase();

        const newEntry: Record<string, any> = {};
        Object.entries(formData).forEach(([key, val]) => {
            if (val === '') return;
            const numVal = parseFloat(val);
            if (!isNaN(numVal)) {
                // If it's a percentage in UI, store as decimal in backend (except for graham_growth which is usually a whole number in formula)
                // Actually, let's keep it simple: DCF parameters are decimals, Graham growth is literal % number.
                const isPercentField = PARAM_INFO[key as keyof typeof PARAM_INFO]?.isPercent;
                newEntry[key] = isPercentField ? numVal / 100 : numVal;
            }
        });

        if (Object.keys(newEntry).length === 0) return;

        currentOverrides[symUpper] = newEntry;

        try {
            await updateSettings({ valuation_overrides: currentOverrides });
            await queryClient.invalidateQueries({ queryKey: ['settings'] });
            setSymbol('');
            setFormData({});
        } catch (err) {
            alert("Failed to save valuation override");
        }
    };

    const handleRemoveOverride = async (sym: string) => {
        const currentOverrides = { ...valuationOverrides };
        delete currentOverrides[sym];

        try {
            await updateSettings({ valuation_overrides: currentOverrides });
            await queryClient.invalidateQueries({ queryKey: ['settings'] });
        } catch (err) {
            alert("Failed to remove override");
        }
    };

    const inputClassName = "w-full rounded-md border border-border bg-secondary text-foreground shadow-sm focus:border-cyan-500 focus:ring-cyan-500 px-3 py-2 text-sm outline-none focus:ring-1";
    const labelClassName = "flex items-center gap-1 text-[10px] font-bold text-muted-foreground mb-1 uppercase tracking-wider";

    return (
        <div className="space-y-6">
            <div className="bg-secondary/30 p-6 rounded-xl border border-dotted border-border">
                <h3 className="text-sm font-semibold mb-4">Add Custom Valuation Parameters</h3>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="md:col-span-4 max-w-xs">
                        <label className={labelClassName}>Symbol</label>
                        <div className="relative">
                            <input
                                type="text"
                                value={symbol}
                                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                                placeholder="e.g. AAPL"
                                className={cn(inputClassName, isLoadingDefaults && "pr-10")}
                            />
                            {isLoadingDefaults && (
                                <div className="absolute right-3 top-1/2 -translate-y-1/2">
                                    <Loader2 className="w-4 h-4 animate-spin text-cyan-500" />
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="md:col-span-4 flex items-center justify-between bg-cyan-500/5 p-3 rounded-lg border border-cyan-500/10">
                        <div className="text-[11px] text-cyan-600 dark:text-cyan-400 font-medium flex items-center gap-2">
                            <Info className="w-4 h-4" />
                            <span>
                                {valuationOverrides[symbol]
                                    ? `Editing existing overrides for ${symbol}`
                                    : Object.keys(liveDefaults).length > 0
                                        ? `Fetched live parameters for ${symbol}. You can edit individual fields or load all defaults.`
                                        : "Enter a symbol to fetch current live parameters."}
                            </span>
                        </div>
                        {Object.keys(liveDefaults).length > 0 && (
                            <button
                                onClick={handleFillDefaults}
                                className="text-[10px] bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-600 dark:text-cyan-400 font-bold px-3 py-1.5 rounded uppercase tracking-wider transition-colors border border-cyan-500/20"
                            >
                                Pre-fill from Defaults
                            </button>
                        )}
                    </div>

                    <div className="md:col-span-4 space-y-8">
                        {/* DCF Section */}
                        <div>
                            <h4 className="text-xs font-bold text-cyan-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                                <div className="h-px flex-1 bg-cyan-500/20"></div>
                                DCF Model Parameters
                                <div className="h-px flex-1 bg-cyan-500/20"></div>
                            </h4>
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                                {Object.entries(PARAM_INFO).filter(([key]) => key.startsWith('dcf')).map(([key, info]) => (
                                    <div key={key} className="space-y-1">
                                        <label className={labelClassName}>
                                            {info.label}
                                            <div className="group relative">
                                                <HelpCircle className="w-3.5 h-3.5 cursor-help text-muted-foreground/50 hover:text-cyan-500 transition-colors" />
                                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 p-3 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[11px] rounded-lg shadow-2xl border border-slate-200 dark:border-white/10 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-relaxed">
                                                    {info.description}
                                                    <div className="mt-2 pt-2 border-t border-slate-100 dark:border-white/10 font-bold text-cyan-600 dark:text-cyan-400">Default: {info.default}</div>
                                                </div>
                                            </div>
                                        </label>
                                        <div className="relative">
                                            <input
                                                type="number"
                                                step="0.01"
                                                value={formData[key] || ''}
                                                onChange={(e) => handleInputChange(key, e.target.value)}
                                                placeholder={
                                                    liveDefaults[key] !== undefined
                                                        ? (info.isPercent ? `${(liveDefaults[key] * 100).toFixed(2)}%` : liveDefaults[key].toLocaleString())
                                                        : info.default
                                                }
                                                className={cn(
                                                    inputClassName,
                                                    "h-10",
                                                    info.isPercent && "pr-8",
                                                    liveDefaults[key] !== undefined && "placeholder:text-cyan-500/50"
                                                )}
                                            />
                                            {info.isPercent && (
                                                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground text-xs">%</span>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Graham Section */}
                        <div>
                            <h4 className="text-xs font-bold text-amber-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                                <div className="h-px flex-1 bg-amber-500/20"></div>
                                Graham's Formula Parameters
                                <div className="h-px flex-1 bg-amber-500/20"></div>
                            </h4>
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                                {Object.entries(PARAM_INFO).filter(([key]) => key.startsWith('graham')).map(([key, info]) => (
                                    <div key={key} className="space-y-1">
                                        <label className={labelClassName}>
                                            {info.label}
                                            <div className="group relative">
                                                <HelpCircle className="w-3.5 h-3.5 cursor-help text-muted-foreground/50 hover:text-cyan-500 transition-colors" />
                                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 p-3 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[11px] rounded-lg shadow-2xl border border-slate-200 dark:border-white/10 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-relaxed">
                                                    {info.description}
                                                    <div className="mt-2 pt-2 border-t border-slate-100 dark:border-white/10 font-bold text-cyan-600 dark:text-cyan-400">Default: {info.default}</div>
                                                </div>
                                            </div>
                                        </label>
                                        <div className="relative">
                                            <input
                                                type="number"
                                                step="0.01"
                                                value={formData[key] || ''}
                                                onChange={(e) => handleInputChange(key, e.target.value)}
                                                placeholder={
                                                    liveDefaults[key] !== undefined
                                                        ? (info.isPercent ? `${(liveDefaults[key] * 100).toFixed(2)}%` : liveDefaults[key].toLocaleString())
                                                        : info.default
                                                }
                                                className={cn(
                                                    inputClassName,
                                                    "h-10",
                                                    info.isPercent && "pr-8",
                                                    liveDefaults[key] !== undefined && "placeholder:text-cyan-500/50"
                                                )}
                                            />
                                            {info.isPercent && (
                                                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground text-xs">%</span>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
                <div className="flex justify-end mt-6">
                    <button
                        onClick={handleAddOverride}
                        disabled={!symbol || Object.keys(formData).length === 0}
                        className="px-6 py-2 bg-cyan-500 text-white rounded-md text-sm font-medium hover:bg-cyan-600 disabled:opacity-50 transition-colors"
                    >
                        Save Valuation Parameters
                    </button>
                </div>
            </div>

            <div className="overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b border-border">
                            <th className="text-left py-3 px-4 font-semibold text-muted-foreground text-xs uppercase tracking-wider">Symbol</th>
                            <th className="text-left py-3 px-4 font-semibold text-muted-foreground text-xs uppercase tracking-wider">Parameters</th>
                            <th className="text-right py-3 px-4 font-semibold text-muted-foreground text-xs uppercase tracking-wider">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                        {Object.entries(valuationOverrides).length === 0 ? (
                            <tr>
                                <td colSpan={3} className="py-8 text-center text-muted-foreground italic">No manual valuation parameters set.</td>
                            </tr>
                        ) : (
                            Object.entries(valuationOverrides).map(([sym, data]: [string, any]) => (
                                <tr key={sym} className="group hover:bg-secondary/10 transition-colors">
                                    <td className="py-4 px-4 font-bold text-cyan-500">{sym}</td>
                                    <td className="py-4 px-4">
                                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                                            {Object.entries(data).map(([key, val]: [string, any]) => {
                                                const info = PARAM_INFO[key as keyof typeof PARAM_INFO];
                                                if (!info) return null;
                                                return (
                                                    <div key={key} className="bg-secondary/40 p-2 rounded border border-border/30">
                                                        <div className="text-[9px] text-muted-foreground uppercase font-bold truncate">{info.label}</div>
                                                        <div className="text-xs font-mono">
                                                            {info.isPercent ? `${(val * 100).toFixed(2)}%` : val.toLocaleString()}
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </td>
                                    <td className="py-4 px-4 text-right">
                                        <div className="flex justify-end gap-1">
                                            <button
                                                onClick={() => {
                                                    setSymbol(sym);
                                                    window.scrollTo({ top: 0, behavior: 'smooth' });
                                                }}
                                                className="p-2 text-muted-foreground hover:text-cyan-500 hover:bg-cyan-500/10 rounded transition-colors"
                                                title="Edit parameters"
                                            >
                                                <Edit2 className="w-4 h-4" />
                                            </button>
                                            <button
                                                onClick={() => handleRemoveOverride(sym)}
                                                className="p-2 text-muted-foreground hover:text-red-500 hover:bg-red-500/10 rounded transition-colors"
                                                title="Delete override"
                                            >
                                                <Trash2 className="w-4 h-4" />
                                            </button>
                                        </div>
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
