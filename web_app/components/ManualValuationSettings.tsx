"use client";

import React, { useState, useEffect, useMemo } from 'react';
import { Trash2, Info, HelpCircle, Loader2, Edit2, Save, Plus } from 'lucide-react';
import { updateSettings, Settings as SettingsType, fetchIntrinsicValue } from '../lib/api';
import { useQueryClient } from '@tanstack/react-query';
import { cn } from '../lib/utils';
import { useAuth } from '../context/AuthContext';

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
    target_fcf_margin: {
        label: "Target FCF Margin",
        description: "Override estimated margin for Revenue-based DCF.",
        default: "Historical Avg",
        isPercent: true
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
    const { user } = useAuth();
    const [symbol, setSymbol] = useState('');
    const [formData, setFormData] = useState<Record<string, string>>({});
    const [liveDefaults, setLiveDefaults] = useState<Record<string, number>>({});
    const [isLoadingDefaults, setIsLoadingDefaults] = useState(false);
    const [isEditing, setIsEditing] = useState(false);

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
                    const defaults: Record<string, number> = {};
                    const dcf = results.models.dcf?.parameters;
                    const graham = results.models.graham?.parameters;

                    if (dcf) {
                        defaults.dcf_discount_rate = dcf.discount_rate;
                        defaults.dcf_growth_rate = dcf.growth_rate;
                        defaults.dcf_terminal_growth = dcf.terminal_growth_rate;
                        defaults.dcf_projection_years = dcf.projection_years;
                        defaults.dcf_fcf = dcf.base_fcf;
                        defaults.target_fcf_margin = dcf.fcf_margin;
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

    const valuationOverrides = useMemo(() => settings.valuation_overrides || {}, [settings.valuation_overrides]);

    // Load existing overrides when symbol matches
    useEffect(() => {
        const symUpper = symbol.toUpperCase();
        if (valuationOverrides[symUpper]) {
            const existing = valuationOverrides[symUpper] as Record<string, number>;
            const newFormData: Record<string, string> = {};
            Object.entries(existing).forEach(([key, val]) => {
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

        const newEntry: Record<string, number> = {};
        Object.entries(formData).forEach(([key, val]) => {
            if (val === '') return;
            const numVal = parseFloat(val);
            if (!isNaN(numVal)) {
                const isPercentField = PARAM_INFO[key as keyof typeof PARAM_INFO]?.isPercent;
                newEntry[key] = isPercentField ? numVal / 100 : numVal;
            }
        });

        if (Object.keys(newEntry).length === 0) return;

        currentOverrides[symUpper] = newEntry;

        try {
            await updateSettings({ valuation_overrides: currentOverrides });
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
            setSymbol('');
            setFormData({});
        } catch {
            alert("Failed to save parameters");
        }
    };

    const handleCancel = () => {
        setIsEditing(false);
        setSymbol('');
        setFormData({});
        setLiveDefaults({});
    };

    const handleRemoveOverride = async (symbolToRemove: string) => {
        const currentOverrides = { ...valuationOverrides };
        delete currentOverrides[symbolToRemove];

        try {
            await updateSettings({ valuation_overrides: currentOverrides });
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
        } catch {
            alert("Failed to remove override");
        }
    };

    const inputClassName = "w-full rounded-xl border border-black/10 dark:border-white/10 bg-white/50 dark:bg-black/20 backdrop-blur-sm text-foreground shadow-sm focus:border-cyan-500 focus:ring-cyan-500/50 px-4 py-2.5 text-sm outline-none focus:ring-2 transition-all hover:border-black/20 dark:hover:border-white/20";
    const labelClassName = "flex items-center gap-1.5 text-[11px] font-bold text-muted-foreground mb-1.5 uppercase tracking-wider";
    const cardClassName = "bg-white/60 dark:bg-zinc-900/60 backdrop-blur-xl p-5 sm:p-8 rounded-3xl border border-white/40 dark:border-white/10 shadow-lg relative overflow-hidden";
    const primaryButtonClassName = "px-6 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-white rounded-xl font-medium shadow-md hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 focus:ring-offset-background disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2";

    return (
        <div className="space-y-8 max-w-6xl">
            {isEditing ? (
                <div className={cardClassName}>
                    <h3 className="text-xl font-bold mb-6 text-foreground flex items-center gap-2">
                        <Edit2 className="w-5 h-5 text-purple-500" />
                        {symbol && valuationOverrides[symbol.toUpperCase()] ? 'Edit Valuation' : 'Customize Valuation'}
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <div className="md:col-span-4 lg:col-span-1">
                        <label className={labelClassName}>Symbol</label>
                        <div className="relative">
                            <input
                                type="text"
                                value={symbol}
                                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                                placeholder="e.g. AAPL"
                                className={cn(inputClassName, isLoadingDefaults && "pr-10")}
                                disabled={symbol !== '' && valuationOverrides.hasOwnProperty(symbol.toUpperCase())}
                            />
                            {isLoadingDefaults && (
                                <div className="absolute right-3 top-1/2 -translate-y-1/2">
                                    <Loader2 className="w-5 h-5 animate-spin text-cyan-500" />
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="md:col-span-4 lg:col-span-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-cyan-500/10 dark:bg-cyan-500/5 backdrop-blur-sm p-4 rounded-xl border border-cyan-500/20 lg:self-end lg:min-h-[46px]">
                        <div className="text-sm text-cyan-700 dark:text-cyan-400 font-medium flex items-start sm:items-center gap-2 min-w-0">
                            <Info className="w-5 h-5 flex-shrink-0 mt-0.5 sm:mt-0" />
                            <span>
                                {valuationOverrides[symbol]
                                    ? `Editing existing overrides for ${symbol}`
                                    : Object.keys(liveDefaults).length > 0
                                        ? `Fetched live parameters for ${symbol}. Edit fields or load defaults.`
                                        : "Enter a symbol to fetch current live parameters."}
                            </span>
                        </div>
                        {Object.keys(liveDefaults).length > 0 && (
                            <button
                                onClick={handleFillDefaults}
                                className="text-xs bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-700 dark:text-cyan-400 font-bold px-4 py-2 rounded-lg uppercase tracking-wider transition-colors w-full sm:w-auto sm:ml-4 flex-shrink-0"
                            >
                                Pre-fill Defaults
                            </button>
                        )}
                    </div>

                    <div className="md:col-span-4 space-y-6 sm:space-y-10 mt-2 sm:mt-4">
                        {/* DCF Section */}
                        <div className="bg-white/40 dark:bg-black/20 p-4 sm:p-6 rounded-2xl border border-black/5 dark:border-white/5">
                            <h4 className="text-sm font-bold text-cyan-600 dark:text-cyan-400 uppercase tracking-widest mb-6 flex items-center gap-3">
                                <div className="w-2 h-2 rounded-full bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
                                Discounted Cash Flow (DCF)
                                <div className="h-px flex-1 bg-gradient-to-r from-cyan-500/20 to-transparent"></div>
                            </h4>
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                                {Object.entries(PARAM_INFO).filter(([key]) => key.startsWith('dcf') || key === 'target_fcf_margin').map(([key, info]) => (
                                    <div key={key} className="space-y-1.5">
                                        <label className={labelClassName}>
                                            {info.label}
                                            <div className="group relative">
                                                <HelpCircle className="w-4 h-4 cursor-help text-muted-foreground/50 hover:text-cyan-500 transition-colors" />
                                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-4 bg-white dark:bg-zinc-800 text-foreground text-xs rounded-xl shadow-xl border border-black/5 dark:border-white/10 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-relaxed">
                                                    {info.description}
                                                    <div className="mt-3 pt-3 border-t border-black/5 dark:border-white/10 font-bold text-cyan-600 dark:text-cyan-400">Default: {info.default}</div>
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
                                                        ? (info.isPercent ? `${(liveDefaults[key] * 100).toFixed(2)}` : liveDefaults[key].toLocaleString())
                                                        : info.default
                                                }
                                                className={cn(
                                                    inputClassName,
                                                    info.isPercent && "pr-8",
                                                    liveDefaults[key] !== undefined && "placeholder:text-cyan-500/40"
                                                )}
                                            />
                                            {info.isPercent && (
                                                <span className="absolute right-4 top-1/2 -translate-y-1/2 text-muted-foreground text-xs font-bold pointer-events-none">%</span>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Graham Section */}
                        <div className="bg-white/40 dark:bg-black/20 p-4 sm:p-6 rounded-2xl border border-black/5 dark:border-white/5">
                            <h4 className="text-sm font-bold text-amber-600 dark:text-amber-400 uppercase tracking-widest mb-6 flex items-center gap-3">
                                <div className="w-2 h-2 rounded-full bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.8)]" />
                                Graham&apos;s Formula
                                <div className="h-px flex-1 bg-gradient-to-r from-amber-500/20 to-transparent"></div>
                            </h4>
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                                {Object.entries(PARAM_INFO).filter(([key]) => key.startsWith('graham')).map(([key, info]) => (
                                    <div key={key} className="space-y-1.5">
                                        <label className={labelClassName}>
                                            {info.label}
                                            <div className="group relative">
                                                <HelpCircle className="w-4 h-4 cursor-help text-muted-foreground/50 hover:text-amber-500 transition-colors" />
                                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-4 bg-white dark:bg-zinc-800 text-foreground text-xs rounded-xl shadow-xl border border-black/5 dark:border-white/10 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-relaxed">
                                                    {info.description}
                                                    <div className="mt-3 pt-3 border-t border-black/5 dark:border-white/10 font-bold text-amber-600 dark:text-amber-400">Default: {info.default}</div>
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
                                                        ? (info.isPercent ? `${(liveDefaults[key] * 100).toFixed(2)}` : liveDefaults[key].toLocaleString())
                                                        : info.default
                                                }
                                                className={cn(
                                                    inputClassName,
                                                    info.isPercent && "pr-8",
                                                    liveDefaults[key] !== undefined && "placeholder:text-amber-500/40"
                                                )}
                                            />
                                            {info.isPercent && (
                                                <span className="absolute right-4 top-1/2 -translate-y-1/2 text-muted-foreground text-xs font-bold pointer-events-none">%</span>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                    </div>
                    <div className="flex justify-between items-center mt-8 pt-6 border-t border-black/5 dark:border-white/5">
                        <button
                            onClick={handleCancel}
                            className="px-6 py-2.5 bg-black/5 dark:bg-white/5 hover:bg-black/10 dark:hover:bg-white/10 text-foreground rounded-xl font-medium shadow-sm transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={() => { handleAddOverride(); setIsEditing(false); }}
                            disabled={!symbol || Object.keys(formData).length === 0}
                            className={primaryButtonClassName}
                        >
                            <Save className="w-5 h-5" />
                            Save Parameters
                        </button>
                    </div>
                </div>
            ) : (
                <div className="flex justify-end">
                    <button
                        type="button"
                        onClick={() => {
                            setSymbol(''); setFormData({}); setLiveDefaults({});
                            setIsEditing(true);
                        }}
                        className="px-6 py-2.5 bg-purple-500 hover:bg-purple-600 text-white rounded-xl font-medium shadow-sm transition-colors flex items-center gap-2"
                    >
                        <Plus className="w-4 h-4" />
                        Add Valuation Override
                    </button>
                </div>
            )}

            {Object.entries(valuationOverrides).length === 0 ? (
                <div className="bg-white/40 dark:bg-zinc-900/40 backdrop-blur-xl rounded-3xl border border-white/40 dark:border-white/10 shadow-lg py-16 px-6 text-center text-muted-foreground">
                    <div className="flex flex-col items-center justify-center gap-3">
                        <Edit2 className="w-8 h-8 opacity-30" />
                        <p>No manual valuation parameters set.</p>
                    </div>
                </div>
            ) : (
                <div className="space-y-4">
                    {Object.entries(valuationOverrides as Record<string, Record<string, number>>).map(([sym, data]) => (
                        <div key={sym} className="bg-white/40 dark:bg-zinc-900/40 backdrop-blur-xl rounded-3xl border border-white/40 dark:border-white/10 shadow-lg p-5 sm:p-6">
                            <div className="flex items-center justify-between gap-3 mb-5">
                                <span className="font-black text-xl sm:text-2xl text-purple-600 dark:text-purple-400">{sym}</span>
                                <div className="flex gap-2 flex-shrink-0">
                                    <button
                                        onClick={() => {
                                            setSymbol(sym);
                                            setIsEditing(true);
                                            window.scrollTo({ top: 0, behavior: 'smooth' });
                                        }}
                                        className="p-2.5 bg-white dark:bg-white/5 shadow-sm border border-black/5 dark:border-white/5 text-muted-foreground hover:text-purple-500 hover:border-purple-500/30 rounded-xl transition-all"
                                        title="Edit parameters"
                                    >
                                        <Edit2 className="w-5 h-5" />
                                    </button>
                                    <button
                                        onClick={() => handleRemoveOverride(sym)}
                                        className="p-2.5 bg-white dark:bg-white/5 shadow-sm border border-black/5 dark:border-white/5 text-muted-foreground hover:text-red-500 hover:border-red-500/30 rounded-xl transition-all"
                                        title="Delete override"
                                    >
                                        <Trash2 className="w-5 h-5" />
                                    </button>
                                </div>
                            </div>
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
                                {/* DCF Group */}
                                {Object.entries(data).some(([k]) => k.startsWith('dcf') || k === 'target_fcf_margin') && (
                                    <div className="bg-white/60 dark:bg-black/20 backdrop-blur-sm rounded-2xl p-4 sm:p-5 border border-cyan-500/20 shadow-sm relative overflow-hidden">
                                        <div className="absolute top-0 inset-x-0 h-1 bg-gradient-to-r from-cyan-400 to-blue-500 opacity-50" />
                                        <div className="text-xs font-bold text-cyan-600 dark:text-cyan-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                                            DCF Model
                                        </div>
                                        <div className="grid grid-cols-1 gap-x-6 gap-y-3">
                                            {Object.entries(data).filter(([k]) => k.startsWith('dcf') || k === 'target_fcf_margin').map(([key, val]) => {
                                                const info = PARAM_INFO[key as keyof typeof PARAM_INFO];
                                                if (!info) return null;

                                                let label = info.label.replace(' (DCF)', '');
                                                let displayVal = info.isPercent ? `${(val * 100).toFixed(2)}%` : val.toLocaleString();

                                                if (key === 'dcf_fcf') {
                                                    label = 'Base FCF';
                                                    displayVal = `${(val / 1_000_000).toLocaleString(undefined, { maximumFractionDigits: 1 })}M`;
                                                } else if (key === 'dcf_terminal_growth') {
                                                    label = 'Term. Growth';
                                                } else if (key === 'dcf_projection_years') {
                                                    label = 'Proj. Years';
                                                }

                                                return (
                                                    <div key={key} className="flex justify-between items-center gap-3 text-sm border-b border-black/5 dark:border-white/5 pb-2">
                                                        <span className="text-muted-foreground font-medium">{label}</span>
                                                        <span className="font-mono font-bold text-foreground bg-black/5 dark:bg-white/5 px-2 py-0.5 rounded">{displayVal}</span>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>
                                )}

                                {/* Graham Group */}
                                {Object.entries(data).some(([k]) => k.startsWith('graham')) && (
                                    <div className="bg-white/60 dark:bg-black/20 backdrop-blur-sm rounded-2xl p-4 sm:p-5 border border-amber-500/20 shadow-sm relative overflow-hidden">
                                        <div className="absolute top-0 inset-x-0 h-1 bg-gradient-to-r from-amber-400 to-orange-500 opacity-50" />
                                        <div className="text-xs font-bold text-amber-600 dark:text-amber-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                                            Graham&apos;s Formula
                                        </div>
                                        <div className="grid grid-cols-1 gap-x-6 gap-y-3">
                                            {Object.entries(data).filter(([k]) => k.startsWith('graham')).map(([key, val]) => {
                                                const info = PARAM_INFO[key as keyof typeof PARAM_INFO];
                                                if (!info) return null;

                                                let label = info.label.replace('Graham ', '').replace(' (Y)', '');
                                                let displayVal = info.isPercent ? `${(val * 100).toFixed(2)}%` : val.toLocaleString();

                                                if (key === 'graham_bond_yield') {
                                                    label = 'Bond Yield';
                                                    displayVal = `${val}%`;
                                                } else if (key === 'graham_growth_rate') {
                                                    displayVal = `${Number(val).toFixed(2)}%`;
                                                }

                                                return (
                                                    <div key={key} className="flex justify-between items-center gap-3 text-sm border-b border-black/5 dark:border-white/5 pb-2">
                                                        <span className="text-muted-foreground font-medium">{label}</span>
                                                        <span className="font-mono font-bold text-foreground bg-black/5 dark:bg-white/5 px-2 py-0.5 rounded">{displayVal}</span>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
