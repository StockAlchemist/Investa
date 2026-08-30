import React, { useState, useMemo } from 'react';
import { Save, Plus, Pencil, Trash2 } from 'lucide-react';
import { Settings as SettingsType, Holding, ManualOverride, ManualOverrideData, updateSettings } from '../../../lib/api';
import { useQueryClient } from '@tanstack/react-query';
import { useAuth } from '../../../context/AuthContext';
import { COUNTRIES, ALL_INDUSTRIES } from '../../../lib/constants';
import {
    ASSET_TYPES,
    SECTORS,
    cardClassName,
    cardHeadClassName,
    sectionTitleClassName,
    countBadgeClassName,
    secondaryButtonClassName,
    labelClassName,
    inputClassName,
    primaryButtonClassName
} from '../constants';

interface OverridesTabProps {
    settings: SettingsType | null;
    holdings: Holding[];
}

export const OverridesTab: React.FC<OverridesTabProps> = ({ settings, holdings }) => {
    const queryClient = useQueryClient();
    const { user } = useAuth();

    const [isEditingOverrides, setIsEditingOverrides] = useState(false);
    const [overrideSymbol, setOverrideSymbol] = useState('');
    const [overridePrice, setOverridePrice] = useState('');
    const [overrideAssetType, setOverrideAssetType] = useState('');
    const [overrideSector, setOverrideSector] = useState('');
    const [overrideGeo, setOverrideGeo] = useState('');
    const [overrideIndustry, setOverrideIndustry] = useState('');
    const [overrideExchange, setOverrideExchange] = useState('');

    const overrides = settings?.manual_overrides || {};
    const availableCountries = COUNTRIES;

    const portfolioCountries = useMemo(() => {
        const usedCountries = new Set<string>();
        holdings.forEach(h => {
            if (h.Country && h.Country !== 'N/A') {
                usedCountries.add(h.Country);
            }
        });
        return Array.from(usedCountries).sort();
    }, [holdings]);

    const handleEdit = (symbol: string, data: ManualOverride) => {
        setOverrideSymbol(symbol);
        if (typeof data === 'object' && data !== null && 'price' in data) {
            const p = data.price;
            setOverridePrice(p !== undefined && p !== null && p !== 0 ? p.toString() : '');
            setOverrideAssetType(data.asset_type || '');
            setOverrideSector(data.sector || '');
            setOverrideGeo(data.geography || '');
            setOverrideIndustry(data.industry || '');
            setOverrideExchange(data.exchange || '');
        } else {
            const p = typeof data === 'number' ? data : undefined;
            setOverridePrice(p !== undefined && p !== null && p !== 0 ? p.toString() : '');
            setOverrideAssetType('');
            setOverrideSector('');
            setOverrideGeo('');
            setOverrideIndustry('');
            setOverrideExchange('');
        }
        setIsEditingOverrides(true);
    };

    const addOverride = async () => {
        if (!overrideSymbol) return;
        try {
            const currentOverrides = settings?.manual_overrides || {};
            const newOverrides: Record<string, ManualOverride> = { ...currentOverrides };
            // A blank price omits the key rather than writing 0: an override may
            // carry metadata only, for a holding that is priced from its own
            // transactions or a published NAV. Writing 0 works too — every
            // backend fallback treats it as absent — but it renders as a real
            // field the next reader has to interpret. The macOS client already
            // omits it, so the two round-trip to the same file.
            const parsed = overridePrice !== '' ? parseFloat(overridePrice) : undefined;
            const priceNum = parsed !== undefined && !isNaN(parsed) && parsed > 0 ? parsed : undefined;
            const newData: ManualOverrideData = {
                price: priceNum,
                asset_type: overrideAssetType || undefined,
                sector: overrideSector || undefined,
                geography: overrideGeo || undefined,
                industry: overrideIndustry || undefined,
                exchange: overrideExchange || undefined,
            };

            newOverrides[overrideSymbol.toUpperCase()] = newData;

            const cleanedOverrides: Record<string, ManualOverride> = {};
            Object.entries(newOverrides).forEach(([k, v]) => {
                if (typeof v === 'number') {
                    if (v > 0) cleanedOverrides[k] = v;
                } else {
                    const rest = { ...v };
                    if ('currency' in rest) delete (rest as Record<string, unknown>).currency;
                    if (rest.price === undefined || rest.price === null || rest.price === 0) delete rest.price;
                    if (rest.asset_type === undefined) delete rest.asset_type;
                    if (rest.sector === undefined) delete rest.sector;
                    if (rest.geography === undefined) delete rest.geography;
                    if (rest.industry === undefined) delete rest.industry;
                    if (rest.exchange === undefined) delete rest.exchange;
                    cleanedOverrides[k] = rest;
                }
            });

            await updateSettings({ manual_price_overrides: cleanedOverrides });
            queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
            queryClient.invalidateQueries({ queryKey: ['holdings'] });
            queryClient.invalidateQueries({ queryKey: ['summary'] });

            setOverrideSymbol('');
            setOverridePrice('');
            setOverrideAssetType('');
            setOverrideSector('');
            setOverrideGeo('');
            setOverrideIndustry('');
            setOverrideExchange('');
            setIsEditingOverrides(false);
        } catch {
            alert("Failed to save override");
        }
    };

    const removeOverride = async (symbol: string) => {
        try {
            const currentOverrides = settings?.manual_overrides || {};
            const cleanedOverrides: Record<string, ManualOverride> = {};

            Object.entries(currentOverrides).forEach(([k, v]) => {
                if (k !== symbol) {
                    if (typeof v === 'number') {
                        cleanedOverrides[k] = v;
                    } else {
                        const rest = { ...v };
                        if ('currency' in rest) delete (rest as Record<string, unknown>).currency;
                        cleanedOverrides[k] = rest;
                    }
                }
            });

            await updateSettings({ manual_price_overrides: cleanedOverrides });
            queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
            queryClient.invalidateQueries({ queryKey: ['holdings'] });
            queryClient.invalidateQueries({ queryKey: ['summary'] });
        } catch {
            alert("Failed to delete override");
        }
    };

    return (
        <div className="space-y-6">
            {isEditingOverrides ? (
                <div className={cardClassName}>
                    <div className={cardHeadClassName}>
                        <h3 className={sectionTitleClassName}>{overrideSymbol ? 'Edit Override' : 'Add Override'}</h3>
                    </div>
                    <p className="text-xs text-muted-foreground mb-4">Set a manual price, asset type, or any metadata field for a symbol.</p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
                        <div className="space-y-1.5">
                            <label className={labelClassName}>Symbol</label>
                            <input
                                type="text"
                                value={overrideSymbol}
                                onChange={(e) => setOverrideSymbol(e.target.value.toUpperCase())}
                                placeholder="AAPL"
                                className={inputClassName}
                                disabled={overrideSymbol !== '' && overrides.hasOwnProperty(overrideSymbol)}
                            />
                        </div>
                        <div className="space-y-1.5">
                            <label className={labelClassName}>Price</label>
                            <input
                                type="number"
                                step="0.0001"
                                value={overridePrice}
                                onChange={(e) => setOverridePrice(e.target.value)}
                                placeholder="0.00"
                                className={inputClassName}
                            />
                        </div>
                        <div className="space-y-1.5">
                            <label className={labelClassName}>Asset Type</label>
                            <select
                                aria-label="Asset Type"
                                value={overrideAssetType}
                                onChange={(e) => setOverrideAssetType(e.target.value)}
                                className={inputClassName}
                            >
                                {ASSET_TYPES.map(t => <option key={t} value={t} className="bg-background text-foreground">{t || "Select..."}</option>)}
                            </select>
                        </div>
                        <div className="space-y-1.5">
                            <label className={labelClassName}>Sector</label>
                            <select
                                aria-label="Sector"
                                value={overrideSector}
                                onChange={(e) => setOverrideSector(e.target.value)}
                                className={inputClassName}
                            >
                                {SECTORS.map(s => <option key={s} value={s} className="bg-background text-foreground">{s || "Select..."}</option>)}
                            </select>
                        </div>
                        <div className="space-y-1.5">
                            <label className={labelClassName}>Country</label>
                            <select
                                aria-label="Country"
                                value={overrideGeo}
                                onChange={(e) => setOverrideGeo(e.target.value)}
                                className={inputClassName}
                            >
                                <option value="" className="bg-background text-foreground">Select...</option>
                                {portfolioCountries.length > 0 && (
                                    <optgroup label="In Portfolio" className="bg-muted text-foreground">
                                        {portfolioCountries.map(c => <option key={c} value={c} className="bg-background">{c}</option>)}
                                    </optgroup>
                                )}
                                <optgroup label="All Countries" className="bg-muted text-foreground">
                                    {availableCountries.map(c => <option key={c} value={c} className="bg-background">{c}</option>)}
                                </optgroup>
                            </select>
                        </div>
                        <div className="space-y-1.5">
                            <label className={labelClassName}>Industry</label>
                            <select
                                aria-label="Industry"
                                value={overrideIndustry}
                                onChange={(e) => setOverrideIndustry(e.target.value)}
                                className={inputClassName}
                            >
                                <option value="" className="bg-background text-foreground">Select...</option>
                                {ALL_INDUSTRIES.map(i => <option key={i} value={i} className="bg-background text-foreground">{i}</option>)}
                            </select>
                        </div>
                        <div className="space-y-1.5">
                            <label className={labelClassName}>Market</label>
                            <input
                                type="text"
                                value={overrideExchange}
                                onChange={(e) => setOverrideExchange(e.target.value)}
                                placeholder="NASDAQ"
                                className={inputClassName}
                            />
                        </div>
                    </div>
                    <div className="flex justify-between items-center mt-6">
                        <button
                            type="button"
                            onClick={() => {
                                setIsEditingOverrides(false);
                                setOverrideSymbol(''); setOverridePrice(''); setOverrideAssetType(''); setOverrideSector(''); setOverrideGeo(''); setOverrideIndustry(''); setOverrideExchange('');
                            }}
                            className={secondaryButtonClassName}
                        >
                            Cancel
                        </button>
                        <button
                            type="button"
                            onClick={addOverride}
                            disabled={!overrideSymbol || (!overridePrice && !overrideAssetType && !overrideSector && !overrideGeo && !overrideIndustry && !overrideExchange)}
                            className={primaryButtonClassName}
                        >
                            <Save className="w-4 h-4" />
                            Save Override
                        </button>
                    </div>
                </div>
            ) : (
                <div className="flex justify-end">
                    <button
                        type="button"
                        onClick={() => {
                            setOverrideSymbol(''); setOverridePrice(''); setOverrideAssetType(''); setOverrideSector(''); setOverrideGeo(''); setOverrideIndustry(''); setOverrideExchange('');
                            setIsEditingOverrides(true);
                        }}
                        className={primaryButtonClassName}
                    >
                        <Plus className="w-4 h-4" />
                        Add New Override
                    </button>
                </div>
            )}

            <div className={`${cardClassName} !p-0`}>
                <div className="flex items-center gap-2.5 px-6 py-4 border-b border-border">
                    <h3 className={sectionTitleClassName}>Active Overrides</h3>
                    <span className={countBadgeClassName}>{Object.entries(overrides).length}</span>
                </div>
                <div className="overflow-x-auto">
                    <table className="min-w-full text-sm">
                        <thead className="bg-muted/60 border-b border-border">
                            <tr>
                                <th className="sticky left-0 z-20 px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs bg-zinc-100 dark:bg-zinc-800 shadow-[1px_0_0_0_rgba(0,0,0,0.06)] dark:shadow-[1px_0_0_0_rgba(255,255,255,0.08)]">Symbol</th>
                                <th className="px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs">Price</th>
                                <th className="px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs">Asset Type</th>
                                <th className="px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs">Sector</th>
                                <th className="px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs">Country</th>
                                <th className="px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs">Industry</th>
                                <th className="px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs">Market</th>
                                <th className="px-6 py-3 text-right font-semibold text-muted-foreground uppercase tracking-wider text-xs">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                            {Object.entries(overrides).length === 0 ? (
                                <tr>
                                    <td colSpan={8} className="px-6 py-12 text-center text-muted-foreground">
                                        No manual overrides defined.
                                    </td>
                                </tr>
                            ) : (
                                Object.entries(overrides)
                                    .sort((a, b) => a[0].localeCompare(b[0]))
                                    .map(([symbol, data]) => {
                                        const isObj = typeof data !== 'number';
                                        const price = isObj ? (data as ManualOverrideData).price : (data as number);
                                        const assetType = isObj ? (data as ManualOverrideData).asset_type : '';
                                        const sector = isObj ? (data as ManualOverrideData).sector : '';
                                        const geo = isObj ? (data as ManualOverrideData).geography : '';
                                        const industry = isObj ? (data as ManualOverrideData).industry : '';
                                        const exchange = isObj ? (data as ManualOverrideData).exchange : '';
                                        const currency = isObj ? (data as ManualOverrideData).currency : 'USD';

                                        return (
                                            <tr key={symbol} className="hover:bg-muted/50 transition-colors group">
                                                <td className="sticky left-0 z-10 px-6 py-4 whitespace-nowrap font-bold text-foreground bg-white dark:bg-zinc-900 group-hover:bg-zinc-50 dark:group-hover:bg-zinc-800 transition-colors shadow-[1px_0_0_0_rgba(0,0,0,0.06)] dark:shadow-[1px_0_0_0_rgba(255,255,255,0.08)]">{symbol}</td>
                                                <td className="px-6 py-4 whitespace-nowrap text-muted-foreground font-mono">
                                                    {(!price || price === 0)
                                                        ? <span className="opacity-50">-</span>
                                                        : <span className="text-up font-medium">{currency === 'THB' ? '฿' : '$'}{price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}</span>
                                                    }
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    {assetType ? <span className="bg-black/5 dark:bg-white/5 px-2.5 py-1 rounded-md text-xs font-medium text-foreground">{assetType}</span> : <span className="opacity-50">-</span>}
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    {sector ? <span className="bg-black/5 dark:bg-white/5 px-2.5 py-1 rounded-md text-xs font-medium text-foreground">{sector}</span> : <span className="opacity-50">-</span>}
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    {geo ? <span className="bg-black/5 dark:bg-white/5 px-2.5 py-1 rounded-md text-xs font-medium text-foreground">{geo}</span> : <span className="opacity-50">-</span>}
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    {industry ? <span className="bg-black/5 dark:bg-white/5 px-2.5 py-1 rounded-md text-xs font-medium text-foreground">{industry}</span> : <span className="opacity-50">-</span>}
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    {exchange ? <span className="bg-black/5 dark:bg-white/5 px-2.5 py-1 rounded-md text-xs font-medium text-foreground">{exchange}</span> : <span className="opacity-50">-</span>}
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap text-right">
                                                    <div className="flex justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                                        <button
                                                            type="button"
                                                            onClick={() => handleEdit(symbol, data)}
                                                            className="p-2 text-cyan-500 hover:bg-cyan-500/10 rounded-lg transition-colors cursor-pointer"
                                                            title="Edit override"
                                                        >
                                                            <Pencil className="w-4 h-4" />
                                                        </button>
                                                        <button
                                                            type="button"
                                                            onClick={() => removeOverride(symbol)}
                                                            className="p-2 text-down hover:bg-down/12 rounded-lg transition-colors cursor-pointer"
                                                            title="Delete override"
                                                        >
                                                            <Trash2 className="w-4 h-4" />
                                                        </button>
                                                    </div>
                                                </td>
                                            </tr>
                                        );
                                    })
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};
