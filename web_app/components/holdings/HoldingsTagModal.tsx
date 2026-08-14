import React from 'react';
import { Tag, X, Save } from 'lucide-react';

interface HoldingsTagModalProps {
    editingTags: { symbol: string; account: string; currentTags: string } | null;
    setEditingTags: (val: { symbol: string; account: string; currentTags: string } | null) => void;
    tagsInput: string;
    setTagsInput: (val: string) => void;
    handleSaveTags: () => void;
    isPending: boolean;
}

export const HoldingsTagModal: React.FC<HoldingsTagModalProps> = ({
    editingTags,
    setEditingTags,
    tagsInput,
    setTagsInput,
    handleSaveTags,
    isPending,
}) => {
    if (!editingTags) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
            <div className="bg-card rounded-lg shadow-none w-full max-w-sm p-4 space-y-4">
                <div className="flex justify-between items-center">
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                        <Tag className="w-4 h-4" />
                        Edit Tags
                    </h3>
                    <button onClick={() => setEditingTags(null)} className="text-muted-foreground hover:text-foreground">
                        <X className="w-4 h-4" />
                    </button>
                </div>
                <div className="space-y-2">
                    <div className="text-sm text-muted-foreground">
                        Tags for <span className="font-medium text-foreground">{editingTags.symbol}</span> ({editingTags.account})
                    </div>
                    <input
                        type="text"
                        value={tagsInput}
                        onChange={(e) => setTagsInput(e.target.value)}
                        placeholder="Enter tags separated by commas..."
                        className="w-full px-3 py-2 bg-secondary border-none rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
                        autoFocus
                    />
                    <p className="text-xs text-muted-foreground">
                        Separate multiple tags with commas (e.g. &quot;Long Term, High Risk&quot;).
                    </p>
                </div>
                <div className="flex justify-end gap-2">
                    <button
                        onClick={() => setEditingTags(null)}
                        className="px-3 py-1.5 text-sm bg-secondary text-foreground rounded hover:bg-accent/50 transition-colors"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSaveTags}
                        disabled={isPending}
                        className="px-3 py-1.5 text-sm bg-[#0097b2] text-white rounded hover:bg-[#0086a0] transition-colors flex items-center gap-2 disabled:opacity-50"
                    >
                        {isPending ? "Saving..." : <><Save className="w-3 h-3" /> Save</>}
                    </button>
                </div>
            </div>
        </div>
    );
};
