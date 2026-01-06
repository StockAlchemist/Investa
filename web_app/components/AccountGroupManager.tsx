import React, { useState } from 'react';
import { Settings, updateSettings } from '../lib/api';

interface AccountGroupManagerProps {
    settings: Settings;
    availableAccounts: string[];
    onUpdate: () => void;
}

export default function AccountGroupManager({ settings, availableAccounts, onUpdate }: AccountGroupManagerProps) {
    const [newGroupName, setNewGroupName] = useState('');
    const [selectedAccounts, setSelectedAccounts] = useState<string[]>([]);
    const [isCreating, setIsCreating] = useState(false);

    const groups = settings.account_groups || {};

    const handleCreateGroup = async () => {
        if (!newGroupName || selectedAccounts.length === 0) return;

        const newGroups = { ...groups, [newGroupName]: selectedAccounts };

        try {
            await updateSettings({ account_groups: newGroups });
            setNewGroupName('');
            setSelectedAccounts([]);
            setIsCreating(false);
            onUpdate();
        } catch (error) {
            console.error("Failed to create group", error);
            alert("Failed to create group");
        }
    };

    const handleDeleteGroup = async (groupName: string) => {
        if (!confirm(`Are you sure you want to delete the group "${groupName}"?`)) return;

        const newGroups = { ...groups };
        delete newGroups[groupName];

        try {
            await updateSettings({ account_groups: newGroups });
            onUpdate();
        } catch (error) {
            console.error("Failed to delete group", error);
            alert("Failed to delete group");
        }
    };

    const toggleAccountSelection = (account: string) => {
        if (selectedAccounts.includes(account)) {
            setSelectedAccounts(selectedAccounts.filter(a => a !== account));
        } else {
            setSelectedAccounts([...selectedAccounts, account]);
        }
    };

    const inputClassName = "w-full rounded-md border border-border bg-secondary text-foreground shadow-sm focus:border-cyan-500 focus:ring-cyan-500 px-3 py-2 text-sm outline-none focus:ring-1";
    const labelClassName = "block text-xs font-medium text-muted-foreground mb-1 uppercase tracking-wide";

    return (
        <div>
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h3 className="text-lg font-medium text-foreground">Account Groups</h3>
                    <p className="text-sm text-muted-foreground">
                        Create custom groups of accounts for quick filtering.
                    </p>
                </div>
                {!isCreating && (
                    <button
                        onClick={() => setIsCreating(true)}
                        className="px-4 py-2 bg-[#0097b2] text-white rounded-md hover:bg-[#0086a0] transition-colors text-sm font-medium shadow-sm"
                    >
                        Create New Group
                    </button>
                )}
            </div>

            {isCreating && (
                <div className="bg-secondary p-4 rounded-lg border border-border mb-6">
                    <h4 className="font-medium text-foreground mb-4">New Group</h4>

                    <div className="mb-4">
                        <label className={labelClassName}>Group Name</label>
                        <input
                            type="text"
                            value={newGroupName}
                            onChange={(e) => setNewGroupName(e.target.value)}
                            placeholder="e.g. Retirement, Short Term"
                            className={inputClassName}
                        />
                    </div>

                    <div className="mb-4">
                        <label className={labelClassName}>Select Accounts</label>
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 max-h-60 overflow-y-auto p-2 border border-border rounded-md bg-background/50">
                            {availableAccounts.map(account => (
                                <label key={account} className="flex items-center space-x-2 p-2 rounded hover:bg-accent/10 cursor-pointer">
                                    <input
                                        type="checkbox"
                                        checked={selectedAccounts.includes(account)}
                                        onChange={() => toggleAccountSelection(account)}
                                        className="rounded border-gray-300 text-cyan-600 focus:ring-cyan-500"
                                    />
                                    <span className="text-sm text-foreground">{account}</span>
                                </label>
                            ))}
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">Selected: {selectedAccounts.length}</p>
                    </div>

                    <div className="flex justify-end gap-2">
                        <button
                            onClick={() => setIsCreating(false)}
                            className="px-4 py-2 border border-border text-foreground rounded-md hover:bg-accent/10 transition-colors text-sm font-medium"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleCreateGroup}
                            disabled={!newGroupName || selectedAccounts.length === 0}
                            className="px-4 py-2 bg-[#0097b2] text-white rounded-md hover:bg-[#0086a0] transition-colors text-sm font-medium shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            Save Group
                        </button>
                    </div>
                </div>
            )}

            <div className="grid gap-4">
                {Object.entries(groups).length === 0 ? (
                    <div className="text-center p-8 text-muted-foreground italic border border-dashed border-border rounded-lg">
                        No groups defined yet.
                    </div>
                ) : (
                    Object.entries(groups).map(([name, accounts]) => (
                        <div key={name} className="flex items-center justify-between p-4 bg-card border border-border rounded-lg shadow-sm">
                            <div>
                                <h4 className="font-medium text-foreground text-base">{name}</h4>
                                <p className="text-sm text-muted-foreground mt-1">
                                    {accounts.join(", ")}
                                </p>
                            </div>
                            <button
                                onClick={() => handleDeleteGroup(name)}
                                className="text-muted-foreground hover:text-rose-500 hover:bg-rose-500/10 p-2 rounded-full transition-all"
                                title="Delete Group"
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                    <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                                </svg>
                            </button>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
