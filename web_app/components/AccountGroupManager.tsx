import React, { useState, useEffect } from 'react';
import { Trash2, Loader2, Pencil, X, GripVertical } from 'lucide-react';
import { updateSettings, Settings, fetchSettings } from '../lib/api';
import { useQueryClient } from '@tanstack/react-query';
import { useAuth } from '../context/AuthContext';
import {
    DndContext,
    closestCenter,
    KeyboardSensor,
    PointerSensor,
    useSensor,
    useSensors,
    DragEndEvent
} from '@dnd-kit/core';
import {
    arrayMove,
    SortableContext,
    sortableKeyboardCoordinates,
    verticalListSortingStrategy,
    useSortable
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

interface AccountGroupManagerProps {
    availableAccounts: string[];
    settings: Settings | null;
    onUpdate?: () => void;
}

interface SortableItemProps {
    id: string;
    name: string;
    accounts: string[];
    onEdit: (name: string, accounts: string[]) => void;
    onDelete: (name: string) => void;
}

function SortableGroupItem({ id, name, accounts, onEdit, onDelete }: SortableItemProps) {
    const {
        attributes,
        listeners,
        setNodeRef,
        transform,
        transition,
        isDragging
    } = useSortable({ id });

    const style = {
        transform: CSS.Transform.toString(transform),
        transition,
        zIndex: isDragging ? 10 : 1,
    };

    return (
        <div
            ref={setNodeRef}
            style={style}
            className={`flex items-center justify-between p-4 bg-card border border-border rounded-lg shadow-sm hover:border-cyan-500/50 transition-colors group ${isDragging ? 'opacity-50 border-cyan-500 ring-2 ring-cyan-500/20' : ''}`}
        >
            <div className="flex items-center gap-3 flex-1 overflow-hidden">
                <button
                    {...attributes}
                    {...listeners}
                    className="cursor-grab active:cursor-grabbing p-1 text-muted-foreground hover:text-foreground touch-none"
                    title="Drag to reorder"
                >
                    <GripVertical className="w-5 h-5" />
                </button>
                <div className="overflow-hidden">
                    <h4 className="font-medium text-foreground text-base truncate">{name}</h4>
                    <p className="text-sm text-muted-foreground mt-1 truncate">
                        {Array.isArray(accounts) ? accounts.join(", ") : ""}
                    </p>
                </div>
            </div>
            <div className="flex items-center gap-2 ml-4">
                <button
                    onClick={() => onEdit(name, accounts as string[])}
                    className="text-muted-foreground hover:text-cyan-500 hover:bg-cyan-500/10 p-2 rounded transition-colors"
                    title="Edit Group"
                >
                    <Pencil className="w-5 h-5" />
                </button>
                <button
                    onClick={() => handleDeleteGroup(name)}
                    className="text-muted-foreground hover:text-red-500 hover:bg-red-500/10 p-2 rounded transition-colors"
                    title="Delete Group"
                >
                    <Trash2 className="w-5 h-5" />
                </button>
            </div>
        </div>
    );

    // This is a bit of a hack since handleDeleteGroup is in the parent scope
    // We'll pass it down properly in the main component
    function handleDeleteGroup(name: string) {
        onDelete(name);
    }
}

export default function AccountGroupManager({ availableAccounts, settings, onUpdate }: AccountGroupManagerProps) {
    const queryClient = useQueryClient();
    const { user } = useAuth();
    const [newGroupName, setNewGroupName] = useState('');
    const [selectedAccounts, setSelectedAccounts] = useState<string[]>([]);
    const [isCreating, setIsCreating] = useState(false);
    const [editingGroupName, setEditingGroupName] = useState<string | null>(null);
    const [groupList, setGroupList] = useState<{name: string, accounts: string[]}[]>([]);

    const sensors = useSensors(
        useSensor(PointerSensor),
        useSensor(KeyboardSensor, {
            coordinateGetter: sortableKeyboardCoordinates,
        })
    );

    useEffect(() => {
        if (settings?.account_groups) {
            const rawGroups = Object.entries(settings.account_groups).map(([name, accounts]) => ({
                name,
                accounts: accounts as string[]
            }));
            
            // Apply order if available
            const order = settings.account_group_order || [];
            if (order.length > 0) {
                rawGroups.sort((a, b) => {
                    const idxA = order.indexOf(a.name);
                    const idxB = order.indexOf(b.name);
                    
                    // If one is not in the order list, put it at the end
                    if (idxA === -1 && idxB === -1) return 0;
                    if (idxA === -1) return 1;
                    if (idxB === -1) return -1;
                    
                    return idxA - idxB;
                });
            }
            
            setGroupList(rawGroups);
        }
    }, [settings?.account_groups, settings?.account_group_order]);

    const handleSaveOrder = async (newList: {name: string, accounts: string[]}[]) => {
        if (!settings) return;

        const order = newList.map(item => item.name);
        const newGroups: Record<string, string[]> = {};
        newList.forEach(item => {
            newGroups[item.name] = item.accounts;
        });

        try {
            await updateSettings({ 
                account_groups: newGroups,
                account_group_order: order
            });
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
            if (onUpdate) onUpdate();
        } catch (error) {
            console.error("Failed to update group order", error);
        }
    };

    const handleDragEnd = (event: DragEndEvent) => {
        const { active, over } = event;

        if (over && active.id !== over.id) {
            const oldIndex = groupList.findIndex(item => item.name === active.id);
            const newIndex = groupList.findIndex(item => item.name === over.id);

            const newList = arrayMove(groupList, oldIndex, newIndex);
            setGroupList(newList);
            handleSaveOrder(newList);
        }
    };

    const handleCreateGroup = async () => {
        if (!newGroupName || selectedAccounts.length === 0 || !settings) return;

        let newGroups: Record<string, string[]> = { ...settings.account_groups };
        let newOrder = [...(settings.account_group_order || groupList.map(g => g.name))];
        
        if (editingGroupName) {
            // Rename in groups object
            if (editingGroupName !== newGroupName) {
                delete newGroups[editingGroupName];
                // Update in order list
                const idx = newOrder.indexOf(editingGroupName);
                if (idx !== -1) {
                    newOrder[idx] = newGroupName;
                }
            }
            newGroups[newGroupName] = selectedAccounts;
        } else {
            // Add new
            newGroups[newGroupName] = selectedAccounts;
            if (!newOrder.includes(newGroupName)) {
                newOrder.push(newGroupName);
            }
        }

        try {
            await updateSettings({ 
                account_groups: newGroups,
                account_group_order: newOrder
            });
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
            setNewGroupName('');
            setSelectedAccounts([]);
            setIsCreating(false);
            setEditingGroupName(null);
            if (onUpdate) onUpdate();
        } catch (error) {
            console.error("Failed to save group", error);
            alert("Failed to save group");
        }
    };

    const handleDeleteGroup = async (groupName: string) => {
        if (!confirm(`Are you sure you want to delete the group "${groupName}"?`) || !settings) return;

        const newGroups = { ...settings.account_groups };
        delete newGroups[groupName];
        
        const newOrder = (settings.account_group_order || groupList.map(g => g.name)).filter(n => n !== groupName);

        try {
            await updateSettings({ 
                account_groups: newGroups,
                account_group_order: newOrder
            });
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
            if (onUpdate) onUpdate();
        } catch (error) {
            console.error("Failed to delete group", error);
            alert("Failed to delete group");
        }
    };

    const handleStartEdit = (name: string, accounts: string[]) => {
        setNewGroupName(name);
        setSelectedAccounts(accounts);
        setEditingGroupName(name);
        setIsCreating(true);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    const handleCancel = () => {
        setIsCreating(false);
        setEditingGroupName(null);
        setNewGroupName('');
        setSelectedAccounts([]);
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

    if (!settings) {
        return (
            <div className="flex justify-center p-8 text-muted-foreground italic">
                No settings data available.
            </div>
        );
    }

    return (
        <div>
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h3 className="text-lg font-medium text-foreground">Account Groups</h3>
                    <p className="text-sm text-muted-foreground">
                        Create custom groups of accounts for quick filtering. Drag to reorder.
                    </p>
                </div>
                {!isCreating && (
                    <button
                        onClick={() => {
                            setIsCreating(true);
                            setEditingGroupName(null);
                            setNewGroupName('');
                            setSelectedAccounts([]);
                        }}
                        className="px-4 py-2 bg-[#0097b2] text-white rounded-md hover:bg-[#0086a0] transition-colors text-sm font-medium shadow-sm"
                    >
                        Create New Group
                    </button>
                )}
            </div>

            {isCreating && (
                <div className="bg-secondary p-4 rounded-lg border border-border mb-6 shadow-md animate-in fade-in slide-in-from-top-4 duration-200">
                    <div className="flex justify-between items-center mb-4">
                        <h4 className="font-medium text-foreground">{editingGroupName ? 'Edit Group' : 'New Group'}</h4>
                        <button onClick={handleCancel} className="text-muted-foreground hover:text-foreground">
                            <X className="w-4 h-4" />
                        </button>
                    </div>

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
                            onClick={handleCancel}
                            className="px-4 py-2 border border-border text-foreground rounded-md hover:bg-accent/10 transition-colors text-sm font-medium"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleCreateGroup}
                            disabled={!newGroupName || selectedAccounts.length === 0}
                            className="px-4 py-2 bg-[#0097b2] text-white rounded-md hover:bg-[#0086a0] transition-colors text-sm font-medium shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {editingGroupName ? 'Update Group' : 'Save Group'}
                        </button>
                    </div>
                </div>
            )}

            <div className="grid gap-4">
                {groupList.length === 0 ? (
                    <div className="text-center p-8 text-muted-foreground italic border border-dashed border-border rounded-lg">
                        No groups defined yet.
                    </div>
                ) : (
                    <DndContext
                        sensors={sensors}
                        collisionDetection={closestCenter}
                        onDragEnd={handleDragEnd}
                    >
                        <SortableContext
                            items={groupList.map(g => g.name)}
                            strategy={verticalListSortingStrategy}
                        >
                            <div className="grid gap-3">
                                {groupList.map((group) => (
                                    <SortableGroupItem
                                        key={group.name}
                                        id={group.name}
                                        name={group.name}
                                        accounts={group.accounts}
                                        onEdit={handleStartEdit}
                                        onDelete={handleDeleteGroup}
                                    />
                                ))}
                            </div>
                        </SortableContext>
                    </DndContext>
                )}
            </div>
        </div>
    );
}
