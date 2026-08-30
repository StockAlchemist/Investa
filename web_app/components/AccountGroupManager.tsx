import React, { useState, useEffect } from 'react';
import { Trash2, Pencil, X, GripVertical, Plus, Users } from 'lucide-react';
import { updateSettings, Settings } from '../lib/api';
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
            className={`flex items-center justify-between card-inset p-4 hover:border-primary/30 transition-colors group ${isDragging ? 'opacity-80 border-primary ring-2 ring-ring/25' : ''}`}
        >
            <div className="flex items-center gap-4 flex-1 overflow-hidden">
                <button
                    {...attributes}
                    {...listeners}
                    className="cursor-grab active:cursor-grabbing p-1.5 text-muted-foreground/50 hover:text-foreground hover:bg-black/5 dark:hover:bg-white/5 rounded-lg touch-none transition-colors"
                    title="Drag to reorder"
                >
                    <GripVertical className="w-5 h-5" />
                </button>
                <div className="overflow-hidden">
                    <h4 className="font-bold text-foreground text-base truncate">{name}</h4>
                    <p className="text-sm text-muted-foreground mt-0.5 truncate font-mono">
                        {Array.isArray(accounts) ? accounts.join(", ") : ""}
                    </p>
                </div>
            </div>
            <div className="flex items-center gap-1.5 ml-4 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                    onClick={() => onEdit(name, accounts as string[])}
                    className="text-muted-foreground hover:text-primary hover:bg-primary/10 p-2.5 rounded-control transition-colors"
                    title="Edit Group"
                >
                    <Pencil className="w-4 h-4" />
                </button>
                <button
                    onClick={() => handleDeleteGroup(name)}
                    className="text-muted-foreground hover:text-down hover:bg-down/12 p-2.5 rounded-xl transition-colors"
                    title="Delete Group"
                >
                    <Trash2 className="w-4 h-4" />
                </button>
            </div>
        </div>
    );

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
            
            const order = settings.account_group_order || [];
            if (order.length > 0) {
                rawGroups.sort((a, b) => {
                    const idxA = order.indexOf(a.name);
                    const idxB = order.indexOf(b.name);
                    
                    if (idxA === -1 && idxB === -1) return 0;
                    if (idxA === -1) return 1;
                    if (idxB === -1) return -1;
                    
                    return idxA - idxB;
                });
            }
            
            // eslint-disable-next-line react-hooks/set-state-in-effect -- syncs the editable local list when the saved account groups/order change
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

        const newGroups: Record<string, string[]> = { ...settings.account_groups };
        const newOrder = [...(settings.account_group_order || groupList.map(g => g.name))];
        
        if (editingGroupName) {
            if (editingGroupName !== newGroupName) {
                delete newGroups[editingGroupName];
                const idx = newOrder.indexOf(editingGroupName);
                if (idx !== -1) {
                    newOrder[idx] = newGroupName;
                }
            }
            newGroups[newGroupName] = selectedAccounts;
        } else {
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

    const inputClassName = "w-full rounded-xl border border-black/10 dark:border-white/10 bg-white/50 dark:bg-black/20 backdrop-blur-sm text-foreground shadow-sm focus:border-indigo-500 focus:ring-indigo-500/50 px-4 py-2.5 text-sm outline-none focus:ring-2 transition-all hover:border-black/20 dark:hover:border-white/20";
    const labelClassName = "block text-[11px] font-bold text-muted-foreground mb-1.5 uppercase tracking-wider";

    if (!settings) {
        return (
            <div className="flex justify-center p-8 text-muted-foreground italic">
                No settings data available.
            </div>
        );
    }

    return (
        <div className="card-standard p-6 space-y-5">
            <div className="flex items-center gap-2.5">
                <h3 className="section-label">Account Groups</h3>
                <span className="text-[11px] font-bold tabular-nums text-primary-ink bg-primary/12 rounded-full px-2 py-0.5 shrink-0">
                    {groupList.length}
                </span>
                <span className="text-xs text-muted-foreground ml-auto hidden sm:block">Group accounts for aggregate filtering — drag to reorder</span>
                {!isCreating && (
                    <button
                        onClick={() => {
                            setIsCreating(true);
                            setEditingGroupName(null);
                            setNewGroupName('');
                            setSelectedAccounts([]);
                        }}
                        className="h-8 px-3 rounded-control bg-primary text-primary-foreground hover:bg-primary-hover text-xs font-semibold transition-colors inline-flex items-center gap-1.5 shrink-0 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ring-offset-background"
                    >
                        <Plus className="w-3.5 h-3.5" />
                        New group
                    </button>
                )}
            </div>

            {isCreating && (
                <div className="card-inset p-5 animate-in fade-in slide-in-from-top-2 duration-200">
                    <div className="flex justify-between items-center mb-5 border-b border-border pb-3">
                        <h4 className="text-sm font-semibold text-foreground flex items-center gap-2">
                            {editingGroupName ? <Pencil className="w-4 h-4 text-primary"/> : <Plus className="w-4 h-4 text-primary"/>}
                            {editingGroupName ? 'Edit Group' : 'New Account Group'}
                        </h4>
                        <button onClick={handleCancel} className="text-muted-foreground hover:text-foreground hover:bg-black/5 dark:hover:bg-white/5 p-2 rounded-xl transition-colors">
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    <div className="mb-6">
                        <label className={labelClassName}>Group Name</label>
                        <input
                            type="text"
                            value={newGroupName}
                            onChange={(e) => setNewGroupName(e.target.value)}
                            placeholder="e.g. Retirement, Short Term"
                            className={inputClassName}
                        />
                    </div>

                    <div className="mb-8">
                        <label className={labelClassName}>Select Accounts</label>
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 max-h-72 overflow-y-auto p-3 border border-border rounded-inset bg-background">
                            {availableAccounts.map(account => {
                                const isSelected = selectedAccounts.includes(account);
                                return (
                                    <label key={account} className={`flex items-center gap-3 h-11 px-3 rounded-control cursor-pointer transition-colors border ${isSelected ? 'bg-primary/12 border-primary/30' : 'border-border hover:bg-muted'}`}>
                                        <input
                                            type="checkbox"
                                            checked={isSelected}
                                            onChange={() => toggleAccountSelection(account)}
                                            className="w-4 h-4 rounded border-border text-primary focus:ring-ring"
                                        />
                                        <span className={`text-sm font-medium truncate ${isSelected ? 'text-primary-ink' : 'text-foreground'}`}>{account}</span>
                                    </label>
                                );
                            })}
                        </div>
                        <p className="text-xs text-muted-foreground mt-2 font-medium bg-muted inline-block px-3 py-1 rounded-full">
                            {selectedAccounts.length} account{selectedAccounts.length !== 1 && 's'} selected
                        </p>
                    </div>

                    <div className="flex justify-end gap-3 pt-4 border-t border-border">
                        <button
                            onClick={handleCancel}
                            className="h-9 px-3.5 rounded-control border border-border bg-background text-foreground hover:bg-muted transition-colors text-sm font-medium cursor-pointer"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleCreateGroup}
                            disabled={!newGroupName || selectedAccounts.length === 0}
                            className="h-9 px-3.5 rounded-control bg-primary text-primary-foreground hover:bg-primary-hover text-sm font-medium disabled:opacity-50 disabled:pointer-events-none transition-colors cursor-pointer"
                        >
                            {editingGroupName ? 'Update Group' : 'Save Group'}
                        </button>
                    </div>
                </div>
            )}

            <div className="grid gap-4">
                {groupList.length === 0 ? (
                    <div className="text-center p-12 text-sm text-muted-foreground border border-dashed border-border rounded-inset flex flex-col items-center justify-center gap-3">
                        <Users className="w-8 h-8 opacity-50" />
                        <p>No groups defined yet.</p>
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
