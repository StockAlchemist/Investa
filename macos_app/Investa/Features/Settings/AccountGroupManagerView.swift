import SwiftUI

/// Component managing custom account group creation, editing, ordering, and deletion.
/// Brings 100% feature parity with the web application's `AccountGroupManager.tsx`.
struct AccountGroupManagerView: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?
    let availableAccounts: [String]
    @ObservedObject var appState: AppState

    @State private var isCreating = false
    @State private var editingGroupName: String? = nil
    @State private var groupNameInput = ""
    @State private var selectedAccounts: Set<String> = []
    @State private var groupToDelete: String? = nil
    @State private var showingDeleteAlert = false

    private var configurableAccounts: [String] {
        availableAccounts.filter { $0 != "All Accounts" }
    }

    private var groupsMap: [String: [String]] {
        settings?.accountGroups ?? [:]
    }

    /// List of group items ordered according to `account_group_order`, appending any unlisted groups.
    private var orderedGroupNames: [String] {
        var order = (settings?.accountGroupOrder ?? groupsMap.keys.sorted())
            .filter { groupsMap[$0] != nil }
        order += groupsMap.keys.filter { !order.contains($0) }.sorted()
        return order
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            headerView

            if isCreating {
                groupFormView
                    .transition(.move(edge: .top).combined(with: .opacity))
            }

            groupListView
        }
        .animation(.easeInOut(duration: 0.25), value: isCreating)
        .alert("Delete Account Group", isPresented: $showingDeleteAlert, presenting: groupToDelete) { name in
            Button("Delete", role: .destructive) {
                deleteGroup(name)
            }
            Button("Cancel", role: .cancel) {}
        } message: { name in
            Text("Are you sure you want to delete the group \"\(name)\"?")
        }
    }

    // MARK: - Header

    private var headerView: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Image(systemName: "person.2.fill")
                        .foregroundStyle(Color.indigo)
                        .font(.title3)
                    Text("Custom Account Groups")
                        .font(.title3.bold())
                }
                Text("Create custom groups of accounts for quick filtering. Drag or use arrows to reorder.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if !isCreating {
                Button {
                    startCreating()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "plus")
                        Text("Create Group")
                    }
                    .font(.system(size: 13, weight: .semibold))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                }
                .buttonStyle(.borderedProminent)
                .tint(.indigo)
            }
        }
    }

    // MARK: - Create / Edit Form

    private var groupFormView: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Form Header
            HStack {
                HStack(spacing: 8) {
                    Image(systemName: editingGroupName != nil ? "pencil" : "plus.circle.fill")
                        .foregroundStyle(Color.indigo)
                        .font(.headline)
                    Text(editingGroupName != nil ? "Edit Group" : "New Account Group")
                        .font(.headline.bold())
                }

                Spacer()

                Button {
                    cancelForm()
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(.secondary)
                        .padding(6)
                        .background(Color.primary.opacity(0.06), in: Circle())
                }
                .buttonStyle(.plain)
            }
            .padding(.bottom, 4)

            Divider()

            // Group Name Input
            VStack(alignment: .leading, spacing: 6) {
                Text("GROUP NAME")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(.secondary)

                TextField("e.g. Retirement, Short Term", text: $groupNameInput)
                    .textFieldStyle(.roundedBorder)
                    .font(.body)
            }

            // Account Selection Section
            VStack(alignment: .leading, spacing: 8) {
                Text("SELECT ACCOUNTS")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(.secondary)

                if configurableAccounts.isEmpty {
                    Text("No configurable accounts available.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    let columns = [GridItem(.adaptive(minimum: 150), spacing: 8)]
                    LazyVGrid(columns: columns, alignment: .leading, spacing: 8) {
                        ForEach(configurableAccounts, id: \.self) { acc in
                            let isSelected = selectedAccounts.contains(acc)
                            Button {
                                toggleAccount(acc)
                            } label: {
                                HStack(spacing: 8) {
                                    Image(systemName: isSelected ? "checkmark.square.fill" : "square")
                                        .font(.system(size: 15, weight: .medium))
                                        .foregroundStyle(isSelected ? Color.indigo : Color.secondary)

                                    Text(acc)
                                        .font(.system(size: 13, weight: isSelected ? .semibold : .regular))
                                        .foregroundStyle(isSelected ? Color.indigo : Color.primary)
                                        .lineLimit(1)

                                    Spacer(minLength: 0)
                                }
                                .padding(.horizontal, 10)
                                .padding(.vertical, 8)
                                .background(
                                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                                        .fill(isSelected ? Color.indigo.opacity(0.12) : Color.primary.opacity(0.04))
                                )
                                .overlay(
                                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                                        .strokeBorder(isSelected ? Color.indigo.opacity(0.4) : Color.primary.opacity(0.08), lineWidth: 1)
                                )
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(12)
                    .background(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(Color.primary.opacity(0.02))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
                    )

                    HStack {
                        Text("\(selectedAccounts.count) account\(selectedAccounts.count == 1 ? "" : "s") selected")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(Color.primary.opacity(0.06), in: Capsule())
                        Spacer()
                    }
                }
            }

            Divider()

            // Form Action Buttons
            HStack {
                Spacer()

                Button("Cancel") {
                    cancelForm()
                }
                .buttonStyle(.bordered)

                Button(editingGroupName != nil ? "Update Group" : "Save Group") {
                    saveGroup()
                }
                .buttonStyle(.borderedProminent)
                .tint(.indigo)
                .disabled(groupNameInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || selectedAccounts.isEmpty)
            }
        }
        .padding(18)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.primary.opacity(0.03))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.indigo.opacity(0.3), lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.04), radius: 8, x: 0, y: 3)
    }

    // MARK: - Group List

    private var groupListView: some View {
        VStack(spacing: 10) {
            if orderedGroupNames.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "person.2")
                        .font(.system(size: 32))
                        .foregroundStyle(Color.secondary.opacity(0.6))
                    Text("No groups defined yet.")
                        .font(.body)
                        .foregroundStyle(.secondary)
                    Text("Create custom account groups to quickly filter dashboard metrics and views.")
                        .font(.caption)
                        .foregroundStyle(Color.secondary.opacity(0.8))
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 36)
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [5]))
                        .foregroundStyle(Color.primary.opacity(0.15))
                )
            } else {
                ForEach(Array(orderedGroupNames.enumerated()), id: \.element) { index, name in
                    groupCard(name: name, index: index)
                }
            }
        }
    }

    @ViewBuilder
    private func groupCard(name: String, index: Int) -> some View {
        let members = groupsMap[name] ?? []

        HStack(spacing: 12) {
            // Drag / Grip indicator & Reorder buttons
            HStack(spacing: 4) {
                Image(systemName: "line.3.horizontal")
                    .font(.system(size: 14))
                    .foregroundStyle(Color.secondary.opacity(0.7))
                    .frame(width: 18)

                VStack(spacing: 2) {
                    Button {
                        moveGroup(name, by: -1)
                    } label: {
                        Image(systemName: "chevron.up")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundStyle(index > 0 ? Color.primary : Color.secondary.opacity(0.3))
                    }
                    .buttonStyle(.plain)
                    .disabled(index == 0)

                    Button {
                        moveGroup(name, by: 1)
                    } label: {
                        Image(systemName: "chevron.down")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundStyle(index < orderedGroupNames.count - 1 ? Color.primary : Color.secondary.opacity(0.3))
                    }
                    .buttonStyle(.plain)
                    .disabled(index == orderedGroupNames.count - 1)
                }
            }

            // Group Info
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(name)
                        .font(.headline.bold())

                    Text("\(members.count)")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.secondary.opacity(0.15), in: Capsule())
                }

                if members.isEmpty {
                    Text("No accounts assigned")
                        .font(.caption)
                        .foregroundStyle(Color.secondary.opacity(0.7))
                        .italic()
                } else {
                    Text(members.joined(separator: ", "))
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }

            Spacer()

            // Actions: Edit and Delete
            HStack(spacing: 8) {
                Button {
                    startEditing(name: name, accounts: members)
                } label: {
                    Image(systemName: "pencil")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(Color.indigo)
                        .padding(8)
                        .background(Color.indigo.opacity(0.1), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
                .buttonStyle(.plain)
                .help("Edit Group")

                Button {
                    groupToDelete = name
                    showingDeleteAlert = true
                } label: {
                    Image(systemName: "trash")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(Color.red)
                        .padding(8)
                        .background(Color.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
                .buttonStyle(.plain)
                .help("Delete Group")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color.primary.opacity(0.03))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        )
    }

    // MARK: - Actions & State Handlers

    private func startCreating() {
        editingGroupName = nil
        groupNameInput = ""
        selectedAccounts = []
        isCreating = true
    }

    private func startEditing(name: String, accounts: [String]) {
        editingGroupName = name
        groupNameInput = name
        selectedAccounts = Set(accounts)
        isCreating = true
    }

    private func cancelForm() {
        isCreating = false
        editingGroupName = nil
        groupNameInput = ""
        selectedAccounts = []
    }

    private func toggleAccount(_ account: String) {
        if selectedAccounts.contains(account) {
            selectedAccounts.remove(account)
        } else {
            selectedAccounts.insert(account)
        }
    }

    private func saveGroup() {
        let name = groupNameInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, !selectedAccounts.isEmpty else { return }

        var newGroups = groupsMap
        var newOrder = orderedGroupNames

        let accountsList = Array(selectedAccounts).sorted()

        if let oldName = editingGroupName {
            if oldName != name {
                newGroups.removeValue(forKey: oldName)
                if let idx = newOrder.firstIndex(of: oldName) {
                    newOrder[idx] = name
                } else {
                    newOrder.append(name)
                }
            }
            newGroups[name] = accountsList
        } else {
            newGroups[name] = accountsList
            if !newOrder.contains(name) {
                newOrder.append(name)
            }
        }

        save(groups: newGroups, order: newOrder)
        cancelForm()
    }

    private func deleteGroup(_ name: String) {
        var newGroups = groupsMap
        newGroups.removeValue(forKey: name)
        let newOrder = orderedGroupNames.filter { $0 != name }
        save(groups: newGroups, order: newOrder)
    }

    private func moveGroup(_ name: String, by offset: Int) {
        var order = orderedGroupNames
        guard let currentIndex = order.firstIndex(of: name) else { return }
        let targetIndex = currentIndex + offset
        guard order.indices.contains(targetIndex) else { return }

        order.swapAt(currentIndex, targetIndex)
        save(groups: groupsMap, order: order)
    }

    private func save(groups: [String: [String]], order: [String]) {
        Task {
            await vm.updateGroups(groups, order: order)
            await appState.loadSettings(initial: false)
        }
    }
}
