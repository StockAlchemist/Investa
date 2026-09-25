import SwiftUI

/// Watchlist membership for the stock on screen, and the calls that change it.
@MainActor
final class StockWatchlistModel: ObservableObject {
    @Published private(set) var watchlists: [WatchlistMeta] = []
    @Published private(set) var memberIDs: Set<Int> = []
    @Published private(set) var isBusy = false
    @Published var errorMessage: String?

    let symbol: String
    private let api: APIClient

    init(symbol: String, api: APIClient = .shared) {
        self.symbol = symbol.uppercased()
        self.api = api
    }

    var favorites: WatchlistMeta? { watchlists.first { $0.isFavorites == true } }
    var isFavorite: Bool { favorites.map { memberIDs.contains($0.id) } ?? false }

    /// Two cheap reads: the list catalogue and this symbol's membership. The
    /// items of each list are not fetched — `GET /watchlist` enriches every row
    /// with market data, which the stock window has no use for.
    func load() async {
        watchlists = (try? await api.get("/watchlists")) ?? watchlists
        if let membership: WatchlistMembership = try? await api.get(
            "/watchlists/membership/\(encoded(symbol))"
        ) {
            memberIDs = Set(membership.watchlistIDs)
        }
    }

    func toggle(_ list: WatchlistMeta) async {
        await setMember(!memberIDs.contains(list.id), listID: list.id)
    }

    /// Toggles Favorites, creating the list the first time it is used.
    func toggleFavorite() async {
        if let favorites {
            await toggle(favorites)
            return
        }
        isBusy = true
        defer { isBusy = false }
        do {
            let created: WatchlistMeta = try await api.send(method: "POST", path: "/watchlists/favorites")
            watchlists.insert(created, at: 0)
            await setMember(true, listID: created.id)
        } catch {
            errorMessage = describe(error, fallback: "Could not open Favorites.")
        }
    }

    /// Creates a list and puts the stock on it. Returns false if the server
    /// refused, with the reason in `errorMessage` (the Favorites name is reserved).
    @discardableResult
    func createList(named name: String) async -> Bool {
        struct Body: Encodable { let name: String }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        isBusy = true
        defer { isBusy = false }
        do {
            let created: WatchlistMeta = try await api.send(method: "POST", path: "/watchlists",
                                                            body: Body(name: trimmed))
            watchlists.append(created)
            await setMember(true, listID: created.id)
            return true
        } catch {
            errorMessage = describe(error, fallback: "Could not create the list.")
            return false
        }
    }

    /// Optimistic: the check or heart flips at once and flips back if the
    /// server refuses, so the control never sits in a state nobody saved.
    private func setMember(_ member: Bool, listID: Int) async {
        struct Body: Encodable { let symbol: String; let note: String; let watchlist_id: Int }
        let before = memberIDs
        if member { memberIDs.insert(listID) } else { memberIDs.remove(listID) }
        do {
            if member {
                let _: StatusResponse = try await api.send(
                    method: "POST", path: "/watchlist",
                    body: Body(symbol: symbol, note: "", watchlist_id: listID))
            } else {
                let _: StatusResponse = try await api.send(
                    method: "DELETE", path: "/watchlist/\(encoded(symbol))",
                    query: [URLQueryItem(name: "id", value: String(listID))])
            }
        } catch {
            memberIDs = before
            errorMessage = describe(error, fallback: "Could not update the watchlist.")
        }
    }

    private func encoded(_ symbol: String) -> String {
        symbol.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? symbol
    }

    private func describe(_ error: Error, fallback: String) -> String {
        (error as? APIError)?.errorDescription ?? fallback
    }
}

/// The favourite heart and the watchlist menu, for the stock window's header.
///
/// The heart is the one-tap path into the built-in Favorites list; the menu
/// covers every other list (a check marks the ones already holding the stock)
/// and can make a new list with the stock already on it.
struct StockWatchlistControls: View {
    @StateObject private var model: StockWatchlistModel
    @State private var showingNewList = false
    @State private var newListName = ""

    init(symbol: String) {
        _model = StateObject(wrappedValue: StockWatchlistModel(symbol: symbol))
    }

    var body: some View {
        HStack(spacing: 8) {
            favoriteButton
            listMenu
        }
        .lineLimit(1)
        .minimumScaleFactor(0.8)
        .task { await model.load() }
        .alert("New Watchlist", isPresented: $showingNewList) {
            TextField("List name", text: $newListName)
            Button("Cancel", role: .cancel) { newListName = "" }
            Button("Create & Add") {
                let name = newListName
                newListName = ""
                Task { await model.createList(named: name) }
            }
        } message: {
            Text("\(model.symbol) will be added to the new list.")
        }
        .alert("Watchlist", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.errorMessage ?? "")
        }
    }

    private var favoriteButton: some View {
        Button {
            Task { await model.toggleFavorite() }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: model.isFavorite ? "heart.fill" : "heart")
                if !isPhoneLayout {
                    Text(model.isFavorite ? "Favorited" : "Favorite")
                }
            }
            .appFont(.system(size: 13, weight: .semibold))
            .foregroundStyle(model.isFavorite ? Color.brand : Color.primary)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(model.isFavorite ? Color.brand.opacity(0.12) : Color.cardBorder.opacity(0.2),
                        in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(model.isBusy)
        .accessibilityLabel(model.isFavorite ? "Remove \(model.symbol) from Favorites"
                                             : "Add \(model.symbol) to Favorites")
        .accessibilityAddTraits(model.isFavorite ? .isSelected : [])
    }

    private var listMenu: some View {
        Menu {
            Section("Add \(model.symbol) to") {
                ForEach(model.watchlists) { list in
                    Button {
                        Task { await model.toggle(list) }
                    } label: {
                        if model.memberIDs.contains(list.id) {
                            Label(list.name, systemImage: "checkmark")
                        } else {
                            Text(list.name)
                        }
                    }
                }
            }
            Divider()
            Button {
                showingNewList = true
            } label: {
                Label("New List…", systemImage: "plus")
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "text.badge.plus")
                if !isPhoneLayout { Text("Watchlists") }
                if !model.memberIDs.isEmpty {
                    Text("\(model.memberIDs.count)")
                        .monospacedDigit()
                        .padding(.horizontal, 5)
                        .background(Color.inset, in: Capsule())
                }
            }
            .appFont(.system(size: 13, weight: .semibold))
            .foregroundStyle(.primary)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Color.cardBorder.opacity(0.2), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .accessibilityLabel("Watchlists for \(model.symbol)")
    }
}
