import SwiftUI

struct ProfileSecuritySettingsView: View {
    @ObservedObject var vm: SettingsViewModel
    @EnvironmentObject private var auth: AuthViewModel

    var embedded: Bool = false
    @State private var alias = ""
    @State private var showPassword = false
    @State private var confirmDelete = false
    @State private var isSavingAlias = false

    var body: some View {
        Group {
            if embedded {
                mainContent
            } else {
                ScrollView {
                    mainContent
                        .padding(16)
                }
                .navigationTitle("Profile & Security")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
            }
        }
        .onAppear {
            alias = auth.currentUser?.alias ?? ""
        }
        .onChange(of: auth.currentUser?.alias) { _, new in
            alias = new ?? alias
        }
        .sheet(isPresented: $showPassword) {
            ChangePasswordView()
                .environmentObject(auth)
        }
        .alert("Delete Account?", isPresented: $confirmDelete) {
            Button("Permanently Delete", role: .destructive) {
                Task {
                    await vm.deleteAccount()
                    auth.logout()
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This permanently deletes your account and all associated transactions, portfolios, and settings. This action cannot be undone.")
        }
    }

    private var mainContent: some View {
        VStack(spacing: 20) {
            // Profile Information Card
            profileCard

            // Security & Password Card
            securityCard

            // Session & Danger Zone
            sessionAndDangerSection
        }
    }

    private var profileCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Image(systemName: "person.crop.circle.fill")
                    .foregroundStyle(Color.cyan)
                    .appFont(.title3)
                Text("Profile Information")
                    .appFont(.headline.bold())
                Spacer()
            }

            if let user = auth.currentUser {
                VStack(spacing: 12) {
                    HStack {
                        Text("Username")
                            .appFont(.subheadline)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(user.username)
                            .appFont(.body.monospaced())
                            .fontWeight(.medium)
                    }

                    HStack {
                        Text("User ID")
                            .appFont(.subheadline)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text("\(user.id)")
                            .appFont(.body.monospaced())
                            .foregroundStyle(.secondary)
                    }

                    Divider()

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Display Name (Alias)")
                            .appFont(.caption2.bold())
                            .foregroundStyle(.secondary)

                        HStack(spacing: 8) {
                            TextField("e.g. My Portfolio", text: $alias)
                                .textFieldStyle(.roundedBorder)

                            Button("Save") {
                                saveAlias()
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.cyan)
                            .disabled(alias == user.alias || isSavingAlias)
                        }

                        Text("Shown across the user menu and reports. Leave empty to use username.")
                            .appFont(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .padding(18)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.primary.opacity(0.03))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        )
    }

    private var securityCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Image(systemName: "lock.shield.fill")
                    .foregroundStyle(Color.orange)
                    .appFont(.title3)
                Text("Login Security")
                    .appFont(.headline.bold())
                Spacer()
            }

            Text("Protect your portfolio data with a strong, updated login password.")
                .appFont(.caption)
                .foregroundStyle(.secondary)

            Button {
                showPassword = true
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "key.horizontal")
                    Text("Change Password...")
                }
                .frame(maxWidth: .infinity)
                .fontWeight(.semibold)
            }
            .buttonStyle(.bordered)
        }
        .padding(18)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.primary.opacity(0.03))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        )
    }

    private var sessionAndDangerSection: some View {
        VStack(spacing: 12) {
            // Sign Out
            Button(role: .destructive) {
                auth.logout()
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "rectangle.portrait.and.arrow.right")
                    Text("Sign Out of Device")
                }
                .frame(maxWidth: .infinity)
                .fontWeight(.semibold)
                .padding(.vertical, 4)
            }
            .buttonStyle(.bordered)

            // Danger Zone
            Button(role: .destructive) {
                confirmDelete = true
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "trash.fill")
                    Text("Delete Account & Data")
                }
                .frame(maxWidth: .infinity)
                .fontWeight(.semibold)
                .padding(.vertical, 4)
            }
            .buttonStyle(.borderedProminent)
            .tint(.red)
        }
        .padding(.top, 4)
    }

    private func saveAlias() {
        isSavingAlias = true
        Task {
            await vm.updateProfile(alias: alias)
            isSavingAlias = false
            ToastManager.shared.show(message: "Display name updated", style: .success)
        }
    }
}
