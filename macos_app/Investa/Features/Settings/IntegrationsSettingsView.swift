import SwiftUI

struct IntegrationsSettingsView: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?

    var embedded: Bool = false
    @State private var ibkrToken = ""
    @State private var ibkrQuery = ""
    @State private var refreshSecret = ""
    @State private var isSavingIBKR = false

    var body: some View {
        Group {
            if embedded {
                mainContent
            } else {
                ScrollView {
                    mainContent
                        .padding(16)
                }
                .navigationTitle("Integrations")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
            }
        }
        .onAppear {
            ibkrToken = settings?.ibkrToken ?? ""
            ibkrQuery = settings?.ibkrQueryId ?? ""
        }
        .onChange(of: settings?.ibkrToken) { _, new in ibkrToken = new ?? ibkrToken }
        .onChange(of: settings?.ibkrQueryId) { _, new in ibkrQuery = new ?? ibkrQuery }
    }

    private var mainContent: some View {
        VStack(spacing: 20) {
            ibkrSection
            webhookSection
        }
    }

    private var ibkrSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                SectionLabel(title: "Interactive Brokers (IBKR)")
                Spacer(minLength: 0)
            }

            Text("Automatically sync your IBKR trades and cash movements using the Flex Web Service. Requires an active Activity Flex Query.")
                .appFont(.caption)
                .foregroundStyle(.secondary)

            VStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Flex Token")
                        .appFont(.caption2.bold())
                        .foregroundStyle(.secondary)
                    SecureField("Enter Flex Token", text: $ibkrToken)
                        .textFieldStyle(.roundedBorder)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("Query ID")
                        .appFont(.caption2.bold())
                        .foregroundStyle(.secondary)
                    TextField("Enter Query ID", text: $ibkrQuery)
                        .textFieldStyle(.roundedBorder)
                        #if os(iOS)
                        .keyboardType(.numberPad)
                        #endif
                }

                HStack(spacing: 12) {
                    Button {
                        saveIBKRCredentials()
                    } label: {
                        HStack(spacing: 6) {
                            if isSavingIBKR { ProgressView().controlSize(.small) }
                            Text("Save Credentials")
                        }
                        .frame(maxWidth: .infinity)
                        .fontWeight(.semibold)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isSavingIBKR || ibkrToken.isEmpty || ibkrQuery.isEmpty)

                    Button {
                        syncIBKRNow()
                    } label: {
                        HStack(spacing: 6) {
                            if vm.isSyncingIbkr {
                                ProgressView().controlSize(.small)
                                Text("Syncing…")
                            } else {
                                Image(systemName: "arrow.clockwise")
                                Text("Sync Now")
                            }
                        }
                        .frame(maxWidth: .infinity)
                        .fontWeight(.semibold)
                    }
                    .buttonStyle(.bordered)
                    .disabled(vm.isSyncingIbkr)
                }
                .padding(.top, 4)
            }
        }
        .padding(18)
        .card()
    }

    private var webhookSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                SectionLabel(title: "Webhook Integration")
                Spacer(minLength: 0)
            }

            Text("Trigger a background data refresh externally (e.g. from GitHub Actions or Cron) by providing your webhook secret key.")
                .appFont(.caption)
                .foregroundStyle(.secondary)

            VStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Webhook Secret")
                        .appFont(.caption2.bold())
                        .foregroundStyle(.secondary)
                    SecureField("Enter Webhook Secret", text: $refreshSecret)
                        .textFieldStyle(.roundedBorder)
                }

                Button {
                    triggerWebhook()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "bolt.fill")
                        Text("Test Trigger Webhook")
                    }
                    .frame(maxWidth: .infinity)
                    .fontWeight(.semibold)
                }
                .buttonStyle(.bordered)
                .disabled(refreshSecret.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(18)
        .card()
    }

    private func saveIBKRCredentials() {
        isSavingIBKR = true
        Task {
            let saved = await vm.updateIBKR(token: ibkrToken, queryId: ibkrQuery)
            isSavingIBKR = false
            guard saved else { return }
            ToastManager.shared.show(message: "IBKR credentials saved", style: .success)
        }
    }

    private func syncIBKRNow() {
        Task {
            await vm.syncIbkr()
            if let status = vm.status {
                ToastManager.shared.show(message: status, style: .info)
            }
        }
    }

    private func triggerWebhook() {
        Task {
            guard await vm.triggerRefresh(secret: refreshSecret) else { return }
            ToastManager.shared.show(message: "Webhook refresh triggered", style: .success)
        }
    }
}
