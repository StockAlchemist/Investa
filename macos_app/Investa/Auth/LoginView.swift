import SwiftUI

struct LoginView: View {
    @EnvironmentObject private var auth: AuthViewModel

    @State private var username = ""
    @State private var password = ""
    @State private var isRegistering = false
    @State private var showingServerSettings = false
    @ObservedObject private var servers = ServerStore.shared
    @State private var serverURL = APIConfig.baseURL

    var body: some View {
        VStack(spacing: 24) {
            VStack(spacing: 8) {
                Image("AppLogoNoText")
                    .resizable()
                    .scaledToFit()
                    .frame(height: 88)
                Text("Investa")
                    .appFont(.largeTitle.bold())
                Text(isRegistering ? "Create an account" : "Sign in to your portfolio")
                    .foregroundStyle(.secondary)
            }

            VStack(spacing: 12) {
                TextField("Username", text: $username)
                    .textContentType(.username)
                    .textFieldStyle(.roundedBorder)
                SecureField("Password", text: $password)
                    .textContentType(.password)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(submit)

                if let error = auth.errorMessage {
                    Text(error)
                        .appFont(.callout)
                        .foregroundStyle(.down)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                Button(action: submit) {
                    if auth.isSubmitting {
                        ProgressView().controlSize(.small)
                    } else {
                        Text(isRegistering ? "Create Account" : "Log In").frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(!canSubmit)

                Button(isRegistering ? "Have an account? Log in" : "Create an account") {
                    auth.errorMessage = nil
                    isRegistering.toggle()
                }
                .buttonStyle(.plain)
                .appFont(.callout)
                .foregroundStyle(.tint)
            }
            .frame(width: 280)

            Button {
                serverURL = APIConfig.baseURL
                showingServerSettings = true
            } label: {
                Label("Server: \(servers.activeServer?.name ?? servers.activeURL)", systemImage: "network")
                    .appFont(.caption)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
        }
        .padding(40)
        .macMinSize(width: 420, height: 460)
        .sheet(isPresented: $showingServerSettings) {
            serverSettingsSheet
        }
    }

    private var canSubmit: Bool {
        !username.isEmpty && !password.isEmpty && !auth.isSubmitting
    }

    private func submit() {
        guard canSubmit else { return }
        Task {
            if isRegistering {
                await auth.register(username: username, password: password)
            } else {
                await auth.login(username: username, password: password)
            }
        }
    }

    private var serverSettingsSheet: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Backend Server").appFont(.headline)
            Text("The address of the Investa FastAPI backend.")
                .appFont(.callout)
                .foregroundStyle(.secondary)
            if !servers.servers.isEmpty {
                // Picking one fills the field; Save connects, as for a typed URL.
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(servers.servers) { server in
                        let picked = APIConfig.normalized(serverURL) == APIConfig.normalized(server.url)
                        Button { serverURL = server.url } label: {
                            HStack(spacing: 8) {
                                Image(systemName: picked ? "checkmark.circle.fill" : "circle")
                                    .foregroundStyle(picked ? Color.brand : Color.ink3)
                                Text(server.name).appFont(.callout.weight(.medium))
                                Spacer(minLength: 8)
                                Text(ServerStore.defaultName(for: server.url))
                                    .appFont(.caption).monospacedDigit()
                                    .foregroundStyle(.secondary)
                            }
                            .lineLimit(1)
                            .minimumScaleFactor(0.7)
                            .padding(.vertical, 4)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            TextField("http://localhost:8000/api", text: $serverURL)
                .textFieldStyle(.roundedBorder)
                #if os(macOS)
                .frame(width: 360)
                #endif
            HStack {
                Button("Reset to Default") {
                    serverURL = APIConfig.fallbackBaseURL
                }
                Spacer()
                Button("Cancel") { showingServerSettings = false }
                Button("Save") {
                    servers.switchTo(serverURL)
                    showingServerSettings = false
                }
                .buttonStyle(.borderedProminent)
                .disabled(serverURL.isEmpty)
            }
        }
        .padding(24)
        #if os(macOS)
        .frame(width: 420)
        #else
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        #endif
    }
}
