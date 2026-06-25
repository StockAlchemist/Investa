import SwiftUI

struct LoginView: View {
    @EnvironmentObject private var auth: AuthViewModel

    @State private var username = ""
    @State private var password = ""
    @State private var isRegistering = false
    @State private var showingServerSettings = false
    @State private var serverURL = APIConfig.baseURL
    @State private var activeServerURL = APIConfig.baseURL

    var body: some View {
        VStack(spacing: 24) {
            VStack(spacing: 8) {
                Image("AppLogoNoText")
                    .resizable()
                    .scaledToFit()
                    .frame(height: 44)
                Text("Investa")
                    .font(.largeTitle.bold())
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
                        .font(.callout)
                        .foregroundStyle(.red)
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
                .font(.callout)
                .foregroundStyle(.tint)
            }
            .frame(width: 280)

            Button {
                serverURL = APIConfig.baseURL
                showingServerSettings = true
            } label: {
                Label("Server: \(activeServerURL)", systemImage: "network")
                    .font(.caption)
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
            Text("Backend Server").font(.headline)
            Text("The address of the Investa FastAPI backend.")
                .font(.callout)
                .foregroundStyle(.secondary)
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
                    APIConfig.baseURL = serverURL
                    activeServerURL = serverURL
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
