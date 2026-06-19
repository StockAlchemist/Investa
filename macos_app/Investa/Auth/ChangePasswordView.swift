import SwiftUI

struct ChangePasswordView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var auth: AuthViewModel

    @State private var current = ""
    @State private var newPassword = ""
    @State private var confirm = ""
    @State private var error: String?
    @State private var isSaving = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Change Password").font(.title2.bold())
            SecureField("Current password", text: $current).textFieldStyle(.roundedBorder)
            SecureField("New password", text: $newPassword).textFieldStyle(.roundedBorder)
            SecureField("Confirm new password", text: $confirm).textFieldStyle(.roundedBorder)
            if let error { Text(error).foregroundStyle(.red).font(.callout) }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Change") { submit() }
                    .buttonStyle(.borderedProminent)
                    .disabled(isSaving || !isValid)
            }
        }
        .padding(24)
        #if os(macOS)
        .frame(width: 380)
        #else
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        #endif
    }

    private var isValid: Bool {
        !current.isEmpty && newPassword.count >= 1 && newPassword == confirm
    }

    private func submit() {
        guard isValid else { error = "New passwords don't match."; return }
        error = nil; isSaving = true
        Task {
            let result = await auth.changePassword(current: current, new: newPassword)
            isSaving = false
            if let result { error = result } else { dismiss() }
        }
    }
}
