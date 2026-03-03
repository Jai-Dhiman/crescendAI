import AuthenticationServices
import SwiftUI

struct SignInView: View {
    let authService: AuthService
    @State private var error: String?
    @State private var isLoading = false

    var body: some View {
        ZStack {
            CrescendColor.background
                .ignoresSafeArea()

            VStack(spacing: CrescendSpacing.space8) {
                Spacer()

                VStack(spacing: CrescendSpacing.space4) {
                    Text("CrescendAI")
                        .font(CrescendFont.displayXL())
                        .foregroundStyle(CrescendColor.foreground)

                    Text("A teacher for every pianist.")
                        .font(CrescendFont.bodyLG())
                        .foregroundStyle(CrescendColor.secondaryText)
                }

                Spacer()

                VStack(spacing: CrescendSpacing.space3) {
                    SignInWithAppleButton(.signIn) { request in
                        request.requestedScopes = [.email]
                    } onCompletion: { result in
                        Task {
                            isLoading = true
                            error = nil
                            do {
                                try await authService.handleAuthorization(result: result)
                            } catch {
                                self.error = error.localizedDescription
                            }
                            isLoading = false
                        }
                    }
                    .signInWithAppleButtonStyle(.black)
                    .frame(height: 50)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .disabled(isLoading)

                    if isLoading {
                        ProgressView()
                            .tint(CrescendColor.foreground)
                    }

                    if let error {
                        Text(error)
                            .font(CrescendFont.bodySM())
                            .foregroundStyle(.red.opacity(0.8))
                            .multilineTextAlignment(.center)
                    }
                }
                .padding(.horizontal, CrescendSpacing.space6)

                Spacer()
                    .frame(height: CrescendSpacing.space12)
            }
        }
    }
}

#Preview {
    SignInView(authService: AuthService())
        .crescendTheme()
}
