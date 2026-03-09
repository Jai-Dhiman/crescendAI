import AuthenticationServices
import SwiftUI

struct SignInView: View {
    let authService: AuthService
    @State private var error: String?
    @State private var isLoading = false
    @State private var cardOpacity = 0.0

    var body: some View {
        ZStack {
            // Full-bleed photo background
            Image("Image5")
                .resizable()
                .aspectRatio(contentMode: .fill)
                .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
                .clipped()
                .ignoresSafeArea()

            // Centered sign-in card
            VStack(spacing: 0) {
                // App logo
                Image("AppLogo")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 56, height: 56)
                    .clipShape(RoundedRectangle(cornerRadius: 14))

                // Title
                Text("crescend")
                    .font(CrescendFont.displayXL())
                    .foregroundStyle(CrescendColor.foreground)
                    .padding(.top, CrescendSpacing.space4)

                // Tagline
                Text("A teacher for every pianist.")
                    .font(CrescendFont.bodyLG())
                    .foregroundStyle(CrescendColor.secondaryText)
                    .padding(.top, CrescendSpacing.space2)

                // Error
                if let error {
                    Text(error)
                        .font(CrescendFont.bodySM())
                        .foregroundStyle(.red.opacity(0.8))
                        .multilineTextAlignment(.center)
                        .padding(.top, CrescendSpacing.space4)
                }

                // Sign in with Apple button (cream-styled)
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
                .signInWithAppleButtonStyle(.white)
                .frame(maxWidth: 260, minHeight: 50, maxHeight: 50)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .tint(CrescendColor.foreground)
                .disabled(isLoading)
                .padding(.top, CrescendSpacing.space8)

                if isLoading {
                    ProgressView()
                        .tint(CrescendColor.foreground)
                        .padding(.top, CrescendSpacing.space3)
                }

                // Terms disclaimer
                Text("By signing in, you agree to our Terms of Service")
                    .font(CrescendFont.labelSM())
                    .foregroundStyle(CrescendColor.tertiaryText)
                    .padding(.top, CrescendSpacing.space6)
            }
            .padding(.horizontal, CrescendSpacing.space8)
            .padding(.vertical, 48)
            .frame(maxWidth: 340)
            .background(Color.clear)
            .opacity(cardOpacity)
        }
        .onAppear {
            withAnimation(.easeOut(duration: 0.6).delay(0.2)) {
                cardOpacity = 1.0
            }
        }
    }
}

#Preview {
    SignInView(authService: AuthService())
        .crescendTheme()
}
