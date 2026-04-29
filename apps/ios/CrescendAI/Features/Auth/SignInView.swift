import AuthenticationServices
import SwiftUI

struct SignInView: View {
    let authService: AuthService
    @State private var error: String?
    @State private var isLoading = false
    @State private var cardOpacity = 0.0
    @State private var cardOffset: CGFloat = 20

    var body: some View {
        ZStack {
            // Full-bleed photo background
            Image("Image5")
                .resizable()
                .aspectRatio(contentMode: .fill)
                .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
                .clipped()
                .ignoresSafeArea()

            // Radial gradient overlay (matches web: rgba(45,41,38,0.4) -> rgba(45,41,38,0.85))
            RadialGradient(
                colors: [
                    CrescendColor.background.opacity(0.4),
                    CrescendColor.background.opacity(0.85),
                ],
                center: .center,
                startRadius: 0,
                endRadius: 420
            )
            .ignoresSafeArea()

            // Sign-in card
            VStack(spacing: 0) {
                Text("crescend")
                    .font(CrescendFont.displayXL())
                    .foregroundStyle(CrescendColor.foreground)

                Text("A teacher for every pianist.")
                    .font(CrescendFont.bodyLG())
                    .foregroundStyle(CrescendColor.secondaryText)
                    .padding(.top, CrescendSpacing.space2)

                if let error {
                    Text(error)
                        .font(CrescendFont.bodySM())
                        .foregroundStyle(.red.opacity(0.8))
                        .multilineTextAlignment(.center)
                        .padding(.top, CrescendSpacing.space4)
                }

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
                .frame(maxWidth: .infinity, minHeight: 50, maxHeight: 50)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .disabled(isLoading)
                .padding(.top, CrescendSpacing.space8)

                if isLoading {
                    ProgressView()
                        .tint(CrescendColor.foreground)
                        .padding(.top, CrescendSpacing.space3)
                }

                Text("By signing in, you agree to our Terms of Service")
                    .font(CrescendFont.labelSM())
                    .foregroundStyle(CrescendColor.tertiaryText)
                    .padding(.top, CrescendSpacing.space6)
            }
            .multilineTextAlignment(.center)
            .padding(.horizontal, CrescendSpacing.space8)
            .padding(.vertical, 56)
            .frame(maxWidth: 360)
            .background {
                ZStack {
                    RoundedRectangle(cornerRadius: 16)
                        .fill(.ultraThinMaterial)
                    RoundedRectangle(cornerRadius: 16)
                        .fill(CrescendColor.surface.opacity(0.7))
                }
            }
            .overlay {
                RoundedRectangle(cornerRadius: 16)
                    .stroke(CrescendColor.border, lineWidth: 1)
            }
            .padding(.horizontal, CrescendSpacing.space6)
            .opacity(cardOpacity)
            .offset(y: cardOffset)
        }
        .onAppear {
            withAnimation(.easeOut(duration: 0.6).delay(0.2)) {
                cardOpacity = 1.0
                cardOffset = 0
            }
        }
    }
}

#Preview {
    SignInView(authService: AuthService())
        .crescendTheme()
}
