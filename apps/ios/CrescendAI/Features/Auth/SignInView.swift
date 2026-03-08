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
                .ignoresSafeArea()

            // Gradient overlay matching web app treatment
            RadialGradient(
                colors: [
                    Color(red: 0x2D / 255.0, green: 0x29 / 255.0, blue: 0x26 / 255.0, opacity: 0.4),
                    Color(red: 0x2D / 255.0, green: 0x29 / 255.0, blue: 0x26 / 255.0, opacity: 0.85),
                ],
                center: .center,
                startRadius: 50,
                endRadius: 400
            )
            .ignoresSafeArea()

            // Floating sign-in card
            VStack(spacing: CrescendSpacing.space6) {
                VStack(spacing: CrescendSpacing.space3) {
                    Text("CrescendAI")
                        .font(CrescendFont.displayXL())
                        .foregroundStyle(CrescendColor.foreground)

                    Text("A teacher for every pianist.")
                        .font(CrescendFont.bodyLG())
                        .foregroundStyle(CrescendColor.secondaryText)
                }

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
                    .signInWithAppleButtonStyle(.white)
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
            }
            .padding(CrescendSpacing.space8)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(CrescendColor.border, lineWidth: 1)
            )
            .padding(.horizontal, CrescendSpacing.space6)
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
