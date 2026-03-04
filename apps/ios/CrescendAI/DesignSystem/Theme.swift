import SwiftUI

private struct CrescendThemeKey: EnvironmentKey {
    static let defaultValue = true
}

extension EnvironmentValues {
    var isCrescendThemed: Bool {
        get { self[CrescendThemeKey.self] }
        set { self[CrescendThemeKey.self] = newValue }
    }
}

extension View {
    /// Apply the CrescendAI dark theme to the view hierarchy.
    func crescendTheme() -> some View {
        self
            .environment(\.isCrescendThemed, true)
            .preferredColorScheme(.dark)
            .tint(CrescendColor.foreground)
    }
}
