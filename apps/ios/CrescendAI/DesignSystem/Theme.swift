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
    /// Apply the CrescendAI theme to the view hierarchy.
    /// Sets background color, preferred color scheme, and tint.
    func crescendTheme() -> some View {
        self
            .environment(\.isCrescendThemed, true)
            .preferredColorScheme(.light)
            .tint(CrescendColor.foreground)
    }
}
