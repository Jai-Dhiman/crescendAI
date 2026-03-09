import SwiftUI

enum CrescendColor {
    // MARK: - Core palette (dark theme -- espresso/cream)

    /// Espresso background (#2D2926)
    static let background = Color(red: 0x2D / 255.0, green: 0x29 / 255.0, blue: 0x26 / 255.0)

    /// Cream foreground/text (#FDF8F0)
    static let foreground = Color(red: 0xFD / 255.0, green: 0xF8 / 255.0, blue: 0xF0 / 255.0)

    /// Surface for cards and sidebar (#3A3633)
    static let surface = Color(red: 0x3A / 255.0, green: 0x36 / 255.0, blue: 0x33 / 255.0)

    /// Hover/pressed state surface (#454140)
    static let surface2 = Color(red: 0x45 / 255.0, green: 0x41 / 255.0, blue: 0x40 / 255.0)

    /// Subtle dividers (#504B48)
    static let border = Color(red: 0x50 / 255.0, green: 0x4B / 255.0, blue: 0x48 / 255.0)

    /// Sidebar background, slightly darker than bg (#252220)
    static let sidebarBackground = Color(red: 0x25 / 255.0, green: 0x22 / 255.0, blue: 0x20 / 255.0)

    // MARK: - Text hierarchy

    /// Stone-400 secondary text (#A8A29E)
    static let secondaryText = Color(red: 0xA8 / 255.0, green: 0xA2 / 255.0, blue: 0x9E / 255.0)

    /// Stone-500 tertiary text (#78716C)
    static let tertiaryText = Color(red: 0x78 / 255.0, green: 0x71 / 255.0, blue: 0x6C / 255.0)

    // MARK: - Functional

    /// Input field background (matches surface)
    static let inputBackground = surface

    /// Subtle fill for icon backgrounds
    static let subtleFill = foreground.opacity(0.05)

    // MARK: - Waveform

    /// Waveform bars during recording (cream @ 80%)
    static let waveformActive = foreground.opacity(0.8)

    /// Waveform bars at rest (cream @ 20%)
    static let waveformInactive = foreground.opacity(0.2)

    // MARK: - Brand accent (muted sage)

    static let accent = Color(red: 0x7A / 255.0, green: 0x9A / 255.0, blue: 0x82 / 255.0)
    static let accentLighter = Color(red: 0xA3 / 255.0, green: 0xBD / 255.0, blue: 0xA9 / 255.0)
    static let accentDarker = Color(red: 0x5F / 255.0, green: 0x7D / 255.0, blue: 0x66 / 255.0)

    // MARK: - Dimension accent colors (muted, low-saturation)

    /// Warm gold for dynamics
    static let dimDynamics = Color(red: 0xC4 / 255.0, green: 0xA8 / 255.0, blue: 0x82 / 255.0)

    /// Sage for timing
    static let dimTiming = Color(red: 0xA8 / 255.0, green: 0xB4 / 255.0, blue: 0xA0 / 255.0)

    /// Muted lavender for pedaling
    static let dimPedaling = Color(red: 0xB0 / 255.0, green: 0xA4 / 255.0, blue: 0xB8 / 255.0)

    /// Steel blue for articulation
    static let dimArticulation = Color(red: 0xA0 / 255.0, green: 0xB8 / 255.0, blue: 0xC4 / 255.0)

    /// Dusty rose for phrasing
    static let dimPhrasing = Color(red: 0xC4 / 255.0, green: 0xA0 / 255.0, blue: 0xA0 / 255.0)

    /// Warm stone for interpretation
    static let dimInterpretation = Color(red: 0xB8 / 255.0, green: 0xB4 / 255.0, blue: 0xA0 / 255.0)

    /// Returns the accent color for a given dimension name
    static func dimensionColor(for dimension: String) -> Color {
        switch dimension.lowercased() {
        case "dynamics": dimDynamics
        case "timing": dimTiming
        case "pedaling": dimPedaling
        case "articulation": dimArticulation
        case "phrasing": dimPhrasing
        case "interpretation": dimInterpretation
        default: secondaryText
        }
    }
}
