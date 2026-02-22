import SwiftUI

enum CrescendColor {
    /// Warm cream background (#FDF8F0)
    static let background = Color(red: 0xFD / 255.0, green: 0xF8 / 255.0, blue: 0xF0 / 255.0)

    /// Dark warm gray foreground (#2D2926)
    static let foreground = Color(red: 0x2D / 255.0, green: 0x29 / 255.0, blue: 0x26 / 255.0)

    /// Foreground at reduced opacity for secondary text
    static let secondaryText = foreground.opacity(0.6)

    /// Foreground at reduced opacity for borders
    static let border = foreground.opacity(0.12)

    /// Foreground at reduced opacity for subtle fills
    static let subtleFill = foreground.opacity(0.05)
}
