import SwiftUI

enum CrescendFont {
    // MARK: - Display

    /// 40pt Lora Bold - hero headlines
    static func displayXL() -> Font {
        .custom("Lora", fixedSize: 40).weight(.bold)
    }

    /// 32pt Lora Bold - section headlines
    static func displayLG() -> Font {
        .custom("Lora", fixedSize: 32).weight(.bold)
    }

    /// 28pt Lora SemiBold - subsection headlines
    static func displayMD() -> Font {
        .custom("Lora", fixedSize: 28).weight(.semibold)
    }

    // MARK: - Heading

    /// 24pt Lora SemiBold
    static func headingXL() -> Font {
        .custom("Lora", fixedSize: 24).weight(.semibold)
    }

    /// 20pt Lora SemiBold
    static func headingLG() -> Font {
        .custom("Lora", fixedSize: 20).weight(.semibold)
    }

    /// 18pt Lora Medium
    static func headingMD() -> Font {
        .custom("Lora", fixedSize: 18).weight(.medium)
    }

    // MARK: - Body (Figtree)

    /// 18pt Figtree Regular
    static func bodyLG() -> Font {
        .custom("Figtree", fixedSize: 18)
    }

    /// 16pt Figtree Regular
    static func bodyMD() -> Font {
        .custom("Figtree", fixedSize: 16)
    }

    /// 14pt Figtree Regular
    static func bodySM() -> Font {
        .custom("Figtree", fixedSize: 14)
    }

    // MARK: - Label (Figtree)

    /// 14pt Figtree Medium
    static func labelLG() -> Font {
        .custom("Figtree", fixedSize: 14).weight(.medium)
    }

    /// 12pt Figtree Medium
    static func labelMD() -> Font {
        .custom("Figtree", fixedSize: 12).weight(.medium)
    }

    /// 11pt Figtree Medium
    static func labelSM() -> Font {
        .custom("Figtree", fixedSize: 11).weight(.medium)
    }
}
