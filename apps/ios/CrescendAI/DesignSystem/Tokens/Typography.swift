import SwiftUI

enum CrescendFont {
    // MARK: - Display

    /// 40pt Lora Bold - hero headlines
    static func displayXL(_ weight: Font.Weight = .bold) -> Font {
        .custom("Lora", size: 40).weight(weight)
    }

    /// 32pt Lora Bold - section headlines
    static func displayLG(_ weight: Font.Weight = .bold) -> Font {
        .custom("Lora", size: 32).weight(weight)
    }

    /// 28pt Lora SemiBold - subsection headlines
    static func displayMD(_ weight: Font.Weight = .semibold) -> Font {
        .custom("Lora", size: 28).weight(weight)
    }

    // MARK: - Heading

    /// 24pt Lora SemiBold
    static func headingXL(_ weight: Font.Weight = .semibold) -> Font {
        .custom("Lora", size: 24).weight(weight)
    }

    /// 20pt Lora SemiBold
    static func headingLG(_ weight: Font.Weight = .semibold) -> Font {
        .custom("Lora", size: 20).weight(weight)
    }

    /// 18pt Lora Medium
    static func headingMD(_ weight: Font.Weight = .medium) -> Font {
        .custom("Lora", size: 18).weight(weight)
    }

    // MARK: - Body

    /// 18pt Lora Regular
    static func bodyLG(_ weight: Font.Weight = .regular) -> Font {
        .custom("Lora", size: 18).weight(weight)
    }

    /// 16pt Lora Regular
    static func bodyMD(_ weight: Font.Weight = .regular) -> Font {
        .custom("Lora", size: 16).weight(weight)
    }

    /// 14pt Lora Regular
    static func bodySM(_ weight: Font.Weight = .regular) -> Font {
        .custom("Lora", size: 14).weight(weight)
    }

    // MARK: - Label

    /// 14pt Lora Medium
    static func labelLG(_ weight: Font.Weight = .medium) -> Font {
        .custom("Lora", size: 14).weight(weight)
    }

    /// 12pt Lora Medium
    static func labelMD(_ weight: Font.Weight = .medium) -> Font {
        .custom("Lora", size: 12).weight(weight)
    }

    /// 11pt Lora Medium
    static func labelSM(_ weight: Font.Weight = .medium) -> Font {
        .custom("Lora", size: 11).weight(weight)
    }
}
