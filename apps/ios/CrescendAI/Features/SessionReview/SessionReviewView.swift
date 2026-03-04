import SwiftUI

struct SessionReviewView: View {
    @Environment(\.dismiss) private var dismiss
    let messages: [ChatMessage]
    @State private var checkInText = ""
    @State private var expandedId: UUID?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: CrescendSpacing.space6) {
                    // Header
                    VStack(alignment: .leading, spacing: CrescendSpacing.space2) {
                        Text("Session Complete")
                            .font(CrescendFont.displayMD())
                            .foregroundStyle(CrescendColor.foreground)

                        Text(Date.now, style: .date)
                            .font(CrescendFont.bodyMD())
                            .foregroundStyle(CrescendColor.secondaryText)
                    }
                    .padding(.horizontal, CrescendSpacing.space4)

                    // Observation timeline
                    if !messages.isEmpty {
                        VStack(alignment: .leading, spacing: CrescendSpacing.space3) {
                            Text("Observations")
                                .font(CrescendFont.headingMD())
                                .foregroundStyle(CrescendColor.foreground)
                                .padding(.horizontal, CrescendSpacing.space4)

                            ForEach(messages) { message in
                                compactObservation(message)
                                    .padding(.horizontal, CrescendSpacing.space4)
                            }
                        }
                    }

                    // Summary placeholder
                    VStack(alignment: .leading, spacing: CrescendSpacing.space2) {
                        Text("A productive session with good dynamic contrast and clean pedaling.")
                            .font(CrescendFont.bodyLG())
                            .italic()
                            .foregroundStyle(CrescendColor.secondaryText)
                    }
                    .padding(.horizontal, CrescendSpacing.space4)

                    // Check-in
                    VStack(alignment: .leading, spacing: CrescendSpacing.space3) {
                        Text("How did that feel?")
                            .font(CrescendFont.headingMD())
                            .foregroundStyle(CrescendColor.foreground)

                        TextField("Share your thoughts...", text: $checkInText, axis: .vertical)
                            .font(CrescendFont.bodyMD())
                            .foregroundStyle(CrescendColor.foreground)
                            .lineLimit(2...6)
                            .padding(CrescendSpacing.space3)
                            .background(CrescendColor.inputBackground)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                            .overlay(
                                RoundedRectangle(cornerRadius: 12)
                                    .stroke(CrescendColor.border, lineWidth: 1)
                            )
                    }
                    .padding(.horizontal, CrescendSpacing.space4)

                    // Done button
                    CrescendButton("Done", style: .primary) {
                        dismiss()
                    }
                    .padding(.horizontal, CrescendSpacing.space4)
                    .padding(.bottom, CrescendSpacing.space6)
                }
                .padding(.top, CrescendSpacing.space4)
            }
            .background(CrescendColor.background)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                        .foregroundStyle(CrescendColor.foreground)
                }
            }
            .toolbarBackground(CrescendColor.surface, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
        }
    }

    private func compactObservation(_ message: ChatMessage) -> some View {
        CrescendCard(style: .interactive, action: {
            withAnimation(.easeOut(duration: 0.2)) {
                expandedId = expandedId == message.id ? nil : message.id
            }
        }) {
            VStack(alignment: .leading, spacing: CrescendSpacing.space2) {
                HStack {
                    if let dimension = message.dimension {
                        DimensionPill(dimension: dimension)
                    }
                    Spacer()
                    Text(message.timestamp, style: .time)
                        .font(CrescendFont.labelSM())
                        .foregroundStyle(CrescendColor.tertiaryText)
                }

                if expandedId == message.id {
                    Text(message.text)
                        .font(CrescendFont.bodyMD())
                        .foregroundStyle(CrescendColor.foreground)
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    Text(message.text)
                        .font(CrescendFont.bodySM())
                        .foregroundStyle(CrescendColor.secondaryText)
                        .lineLimit(1)
                }
            }
        }
    }
}

#Preview {
    SessionReviewView(messages: [
        ChatMessage(role: .observation, text: "Your dynamics showed nice contrast between the forte and piano sections.", dimension: "dynamics"),
        ChatMessage(role: .observation, text: "The pedaling through the transition was clean.", dimension: "pedaling"),
    ])
    .crescendTheme()
}
