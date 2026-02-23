import Foundation

enum DemoScenarios {

    // MARK: - Scenario 1: First-time user, general feedback

    static let firstTimeGreeting = ChatMessage(
        sender: .ai,
        blocks: [
            .text("Nice -- I heard something lyrical, with a gentle touch in the upper register. Here's what stood out to me:"),
            .text("Your legato phrasing is naturally musical -- the way you connected the melodic line showed real sensitivity. The dynamics could use more contrast though. The quiet sections are beautiful, but the crescendo moments don't quite arrive where they want to go."),
            .dimensionCard(label: "Dynamics", score: 5.8, interpretation: "Room to grow -- your quiet playing is expressive, but the louder passages need more commitment."),
            .text("Try exaggerating the dynamic range for a few run-throughs. It often feels like too much at the piano, but from the audience's perspective, it reads as just right."),
        ]
    )

    static let firstTimeChips: [SuggestionChip] = [
        SuggestionChip(label: "Focus on dynamics", actionKey: "start_focus_dynamics"),
        SuggestionChip(label: "What should I work on?", actionKey: "what_to_work_on"),
        SuggestionChip(label: "How does this work?", actionKey: "explain_app"),
    ]

    static let howItWorks = ChatMessage(
        sender: .ai,
        blocks: [
            .text("I listen to how you play -- not just the notes, but how they sound. Things like your dynamics, tone quality, phrasing, pedaling, and articulation."),
            .text("I can focus on one thing at a time (like just dynamics), or give you a general picture. When it helps, I can play a reference so you can hear what I'm describing."),
            .text("Just play whenever you're ready. I'm always listening."),
        ]
    )

    static let howItWorksChips: [SuggestionChip] = [
        SuggestionChip(label: "Focus on dynamics", actionKey: "start_focus_dynamics"),
        SuggestionChip(label: "Play it again", actionKey: "dismiss_chat"),
    ]

    static let whatToWorkOn = ChatMessage(
        sender: .ai,
        blocks: [
            .text("Based on what I just heard, I'd suggest focusing on dynamics first. Your phrasing instincts are strong, but the dynamic palette is narrow -- opening that up will make everything else shine more."),
            .text("Want to try a focused session on dynamics? I'll only give feedback on dynamic contrast and shaping."),
        ]
    )

    static let whatToWorkOnChips: [SuggestionChip] = [
        SuggestionChip(label: "Start focus mode", actionKey: "start_focus_dynamics"),
        SuggestionChip(label: "Try something else", actionKey: "dismiss_chat"),
    ]

    // MARK: - Scenario 2: Returning user, focus mode (dynamics)

    static let returningGreeting = "Last time: Chopin Nocturne Op. 9 No. 2 -- Bars 12-18 dynamics. Pick up where you left off?"

    static let focusDynamicsFeedback = ChatMessage(
        sender: .ai,
        blocks: [
            .text("Better! The crescendo into bar 14 has more shape now. But you're pulling back right at the peak -- try sustaining the forte for one more beat before the diminuendo."),
            .dimensionCard(label: "Dynamics", score: 6.4, interpretation: "Improving. The crescendo shape is there, but the peak needs more sustain."),
            .text("Listen to how this passage can build -- notice how the loudest moment lingers before pulling back:"),
            .referencePlayback(label: "Bars 12-18: crescendo with sustained peak", audioFileName: "reference_nocturne_bars12_18"),
        ]
    )

    static let focusDynamicsChips: [SuggestionChip] = [
        SuggestionChip(label: "Play it again", actionKey: "dismiss_chat"),
        SuggestionChip(label: "Show me another reference", actionKey: "another_reference"),
        SuggestionChip(label: "Exit focus mode", actionKey: "exit_focus"),
        SuggestionChip(label: "I'm done for today", actionKey: "done"),
    ]

    // MARK: - Scenario 3: Reference playback follow-up

    static let referenceFollowUp = ChatMessage(
        sender: .ai,
        blocks: [
            .text("Notice two things in that reference: the crescendo builds gradually over three beats (not all at once), and the peak is held with a slight tenuto before the release. That sustained moment is what gives the phrase its emotional weight."),
            .text("Try matching that shape. Gradual build, hold the peak, then release."),
        ]
    )

    static let referenceFollowUpChips: [SuggestionChip] = [
        SuggestionChip(label: "Play it again", actionKey: "dismiss_chat"),
        SuggestionChip(label: "I'm done for today", actionKey: "done"),
    ]

    // MARK: - Done

    static let doneMessage = ChatMessage(
        sender: .ai,
        blocks: [
            .text("Good session. Your dynamics are moving in the right direction -- the crescendo shaping is noticeably better than when we started. Next time, we can pick up right where you left off."),
        ]
    )

    static let doneChips: [SuggestionChip] = [
        SuggestionChip(label: "Play one more time", actionKey: "dismiss_chat"),
    ]
}
