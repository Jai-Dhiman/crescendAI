# Landing Page Updates Design

Status: DESIGNED
Date: 2026-03-04

Three changes to the landing page at crescend.ai.

## 1. Cascading Images Section -- Rework

### Layout

Replace overlapping cascade with a staircase layout. Three images offset down and to the right, touching at corners but never layered on top of each other. Reference: bakery site with stepped image grid.

### Image Arc: Struggle, Guidance, Breakthrough

- **Image 1 (top-left):** The struggle. Practicing alone, frustration, pausing mid-passage.
- **Image 2 (middle):** The guidance. A moment of insight, focused attention, studying a score.
- **Image 3 (bottom-right):** The breakthrough. Confident playing, flow, satisfaction.

Source new stock photography. Current images (Image2-4.jpg) are generic piano stock that don't tell a story. New images should work with the espresso/warm palette.

Keep the pull quote alongside the staircase: "What's the one thing that sounds off that I can't hear myself?"

### Implementation

- Rework CSS positioning in CascadingQuoteSection from overlapping absolute offsets to a staircase grid
- Replace Image2.jpg, Image3.jpg, Image4.jpg with new stock photos
- Image1.jpg (hero background) stays unchanged

## 2. Laptop + Phone Mockup Section -- New

### Position

New section directly below the cascading images section, before the final CTA.

### Content

Dark background. Laptop frame (desktop) and phone frame (mobile) showing the CrescendAI web app chat interface in action.

**Desktop mockup:** Chat conversation about a Bach Invention or Burgmuller etude. Teacher provides feedback. An inline score highlight on-demand component is visible in the chat.

**Mobile mockup:** Same conversation context. An inline exercise set on-demand component (2-3 practice variation cards) is visible.

### Mockup Build Process

Build two standalone HTML mockup files (not embedded in the landing page) that render realistic chat UIs:
- `apps/web/mockups/desktop-chat.html` -- desktop-width chat with score highlight card
- `apps/web/mockups/mobile-chat.html` -- mobile-width chat with exercise set card

Screenshot these files and place the images in the landing page's laptop + phone frame layout.

### Landing Page

- Add new section component (DeviceMockupSection or similar)
- Laptop + phone frames with screenshot images as content
- Placeholder images until mockup screenshots are ready

## 3. Feature Cards -- Rework

### Animations (Jitter)

Three abstract/conceptual animations, designed in Jitter, exported as looping video (MP4/WebM) or Lottie:

1. **Audio waveform into teacher response** -- Abstract waveform animating as audio is captured, then a chat bubble or feedback element fades/slides in. Communicates: the app listens and responds.
2. **Score with annotations appearing** -- Abstract bars of sheet music where colored highlights and annotation marks fade in on specific passages. Communicates: the teacher points to exactly where to focus.
3. **Exercise cards generating** -- A prompt or message spawns 2-3 practice variation cards that slide in. Communicates: personalized drills, not generic exercises.

Style: abstract and conceptual (not realistic app UI mockups). Shapes, motion, and color over literal screenshots. Can pivot to realistic UI style later if the abstract approach doesn't land.

### Copy

Titles and descriptions written after animations are designed, driven by what each animation communicates visually. Short title + one-line description per card, matching the [untitled] reference format.

Current placeholder text ("Your teacher is listening", "Exercises built for you", "See what you hear") may change.

### Implementation

- Embed Jitter exports (video or Lottie) in the existing three-card grid
- Update titles and descriptions once finalized
- Keep the card layout structure (4:3 aspect ratio visual + text below)

## Implementation Order

1. Staircase layout CSS change (can ship before new images arrive)
2. Standalone chat mockup HTML files (desktop + mobile)
3. New device mockup section on landing page (with placeholder images)
4. Jitter animations (design externally, embed when ready)
5. Source stock photography for cascading section
6. Final copy pass on feature card titles/descriptions
