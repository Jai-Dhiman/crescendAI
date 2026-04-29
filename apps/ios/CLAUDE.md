# iOS Practice Companion

Native iOS app -- the primary CrescendAI product.

## Architecture

See `docs/architecture.md` for the full system design.
See `docs/apps/01-product-vision.md` for the product vision.
See `docs/apps/00-status.md` through `docs/apps/05-ui-system.md` for apps documentation.

## Stack

- **UI:** SwiftUI (iOS 17+ minimum, @Observable pattern)
- **Audio:** AVAudioEngine with input tap, 24kHz mono, ring buffer for continuous capture
- **Inference:** Core ML (finetuned MuQ model, 6-dimension output, runs on Neural Engine)
- **On-device scoring:** STOP classifier (logistic regression, 6 weights + bias)
- **Persistence:** SwiftData (local-first, syncs to Cloudflare D1)
- **Auth:** Sign in with Apple (JWT stored in Keychain)
- **Networking:** URLSession async/await (POST /api/ask, POST /api/sync)
- **Observability:** Sentry (`sentry-cocoa`) -- crash reporting, error capture, breadcrumbs, MetricKit

## 6 Dimensions

The model outputs these directly (no mapping from 19):
1. dynamics
2. timing
3. pedaling
4. articulation
5. phrasing
6. interpretation

## Patterns

- MVVM: View + ViewModel + SwiftData Models
- @Observable (not ObservableObject) for ViewModels
- Actor-based networking (APIClient)
- Explicit state enums in ViewModels (idle, recording, processing, error)
- Async/await throughout, MainActor for UI updates
- Explicit exception handling -- no silent fallbacks
- Capture errors at call sites with `SentrySDK.capture(error:)`, not inside error enum definitions
- Set user context (`SentrySDK.setUser`) after auth

## Structure

```
CrescendAI/
  App/                    - Entry point, navigation
  DesignSystem/
    Components/           - CrescendButton, CrescendCard, etc.
    Tokens/               - Colors, typography, spacing
  Features/
    Practice/             - Active session screen (audio capture + inference)
    Observation/          - Teaching moment display
    SessionReview/        - Post-practice timeline
    FocusMode/            - Guided exercise flow
    Profile/              - Settings, goals, dimension trends
  Models/                 - SwiftData models (Student, Session, Chunk, etc.)
  Services/
    AudioEngine/          - AVAudioEngine, ring buffer, chunking
    Inference/            - Core ML model loading + prediction
    StopClassifier/       - Teaching moment scoring
    TeachingMoment/       - Selection + blind spot detection
    Sync/                 - D1 sync protocol
    Auth/                 - Sign in with Apple + JWT management
  Networking/             - API client, endpoints, request/response models
  Resources/
    Fonts/                - Lora font family
```

## Cloud Endpoints (Cloudflare Workers)

- `POST /api/auth/apple` - Validate Apple ID token, issue JWT
- `POST /api/ask` - Send teaching moment context, receive LLM observation (two-stage: subagent analysis + teacher voice, see `docs/apps/02-pipeline.md`)
- `POST /api/sync` - Push student/session deltas, receive exercise updates
- `GET /api/exercises` - Fetch exercise catalog

## Adding New Swift Files (REQUIRED)

Xcode compiles only files listed in `CrescendAI.xcodeproj/project.pbxproj` — files on disk that aren't registered are silently ignored. Since development happens via Claude Code (not Xcode GUI), every new `.swift` file MUST be manually registered.

**Every new file needs entries in 4 places in `project.pbxproj`:**

1. **PBXBuildFile** — `AxxxxxxX /* Foo.swift in Sources */ = {isa = PBXBuildFile; fileRef = BxxxxxxX /* Foo.swift */; };`
2. **PBXFileReference** — `BxxxxxxX /* Foo.swift */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = Foo.swift; sourceTree = "<group>"; };`
3. **PBXGroup** — add `BxxxxxxX /* Foo.swift */,` under the correct group (Models, Chat, Services/Chat, etc.)
4. **PBXSourcesBuildPhase** — add `AxxxxxxX /* Foo.swift in Sources */,` in the correct target's Sources list (app target = `F100000002`, test target = `F300000002`)

**ID conventions used in this project:**
- App source IDs: `A400000028`+ / `B400000028`+ (increment from last used)
- Test source IDs: `A200000003`+ / `B200000003`+

**Verify with:** `just check-ios` — prints any `.swift` files on disk not in the project manifest.

## Notes

- Core ML model is downloaded on first launch (too large for App Store binary)
- Background audio mode requires `UIBackgroundModes: audio` in Info.plist
- Feature flag `useOnDeviceInference` switches between Core ML and cloud fallback
- Audio never leaves the device for scoring (only structured dimension scores sent to cloud)
- Score alignment (chunk timestamps to bar/measure numbers) will be student-reported initially, automated later. See `docs/apps/02-pipeline.md`
