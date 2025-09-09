# CrescendAI iOS App

This directory contains the iOS native Swift application for CrescendAI - an AI-powered piano performance analysis tool.

## Project Structure

```
ios/
├── CrescendAI/
│   ├── Sources/
│   │   ├── Models/           # Data models (User, Recording, Analysis, etc.)
│   │   ├── Services/         # API client, networking, core services
│   │   ├── ViewControllers/  # Main view controllers for each screen
│   │   ├── Views/           # Custom UI components and views
│   │   ├── Utils/           # Utility classes and extensions
│   │   ├── AppDelegate.swift
│   │   └── SceneDelegate.swift
│   ├── Resources/
│   │   ├── Assets/          # Images, icons, color sets
│   │   ├── Storyboards/     # Interface Builder files
│   │   └── Fonts/           # Custom fonts
│   ├── Tests/
│   │   ├── Unit/           # Unit tests
│   │   └── UI/             # UI tests
│   └── Supporting Files/
│       └── Info.plist      # App configuration
└── README.md
```

## Architecture

The iOS app follows the **MVVM (Model-View-ViewModel)** pattern:

- **Models**: Data structures that match the shared API types
- **Services**: Network layer, data persistence, business logic
- **View Controllers**: Manage UI lifecycle and user interactions
- **Views**: Custom UI components and layouts

## Key Features

- **Native Swift UI**: Optimized for iOS performance and user experience
- **Shared API**: Connects to the same backend as the web app
- **Audio Recording**: Native iOS audio recording capabilities
- **Real-time Analysis**: Display piano performance analysis results
- **Progress Tracking**: Track improvement over time

## Development Setup

### Prerequisites

- **Xcode 15.0+** (latest stable version)
- **iOS 15.0+** deployment target
- **Swift 5.9+**

### Getting Started

1. Open the project in Xcode:
   ```bash
   open ios/CrescendAI.xcodeproj
   ```

2. Configure your development team and bundle identifier in the project settings

3. Install any dependencies (if using Swift Package Manager):
   - Alamofire (for networking)
   - SwiftUI (if using SwiftUI components)

4. Build and run on simulator or device

### API Integration

The iOS app communicates with the same backend API as the web application:

- **Base URL**: `https://api.pianoanalyzer.com/v1`
- **Authentication**: JWT tokens (same as web)
- **Data Models**: Swift models matching the shared TypeScript interfaces

### Build Configuration

- **Development**: Points to local/staging API
- **Production**: Points to production API
- **Bundle ID**: `com.crescendai.client`

## Next Steps

1. **Complete the API Service Layer** - Implement Swift API client
2. **Create View Controllers** - Build main app screens
3. **Implement Audio Recording** - Add native audio capture
4. **Add UI Components** - Create custom Swift UI components
5. **Setup Navigation** - Configure tab bar and navigation flow

## Notes

- The project structure is ready for development
- Core app delegates and configuration files are in place
- Models should match the shared TypeScript types for consistency
- Use the same API endpoints as the web application
