# Piano Performance Analyzer Mobile App

React Native mobile application built with Expo for piano performance analysis and AI-powered feedback.

## Features

- **Audio Recording**: High-quality piano performance recording
- **Google Authentication**: Secure OAuth login
- **Performance Analysis**: AI-powered feedback (backend integration)
- **Progress Tracking**: Monitor improvement over time
- **Local Storage**: MMKV-based persistent storage
- **Cross-Platform**: iOS, Android, and Web support

## Tech Stack

- **Framework**: Expo (~50.0.17) + React Native (0.73.6)
- **Navigation**: Expo Router (~3.4.8) with file-based routing
- **State Management**: Zustand (^4.5.2) with MMKV persistence
- **Data Fetching**: TanStack Query (^5.29.0)
- **Forms**: TanStack Form (^0.20.1)
- **Authentication**: expo-auth-session with Google OAuth
- **Storage**: react-native-mmkv (^2.12.2)
- **Audio**: expo-av (~13.10.5)
- **TypeScript**: Full type safety with strict mode

## Getting Started

### Prerequisites

- Node.js 18+
- Bun (recommended) or npm/yarn
- Expo CLI: `npm install -g @expo/cli`
- For iOS: Xcode and iOS Simulator
- For Android: Android Studio and Android Emulator

### Installation

```bash
# Install dependencies
bun install

# Start development server
bun run start
# or
expo start

# Run on specific platforms
bun run ios     # iOS simulator
bun run android # Android emulator
bun run web     # Web browser
```

### Environment Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Configure Google OAuth:
   - Create a Google Cloud Console project
   - Enable Google+ API
   - Create OAuth 2.0 credentials
   - Add your client IDs to `.env`

```bash
EXPO_PUBLIC_GOOGLE_CLIENT_ID=your-google-client-id-here
EXPO_PUBLIC_GOOGLE_CLIENT_ID_WEB=your-google-web-client-id-here
EXPO_PUBLIC_API_URL=https://api.pianoanalyzer.com
```

## Project Structure

```
mobile/
├── app/                    # Expo Router pages
│   ├── (tabs)/            # Tab navigation
│   │   ├── record.tsx     # Recording screen
│   │   ├── recordings.tsx # Recordings list
│   │   ├── progress.tsx   # Progress tracking
│   │   └── profile.tsx    # User profile
│   ├── _layout.tsx        # Root layout
│   ├── index.tsx          # Landing page
│   └── auth.tsx           # Authentication
├── src/
│   ├── components/        # Reusable components
│   ├── hooks/             # Custom hooks
│   ├── services/          # API and external services
│   ├── stores/            # Zustand stores
│   ├── types/             # TypeScript definitions
│   └── utils/             # Utility functions
├── assets/                # Images, fonts, etc.
└── app.json              # Expo configuration
```

## Key Features Implementation

### Authentication Flow
- Google OAuth via expo-auth-session
- Token management with automatic refresh
- Persistent auth state with MMKV

### Recording System
- High-quality audio recording with expo-av
- Local file storage with expo-file-system
- Upload progress tracking
- Recording metadata management

### State Management
- Zustand stores with MMKV persistence
- Separate stores for auth, recordings, and settings
- Optimistic updates for better UX

### Data Fetching
- TanStack Query for server state management
- Automatic caching and background updates
- Error handling and retry logic
- Pagination support

## Development Guidelines

### Code Style
- TypeScript strict mode enabled
- Functional components with hooks
- Custom hooks for business logic
- Modular store architecture

### Testing
```bash
# Type checking
bun run typecheck

# Linting
bun run lint
```

### Building
```bash
# Development build
expo build

# Production build
expo build --release-channel production
```

## Configuration

### App Configuration (app.json)
- App metadata and icons
- Platform-specific settings
- Deep linking scheme
- Expo Router configuration

### Authentication Setup
The app uses Google OAuth with the following redirect URI pattern:
- Scheme: `piano-analyzer://auth`
- Web: `https://yourdomain.com/auth`

## Troubleshooting

### Common Issues

1. **Metro bundler issues**: Clear cache with `expo r -c`
2. **Android build issues**: Check SDK and emulator setup
3. **OAuth not working**: Verify client IDs and redirect URIs
4. **Audio recording issues**: Check microphone permissions

### Debug Information
- Use the Profile screen's "Configuration Status" to check OAuth setup
- Console logs available in development mode
- React Query DevTools for API debugging (development only)

## Deployment

### Development
- Use Expo Go app for quick testing
- Development builds for native features

### Production
- EAS Build for production apps
- App Store / Google Play deployment
- Web deployment via static hosting

## Contributing

1. Follow the existing code structure and patterns
2. Use TypeScript for all new code
3. Implement proper error handling
4. Add loading states for async operations
5. Keep components unstyled (as per project requirements)

## Next Steps

1. Implement audio analysis visualization components
2. Add offline capability with sync
3. Implement push notifications
4. Add more detailed progress analytics
5. Integrate with piano learning resources
