# CrescendAI Architecture Documentation

## Overview

CrescendAI is a cross-platform piano performance analysis application with a modern, scalable architecture that supports both web and iOS native platforms. This document outlines the complete system architecture, data flow, and development workflow.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│   Web Client    │    │   iOS Client    │    │  Future Clients │
│   (SvelteKit)   │    │    (Swift)      │    │   (Android?)    │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │                 │
                    │   Backend API   │
                    │   (Server)      │
                    │                 │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │                 │
                    │   ML Pipeline   │
                    │   (Model)       │
                    │                 │
                    └─────────────────┘
```

## Directory Structure

```
crescendai/
├── web/                    # SvelteKit web application
│   ├── src/
│   │   ├── lib/
│   │   │   ├── components/     # Svelte components
│   │   │   │   ├── ui/         # Basic UI components (Button, Card, etc.)
│   │   │   │   └── sketchy/    # Styled visualization components
│   │   │   └── services/       # Web-specific services
│   │   └── routes/             # SvelteKit routes
│   └── package.json
│
├── ios/                    # iOS Swift application
│   ├── CrescendAI/
│   │   ├── Sources/
│   │   │   ├── Models/         # Swift data models
│   │   │   ├── Services/       # API client, auth service
│   │   │   ├── ViewControllers/# Main view controllers
│   │   │   ├── Views/          # Custom UI components
│   │   │   └── Utils/          # Utility classes
│   │   ├── Resources/          # Assets, storyboards
│   │   ├── Tests/              # Unit and UI tests
│   │   └── Supporting Files/   # Info.plist, etc.
│   └── README.md
│
├── shared/                 # Cross-platform shared code
│   ├── types/              # TypeScript interface definitions
│   ├── utils/              # Platform-agnostic utilities
│   ├── theme/              # Design tokens and colors
│   ├── stores/             # State management (Zustand)
│   └── components/         # Shared Svelte components
│
├── server/                 # Backend API service
│   ├── src/                # Server source code
│   └── ...
│
├── model/                  # ML pipeline and model code
│   ├── src/                # Python ML code
│   └── ...
│
└── docs/                   # Project documentation
```

## Platform-Specific Implementations

### Web Application (SvelteKit)

**Technology Stack:**

- **Frontend**: SvelteKit 2.x with Svelte 5
- **Styling**: TailwindCSS 4.x
- **State Management**: Zustand (shared)
- **Build Tool**: Vite + bun
- **Forms**: @tanstack/svelte-form
- **API Queries**: @tanstack/svelte-query

**Key Features:**

- Server-side rendering (SSR)
- Progressive web app (PWA) capabilities
- Responsive design for desktop and mobile browsers
- Real-time audio visualization
- File upload with drag-and-drop support

**Component Architecture:**

- **UI Components**: Basic building blocks (Button, Card, Typography)
- **Sketchy Components**: Stylized components with hand-drawn aesthetics
- **Page Components**: Route-specific components
- **Layout Components**: App shell, navigation, etc.

### iOS Application (Swift)

**Technology Stack:**

- **Language**: Swift 5.9+
- **Minimum iOS**: 15.0
- **Architecture**: MVVM (Model-View-ViewModel)
- **Networking**: URLSession (custom API client)
- **Storage**: Keychain (authentication), UserDefaults (settings)
- **UI**: UIKit + programmatic layouts

**Key Features:**

- Native iOS performance and user experience
- Core Audio integration for recording
- Keychain secure storage for authentication
- Background processing for uploads
- Push notifications for analysis completion

**Architecture Patterns:**

- **Models**: Swift structs matching shared TypeScript interfaces
- **Services**: API client, auth service, data managers
- **ViewModels**: Business logic and state management
- **ViewControllers**: UI lifecycle and user interaction
- **Views**: Custom UI components

## Shared Resources

### Types and Interfaces (`/shared/types/`)

All data models are defined once as TypeScript interfaces and then implemented in each platform:

```typescript
// TypeScript (source of truth)
export interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  createdAt: string;
  updatedAt: string;
}
```

```swift
// Swift (matches TypeScript)
struct User: Codable {
    let id: String
    let email: String
    let name: String
    let avatar: String?
    let createdAt: String
    let updatedAt: String
}
```

### API Integration

Both platforms communicate with the same backend API using consistent patterns:

**Base URL**: `https://api.pianoanalyzer.com/v1`

**Authentication**: JWT tokens with automatic refresh

- Access tokens for API requests
- Refresh tokens for token renewal
- Secure storage on each platform

**Common Endpoints**:

- `POST /auth/google` - Google OAuth authentication
- `GET /users/{id}` - Get user profile
- `GET /users/{id}/recordings` - Get user recordings
- `POST /recordings` - Create new recording
- `POST /recordings/{id}/upload` - Upload audio file
- `GET /recordings/{id}/analysis` - Get analysis results

### State Management

**Web**: Zustand stores with localStorage persistence
**iOS**: ObservableObject classes with Keychain/UserDefaults persistence

Both platforms maintain consistent state structure:

```typescript
interface AuthState {
  user: User | null;
  tokens: AuthTokens | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}
```

### Theme and Design System

Shared design tokens ensure visual consistency:

- **Colors**: Primary, secondary, semantic colors
- **Typography**: Font sizes, weights, line heights
- **Spacing**: Consistent spacing scale
- **Border Radius**: Rounded corner values

## Data Flow

### Authentication Flow

```
1. User initiates sign-in (Google OAuth)
2. Platform-specific OAuth flow
3. Exchange code for JWT tokens via API
4. Store tokens securely (Keychain/localStorage)
5. Update auth state across app
6. Auto-refresh tokens when needed
```

### Recording and Analysis Flow

```
1. User records audio (native recording APIs)
2. Create recording entry via API
3. Upload audio file to server
4. Server processes audio through ML pipeline
5. Analysis results returned via API
6. Update UI with results and visualizations
7. Cache results locally for offline viewing
```

## Development Workflow

### Getting Started

1. **Clone Repository**

   ```bash
   git clone <repository-url>
   cd crescendai
   ```

2. **Setup Web Development**

   ```bash
   cd web
   bun install
   bun run dev
   ```

3. **Setup iOS Development**

   ```bash
   cd ios
   open CrescendAI.xcodeproj
   # Configure signing and build
   ```

4. **Setup Shared Dependencies**

   ```bash
   cd shared
   # Shared code is automatically linked to web via bun
   ```

### Development Guidelines

**Code Organization**:

- Keep platform-specific code in respective directories
- Share types, utilities, and business logic via `/shared`
- Use consistent naming conventions across platforms
- Follow platform-specific best practices (SwiftLint, ESLint)

**API Integration**:

- Define API contracts in `/shared/types`
- Implement platform-specific HTTP clients
- Use consistent error handling patterns
- Test API integration on both platforms

**State Management**:

- Keep auth state consistent across platforms
- Use platform-appropriate persistence mechanisms
- Implement proper state cleanup on logout
- Handle network connectivity changes gracefully

### Build and Deployment

**Web Application**:

```bash
cd web
bun run build
bun run preview
# Deploy to Vercel, Netlify, or similar
```

**iOS Application**:

```bash
# In Xcode
1. Archive for distribution
2. Upload to App Store Connect
3. TestFlight distribution for testing
4. App Store release
```

## Testing Strategy

### Web Testing

- **Unit Tests**: Component testing with Vitest
- **Integration Tests**: API integration testing  
- **E2E Tests**: Playwright for user workflows
- **Performance**: Lighthouse CI for web vitals

### iOS Testing

- **Unit Tests**: XCTest for business logic
- **UI Tests**: XCUITest for user interactions
- **Network Tests**: Mock API responses
- **Performance**: Xcode Instruments profiling

### Cross-Platform Testing

- **API Contract Tests**: Ensure consistent API behavior
- **Data Model Tests**: Verify type compatibility
- **Auth Flow Tests**: Test authentication on both platforms

## Security Considerations

### Authentication

- Secure token storage (Keychain/secure localStorage)
- Automatic token refresh
- Proper logout and session cleanup
- OAuth 2.0 with PKCE for mobile

### Data Protection

- HTTPS for all API communication
- Certificate pinning (iOS)
- Content Security Policy (web)
- Input validation and sanitization

### Privacy

- Microphone permission handling
- User data encryption at rest
- GDPR compliance for EU users
- Clear privacy policy and data usage

## Performance Optimizations

### Web

- Code splitting and lazy loading
- Service Worker for offline functionality
- Image optimization and WebP support
- Bundle size optimization

### iOS  

- Lazy loading of view controllers
- Background queue for API requests
- Core Data for local caching
- Memory management and leak prevention

## Monitoring and Analytics

### Error Tracking

- Web: Sentry integration
- iOS: Crashlytics integration
- API: Structured logging

### Performance Monitoring

- Web: Core Web Vitals tracking
- iOS: Performance metrics collection
- API: Response time monitoring

### User Analytics

- Privacy-compliant usage tracking
- Feature adoption metrics
- User journey analysis

## Future Considerations

### Scalability

- CDN integration for global performance
- Database sharding for user data
- Microservices architecture for API
- Auto-scaling for ML pipeline

### Platform Expansion

- Android native application
- Desktop applications (Electron/Tauri)
- API versioning for backward compatibility
- Feature flag system for gradual rollouts

### Technology Evolution

- SwiftUI adoption for iOS
- Web Assembly for performance-critical features
- Real-time collaboration features
- Offline-first architecture

## Conclusion

This architecture provides a solid foundation for CrescendAI's cross-platform development while maintaining code quality, performance, and user experience consistency. The separation of concerns between platforms, shared business logic, and common API ensures maintainability and scalability as the application grows.

For specific implementation details, refer to the README files in each platform directory and the inline code documentation.
