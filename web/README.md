# CrescendAI Web Application

This directory contains the SvelteKit web application for CrescendAI - an AI-powered piano performance analysis tool.

## Project Structure

```
web/
├── src/
│   ├── lib/                     # Main library directory ($lib alias)
│   │   ├── components/          # Svelte components
│   │   │   ├── ui/              # Basic UI components (Button, Card, etc.)
│   │   │   └── sketchy/         # Styled visualization components
│   │   ├── services/            # API clients and service layer
│   │   ├── stores/              # Zustand state management
│   │   ├── styles/              # Theme, colors, typography
│   │   ├── types/               # TypeScript type definitions
│   │   ├── utils/               # Utility functions
│   │   └── index.ts             # Main library exports
│   ├── routes/                  # SvelteKit routes
│   │   ├── +layout.svelte       # App layout
│   │   └── +page.svelte         # Home page
│   ├── app.css                  # Global styles
│   ├── app.d.ts                 # Type declarations
│   └── app.html                 # HTML template
├── static/                      # Static assets
├── package.json                 # Dependencies and scripts
├── svelte.config.js             # SvelteKit configuration
├── tailwind.config.js           # TailwindCSS configuration
├── tsconfig.json                # TypeScript configuration
└── vite.config.ts               # Vite configuration
```

## Technology Stack

- **Framework**: SvelteKit 2.x with Svelte 5
- **Styling**: TailwindCSS 4.x
- **State Management**: Zustand
- **Build Tool**: Vite + Bun
- **Type Safety**: TypeScript
- **Forms**: @tanstack/svelte-form  
- **API Queries**: @tanstack/svelte-query
- **Package Manager**: Bun

## Development

### Getting Started

1. **Install dependencies**:
   ```bash
   cd web
   bun install
   ```

2. **Start development server**:
   ```bash
   bun run dev
   ```

3. **Build for production**:
   ```bash
   bun run build
   ```

4. **Preview production build**:
   ```bash
   bun run preview
   ```

### Import Patterns

All components and utilities can be imported using the `$lib` alias:

```typescript
// Components
import { Button, Card, Typography } from '$lib';
import FeatureCard from '$lib/components/FeatureCard.svelte';

// Services  
import { apiClient, useAuthStore } from '$lib';

// Types
import type { User, Recording } from '$lib/types';

// Utils
import { formatDuration, validateAudioFile } from '$lib/utils';

// Styles
import { colors, typography } from '$lib/styles';
```

## Key Features

- **Server-side rendering (SSR)** with SvelteKit
- **Progressive Web App (PWA)** capabilities
- **Responsive design** for desktop and mobile browsers
- **Real-time audio visualization** components
- **File upload** with drag-and-drop support
- **Type-safe API integration** with the backend
- **Persistent state management** with Zustand
- **Component-based architecture** following SvelteKit best practices
