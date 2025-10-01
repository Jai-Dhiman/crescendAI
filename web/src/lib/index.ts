// Main library exports for CrescendAI web app
// Following SvelteKit best practices for lib organization

// Types
export * from './types/index.js';

// Stores
export * from './stores/index.js';

// Utilities
export * from './utils/index.js';

// Styles/Theme
export * from './styles/index.js';

// Services
export * from './services/index.js';

// Components - using explicit exports for better tree shaking
export { default as Button } from './components/Button.svelte';
export { default as Card } from './components/Card.svelte';
export { default as FeatureCard } from './components/FeatureCard.svelte';
export { default as SketchyPianoKeys } from './components/SketchyPianoKeys.svelte';
export { default as SketchySheetMusic } from './components/SketchySheetMusic.svelte';
export { default as SketchyWaveform } from './components/SketchyWaveform.svelte';
export { default as AnalysisVisualization } from './components/AnalysisVisualization.svelte';

// UI Components
export { default as EnhancedButton } from './components/ui/EnhancedButton.svelte';
export { default as UICard } from './components/ui/Card.svelte';
export { default as CardHeader } from './components/ui/CardHeader.svelte';
export { default as CardContent } from './components/ui/CardContent.svelte';
export { default as CardFooter } from './components/ui/CardFooter.svelte';
export { default as Typography } from './components/ui/Typography.svelte';

// Sketchy Components
export { default as SketchyFeatureCard } from './components/sketchy/FeatureCard.svelte';
