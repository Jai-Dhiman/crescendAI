/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.rs",
    "./dist/client/**/*.html",
  ],
  theme: {
    extend: {
      // COLOR PALETTE - Academic Paper Aesthetic
      colors: {
        // Primary: Sepia/Brown spectrum (scholarly accent)
        sepia: {
          50:  '#fdfcfa',
          100: '#f9f6f0',
          200: '#f0e9dc',
          300: '#e3d5c0',
          400: '#c9b89a',
          500: '#a69276',  // Primary accent
          600: '#8b7355',  // Darker sepia
          700: '#6e5a43',
          800: '#554535',
          900: '#3d3228',
          950: '#251f1a',
        },
        // Paper backgrounds (warm whites, parchment)
        paper: {
          50:  '#fefdfb',  // Lightest cream
          100: '#fbf9f5',  // Warm white
          200: '#f5f1e8',  // Light parchment
          300: '#ede6d9',  // Parchment
          400: '#e0d5c3',  // Aged paper
          500: '#cfc3ad',  // Darker parchment
        },
        // Ink tones (text colors)
        ink: {
          50:  '#f8f7f6',
          100: '#edeae6',
          200: '#d9d4cc',
          300: '#b8b0a3',
          400: '#968c7d',
          500: '#746a5c',  // Secondary text
          600: '#5a524a',  // Body text
          700: '#433d38',  // Primary text
          800: '#2d2926',  // Headings
          900: '#1a1816',  // Darkest
        },
        // Highlight colors (for metrics, accents)
        highlight: {
          light: '#fcf8e8',
          DEFAULT: '#c9a227',  // Scholarly gold
          dark: '#9a7b1a',
        },
        // Legacy stone (keeping for compatibility during transition)
        stone: {
          50:  '#fafaf9',
          100: '#f5f5f4',
          150: '#eeede9',
          200: '#e7e5e4',
          300: '#d6d3d1',
          400: '#a8a29e',
          500: '#78716c',
          600: '#57534e',
          700: '#44403c',
          800: '#292524',
          900: '#1c1917',
          950: '#0c0a09',
        },
        // Semantic colors with warm undertones
        success: {
          light: '#ecfdf5',
          DEFAULT: '#059669',
          dark: '#047857',
        },
        warning: {
          light: '#fffbeb',
          DEFAULT: '#d97706',
          dark: '#b45309',
        },
        error: {
          light: '#fef2f2',
          DEFAULT: '#dc2626',
          dark: '#b91c1c',
        },
        info: {
          light: '#f0f9ff',
          DEFAULT: '#0284c7',
          dark: '#0369a1',
        },
      },

      // TYPOGRAPHY - Scholarly + Modern
      fontFamily: {
        display: ['"Lora"', 'Georgia', 'serif'],
        serif: ['"Lora"', 'Georgia', 'serif'],
        sans: ['"Inter"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'Consolas', 'monospace'],
      },
      fontSize: {
        // Display sizes for hero sections
        'display-2xl': ['4.5rem', { lineHeight: '1', letterSpacing: '-0.02em', fontWeight: '500' }],
        'display-xl': ['3.75rem', { lineHeight: '1.05', letterSpacing: '-0.02em', fontWeight: '500' }],
        'display-lg': ['3rem', { lineHeight: '1.1', letterSpacing: '-0.01em', fontWeight: '500' }],
        'display-md': ['2.25rem', { lineHeight: '1.15', letterSpacing: '-0.01em', fontWeight: '500' }],
        'display-sm': ['1.875rem', { lineHeight: '1.2', letterSpacing: '0', fontWeight: '500' }],
        // Heading sizes
        'heading-xl': ['1.5rem', { lineHeight: '1.3', fontWeight: '600' }],
        'heading-lg': ['1.25rem', { lineHeight: '1.4', fontWeight: '600' }],
        'heading-md': ['1.125rem', { lineHeight: '1.4', fontWeight: '600' }],
        'heading-sm': ['1rem', { lineHeight: '1.5', fontWeight: '600' }],
        // Body sizes
        'body-lg': ['1.125rem', { lineHeight: '1.75' }],
        'body-md': ['1rem', { lineHeight: '1.75' }],
        'body-sm': ['0.875rem', { lineHeight: '1.6' }],
        'body-xs': ['0.75rem', { lineHeight: '1.5' }],
        // Label/caption
        'label-lg': ['0.875rem', { lineHeight: '1.4', fontWeight: '500', letterSpacing: '0.025em' }],
        'label-md': ['0.75rem', { lineHeight: '1.4', fontWeight: '500', letterSpacing: '0.05em' }],
        'label-sm': ['0.6875rem', { lineHeight: '1.4', fontWeight: '500', letterSpacing: '0.075em' }],
      },

      // SPACING SCALE
      spacing: {
        '0.5': '0.125rem',
        '1': '0.25rem',
        '1.5': '0.375rem',
        '2': '0.5rem',
        '2.5': '0.625rem',
        '3': '0.75rem',
        '3.5': '0.875rem',
        '4': '1rem',
        '5': '1.25rem',
        '6': '1.5rem',
        '7': '1.75rem',
        '8': '2rem',
        '9': '2.25rem',
        '10': '2.5rem',
        '11': '2.75rem',
        '12': '3rem',
        '14': '3.5rem',
        '16': '4rem',
        '18': '4.5rem',
        '20': '5rem',
        '24': '6rem',
        '28': '7rem',
        '32': '8rem',
        '36': '9rem',
        '40': '10rem',
        '48': '12rem',
        '56': '14rem',
        '64': '16rem',
      },

      // BORDER RADIUS - Slightly more rounded for warmth
      borderRadius: {
        'none': '0',
        'xs': '0.125rem',
        'sm': '0.1875rem',
        'DEFAULT': '0.25rem',
        'md': '0.375rem',
        'lg': '0.5rem',
        'xl': '0.75rem',
        '2xl': '1rem',
        'full': '9999px',
      },

      // SHADOWS & ELEVATION - Warmer shadows
      boxShadow: {
        'elevation-1': '0 1px 2px 0 rgba(37, 31, 26, 0.04)',
        'elevation-2': '0 1px 3px 0 rgba(37, 31, 26, 0.06), 0 1px 2px -1px rgba(37, 31, 26, 0.06)',
        'elevation-3': '0 4px 6px -1px rgba(37, 31, 26, 0.06), 0 2px 4px -2px rgba(37, 31, 26, 0.04)',
        'elevation-4': '0 10px 15px -3px rgba(37, 31, 26, 0.06), 0 4px 6px -4px rgba(37, 31, 26, 0.04)',
        'elevation-5': '0 20px 25px -5px rgba(37, 31, 26, 0.08), 0 8px 10px -6px rgba(37, 31, 26, 0.04)',
        'elevation-6': '0 25px 50px -12px rgba(37, 31, 26, 0.15)',
        'button': '0 1px 2px 0 rgba(37, 31, 26, 0.04), 0 1px 3px 0 rgba(37, 31, 26, 0.08)',
        'button-hover': '0 4px 8px -2px rgba(37, 31, 26, 0.10), 0 2px 4px -2px rgba(37, 31, 26, 0.06)',
        'button-active': '0 1px 2px 0 rgba(37, 31, 26, 0.06)',
        'card': '0 1px 3px 0 rgba(37, 31, 26, 0.04), 0 1px 2px -1px rgba(37, 31, 26, 0.04)',
        'card-hover': '0 8px 16px -4px rgba(37, 31, 26, 0.08), 0 4px 6px -2px rgba(37, 31, 26, 0.04)',
        'sepia': '0 4px 14px -2px rgba(166, 146, 118, 0.20)',
        'sepia-lg': '0 8px 24px -4px rgba(166, 146, 118, 0.25)',
        'inner-sm': 'inset 0 1px 2px 0 rgba(37, 31, 26, 0.04)',
        'inner': 'inset 0 2px 4px 0 rgba(37, 31, 26, 0.06)',
        'glow': '0 0 20px rgba(201, 162, 39, 0.15)',
      },

      // TRANSITIONS & ANIMATIONS
      transitionDuration: {
        '75': '75ms',
        '100': '100ms',
        '150': '150ms',
        '200': '200ms',
        '250': '250ms',
        '300': '300ms',
        '400': '400ms',
        '500': '500ms',
        '600': '600ms',
        '700': '700ms',
      },
      transitionTimingFunction: {
        'ease-out-expo': 'cubic-bezier(0.16, 1, 0.3, 1)',
        'ease-out-quart': 'cubic-bezier(0.25, 1, 0.5, 1)',
        'ease-in-out-quart': 'cubic-bezier(0.76, 0, 0.24, 1)',
        'spring': 'cubic-bezier(0.34, 1.56, 0.64, 1)',
        'smooth': 'cubic-bezier(0.4, 0, 0.2, 1)',
      },
      animation: {
        'fade-in': 'fadeIn 500ms cubic-bezier(0.16, 1, 0.3, 1)',
        'fade-in-up': 'fadeInUp 600ms cubic-bezier(0.16, 1, 0.3, 1)',
        'fade-in-down': 'fadeInDown 500ms cubic-bezier(0.16, 1, 0.3, 1)',
        'scale-in': 'scaleIn 300ms cubic-bezier(0.34, 1.56, 0.64, 1)',
        'scale-out': 'scaleOut 200ms cubic-bezier(0.4, 0, 0.2, 1)',
        'slide-in-right': 'slideInRight 400ms cubic-bezier(0.16, 1, 0.3, 1)',
        'slide-in-left': 'slideInLeft 400ms cubic-bezier(0.16, 1, 0.3, 1)',
        'slide-in-up': 'slideInUp 400ms cubic-bezier(0.16, 1, 0.3, 1)',
        'stagger-1': 'fadeInUp 600ms cubic-bezier(0.16, 1, 0.3, 1) 100ms both',
        'stagger-2': 'fadeInUp 600ms cubic-bezier(0.16, 1, 0.3, 1) 200ms both',
        'stagger-3': 'fadeInUp 600ms cubic-bezier(0.16, 1, 0.3, 1) 300ms both',
        'stagger-4': 'fadeInUp 600ms cubic-bezier(0.16, 1, 0.3, 1) 400ms both',
        'stagger-5': 'fadeInUp 600ms cubic-bezier(0.16, 1, 0.3, 1) 500ms both',
        'stagger-6': 'fadeInUp 600ms cubic-bezier(0.16, 1, 0.3, 1) 600ms both',
        'spin-slow': 'spin 2s linear infinite',
        'pulse-subtle': 'pulseSubtle 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'shimmer': 'shimmer 2s infinite',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeInDown: {
          '0%': { opacity: '0', transform: 'translateY(-16px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        scaleIn: {
          '0%': { opacity: '0', transform: 'scale(0.95)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        scaleOut: {
          '0%': { opacity: '1', transform: 'scale(1)' },
          '100%': { opacity: '0', transform: 'scale(0.95)' },
        },
        slideInRight: {
          '0%': { opacity: '0', transform: 'translateX(24px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        slideInLeft: {
          '0%': { opacity: '0', transform: 'translateX(-24px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        slideInUp: {
          '0%': { opacity: '0', transform: 'translateY(24px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseSubtle: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.6' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },

      // BACKGROUND IMAGES
      backgroundImage: {
        'gradient-sepia': 'linear-gradient(135deg, #a69276 0%, #8b7355 100%)',
        'gradient-sepia-subtle': 'linear-gradient(135deg, #f0e9dc 0%, #e3d5c0 100%)',
        'gradient-paper': 'linear-gradient(180deg, #fefdfb 0%, #f5f1e8 100%)',
        'gradient-warm': 'linear-gradient(180deg, #fbf9f5 0%, #ede6d9 100%)',
        'gradient-page': 'linear-gradient(180deg, #fefdfb 0%, #f5f1e8 100%)',
        'gradient-cta': 'linear-gradient(135deg, #8b7355 0%, #6e5a43 100%)',
        'gradient-hero': 'linear-gradient(180deg, #fdf8f0 0%, #f6e9d5 50%, #f0dfc4 100%)',
        'gradient-warm-mid': 'linear-gradient(180deg, #f6e9d5 0%, #eedcca 100%)',
        'gradient-warm-deep': 'linear-gradient(180deg, #eedcca 0%, #e8d5b8 100%)',
        'gradient-warm-rich': 'linear-gradient(180deg, #e8d5b8 0%, #dfc9a8 100%)',
        'noise': "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E\")",
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}
