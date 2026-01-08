/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.rs",
    "./dist/client/**/*.html",
  ],
  theme: {
    extend: {
      // COLOR PALETTE
      colors: {
        // Primary: Gold/Amber spectrum
        gold: {
          50:  '#fffbeb',
          100: '#fef3c7',
          200: '#fde68a',
          300: '#fcd34d',
          400: '#fbbf24',
          500: '#d4a012',  // Primary accent
          600: '#b8860b',  // Darker gold (DarkGoldenrod)
          700: '#92650a',
          800: '#774d0a',
          900: '#5c3a08',
          950: '#3d2506',
        },
        // Secondary: Deep burgundy for contrast and sophistication
        burgundy: {
          50:  '#fdf2f4',
          100: '#fce7ea',
          200: '#f9d0d8',
          300: '#f4a9b8',
          400: '#ec7a93',
          500: '#df4d6f',
          600: '#c92d54',
          700: '#a82145',
          800: '#8c1e3d',
          900: '#771c38',
          950: '#420a1a',
        },
        // Neutrals: Warm stone palette
        stone: {
          50:  '#fafaf9',
          100: '#f5f5f4',
          150: '#eeede9',  // Custom: subtle background
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
        // Background tones
        cream: {
          50:  '#fefdfb',
          100: '#fdfbf7',
          200: '#f9f5ed',
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

      // TYPOGRAPHY
      fontFamily: {
        display: ['"Cormorant Garamond"', 'Georgia', 'serif'],
        serif: ['"Cormorant Garamond"', 'Georgia', 'serif'],
        sans: ['"DM Sans"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'Consolas', 'monospace'],
      },
      fontSize: {
        // Display sizes for hero sections
        'display-2xl': ['4.5rem', { lineHeight: '1', letterSpacing: '-0.02em', fontWeight: '600' }],
        'display-xl': ['3.75rem', { lineHeight: '1.05', letterSpacing: '-0.02em', fontWeight: '600' }],
        'display-lg': ['3rem', { lineHeight: '1.1', letterSpacing: '-0.01em', fontWeight: '600' }],
        'display-md': ['2.25rem', { lineHeight: '1.15', letterSpacing: '-0.01em', fontWeight: '600' }],
        'display-sm': ['1.875rem', { lineHeight: '1.2', letterSpacing: '0', fontWeight: '600' }],
        // Heading sizes
        'heading-xl': ['1.5rem', { lineHeight: '1.3', fontWeight: '600' }],
        'heading-lg': ['1.25rem', { lineHeight: '1.4', fontWeight: '600' }],
        'heading-md': ['1.125rem', { lineHeight: '1.4', fontWeight: '600' }],
        'heading-sm': ['1rem', { lineHeight: '1.5', fontWeight: '600' }],
        // Body sizes
        'body-lg': ['1.125rem', { lineHeight: '1.7' }],
        'body-md': ['1rem', { lineHeight: '1.7' }],
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

      // BORDER RADIUS
      borderRadius: {
        'none': '0',
        'xs': '0.0625rem',
        'sm': '0.125rem',
        'DEFAULT': '0.1875rem',
        'md': '0.25rem',
        'lg': '0.375rem',
        'xl': '0.5rem',
        'full': '9999px',
      },

      // SHADOWS & ELEVATION
      boxShadow: {
        'elevation-1': '0 1px 2px 0 rgba(28, 25, 23, 0.05)',
        'elevation-2': '0 1px 3px 0 rgba(28, 25, 23, 0.08), 0 1px 2px -1px rgba(28, 25, 23, 0.08)',
        'elevation-3': '0 4px 6px -1px rgba(28, 25, 23, 0.08), 0 2px 4px -2px rgba(28, 25, 23, 0.06)',
        'elevation-4': '0 10px 15px -3px rgba(28, 25, 23, 0.08), 0 4px 6px -4px rgba(28, 25, 23, 0.06)',
        'elevation-5': '0 20px 25px -5px rgba(28, 25, 23, 0.10), 0 8px 10px -6px rgba(28, 25, 23, 0.06)',
        'elevation-6': '0 25px 50px -12px rgba(28, 25, 23, 0.20)',
        'button': '0 1px 2px 0 rgba(28, 25, 23, 0.06), 0 1px 3px 0 rgba(28, 25, 23, 0.10)',
        'button-hover': '0 4px 8px -2px rgba(28, 25, 23, 0.12), 0 2px 4px -2px rgba(28, 25, 23, 0.08)',
        'button-active': '0 1px 2px 0 rgba(28, 25, 23, 0.08)',
        'card': '0 1px 3px 0 rgba(28, 25, 23, 0.06), 0 1px 2px -1px rgba(28, 25, 23, 0.06)',
        'card-hover': '0 8px 16px -4px rgba(28, 25, 23, 0.10), 0 4px 6px -2px rgba(28, 25, 23, 0.06)',
        'gold': '0 4px 14px -2px rgba(212, 160, 18, 0.25)',
        'gold-lg': '0 8px 24px -4px rgba(212, 160, 18, 0.30)',
        'inner-sm': 'inset 0 1px 2px 0 rgba(28, 25, 23, 0.05)',
        'inner': 'inset 0 2px 4px 0 rgba(28, 25, 23, 0.08)',
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
        'fade-in': 'fadeIn 400ms cubic-bezier(0.16, 1, 0.3, 1)',
        'fade-in-up': 'fadeInUp 500ms cubic-bezier(0.16, 1, 0.3, 1)',
        'fade-in-down': 'fadeInDown 500ms cubic-bezier(0.16, 1, 0.3, 1)',
        'scale-in': 'scaleIn 300ms cubic-bezier(0.34, 1.56, 0.64, 1)',
        'scale-out': 'scaleOut 200ms cubic-bezier(0.4, 0, 0.2, 1)',
        'slide-in-right': 'slideInRight 400ms cubic-bezier(0.16, 1, 0.3, 1)',
        'slide-in-left': 'slideInLeft 400ms cubic-bezier(0.16, 1, 0.3, 1)',
        'slide-in-up': 'slideInUp 400ms cubic-bezier(0.16, 1, 0.3, 1)',
        'stagger-1': 'fadeInUp 500ms cubic-bezier(0.16, 1, 0.3, 1) 100ms both',
        'stagger-2': 'fadeInUp 500ms cubic-bezier(0.16, 1, 0.3, 1) 200ms both',
        'stagger-3': 'fadeInUp 500ms cubic-bezier(0.16, 1, 0.3, 1) 300ms both',
        'stagger-4': 'fadeInUp 500ms cubic-bezier(0.16, 1, 0.3, 1) 400ms both',
        'spin-slow': 'spin 2s linear infinite',
        'pulse-subtle': 'pulseSubtle 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'shimmer': 'shimmer 2s infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(16px)' },
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
      },

      // ADDITIONAL UTILITIES
      backgroundImage: {
        'gradient-gold': 'linear-gradient(135deg, #d4a012 0%, #b8860b 100%)',
        'gradient-gold-subtle': 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)',
        'gradient-warm': 'linear-gradient(180deg, #fdfbf7 0%, #f9f5ed 100%)',
        'gradient-page': 'linear-gradient(180deg, #fefdfb 0%, #f5f5f4 100%)',
        'noise': "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%' height='100%' filter='url(%23noise)'/%3E%3C/svg%3E\")",
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}
