import { defineConfig } from 'tailwindcss';

export default defineConfig({
  content: ['./src/**/*.{html,js,svelte,ts}'],
  theme: {
    extend: {
      colors: {
        'cream': '#f5f4f0',
        'charcoal': {
          DEFAULT: '#1a1a1a',
          'light': '#363634'
        },
        'sage': {
          DEFAULT: '#9CAF88',
          'light': '#B8C4A6'
        },
        'warm-gray': '#5a5550',
        'paper-white': '#FEFCF7'
      },
      fontFamily: {
        sans: ['Josefin Sans', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
        handwritten: ['Kalam', 'cursive']
      },
      fontSize: {
        'xs': '0.75rem',
        'sm': '0.875rem',
        'base': '1rem',
        'lg': '1.125rem',
        'xl': '1.25rem',
        '2xl': '1.5rem',
        '3xl': '1.875rem',
        '4xl': '2.25rem'
      },
      spacing: {
        'sm': '0.75rem',
        'md': '1rem',
        'lg': '1.5rem',
        'xl': '2rem',
        '2xl': '3rem'
      }
    },
  },
});