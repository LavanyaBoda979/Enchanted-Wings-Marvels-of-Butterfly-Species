/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        butterfly: {
          50: '#fef7ee',
          100: '#fdedd3',
          200: '#fbd7a5',
          300: '#f8bb6d',
          400: '#f59532',
          500: '#f2770a',
          600: '#e35d05',
          700: '#bc4508',
          800: '#95370e',
          900: '#782f0f',
        },
        nature: {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
        }
      },
      fontFamily: {
        'display': ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'flutter': 'flutter 3s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        flutter: {
          '0%, 100%': { transform: 'rotate(0deg) scale(1)' },
          '25%': { transform: 'rotate(2deg) scale(1.05)' },
          '75%': { transform: 'rotate(-2deg) scale(0.95)' },
        }
      }
    },
  },
  plugins: [],
}