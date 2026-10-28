/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f0f6ff',
          100: '#dbe9ff',
          500: '#3366ff',
          600: '#2451db',
          700: '#1c3fae',
        },
      },
    },
  },
  plugins: [],
}
