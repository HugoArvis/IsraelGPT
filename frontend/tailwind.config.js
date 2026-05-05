/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          bg: "#0d1117",
          card: "#161b22",
          border: "#30363d",
          accent: "#58a6ff",
          green: "#3fb950",
          red: "#f85149",
          yellow: "#d29922",
        },
      },
    },
  },
  plugins: [],
};
