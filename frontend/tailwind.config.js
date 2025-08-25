/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // your existing
        darkGlass: "rgba(255, 255, 255, 0.05)",
        glassBorder: "rgba(255, 255, 255, 0.15)",
        cryptoBlue: "#0d0d2b",
        neonGreen: "#39ff14",
        tiktokBlue: "#00f2ea",
        xBlue: "#1DA1F2",

        // logo palette
        ink: { 900: "#07080e", 800: "#191b30", 600: "#314483" },
        brand: {
          cyan: "#21b4fb",
          blue: "#2078f7",
          violet: "#4237f8",
          purple: "#8a17e4",
          pink: "#d31ab2",
        },
      },
      backgroundImage: {
        "brand-gradient": "var(--grad-brand)",
        "night-sky": "var(--bg-sky)",
      },
      backdropBlur: {
        xs: "2px",
      },
    },
  },
  plugins: [],
};
