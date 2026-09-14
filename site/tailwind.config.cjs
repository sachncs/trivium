/** @type {import('tailwindcss').Config} */
const withAlpha = (variable) => ({ with: (opacity) => `rgb(var(${variable}) / ${opacity})` });

module.exports = {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  darkMode: ["class", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        ink: {
          50: withAlpha("--c-ink-50"),
          100: withAlpha("--c-ink-100"),
          200: withAlpha("--c-ink-200"),
          300: withAlpha("--c-ink-300"),
          400: withAlpha("--c-ink-400"),
          500: withAlpha("--c-ink-500"),
          600: withAlpha("--c-ink-600"),
          700: withAlpha("--c-ink-700"),
          800: withAlpha("--c-ink-800"),
          900: withAlpha("--c-ink-900"),
          950: withAlpha("--c-ink-950"),
        },
        brand: {
          50: withAlpha("--c-brand-50"),
          100: withAlpha("--c-brand-100"),
          200: withAlpha("--c-brand-200"),
          300: withAlpha("--c-brand-300"),
          400: withAlpha("--c-brand-400"),
          500: withAlpha("--c-brand-500"),
          600: withAlpha("--c-brand-600"),
          700: withAlpha("--c-brand-700"),
          800: withAlpha("--c-brand-800"),
          900: withAlpha("--c-brand-900"),
          950: withAlpha("--c-brand-950"),
        },
        accent: {
          50: withAlpha("--c-accent-50"),
          100: withAlpha("--c-accent-100"),
          200: withAlpha("--c-accent-200"),
          300: withAlpha("--c-accent-300"),
          400: withAlpha("--c-accent-400"),
          500: withAlpha("--c-accent-500"),
          600: withAlpha("--c-accent-600"),
          700: withAlpha("--c-accent-700"),
          800: withAlpha("--c-accent-800"),
          900: withAlpha("--c-accent-900"),
        },
        surface: {
          DEFAULT: withAlpha("--c-surface"),
          raised: withAlpha("--c-surface-raised"),
          sunken: withAlpha("--c-surface-sunken"),
          inset: withAlpha("--c-surface-inset"),
        },
        border: {
          DEFAULT: withAlpha("--c-border"),
          subtle: withAlpha("--c-border-subtle"),
          strong: withAlpha("--c-border-strong"),
        },
        text: {
          DEFAULT: withAlpha("--c-text"),
          soft: withAlpha("--c-text-soft"),
          mute: withAlpha("--c-text-mute"),
          faint: withAlpha("--c-text-faint"),
        },
      },
      fontFamily: {
        sans: ["Inter var", "Inter", "system-ui", "-apple-system", "BlinkMacSystemFont", "sans-serif"],
        display: ["Inter Display var", "Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono var", "JetBrains Mono", "ui-monospace", "SF Mono", "Menlo", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.04em" }],
        xs: ["0.75rem", { lineHeight: "1.1rem", letterSpacing: "0.01em" }],
        sm: ["0.875rem", { lineHeight: "1.35rem" }],
        base: ["1rem", { lineHeight: "1.6rem" }],
        lg: ["1.0625rem", { lineHeight: "1.65rem" }],
        xl: ["1.1875rem", { lineHeight: "1.85rem" }],
        "2xl": ["1.4375rem", { lineHeight: "2.05rem" }],
        "3xl": ["1.75rem", { lineHeight: "2.25rem", letterSpacing: "-0.015em" }],
        "4xl": ["2.125rem", { lineHeight: "2.55rem", letterSpacing: "-0.02em" }],
        "5xl": ["2.625rem", { lineHeight: "3rem", letterSpacing: "-0.025em" }],
        "6xl": ["3.25rem", { lineHeight: "3.6rem", letterSpacing: "-0.03em" }],
        "7xl": ["4rem", { lineHeight: "4.25rem", letterSpacing: "-0.035em" }],
        "8xl": ["5rem", { lineHeight: "5.1rem", letterSpacing: "-0.04em" }],
        "9xl": ["6.25rem", { lineHeight: "6.25rem", letterSpacing: "-0.045em" }],
      },
      maxWidth: {
        prose: "68ch",
        screen: "1440px",
        content: "1280px",
        narrow: "960px",
        wide: "1600px",
      },
      borderRadius: {
        "4xl": "2rem",
        "5xl": "2.5rem",
      },
      boxShadow: {
        soft: "0 1px 2px rgb(15 23 42 / 0.04), 0 0 0 1px rgb(15 23 42 / 0.04)",
        card: "0 1px 3px rgb(15 23 42 / 0.04), 0 4px 12px rgb(15 23 42 / 0.04), 0 0 0 1px rgb(15 23 42 / 0.03)",
        pop: "0 4px 12px rgb(15 23 42 / 0.06), 0 16px 36px rgb(15 23 42 / 0.08)",
        lift: "0 10px 30px rgb(15 23 42 / 0.10), 0 30px 80px rgb(15 23 42 / 0.16)",
        glow: "0 0 0 1px rgb(67 56 202 / 0.10), 0 12px 40px -8px rgb(67 56 202 / 0.30)",
        "glow-accent": "0 0 0 1px rgb(234 88 12 / 0.14), 0 12px 40px -8px rgb(234 88 12 / 0.30)",
        inset: "inset 0 1px 0 rgb(255 255 255 / 0.06)",
      },
      backgroundImage: {
        "grid-faint":
          "linear-gradient(to right, rgb(15 23 42 / 0.06) 1px, transparent 1px), linear-gradient(to bottom, rgb(15 23 42 / 0.06) 1px, transparent 1px)",
        "noise":
          "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.35 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")",
      },
      transitionTimingFunction: {
        smooth: "cubic-bezier(0.32, 0.72, 0, 1)",
        spring: "cubic-bezier(0.34, 1.56, 0.64, 1)",
      },
      keyframes: {
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(24px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "scale-in": {
          "0%": { opacity: "0", transform: "scale(0.96)" },
          "100%": { opacity: "1", transform: "scale(1)" },
        },
        "shimmer": {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "marquee": {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
        "drift": {
          "0%, 100%": { transform: "translate(0, 0)" },
          "50%": { transform: "translate(20px, -12px)" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "0.6" },
          "50%": { opacity: "1" },
        },
      },
      animation: {
        "fade-in": "fade-in 0.6s cubic-bezier(0.32, 0.72, 0, 1) both",
        "fade-up": "fade-up 0.8s cubic-bezier(0.32, 0.72, 0, 1) both",
        "scale-in": "scale-in 0.7s cubic-bezier(0.32, 0.72, 0, 1) both",
        "shimmer": "shimmer 3s linear infinite",
        "marquee": "marquee 40s linear infinite",
        "drift": "drift 12s ease-in-out infinite",
        "pulse-soft": "pulse-soft 3s ease-in-out infinite",
      },
    },
  },
  plugins: [require("@tailwindcss/typography"), require("@tailwindcss/forms"), require("@tailwindcss/aspect-ratio")],
};