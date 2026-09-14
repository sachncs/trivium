// @ts-check
import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";

const SITE_URL = "https://sachncs.github.io";
const REPO = "trivium";
const BASE_PATH = `/${REPO}`;

export default defineConfig({
  site: `${SITE_URL}${BASE_PATH}`,
  base: BASE_PATH,
  trailingSlash: "always",
  build: {
    format: "directory",
    inlineStylesheets: "auto",
    assets: "_assets",
  },
  integrations: [
    sitemap({
      changefreq: "monthly",
      priority: 0.7,
    }),
  ],
  image: {
    service: { entrypoint: "astro/assets/services/sharp" },
  },
  vite: {
    css: {
      postcss: "./postcss.config.cjs",
    },
  },
  compressHTML: true,
});