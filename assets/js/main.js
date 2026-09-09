// trivium site: small UX touches that don't require a framework.
// - Theme toggle (light/dark) with localStorage persistence.
// - Mobile nav toggle.
// - Smooth-scroll anchors respecting the sticky header.

(function () {
  "use strict";

  const STORAGE_KEY = "trivium-theme";

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
  }

  function initTheme() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "dark" || saved === "light") {
      applyTheme(saved);
    } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
      applyTheme("dark");
    }
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "light";
    const next = current === "dark" ? "light" : "dark";
    applyTheme(next);
    localStorage.setItem(STORAGE_KEY, next);
  }

  function initMobileNav() {
    const toggle = document.querySelector(".site-nav-toggle");
    const mobile = document.getElementById("mobile-nav");
    if (!toggle || !mobile) return;
    toggle.addEventListener("click", function () {
      const expanded = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", String(!expanded));
      if (expanded) {
        mobile.setAttribute("hidden", "");
      } else {
        mobile.removeAttribute("hidden");
      }
    });
    mobile.querySelectorAll("a").forEach(function (a) {
      a.addEventListener("click", function () {
        toggle.setAttribute("aria-expanded", "false");
        mobile.setAttribute("hidden", "");
      });
    });
  }

  function ensureHeaderOffset() {
    // Account for the sticky header on in-page anchors.
    document.querySelectorAll('a[href^="#"]').forEach(function (a) {
      a.addEventListener("click", function (e) {
        const id = a.getAttribute("href");
        if (!id || id === "#") return;
        const target = document.querySelector(id);
        if (!target) return;
        e.preventDefault();
        const offset = document.querySelector(".site-header").offsetHeight + 12;
        const y = target.getBoundingClientRect().top + window.scrollY - offset;
        window.scrollTo({ top: y, behavior: "smooth" });
        history.pushState(null, "", id);
      });
    });
  }

  function addThemeToggle() {
    const nav = document.querySelector(".site-nav-list");
    if (!nav) return;
    const li = document.createElement("li");
    li.className = "site-nav-item";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "theme-toggle";
    btn.setAttribute("aria-label", "Toggle colour theme");
    btn.textContent = document.documentElement.getAttribute("data-theme") === "dark" ? "☀" : "☾";
    btn.style.cssText = "background:transparent;border:1px solid var(--c-border);padding:6px 10px;border-radius:var(--radius-sm);cursor:pointer;font-size:14px;color:var(--c-text-soft);";
    btn.addEventListener("click", function () {
      toggleTheme();
      btn.textContent = document.documentElement.getAttribute("data-theme") === "dark" ? "☀" : "☾";
    });
    li.appendChild(btn);
    nav.insertBefore(li, nav.lastElementChild);
  }

  initTheme();
  initMobileNav();
  ensureHeaderOffset();
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", addThemeToggle);
  } else {
    addThemeToggle();
  }
})();
