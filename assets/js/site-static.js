(function () {
  const root = document.documentElement;
  const saved = localStorage.getItem("theme");
  if (saved) root.dataset.theme = saved;
  else if (window.matchMedia("(prefers-color-scheme: dark)").matches) root.dataset.theme = "dark";

  const themeButton = document.querySelector("[data-theme-toggle]");
  if (themeButton) {
    themeButton.addEventListener("click", function () {
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      localStorage.setItem("theme", next);
    });
  }

  const menuButton = document.querySelector("[data-menu-toggle]");
  const navLinks = document.querySelector(".nav-links");
  if (menuButton && navLinks) menuButton.addEventListener("click", () => navLinks.classList.toggle("open"));

  const filter = document.querySelector("[data-publication-filter]");
  if (filter) {
    filter.addEventListener("input", function () {
      const query = filter.value.toLowerCase().trim();
      document.querySelectorAll("[data-publication]").forEach(function (item) {
        item.classList.toggle("hidden", !item.textContent.toLowerCase().includes(query));
      });
    });
  }
})();
