/* Add descriptive native tooltips to the repository and home-logo links.
   Re-apply after Material's instant-navigation updates. */
(() => {
  const addDanFlowTooltips = () => {
    const repositoryLink = document.querySelector(
      ".md-header__source a.md-source, .md-header__source a"
    );

    if (repositoryLink) {
      repositoryLink.setAttribute(
        "title",
        "Open the DanFlow repository on GitHub"
      );
      repositoryLink.setAttribute(
        "aria-label",
        "Open the DanFlow repository on GitHub"
      );
    }

    const homeLogoLink = document.querySelector(
      "a.md-header__button.md-logo, a.md-logo"
    );

    if (homeLogoLink) {
      homeLogoLink.setAttribute("title", "DanFlow home");
      homeLogoLink.setAttribute("aria-label", "DanFlow home");
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", addDanFlowTooltips, {
      once: true
    });
  } else {
    addDanFlowTooltips();
  }

  // MkDocs Material can swap page content without a full page reload.
  if ("MutationObserver" in window && document.documentElement) {
    const observer = new MutationObserver(addDanFlowTooltips);
    observer.observe(document.documentElement, {
      childList: true,
      subtree: true
    });
  }
})();