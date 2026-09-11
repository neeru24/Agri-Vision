/**
 * theme.js
 * Extracted from inline script in templates/index.html (issue #749)
 *
 * Responsibilities:
 *  - Reads saved theme from localStorage on DOMContentLoaded
 *  - Applies it to <html data-theme="...">
 *  - Toggles the moon/sun icon on #theme-toggle
 *  - Listens for the toggle button click
 */

document.addEventListener("DOMContentLoaded", () => {
    const themeToggleBtn =
        document.getElementById("theme-toggle");

    function updateThemeIcon(theme) {

        const icon =
            themeToggleBtn.querySelector("i");

        if (theme === "dark") {

            icon.classList.remove("fa-moon");
            icon.classList.add("fa-sun");

        } else {

            icon.classList.remove("fa-sun");
            icon.classList.add("fa-moon");
        }
    }

    // Load saved theme
    const savedTheme =
        localStorage.getItem("theme") || "light";

    document.documentElement.setAttribute(
        "data-theme",
        savedTheme
    );

    updateThemeIcon(savedTheme);

    // Toggle theme
    themeToggleBtn.addEventListener("click", () => {

        const currentTheme =
            document.documentElement.getAttribute(
                "data-theme"
            );

        const newTheme =
            currentTheme === "dark"
            ? "light"
            : "dark";

        document.documentElement.setAttribute(
            "data-theme",
            newTheme
        );

        localStorage.setItem(
            "theme",
            newTheme
        );

        updateThemeIcon(newTheme);
    });

});