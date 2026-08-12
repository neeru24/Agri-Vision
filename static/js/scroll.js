/**
 * scroll.js
 * Extracted from inline script in templates/index.html (issue #749)
 *
 * Responsibilities:
 *  - Shows #scrollTopBtn when scrollY > 300px, hides otherwise
 *  - Smoothly scrolls to top on button click
 *  - Auto-hides flash alert messages after 5 seconds
 */

// Flash message auto-hide
setTimeout(() => {
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        alert.style.transition = 'opacity 0.5s';
        alert.style.opacity = '0';
        setTimeout(() => alert.remove(), 500);
    });
}, 5000);

// Scroll-to-top button
const scrollBtn = document.getElementById("scrollTopBtn");
if (scrollBtn) {
    window.addEventListener("scroll", () => {
        if (window.scrollY > 300) {
            scrollBtn.style.display = "flex";
        } else {
            scrollBtn.style.display = "none";
        }
    });

    scrollBtn.addEventListener("click", () => {
        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    });
}