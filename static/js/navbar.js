/**
 * navbar.js
 * Extracted from inline script in templates/index.html (issue #749)
 *
 * Responsibilities:
 *  - Adds/removes class "scrolled" on #navbar when page is scrolled > 50px
 *  - Toggles mobile nav menu (#nav-menu) open/close via #nav-toggle button
 *  - Swaps hamburger icon (fa-bars <-> fa-times) on toggle
 *  - Closes the menu when any nav link is clicked
 *  - Closes the menu when clicking outside the nav
 */

// Navbar scroll effect
const navbar = document.getElementById('navbar');
if (navbar) {
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });
}

// Mobile Navigation Toggle
const navToggle = document.getElementById('nav-toggle');
const navMenu = document.getElementById('nav-menu');

if (navToggle && navMenu) {
    navToggle.addEventListener('click', () => {
        navMenu.classList.toggle('active');
        const icon = navToggle.querySelector('i');
        if (!icon) return;
        if (navMenu.classList.contains('active')) {
            icon.classList.remove('fa-bars');
            icon.classList.add('fa-times');
        } else {
            icon.classList.remove('fa-times');
            icon.classList.add('fa-bars');
        }
    });

    // Close menu when clicking on a link
    navMenu.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            navMenu.classList.remove('active');
            const icon = navToggle.querySelector('i');
            if (!icon) return;
            icon.classList.remove('fa-times');
            icon.classList.add('fa-bars');
        });
    });

    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!navMenu.contains(e.target) && !navToggle.contains(e.target)) {
            navMenu.classList.remove('active');
            const icon = navToggle.querySelector('i');
            if (!icon) return;
            icon.classList.remove('fa-times');
            icon.classList.add('fa-bars');
        }
    });
}