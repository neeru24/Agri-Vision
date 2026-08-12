/**
 * mouse-trail.js
 * Extracted from inline script in templates/index.html (issue #749)
 *
 * Responsibilities:
 *  - Only runs on non-touch, fine-pointer (desktop) devices
 *  - Appends 15 .trail-dot elements to .trail-container
 *  - Each dot follows the cursor with easing (lerp factor 0.35)
 *  - Dots scale and fade based on their position in the chain
 *
 * Requires: a <div class="trail-container"> in the HTML (already present)
 * and corresponding CSS for .trail-dot in style.css
 */

if (!('ontouchstart' in window) && window.matchMedia('(pointer: fine)').matches) {
    document.addEventListener("DOMContentLoaded", () => {
        const container = document.querySelector(".trail-container");
        const dots = [];
        const totalDots = 15;
        let mouseX = 0;
        let mouseY = 0;

        for (let i = 0; i < totalDots; i++) {
            const dot = document.createElement("div");
            dot.classList.add("trail-dot");
            container.appendChild(dot);
            dots.push({ el: dot, x: 0, y: 0 });
        }

        document.addEventListener("mousemove", (e) => {
            mouseX = e.clientX;
            mouseY = e.clientY;
        });

        function animate() {
            let x = mouseX;
            let y = mouseY;
            dots.forEach((dot, index) => {
                dot.x += (x - dot.x) * 0.35;
                dot.y += (y - dot.y) * 0.35;
                dot.el.style.left = dot.x + "px";
                dot.el.style.top = dot.y + "px";
                const scale = (totalDots - index) / totalDots;
                dot.el.style.transform = `translate(-50%, -50%) scale(${scale})`;
                dot.el.style.opacity = scale;
                x = dot.x;
                y = dot.y;
            });
            requestAnimationFrame(animate);
        }
        animate();
    });
}