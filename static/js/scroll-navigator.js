(function() {
    'use strict';

    function initScrollNavigator() {
        var container = document.querySelector('.scroll-navigator');
        if (!container) return;

        var scrollUpBtn = container.querySelector('.scroll-nav-btn[data-direction="up"]');
        var scrollDownBtn = container.querySelector('.scroll-nav-btn[data-direction="down"]');

        function updateVisibility() {
            var scrollY = window.scrollY;
            var maxScroll = document.documentElement.scrollHeight - window.innerHeight;

            if (scrollY > 300) {
                container.classList.add('visible');
            } else {
                container.classList.remove('visible');
            }

            if (scrollUpBtn) {
                if (scrollY <= 300) {
                    scrollUpBtn.style.opacity = '0.5';
                    scrollUpBtn.style.pointerEvents = 'none';
                } else {
                    scrollUpBtn.style.opacity = '1';
                    scrollUpBtn.style.pointerEvents = 'auto';
                }
            }

            if (scrollDownBtn) {
                if (maxScroll - scrollY <= 100) {
                    scrollDownBtn.style.opacity = '0.5';
                    scrollDownBtn.style.pointerEvents = 'none';
                } else {
                    scrollDownBtn.style.opacity = '1';
                    scrollDownBtn.style.pointerEvents = 'auto';
                }
            }
        }

        function scrollUp() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        function scrollDown() {
            window.scrollTo({
                top: document.documentElement.scrollHeight,
                behavior: 'smooth'
            });
        }

        if (scrollUpBtn) {
            scrollUpBtn.addEventListener('click', scrollUp);
        }

        if (scrollDownBtn) {
            scrollDownBtn.addEventListener('click', scrollDown);
        }

        window.addEventListener('scroll', updateVisibility, { passive: true });
        window.addEventListener('resize', updateVisibility, { passive: true });

        updateVisibility();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initScrollNavigator);
    } else {
        initScrollNavigator();
    }
})();
