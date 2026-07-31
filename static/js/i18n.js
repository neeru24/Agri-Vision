/**
 * Agri-Vision i18n Engine
 * Handles dynamic language switching and localization persistence
 */
const I18nEngine = (function() {
    let currentLang = localStorage.getItem('i18n_lang') || 'en';
    const cache = {};
    const supportedLangs = ['en', 'hi', 'gu', 'mr', 'ta', 'te', 'pa'];

    // Helper to get nested object property
    function getNestedProperty(obj, path) {
        return path.split('.').reduce((acc, part) => acc && acc[part], obj);
    }

    // Apply translations to all elements with data-i18n attribute
    function applyTranslations(translations) {
        const elements = document.querySelectorAll('[data-i18n]');
        elements.forEach(el => {
            const key = el.getAttribute('data-i18n');
            const translation = getNestedProperty(translations, key);
            if (translation) {
                // If element has placeholder attribute (e.g. inputs)
                if (el.hasAttribute('placeholder')) {
                    el.setAttribute('placeholder', translation);
                } else if (el.tagName === 'INPUT' && el.type === 'submit') {
                    el.value = translation;
                } else {
                    el.textContent = translation;
                }
            }
        });
        
        // Dispatch custom event so other components (e.g., charts) can react
        const event = new CustomEvent('i18nLanguageChanged', { detail: { lang: currentLang } });
        document.dispatchEvent(event);
    }

    async function loadLanguage(lang) {
        if (!supportedLangs.includes(lang)) {
            lang = 'en';
        }

        document.documentElement.lang = lang;
        currentLang = lang;
        localStorage.setItem('i18n_lang', lang);
        
        // Update language selector dropdown if present
        const selector = document.getElementById('lang-selector');
        if (selector && selector.value !== lang) {
            selector.value = lang;
        }

        if (cache[lang]) {
            applyTranslations(cache[lang]);
            return;
        }

        try {
            const response = await fetch(`/static/locales/${lang}.json`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const translations = await response.json();
            cache[lang] = translations;
            applyTranslations(translations);
        } catch (error) {
            console.error('Error loading translation file:', error);
            // Fallback to English if translation file fails to load
            if (lang !== 'en') {
                loadLanguage('en');
            }
        }
    }

    // Initialization
    function init() {
        // Find language selector and bind event
        const selector = document.getElementById('lang-selector');
        if (selector) {
            selector.value = currentLang;
            selector.addEventListener('change', (e) => {
                loadLanguage(e.target.value);
            });
        }
        
        // Load initial language
        loadLanguage(currentLang);
    }

    // Expose public API
    return {
        init,
        setLanguage: loadLanguage,
        getCurrentLang: () => currentLang,
        getTranslation: (key) => {
            if (cache[currentLang]) {
                return getNestedProperty(cache[currentLang], key) || key;
            }
            return key;
        }
    };
})();

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    I18nEngine.init();
});
