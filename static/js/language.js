async function loadLanguage(lang) {

    const response =
        await fetch(`/static/lang/${lang}.json`);

    if (!response.ok) throw new Error("Request failed");
const translations =
        await response.json();

    document.querySelectorAll("[data-i18n]")
    .forEach(element => {

        const key =
            element.getAttribute("data-i18n");

        if (translations[key]) {

            element.textContent =
                translations[key];
            const icon =
                element.querySelector("i");

            if (icon) {

                const iconHTML =
                    icon.outerHTML;

                element.innerHTML =
                    `${iconHTML} ${translations[key]}`;

            } else {

                element.textContent =
                    translations[key];
            }
        }
    });

    document.querySelectorAll("[data-i18n-placeholder]")
    .forEach(element => {

        const key =
            element.getAttribute(
                "data-i18n-placeholder"
            );

        if (translations[key]) {

            element.placeholder =
                translations[key];
        }
    });

    localStorage.setItem("language", lang);
}

document.addEventListener("DOMContentLoaded", () => {

    const switcher =
        document.getElementById("languageSwitcher");

    const savedLanguage =
        localStorage.getItem("language") || "en";

    switcher.value = savedLanguage;

    loadLanguage(savedLanguage);

    switcher.addEventListener("change", (e) => {

        loadLanguage(e.target.value);
    });
});