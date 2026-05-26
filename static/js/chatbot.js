/**
 * static/js/chatbot.js
 * Agri-Vision AI Chatbot — Full Premium Controller
 * Phases: Toggle (3), Fetch (4), Rendering (5)
 */

document.addEventListener("DOMContentLoaded", () => {

    // ─── DOM References ────────────────────────────────────────────────────
    const launcher    = document.getElementById("chatbot-launcher");
    const widget      = document.getElementById("chatbot-widget");
    const closeBtn    = document.getElementById("chatbot-close-btn");
    const chatForm    = document.getElementById("chat-form");
    const chatInput   = document.getElementById("chat-input");
    const chatMessages = document.getElementById("chat-messages");

    // Defensive check — abort cleanly on pages without the widget
    if (!launcher || !widget || !closeBtn || !chatForm || !chatInput || !chatMessages) {
        return;
    }

    // ─── Phase 3: DRY Open / Close Helpers ────────────────────────────────

    function openChatbot(e) {
        if (e) e.stopPropagation();
        widget.classList.remove("collapsed");
        launcher.classList.add("launcher-hidden");
        launcher.setAttribute("aria-expanded", "true");
        chatInput.focus();
    }

    function closeChatbot(e) {
        if (e) e.stopPropagation();
        widget.classList.add("collapsed");
        launcher.classList.remove("launcher-hidden");
        launcher.setAttribute("aria-expanded", "false");
        launcher.focus();
    }

    // Bind toggle listeners
    launcher.addEventListener("click", openChatbot);
    closeBtn.addEventListener("click", closeChatbot);

    // Escape key — close if widget is open
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !widget.classList.contains("collapsed")) {
            closeChatbot();
        }
    });

    // ─── Phase 5: Message Rendering Helpers ───────────────────────────────

    /**
     * Append a message bubble. XSS-safe via .textContent.
     * senderType: "user" | "ai" | "error"
     */
    function appendMessage(text, senderType) {
        const msgDiv = document.createElement("div");
        msgDiv.classList.add("chat-message");

        if (senderType === "user") {
            msgDiv.classList.add("user-message");
        } else {
            msgDiv.classList.add("ai-message");
            if (senderType === "error") {
                msgDiv.classList.add("error-message");
            }
        }

        msgDiv.textContent = text;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    /**
     * Show animated bouncing-dot typing indicator.
     * Dots created via DOM (no innerHTML) for safer rendering habits.
     */
    function showLoadingIndicator() {
        const loaderDiv = document.createElement("div");
        loaderDiv.className = "chat-message ai-message loading-indicator";

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement("span");
            dot.className = "dot";
            loaderDiv.appendChild(dot);
        }

        chatMessages.appendChild(loaderDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return loaderDiv;
    }

    /**
     * Safely remove loading indicator from DOM.
     */
    function removeLoadingIndicator(indicator) {
        if (indicator && indicator.parentNode) {
            indicator.parentNode.removeChild(indicator);
        }
    }

    // ─── Phase 4: Fetch /api/chat Integration ─────────────────────────────

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const messageText = chatInput.value.trim();
        if (!messageText) return;

        // Prevent duplicate submits
        chatInput.disabled = true;
        const sendBtn = document.getElementById("chat-send-btn");
        if (sendBtn) sendBtn.disabled = true;

        // Render user message immediately
        appendMessage(messageText, "user");
        chatInput.value = "";

        // Show typing indicator
        const loadingIndicator = showLoadingIndicator();

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: messageText })
            });

            // Parse JSON — handle malformed/non-JSON responses gracefully
            let data;
            try {
                data = await response.json();
            } catch (_) {
                data = null;
            }

            if (!response.ok || !data) {
                throw new Error(`Server error: ${response.status}`);
            }

            appendMessage(data.reply || "I didn't receive a reply.", "ai");

        } catch (error) {
            console.error("Chatbot communication error:", error);
            appendMessage(
                "Sorry, I'm having trouble connecting. Please check your network and try again.",
                "error"
            );
        } finally {
            // Always restore UI — remove loader, re-enable controls, focus input
            removeLoadingIndicator(loadingIndicator);
            chatInput.disabled = false;
            if (sendBtn) sendBtn.disabled = false;
            chatInput.focus();
        }
    });

});
