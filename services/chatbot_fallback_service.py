"""Optional Hugging Face fallback for agricultural chatbot replies."""

from __future__ import annotations

import os
from typing import Optional

import requests


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
API_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{model}"


def get_huggingface_chat_reply(message: str, timeout: int = 12) -> Optional[str]:
    """Return a Hugging Face generated reply when the integration is configured."""
    token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_API_TOKEN")
    if not token:
        return None

    model = os.getenv("HUGGINGFACE_CHAT_MODEL", DEFAULT_MODEL)
    prompt = (
        "You are Agri-Vision, an assistant for cotton farmers. "
        "Give concise, practical, safe agricultural guidance.\n\n"
        f"Farmer question: {message}\n"
        "Answer:"
    )

    response = requests.post(
        API_URL_TEMPLATE.format(model=model),
        headers={"Authorization": f"Bearer {token}"},
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 120,
                "temperature": 0.4,
                "return_full_text": False,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()

    payload = response.json()
    if isinstance(payload, list) and payload:
        generated = payload[0].get("generated_text")
        if generated:
            return generated.strip()
    if isinstance(payload, dict):
        generated = payload.get("generated_text")
        if generated:
            return generated.strip()

    return None
