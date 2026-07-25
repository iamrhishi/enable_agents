"""
Shared "does this API key actually work" checks, used by both the personal
Settings page and the project-level AI key settings so a user finds out a
key is bad before saving it, not on the first real AI call that uses it.
"""
from __future__ import annotations

from typing import Tuple


def test_openai_key(api_key: str) -> Tuple[bool, str]:
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        models = client.models.list()
        return True, f"Connected! Found {len(list(models))} models"
    except Exception as e:
        return False, f"Failed: {str(e)}"


def test_anthropic_key(api_key: str) -> Tuple[bool, str]:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}],
        )
        return True, "Connected successfully!"
    except Exception as e:
        return False, f"Failed: {str(e)}"
