"""
Single shared entry point for every LLM call in the app.

Before this module, ~30 call sites across app.py and agents/*/service.py
each instantiated their own openai.OpenAI()/ChatOpenAI() client, almost
always reading OPENAI_API_KEY straight from the environment - meaning the
whole app ran on one shared platform key with no per-user or per-project
attribution, and no usage/cost tracking at all.

This module:
  - resolves which API key a call should use, in priority order:
      1. the project's own key (core/project_settings.py), if project_id
         is given and the project has one set for this provider
      2. the calling user's personal key (core/settings.py)
      3. the platform default from environment (OPENAI_API_KEY / ANTHROPIC_API_KEY)
  - makes the call
  - logs token usage + an estimated cost to AIUsageLog, tagged with
    user/project/team/agent so usage can be rolled up at any of those levels

Usage (raw OpenAI client style, the majority pattern in this codebase):

    from core.ai_client import ai_chat_completion

    response = ai_chat_completion(
        user_id=g.user_id, project_id=project_id, agent="content_marketing.generate",
        model="gpt-4", messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content

Usage (LangChain ChatOpenAI style):

    from core.ai_client import get_langchain_llm, log_langchain_usage

    llm, key_source = get_langchain_llm(user_id=g.user_id, project_id=project_id, model="gpt-4")
    result = llm.invoke(prompt)
    log_langchain_usage(result, user_id=g.user_id, project_id=project_id,
                         agent="content_marketing.kg_builder", model="gpt-4", key_source=key_source)
    text = result.content
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


def _current_user_id(explicit: Optional[str]) -> Optional[str]:
    """Falls back to the authenticated request's g.user_id when a call site
    doesn't have an explicit user_id handy (e.g. a helper several calls deep
    inside a request). Returns None outside a request context (e.g. Celery
    tasks) - callers there should pass user_id explicitly."""
    if explicit:
        return explicit
    try:
        from flask import g, has_request_context
        if has_request_context():
            return getattr(g, "user_id", None)
    except RuntimeError:
        pass
    return None

# Rough, approximate USD-per-1K-token rates for cost estimation and display
# only - not wired to any provider's real billing API, so treat this as an
# order-of-magnitude estimate, not an invoice. Update as pricing changes.
_COST_PER_1K_TOKENS: Dict[str, Dict[str, float]] = {
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
    "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
    "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
}
_DEFAULT_COST_PER_1K = {"prompt": 0.005, "completion": 0.015}  # unknown-model fallback


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = _COST_PER_1K_TOKENS.get(model, _DEFAULT_COST_PER_1K)
    return (prompt_tokens / 1000) * rates["prompt"] + (completion_tokens / 1000) * rates["completion"]


def resolve_api_key(user_id: Optional[str], project_id: Optional[str], provider: str) -> Tuple[Optional[str], str]:
    """Returns (api_key, source) where source is 'project' | 'user' | 'platform' | 'none'."""
    key_name = f"{provider}_key"

    if project_id:
        from core.project_settings import ProjectSettings
        project_key = ProjectSettings().get(project_id, key_name)
        if project_key:
            return project_key, "project"

    if user_id:
        from core.settings import UserSettings
        user_key = UserSettings().get(user_id, "ai", key_name)
        if user_key:
            return user_key, "user"

    env_var = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    platform_key = os.getenv(env_var)
    if platform_key:
        return platform_key, "platform"

    return None, "none"


def _team_id_for_project(project_id: Optional[str]) -> Optional[str]:
    if not project_id:
        return None
    from core.models import Project
    project = Project.query.filter_by(project_id=project_id).first()
    return project.team_id if project else None


def log_ai_usage(
    user_id: str,
    project_id: Optional[str],
    agent: str,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    key_source: str,
) -> None:
    from core.database import db
    from core.models import AIUsageLog

    entry = AIUsageLog(
        user_id=user_id or "unknown",
        project_id=project_id,
        team_id=_team_id_for_project(project_id),
        agent=agent,
        provider=provider,
        model=model,
        key_source=key_source,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        estimated_cost_usd=estimate_cost_usd(model, prompt_tokens, completion_tokens),
    )
    db.session.add(entry)
    db.session.commit()


class NoApiKeyConfigured(RuntimeError):
    """Raised when no project/user/platform key is available for a provider."""


def ai_chat_completion(
    user_id: Optional[str],
    project_id: Optional[str],
    agent: str,
    model: str,
    messages: List[Dict[str, str]],
    provider: str = "openai",
    **kwargs: Any,
):
    """Raw-client-style chat completion (mirrors openai's
    client.chat.completions.create signature/response shape) with key
    resolution and usage logging built in. user_id falls back to the
    current request's g.user_id if not given explicitly."""
    import openai

    user_id = _current_user_id(user_id)
    api_key, key_source = resolve_api_key(user_id, project_id, provider)
    if not api_key:
        raise NoApiKeyConfigured(
            f"No {provider} API key configured for this project or user, and no platform default is set."
        )

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(model=model, messages=messages, **kwargs)

    usage = getattr(response, "usage", None)
    log_ai_usage(
        user_id=user_id,
        project_id=project_id,
        agent=agent,
        provider=provider,
        model=model,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        key_source=key_source,
    )
    return response


def get_langchain_llm(
    user_id: Optional[str],
    project_id: Optional[str],
    model: str = "gpt-4",
    temperature: float = 0.7,
):
    """Returns (llm, key_source) - a ChatOpenAI instance configured with
    the resolved key. Caller must call log_langchain_usage() after
    invoking it, since usage isn't logged automatically here (LangChain's
    .invoke() returns before we'd have a chance to intercept)."""
    from langchain_openai import ChatOpenAI

    api_key, key_source = resolve_api_key(user_id, project_id, "openai")
    if not api_key:
        raise NoApiKeyConfigured(
            "No OpenAI API key configured for this project or user, and no platform default is set."
        )
    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key), key_source


def log_langchain_usage(ai_message, user_id: Optional[str], project_id: Optional[str], agent: str, model: str, key_source: str) -> None:
    """Extracts token usage from a LangChain AIMessage and logs it.
    Handles both the modern `.usage_metadata` attribute and the older
    `.response_metadata['token_usage']` shape."""
    prompt_tokens = 0
    completion_tokens = 0

    usage_metadata = getattr(ai_message, "usage_metadata", None)
    if usage_metadata:
        prompt_tokens = usage_metadata.get("input_tokens", 0) or 0
        completion_tokens = usage_metadata.get("output_tokens", 0) or 0
    else:
        token_usage = (getattr(ai_message, "response_metadata", None) or {}).get("token_usage", {})
        prompt_tokens = token_usage.get("prompt_tokens", 0) or 0
        completion_tokens = token_usage.get("completion_tokens", 0) or 0

    log_ai_usage(
        user_id=user_id,
        project_id=project_id,
        agent=agent,
        provider="openai",
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        key_source=key_source,
    )
