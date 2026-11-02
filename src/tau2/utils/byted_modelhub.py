"""Explicit ModelHub routes; credentials never enter model kwargs or artifacts."""

import os

import httpx
from litellm import ModelResponse
from openai import AzureOpenAI

from tau2.config import (
    BYTED_CHAT_API_VERSION,
    BYTED_CHAT_ENDPOINT,
    BYTED_ELEVENLABS_BASE_URL,
)


def gateway_key():
    """Resolve the user configured gateway credential without serializing it."""
    key = os.environ.get("TERUMI_BT_AK")
    if not key:
        raise ValueError("TERUMI_BT_AK is required for the ModelHub route")
    return key


def chat_completion(model, messages, tools=None, tool_choice=None, **kwargs):
    """Return a standard response using the documented Azure compatible route."""
    retries = kwargs.pop("num_retries", 0)
    timeout = kwargs.pop("timeout", 180)
    # Credentials may only come from the environment, never logged llm_args.
    for forbidden in ("api_key", "api_base", "base_url", "custom_llm_provider"):
        if forbidden in kwargs:
            raise ValueError(f"Unsupported gateway model option: {forbidden}")
    arguments = {"model": model.removeprefix("byted/"), "messages": messages, **kwargs}
    if tools:
        arguments.update(tools=tools, tool_choice=tool_choice or "auto")
    try:
        with AzureOpenAI(
            azure_endpoint=BYTED_CHAT_ENDPOINT,
            api_version=BYTED_CHAT_API_VERSION,
            api_key=gateway_key(),
            max_retries=retries,
            timeout=timeout,
        ) as client:
            result = client.chat.completions.create(**arguments)
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        raise RuntimeError(
            f"ModelHub chat failed: {type(exc).__name__}; status={status}"
        ) from None
    return ModelResponse(**result.model_dump())


def elevenlabs_request(endpoint, **kwargs):
    """Call documented ElevenLabs gateway without leaking authenticated URLs."""
    try:
        response = httpx.post(
            BYTED_ELEVENLABS_BASE_URL + endpoint,
            params={"ak": gateway_key(), **kwargs.pop("params", {})},
            timeout=120,
            **kwargs,
        )
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"ModelHub ElevenLabs transport failure: {type(exc).__name__}"
        ) from None
    if response.status_code != 200:
        raise RuntimeError(f"ModelHub ElevenLabs HTTP {response.status_code}")
    return response
