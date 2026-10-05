"""Anthropic refusal stop reasons must fail permanently, not enter JSON retries."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.llm_interface import ProviderContentPolicyError


class _StructuredResponse(BaseModel):
    answer: str


def _provider():
    with patch("anthropic.AsyncAnthropic") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM

        provider = AnthropicLLM(
            provider="anthropic",
            api_key="test-key",
            base_url="",
            model="claude-sonnet-test",
        )
    provider._client = MagicMock()
    return provider


def _refusal_response(stop_reason: str):
    return SimpleNamespace(
        content=[],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=8, output_tokens=0, cache_read_input_tokens=0),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["refusal", "content_filter"])
async def test_call_raises_permanent_refusal_before_json_retry(stop_reason):
    provider = _provider()
    create = AsyncMock(return_value=_refusal_response(stop_reason))
    provider._client.messages.create = create

    with pytest.raises(ProviderContentPolicyError, match=f"stop_reason={stop_reason}"):
        await provider.call(
            messages=[{"role": "user", "content": "restricted request"}],
            response_format=_StructuredResponse,
            max_retries=4,
            initial_backoff=0,
            max_backoff=0,
        )

    assert create.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["refusal", "content_filter"])
async def test_call_with_tools_raises_permanent_refusal(stop_reason):
    provider = _provider()
    create = AsyncMock(return_value=_refusal_response(stop_reason))
    provider._client.messages.create = create

    with pytest.raises(ProviderContentPolicyError, match=f"stop_reason={stop_reason}"):
        await provider.call_with_tools(
            messages=[{"role": "user", "content": "restricted request"}],
            tools=[],
            max_retries=4,
            initial_backoff=0,
            max_backoff=0,
        )

    assert create.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["refusal", "content_filter"])
async def test_batch_refusal_raises_permanent_error(stop_reason):
    provider = _provider()
    batch = SimpleNamespace(processing_status="ended")
    provider._client.messages.batches.retrieve = AsyncMock(return_value=batch)
    refusal_message = _refusal_response(stop_reason)
    entry = SimpleNamespace(
        custom_id="chunk_0",
        result=SimpleNamespace(type="succeeded", message=refusal_message),
    )
    provider._client.messages.batches.results = AsyncMock(return_value=_AsyncIter([entry]))

    with pytest.raises(ProviderContentPolicyError, match=f"stop_reason={stop_reason}"):
        await provider.retrieve_batch_results("batch-1")


class _AsyncIter:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration from None
