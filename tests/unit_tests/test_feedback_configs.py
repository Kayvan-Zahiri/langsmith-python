"""Unit tests for the feedback config client methods."""

import uuid
from unittest import mock

import pytest

from langsmith import AsyncClient, Client

_CONFIG = {"type": "continuous", "min": 0.0, "max": 1.0}


def _config_response(**overrides):
    body = {
        "id": str(uuid.uuid4()),
        "feedback_key": "correctness",
        "feedback_config": _CONFIG,
        "tenant_id": str(uuid.uuid4()),
        "modified_at": "2026-10-19T00:00:00Z",
        "is_lower_score_better": False,
        "is_enforced": False,
        "is_first": True,
        "is_deleted": False,
    }
    body.update(overrides)
    return body


def _client() -> Client:
    return Client(api_url="http://localhost:1984", api_key="123", session=mock.Mock())


def test_create_feedback_config_sends_is_enforced() -> None:
    client = _client()
    with mock.patch.object(Client, "request_with_retries") as request:
        request.return_value = mock.Mock(json=lambda: _config_response(is_enforced=True))
        config = client.create_feedback_config("correctness", feedback_config=_CONFIG, is_enforced=True)

    assert request.call_args.kwargs["json"]["is_enforced"] is True
    assert config.is_enforced is True
    assert config.id is not None


def test_create_feedback_config_omits_is_enforced_by_default() -> None:
    client = _client()
    with mock.patch.object(Client, "request_with_retries") as request:
        request.return_value = mock.Mock(json=lambda: _config_response())
        config = client.create_feedback_config("correctness", feedback_config=_CONFIG)

    assert "is_enforced" not in request.call_args.kwargs["json"]
    assert config.is_enforced is False


def test_list_feedback_configs_filters_by_is_enforced() -> None:
    client = _client()
    with mock.patch.object(
        Client,
        "_get_paginated_list",
        return_value=iter([_config_response(is_enforced=True)]),
    ) as paginate:
        configs = list(client.list_feedback_configs(is_enforced=True))

    assert paginate.call_args.kwargs["params"]["is_enforced"] is True
    assert [c.is_enforced for c in configs] == [True]


@pytest.mark.asyncio
async def test_async_create_feedback_config_sends_is_enforced() -> None:
    client = AsyncClient(api_url="http://localhost:1984", api_key="123")
    with mock.patch.object(AsyncClient, "_arequest_with_retries", new_callable=mock.AsyncMock) as request:
        request.return_value = mock.Mock(json=lambda: _config_response(is_enforced=True))
        config = await client.create_feedback_config("correctness", feedback_config=_CONFIG, is_enforced=True)

    assert request.call_args.kwargs["json"]["is_enforced"] is True
    assert config.is_enforced is True


@pytest.mark.asyncio
async def test_async_list_feedback_configs_filters_by_is_enforced() -> None:
    client = AsyncClient(api_url="http://localhost:1984", api_key="123")
    seen_params = {}

    async def paginate(path, params=None):
        seen_params.update(params)
        yield _config_response(is_enforced=False)

    with mock.patch.object(AsyncClient, "_aget_paginated_list", side_effect=paginate):
        configs = [c async for c in client.list_feedback_configs(is_enforced=False)]

    assert seen_params["is_enforced"] is False
    assert [c.is_enforced for c in configs] == [False]
