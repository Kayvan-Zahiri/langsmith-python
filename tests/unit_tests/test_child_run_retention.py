import asyncio
import contextvars
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from langsmith import Client, get_tracing_context, traceable, tracing_context
from langsmith.evaluation._arunner import _aforward
from langsmith.evaluation._runner import _forward
from langsmith.run_trees import RunTree
from langsmith.schemas import Example


@pytest.fixture
def client() -> MagicMock:
    client = MagicMock(spec=Client)
    client.tracing_queue = None
    return client


@pytest.mark.parametrize("retain", [False, True])
def test_retention_context_restores_snapshots(retain: bool) -> None:
    original = get_tracing_context()
    with tracing_context(retain_child_runs=retain):
        assert get_tracing_context()["retain_child_runs"] == retain
        snapshot = contextvars.copy_context()
        assert get_tracing_context(snapshot)["retain_child_runs"] == retain
        with tracing_context(tags=["nested"]):
            assert get_tracing_context()["retain_child_runs"] == retain
        with tracing_context(retain_child_runs=not retain):
            assert get_tracing_context()["retain_child_runs"] != retain
        assert get_tracing_context()["retain_child_runs"] == retain
    assert get_tracing_context() == original


@pytest.mark.parametrize("with_callback", [False, True])
def test_on_end_requests_retention_for_descendants(client: MagicMock, with_callback: bool) -> None:
    collected = []

    @traceable
    def child() -> bool:
        return get_tracing_context()["retain_child_runs"]

    @traceable
    def parent() -> bool:
        return child()

    with tracing_context(enabled="local", client=client, retain_child_runs=False):
        extra = {"on_end": collected.append} if with_callback else {}
        assert parent(langsmith_extra=extra) == with_callback
        assert get_tracing_context()["retain_child_runs"] is False
    if with_callback:
        assert collected[0].child_runs[0].outputs == {"output": True}


async def test_retention_does_not_leak_between_tasks(client: MagicMock) -> None:
    ready = asyncio.Event()

    @traceable
    async def parent() -> bool:
        ready.set()
        await asyncio.sleep(0)
        return get_tracing_context()["retain_child_runs"]

    async def observed() -> bool:
        return await parent(langsmith_extra={"on_end": lambda run: None})

    async def ordinary() -> bool:
        await ready.wait()
        return await parent()

    with tracing_context(enabled="local", client=client, retain_child_runs=False):
        assert await asyncio.gather(observed(), ordinary()) == [True, False]
        assert get_tracing_context()["retain_child_runs"] is False


def test_sync_evaluation_requests_retention() -> None:
    @traceable
    def target(inputs: dict) -> dict:
        assert get_tracing_context()["retain_child_runs"] is True
        return inputs

    example = Example(id=uuid4(), dataset_id=uuid4(), inputs={"value": 1})
    client = MagicMock(spec=Client)
    with tracing_context(retain_child_runs=False):
        result = _forward(target, example, "retention", {}, client, False)
        assert get_tracing_context()["retain_child_runs"] is False
    assert result["run"].outputs == example.inputs


async def test_async_evaluation_requests_retention() -> None:
    @traceable
    async def target(inputs: dict) -> dict:
        assert get_tracing_context()["retain_child_runs"] is True
        return inputs

    example = Example(id=uuid4(), dataset_id=uuid4(), inputs={"value": 1})
    client = MagicMock(spec=Client)
    with tracing_context(retain_child_runs=False):
        result = await _aforward(target, example, "retention", {}, client)
        assert get_tracing_context()["retain_child_runs"] is False
    assert result["run"].outputs == example.inputs


def test_manual_sdk_tree_retention_is_unchanged(client: MagicMock) -> None:
    with tracing_context(retain_child_runs=False):
        parent = RunTree(name="manual", client=client)
        child = parent.create_child(name="child")
    assert parent.child_runs == [child]
