#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import json
import threading
from contextvars import copy_context

import pytest
from elastic_transport import (
    ApiResponseMeta,
    BaseAsyncNode,
    BaseNode,
    HttpHeaders,
    NodeConfig,
)
from elastic_transport._node import NodeApiResponse

from elasticsearch import (
    AsyncElasticsearch,
    Elasticsearch,
    OpAdmit,
    RuntimePolicy,
    RuntimePolicyError,
    ScrollGate,
    WriteGate,
    async_runtime_policy,
    runtime_policy,
)
from elasticsearch._async import helpers as async_helpers
from elasticsearch._runtime_policy import admit_request, push_runtime_policy
from elasticsearch import helpers


class RuntimePolicyNode(BaseNode):
    def __init__(self, config: NodeConfig):
        self.calls = []
        self.clear_status = 200
        super().__init__(config)

    def _response(self, status: int, body):
        return NodeApiResponse(
            ApiResponseMeta(
                status=status,
                headers=HttpHeaders({"X-elastic-product": "Elasticsearch"}),
                http_version="1.1",
                duration=0.0,
                node=self.config,
            ),
            json.dumps(body).encode("utf-8"),
        )

    def perform_request(self, method, target, **kwargs):
        self.calls.append((method, target, kwargs))
        if method == "DELETE" and target.startswith("/_search/scroll"):
            return self._response(self.clear_status, {"succeeded": True})
        if method == "POST" and target.startswith("/_search/scroll"):
            return self._response(
                200,
                {
                    "_scroll_id": "scroll-2",
                    "_shards": {"successful": 1, "failed": 0, "skipped": 0, "total": 1},
                    "hits": {"hits": []},
                },
            )
        if method == "POST" and "/_search" in target and "scroll=" in target:
            return self._response(
                200,
                {
                    "_scroll_id": "scroll-1",
                    "_shards": {"successful": 1, "failed": 0, "skipped": 0, "total": 1},
                    "hits": {"hits": [{"_id": "1"}]},
                },
            )
        if "/_bulk" in target:
            return self._response(
                200,
                {"errors": False, "items": [{"index": {"status": 200}}]},
            )
        return self._response(200, {})


class AsyncRuntimePolicyNode(BaseAsyncNode):
    def __init__(self, config: NodeConfig):
        self.calls = []
        self.clear_status = 200
        super().__init__(config)

    def _response(self, status: int, body):
        return NodeApiResponse(
            ApiResponseMeta(
                status=status,
                headers=HttpHeaders({"X-elastic-product": "Elasticsearch"}),
                http_version="1.1",
                duration=0.0,
                node=self.config,
            ),
            json.dumps(body).encode("utf-8"),
        )

    async def perform_request(self, method, target, **kwargs):
        self.calls.append((method, target, kwargs))
        if method == "DELETE" and target.startswith("/_search/scroll"):
            return self._response(self.clear_status, {"succeeded": True})
        if method == "POST" and target.startswith("/_search/scroll"):
            return self._response(
                200,
                {
                    "_scroll_id": "scroll-async-2",
                    "_shards": {"successful": 1, "failed": 0, "skipped": 0, "total": 1},
                    "hits": {"hits": []},
                },
            )
        if method == "POST" and "/_search" in target and "scroll=" in target:
            return self._response(
                200,
                {
                    "_scroll_id": "scroll-async-1",
                    "_shards": {"successful": 1, "failed": 0, "skipped": 0, "total": 1},
                    "hits": {"hits": [{"_id": "1"}]},
                },
            )
        if "/_bulk" in target:
            return self._response(
                200,
                {"errors": False, "items": [{"index": {"status": 200}}]},
            )
        return self._response(200, {})


@pytest.fixture
def client():
    return Elasticsearch(
        "http://localhost:9200", meta_header=False, node_class=RuntimePolicyNode
    )


@pytest.fixture
def async_client():
    return AsyncElasticsearch(
        "http://localhost:9200", meta_header=False, node_class=AsyncRuntimePolicyNode
    )


def node(client):
    return client.transport.node_pool.get()


def async_node(async_client):
    return async_client.transport.node_pool.get()


def call_endpoint(client, endpoint_id):
    if endpoint_id == "search":
        return client.search(index="test")
    if endpoint_id == "get":
        return client.get(index="test", id="1")
    if endpoint_id == "mget":
        return client.mget(index="test", ids=["1"])
    if endpoint_id == "msearch":
        return client.msearch(searches=[{}, {}])
    if endpoint_id == "count":
        return client.count(index="test")
    if endpoint_id == "scroll":
        return client.scroll(scroll_id="opened", scroll="1m")
    raise AssertionError(endpoint_id)


async def call_async_endpoint(client, endpoint_id):
    if endpoint_id == "search":
        return await client.search(index="test")
    if endpoint_id == "get":
        return await client.get(index="test", id="1")
    if endpoint_id == "mget":
        return await client.mget(index="test", ids=["1"])
    if endpoint_id == "msearch":
        return await client.msearch(searches=[{}, {}])
    if endpoint_id == "count":
        return await client.count(index="test")
    if endpoint_id == "scroll":
        return await client.scroll(scroll_id="opened", scroll="1m")
    raise AssertionError(endpoint_id)


def test_empty_stack_matches_existing_api(client):
    call_endpoint(client, "search")
    call_endpoint(client, "count")
    assert len(node(client).calls) == 2


def test_honor_inner_frame_keeps_outer_deny(client):
    with runtime_policy(RuntimePolicy(op_admit=OpAdmit.deny_new)):
        with runtime_policy():
            with pytest.raises(RuntimePolicyError) as excinfo:
                client.get(index="test", id="1")
    error = excinfo.value
    assert error.policy_axis == "op_admit"
    assert error.policy_mode == "deny_new"
    assert error.endpoint_id == "get"
    assert node(client).calls == []


@pytest.mark.parametrize("endpoint_id", ["search", "get", "mget", "msearch", "count"])
def test_sync_read_endpoints_deny_new(client, endpoint_id):
    with runtime_policy(RuntimePolicy(op_admit=OpAdmit.deny_new)):
        with pytest.raises(RuntimePolicyError) as excinfo:
            call_endpoint(client, endpoint_id)
    assert excinfo.value.endpoint_id == endpoint_id
    assert node(client).calls == []


def test_scroll_advance_denied_by_both_gates(client):
    with runtime_policy(RuntimePolicy(op_admit=OpAdmit.deny_new)):
        with pytest.raises(RuntimePolicyError) as excinfo:
            client.scroll(scroll_id="opened", scroll="1m")
    assert excinfo.value.policy_axis == "op_admit"
    assert node(client).calls == []

    with runtime_policy(RuntimePolicy(scroll_gate=ScrollGate.deny_advance)):
        with pytest.raises(RuntimePolicyError) as excinfo:
            client.scroll(scroll_id="opened", scroll="1m")
    assert excinfo.value.policy_axis == "scroll_gate"
    assert node(client).calls == []


@pytest.mark.parametrize("endpoint_id", ["search", "get", "mget", "msearch", "count"])
@pytest.mark.anyio
async def test_async_read_endpoints_deny_new(async_client, endpoint_id):
    async with async_runtime_policy(
        RuntimePolicy(op_admit=OpAdmit.deny_new)
    ):
        with pytest.raises(RuntimePolicyError) as excinfo:
            await call_async_endpoint(async_client, endpoint_id)
    assert excinfo.value.endpoint_id == endpoint_id
    assert async_node(async_client).calls == []


def test_one_more_allows_exactly_one_sync_request(client):
    with runtime_policy(RuntimePolicy(op_admit=OpAdmit.one_more)):
        client.search(index="test")
        with pytest.raises(RuntimePolicyError):
            client.get(index="test", id="1")
    assert len(node(client).calls) == 1


def test_one_more_is_thread_safe():
    frame = push_runtime_policy(RuntimePolicy(op_admit=OpAdmit.one_more))
    allowed = []
    barrier = threading.Barrier(4)

    def worker():
        barrier.wait()
        try:
            admit_request("get")
        except RuntimePolicyError:
            pass
        else:
            allowed.append(1)

    context = copy_context()
    threads = [
        threading.Thread(target=context.run, args=(worker,)) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(allowed) == 1
    from elasticsearch import pop_runtime_policy

    pop_runtime_policy(frame)


@pytest.mark.parametrize(
    "method_name,kwargs",
    [
        ("index", {"index": "test", "id": "1", "document": {"field": 1}}),
        ("update", {"index": "test", "id": "1", "doc": {"field": 1}}),
        ("delete", {"index": "test", "id": "1"}),
        (
            "bulk",
            {"operations": [b'{"index":{}}', b'{"field":1}']},
        ),
    ],
)
def test_sync_write_gate_deny_and_force_fail(client, method_name, kwargs):
    for mode in (WriteGate.deny_new, WriteGate.force_fail):
        node(client).calls.clear()
        with runtime_policy(RuntimePolicy(write_gate=mode)):
            with pytest.raises(RuntimePolicyError) as excinfo:
                getattr(client, method_name)(**kwargs)
        assert excinfo.value.policy_mode == mode.value
        assert node(client).calls == []


@pytest.mark.anyio
async def test_async_write_force_fail(async_client):
    async with async_runtime_policy(
        RuntimePolicy(write_gate=WriteGate.force_fail)
    ):
        with pytest.raises(RuntimePolicyError):
            await async_client.index(index="test", id="1", document={"field": 1})
    assert async_node(async_client).calls == []


def test_non_managed_endpoint_is_allowed(client):
    with runtime_policy(
        RuntimePolicy(
            op_admit=OpAdmit.deny_new,
            write_gate=WriteGate.force_fail,
            scroll_gate=ScrollGate.deny_advance,
        )
    ):
        client.info()
        client.clear_scroll(scroll_id="opened")
    calls = node(client).calls
    assert [method for method, _target, _kwargs in calls] == ["GET", "DELETE"]


def test_inflight_request_is_not_cancelled(client):
    holder = []

    original_perform_request = type(node(client)).perform_request

    def perform_request(self, method, target, **kwargs):
        result = original_perform_request(self, method, target, **kwargs)
        if not holder and target.startswith("/_search") and "scroll=" not in target:
            holder.append(push_runtime_policy(
                RuntimePolicy(op_admit=OpAdmit.deny_new)
            ))
        return result

    type(node(client)).perform_request = perform_request
    try:
        client.search(index="test")
        assert len(node(client).calls) == 1
        with pytest.raises(RuntimePolicyError):
            client.get(index="test", id="1")
    finally:
        from elasticsearch import pop_runtime_policy

        pop_runtime_policy(holder[0])
        type(node(client)).perform_request = original_perform_request


def test_bulk_helpers_deny_without_success_and_keep_buffer(client):
    actions = [{"field": 1}]
    with runtime_policy(RuntimePolicy(write_gate=WriteGate.deny_new)):
        with pytest.raises(RuntimePolicyError):
            helpers.bulk(client, actions, chunk_size=1)
        with pytest.raises(RuntimePolicyError):
            list(helpers.streaming_bulk(client, list(actions), chunk_size=1))
    assert actions == [{"field": 1}]
    assert node(client).calls == []


def test_parallel_bulk_worker_uses_callers_policy(client):
    actions = [{"field": 1}, {"field": 2}]
    with runtime_policy(RuntimePolicy(write_gate=WriteGate.force_fail)):
        with pytest.raises(RuntimePolicyError):
            list(
                helpers.parallel_bulk(
                    client, actions, chunk_size=1, thread_count=2
                )
            )
    assert [call for call in node(client).calls if "/_bulk" in call[1]] == []


@pytest.mark.anyio
async def test_async_bulk_helper_force_fail(async_client):
    actions = [{"field": 1}]
    async with async_runtime_policy(
        RuntimePolicy(write_gate=WriteGate.force_fail)
    ):
        with pytest.raises(RuntimePolicyError):
            await async_helpers.async_bulk(async_client, actions, chunk_size=1)
    assert async_node(async_client).calls == []


def test_scan_deny_advance_does_not_fake_exhaustion(client):
    with runtime_policy(RuntimePolicy(scroll_gate=ScrollGate.deny_advance)):
        scanner = helpers.scan(client, index="test", query={}, size=1)
        assert next(scanner)["_id"] == "1"
        with pytest.raises(RuntimePolicyError) as excinfo:
            next(scanner)
        scanner.close()
    error = excinfo.value
    assert error.policy_axis == "scroll_gate"
    assert error.policy_mode == "deny_advance"
    scroll_calls = [
        call
        for call in node(client).calls
        if call[0] == "POST" and call[1].startswith("/_search/scroll")
    ]
    assert scroll_calls == []


@pytest.mark.anyio
async def test_async_scan_deny_advance(async_client):
    async with async_runtime_policy(
        RuntimePolicy(scroll_gate=ScrollGate.deny_advance)
    ):
        scanner = async_helpers.async_scan(async_client, index="test", query={}, size=1)
        assert (await scanner.__anext__())["_id"] == "1"
        with pytest.raises(RuntimePolicyError):
            await scanner.__anext__()
        await scanner.aclose()
    assert [
        call
        for call in async_node(async_client).calls
        if call[0] == "POST" and call[1].startswith("/_search/scroll")
    ] == []


def test_force_close_clears_current_scroll_and_prevents_duplicate(client):
    with runtime_policy(RuntimePolicy(scroll_gate=ScrollGate.force_close)):
        scanner = helpers.scan(client, index="test", query={}, size=1)
        assert next(scanner)["_id"] == "1"

    delete_calls = [
        call for call in node(client).calls if call[0] == "DELETE"
    ]
    assert len(delete_calls) == 1
    assert '"scroll-1"' in delete_calls[0][2]["body"]
    scanner.close()
    assert len([call for call in node(client).calls if call[0] == "DELETE"]) == 1


def test_force_close_uses_latest_scroll_id(client):
    with runtime_policy(RuntimePolicy(scroll_gate=ScrollGate.force_close)):
        client.search(index="test", scroll="1m")
        client.scroll(scroll_id="scroll-1", scroll="1m")
    delete_calls = [call for call in node(client).calls if call[0] == "DELETE"]
    assert len(delete_calls) == 1
    assert delete_calls[0][2]["body"] == {"scroll_id": "scroll-2"}


def test_normal_clear_is_not_repeated_on_force_close_pop(client):
    with runtime_policy(RuntimePolicy(scroll_gate=ScrollGate.force_close)):
        client.search(index="test", scroll="1m")
        client.clear_scroll(scroll_id="scroll-1")
    assert len([call for call in node(client).calls if call[0] == "DELETE"]) == 1


def test_force_close_failure_is_observable(client):
    node(client).clear_status = 500
    with pytest.raises(RuntimePolicyError) as excinfo:
        with runtime_policy(RuntimePolicy(scroll_gate=ScrollGate.force_close)):
            client.search(index="test", scroll="1m")
    assert excinfo.value.policy_mode == "force_close"
    assert isinstance(excinfo.value.__context__, Exception)


def test_nested_honor_scroll_transfers_to_outer_force_close(client):
    with runtime_policy(RuntimePolicy(scroll_gate=ScrollGate.force_close)):
        with runtime_policy():
            scanner = helpers.scan(client, index="test", query={}, size=1)
            next(scanner)
        assert [call for call in node(client).calls if call[0] == "DELETE"] == []
    assert len([call for call in node(client).calls if call[0] == "DELETE"]) == 1
    scanner.close()


@pytest.mark.anyio
async def test_async_force_close_clears_scroll(async_client):
    async with async_runtime_policy(
        RuntimePolicy(scroll_gate=ScrollGate.force_close)
    ):
        await async_client.search(index="test", scroll="1m")
    assert [
        call for call in async_node(async_client).calls if call[0] == "DELETE"
    ][0][2]["body"] == {"scroll_id": "scroll-async-1"}


def test_concurrent_thread_scopes_do_not_share_stack(client):
    other_client = Elasticsearch(
        "http://localhost:9200", meta_header=False, node_class=RuntimePolicyNode
    )

    def denied():
        with runtime_policy(RuntimePolicy(op_admit=OpAdmit.deny_new)):
            with pytest.raises(RuntimePolicyError):
                client.get(index="test", id="1")

    thread = threading.Thread(target=denied)
    thread.start()
    thread.join()
    other_client.info()
    assert node(client).calls == []
    assert len(node(other_client).calls) == 1


@pytest.mark.asyncio
async def test_concurrent_async_scopes_do_not_share_stack(async_client):
    async def denied():
        async with async_runtime_policy(
            RuntimePolicy(write_gate=WriteGate.deny_new)
        ):
            with pytest.raises(RuntimePolicyError):
                await async_client.index(index="test", id="1", document={"x": 1})

    async def allowed():
        await async_client.info()

    task_denied = asyncio_create_task(denied())
    task_allowed = asyncio_create_task(allowed())
    await task_denied
    await task_allowed
    assert [call for call in async_node(async_client).calls if call[0] != "GET"] == []


def asyncio_create_task(coro):
    import asyncio

    return asyncio.create_task(coro)
