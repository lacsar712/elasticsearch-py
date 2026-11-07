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

import contextvars
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from .exceptions import RuntimePolicyError

_READ_ENDPOINTS = frozenset({"search", "get", "mget", "msearch", "count"})
_WRITE_ENDPOINTS = frozenset({"index", "update", "delete", "bulk"})
_SCROLL_ENDPOINT = "scroll"
_CLEAR_SCROLL_ENDPOINT = "clear_scroll"


class OpAdmit(str, Enum):
    honor_cfg = "honor_cfg"
    deny_new = "deny_new"
    one_more = "one_more"


class WriteGate(str, Enum):
    honor_cfg = "honor_cfg"
    deny_new = "deny_new"
    force_fail = "force_fail"


class ScrollGate(str, Enum):
    honor_cfg = "honor_cfg"
    deny_advance = "deny_advance"
    force_close = "force_close"


@dataclass(frozen=True)
class RuntimePolicy:
    """The three independently resolved runtime policy axes."""

    op_admit: OpAdmit = OpAdmit.honor_cfg
    write_gate: WriteGate = WriteGate.honor_cfg
    scroll_gate: ScrollGate = ScrollGate.honor_cfg

    def __post_init__(self) -> None:
        if not isinstance(self.op_admit, OpAdmit):
            raise TypeError("'op_admit' must be an OpAdmit value")
        if not isinstance(self.write_gate, WriteGate):
            raise TypeError("'write_gate' must be a WriteGate value")
        if not isinstance(self.scroll_gate, ScrollGate):
            raise TypeError("'scroll_gate' must be a ScrollGate value")


class RuntimePolicyFrame:
    """An opaque pushed runtime policy frame."""

    def __init__(self, policy: RuntimePolicy):
        self.frame_id = uuid.uuid4().hex
        self.policy = policy
        self._lock = RLock()
        self._remaining_op = 1 if policy.op_admit is OpAdmit.one_more else None
        self._active_scrolls = {}

    def __repr__(self) -> str:
        return (
            f"RuntimePolicyFrame(frame_id={self.frame_id!r}, policy={self.policy!r})"
        )

    def _register_scroll(
        self,
        scroll_id: str,
        client: Any,
        is_async: bool,
        old_scroll_id: Optional[str],
    ) -> None:
        with self._lock:
            old_handle = None
            if old_scroll_id is not None:
                old_handle = self._active_scrolls.pop(old_scroll_id, None)
            observer_states = (
                set(old_handle.observer_states) if old_handle is not None else set()
            )
            self._active_scrolls[scroll_id] = _ScrollHandle(
                scroll_id=scroll_id,
                client=client,
                is_async=is_async,
                observer_states=observer_states,
            )


class ScrollState:
    """Helper-owned state used to avoid a second clear after force-close."""

    def __init__(self) -> None:
        self.closed = False

    def bind(self, scroll_id: Optional[str]) -> None:
        if not scroll_id:
            return
        for frame in _current_frames():
            with frame._lock:
                handle = frame._active_scrolls.get(scroll_id)
                if handle is not None:
                    handle.observer_states.add(self)
                    return


@dataclass(frozen=True)
class _ScrollHandle:
    scroll_id: str
    client: Any
    is_async: bool
    closing: bool = False
    observer_states: Set[ScrollState] = field(default_factory=set)


@dataclass(frozen=True)
class _Admission:
    endpoint_id: Optional[str]


_NO_ADMISSION = _Admission(endpoint_id=None)
_STACK: contextvars.ContextVar[Tuple[RuntimePolicyFrame, ...]] = contextvars.ContextVar(
    "elasticsearch_runtime_policy_stack", default=()
)


def _current_frames() -> Tuple[RuntimePolicyFrame, ...]:
    return _STACK.get()


def _effective_frame(axis: str) -> Optional[RuntimePolicyFrame]:
    for frame in reversed(_current_frames()):
        mode = getattr(frame.policy, axis)
        if mode.value != ScrollGate.honor_cfg.value:
            return frame
    return None


def _policy_error(
    message: str,
    *,
    axis: str,
    mode: Union[OpAdmit, WriteGate, ScrollGate],
    endpoint_id: Optional[str] = None,
    frame_id: Optional[str] = None,
) -> RuntimePolicyError:
    return RuntimePolicyError(
        message,
        policy_axis=axis,
        policy_mode=mode.value,
        endpoint_id=endpoint_id,
        frame_id=frame_id,
    )


def _reject(
    axis: str,
    mode: Union[OpAdmit, WriteGate, ScrollGate],
    endpoint_id: str,
    frame: RuntimePolicyFrame,
) -> None:
    raise _policy_error(
        f"Runtime policy {axis}.{mode.value} rejected '{endpoint_id}'",
        axis=axis,
        mode=mode,
        endpoint_id=endpoint_id,
        frame_id=frame.frame_id,
    )


def push_runtime_policy(
    policy: Optional[RuntimePolicy] = None,
    *,
    op_admit: OpAdmit = OpAdmit.honor_cfg,
    write_gate: WriteGate = WriteGate.honor_cfg,
    scroll_gate: ScrollGate = ScrollGate.honor_cfg,
) -> RuntimePolicyFrame:
    """Push a runtime policy frame onto the current execution scope."""

    resolved_policy = policy or RuntimePolicy(
        op_admit=op_admit,
        write_gate=write_gate,
        scroll_gate=scroll_gate,
    )
    frame = RuntimePolicyFrame(resolved_policy)
    _STACK.set(_current_frames() + (frame,))
    return frame


def _transfer_active_scrolls(
    frame: RuntimePolicyFrame, parent: Optional[RuntimePolicyFrame]
) -> None:
    with frame._lock:
        handles = list(frame._active_scrolls.items())
        frame._active_scrolls.clear()
    if parent is None:
        return
    with parent._lock:
        for scroll_id, handle in handles:
            parent._active_scrolls.setdefault(scroll_id, handle)


def pop_runtime_policy(frame: RuntimePolicyFrame) -> None:
    """Pop a synchronous runtime policy frame and run synchronous cleanup."""

    frames = _current_frames()
    if not frames or frames[-1] is not frame:
        raise RuntimeError("Cannot pop a runtime policy frame that is not on top")

    try:
        if frame.policy.scroll_gate is ScrollGate.force_close:
            _close_active_scrolls_sync(frame)
    finally:
        parent = frames[-2] if len(frames) > 1 else None
        _transfer_active_scrolls(frame, parent)
        _STACK.set(frames[:-1])


async def apop_runtime_policy(frame: RuntimePolicyFrame) -> None:
    """Pop an asynchronous runtime policy frame and run asynchronous cleanup."""

    frames = _current_frames()
    if not frames or frames[-1] is not frame:
        raise RuntimeError("Cannot pop a runtime policy frame that is not on top")

    try:
        if frame.policy.scroll_gate is ScrollGate.force_close:
            await _close_active_scrolls_async(frame)
    finally:
        parent = frames[-2] if len(frames) > 1 else None
        _transfer_active_scrolls(frame, parent)
        _STACK.set(frames[:-1])


@contextmanager
def runtime_policy(
    policy: Optional[RuntimePolicy] = None,
    *,
    op_admit: OpAdmit = OpAdmit.honor_cfg,
    write_gate: WriteGate = WriteGate.honor_cfg,
    scroll_gate: ScrollGate = ScrollGate.honor_cfg,
) -> Iterator[RuntimePolicyFrame]:
    frame = push_runtime_policy(
        policy,
        op_admit=op_admit,
        write_gate=write_gate,
        scroll_gate=scroll_gate,
    )
    try:
        yield frame
    finally:
        pop_runtime_policy(frame)


@asynccontextmanager
async def async_runtime_policy(
    policy: Optional[RuntimePolicy] = None,
    *,
    op_admit: OpAdmit = OpAdmit.honor_cfg,
    write_gate: WriteGate = WriteGate.honor_cfg,
    scroll_gate: ScrollGate = ScrollGate.honor_cfg,
) -> AsyncIterator[RuntimePolicyFrame]:
    frame = push_runtime_policy(
        policy,
        op_admit=op_admit,
        write_gate=write_gate,
        scroll_gate=scroll_gate,
    )
    try:
        yield frame
    finally:
        await apop_runtime_policy(frame)


def admit_request(
    endpoint_id: Optional[str], *, body: Any = None
) -> _Admission:
    """Apply the current policy before a request reaches transport."""

    if not isinstance(endpoint_id, str):
        return _NO_ADMISSION

    op_frame: Optional[RuntimePolicyFrame] = None
    op_mode: Optional[OpAdmit] = None
    if endpoint_id in _READ_ENDPOINTS or endpoint_id == _SCROLL_ENDPOINT:
        op_frame = _effective_frame("op_admit")
        if op_frame is not None:
            op_mode = op_frame.policy.op_admit
            if op_mode is OpAdmit.deny_new:
                _reject("op_admit", op_mode, endpoint_id, op_frame)

    if endpoint_id == _SCROLL_ENDPOINT:
        scroll_frame = _effective_frame("scroll_gate")
        if (
            scroll_frame is not None
            and scroll_frame.policy.scroll_gate is ScrollGate.deny_advance
        ):
            _reject(
                "scroll_gate",
                scroll_frame.policy.scroll_gate,
                endpoint_id,
                scroll_frame,
            )

    if endpoint_id in _WRITE_ENDPOINTS:
        write_frame = _effective_frame("write_gate")
        if write_frame is not None:
            mode = write_frame.policy.write_gate
            if mode in (WriteGate.deny_new, WriteGate.force_fail):
                _reject("write_gate", mode, endpoint_id, write_frame)

    if op_frame is not None and op_mode is OpAdmit.one_more:
        with op_frame._lock:
            if op_frame._remaining_op == 0:
                _reject("op_admit", op_mode, endpoint_id, op_frame)
            op_frame._remaining_op -= 1

    if endpoint_id == _CLEAR_SCROLL_ENDPOINT:
        _mark_clear_started(body)

    return _Admission(endpoint_id=endpoint_id)


def record_request_exception(admission: _Admission) -> None:
    if admission.endpoint_id == _CLEAR_SCROLL_ENDPOINT:
        _mark_clear_failed(None)


def record_response(
    admission: _Admission,
    *,
    status: int,
    params: Optional[Mapping[str, Any]],
    request_body: Any,
    response_body: Any,
    client: Any,
    is_async: bool,
) -> None:
    """Update scroll ownership after a response has been received."""

    endpoint_id = admission.endpoint_id
    if endpoint_id == _CLEAR_SCROLL_ENDPOINT:
        if 200 <= status < 300 or status == 404:
            _remove_scrolls(_extract_scroll_ids(request_body))
        else:
            _mark_clear_failed(_extract_scroll_ids(request_body))
        return

    if not 200 <= status < 300 or not isinstance(response_body, Mapping):
        return

    new_scroll_id = response_body.get("_scroll_id")
    if not isinstance(new_scroll_id, str):
        new_scroll_id = None

    if endpoint_id == "search" and _opens_scroll(params, request_body):
        if new_scroll_id:
            frames = _current_frames()
            if frames:
                frames[-1]._register_scroll(
                    new_scroll_id, client, is_async, old_scroll_id=None
                )
    elif endpoint_id == _SCROLL_ENDPOINT:
        old_scroll_id = _scroll_id_from_body(request_body)
        owner = _find_owner(old_scroll_id) if old_scroll_id else None
        if owner is not None and new_scroll_id:
            owner._register_scroll(
                new_scroll_id, client, is_async, old_scroll_id=old_scroll_id
            )
        elif owner is not None:
            _remove_scrolls((old_scroll_id,))


def _opens_scroll(params: Optional[Mapping[str, Any]], request_body: Any) -> bool:
    if isinstance(params, Mapping) and params.get("scroll") is not None:
        return True
    if isinstance(request_body, Mapping) and request_body.get("scroll") is not None:
        return True
    return False


def _scroll_id_from_body(body: Any) -> Optional[str]:
    if not isinstance(body, Mapping):
        return None
    scroll_id = body.get("scroll_id")
    return scroll_id if isinstance(scroll_id, str) else None


def _extract_scroll_ids(body: Any) -> Optional[Sequence[str]]:
    if isinstance(body, Mapping):
        scroll_id = body.get("scroll_id")
        if isinstance(scroll_id, str):
            return (scroll_id,)
        if isinstance(scroll_id, Sequence) and not isinstance(scroll_id, (str, bytes)):
            return tuple(scroll_id)
        if scroll_id == "_all":
            return None
    return ()


def _snapshot_handles(
    scroll_ids: Optional[Sequence[str]],
) -> Sequence[Tuple[RuntimePolicyFrame, str, _ScrollHandle]]:
    result = []
    frames = _current_frames()
    for frame in frames:
        with frame._lock:
            if scroll_ids is None:
                result.extend(
                    (frame, scroll_id, handle)
                    for scroll_id, handle in frame._active_scrolls.items()
                )
            else:
                for scroll_id in scroll_ids:
                    handle = frame._active_scrolls.get(scroll_id)
                    if handle is not None:
                        result.append((frame, scroll_id, handle))
    return result


def _mark_clear_started(body: Any) -> None:
    scroll_ids = _extract_scroll_ids(body)
    if scroll_ids == ():
        return
    for frame, _scroll_id, _handle in _snapshot_handles(scroll_ids):
        with frame._lock:
            handle = frame._active_scrolls.get(_scroll_id)
            if handle is not None and not handle.closing:
                frame._active_scrolls[_scroll_id] = _ScrollHandle(
                    scroll_id=handle.scroll_id,
                    client=handle.client,
                    is_async=handle.is_async,
                    closing=True,
                    observer_states=handle.observer_states,
                )


def _mark_clear_failed(scroll_ids: Optional[Sequence[str]]) -> None:
    for frame, current_id, _handle in _snapshot_handles(scroll_ids):
        with frame._lock:
            handle = frame._active_scrolls.get(current_id)
            if handle is not None and handle.closing:
                frame._active_scrolls[current_id] = _ScrollHandle(
                    scroll_id=handle.scroll_id,
                    client=handle.client,
                    is_async=handle.is_async,
                    closing=False,
                    observer_states=handle.observer_states,
                )


def _remove_scrolls(scroll_ids: Optional[Sequence[str]]) -> None:
    if scroll_ids == ():
        return
    for frame, current_id, _handle in _snapshot_handles(scroll_ids):
        with frame._lock:
            removed = frame._active_scrolls.pop(current_id, None)
        if removed is not None:
            for state in removed.observer_states:
                state.closed = True


def _find_owner(scroll_id: Optional[str]) -> Optional[RuntimePolicyFrame]:
    if not scroll_id:
        return None
    for frame in _current_frames():
        with frame._lock:
            if scroll_id in frame._active_scrolls:
                return frame
    return None


def _active_handles(frame: RuntimePolicyFrame) -> Sequence[_ScrollHandle]:
    with frame._lock:
        return tuple(frame._active_scrolls.values())


def _cleanup_error(frame: RuntimePolicyFrame, scroll_id: str) -> RuntimePolicyError:
    return _policy_error(
        f"Runtime policy scroll_gate.force_close failed to clear scroll {scroll_id!r}",
        axis="scroll_gate",
        mode=ScrollGate.force_close,
        endpoint_id=_CLEAR_SCROLL_ENDPOINT,
        frame_id=frame.frame_id,
    )


def _pending_cleanup_error(frame: RuntimePolicyFrame) -> RuntimePolicyError:
    return _policy_error(
        "Runtime policy scroll_gate.force_close cannot wait for an in-flight clear",
        axis="scroll_gate",
        mode=ScrollGate.force_close,
        endpoint_id=_CLEAR_SCROLL_ENDPOINT,
        frame_id=frame.frame_id,
    )


def _close_active_scrolls_sync(frame: RuntimePolicyFrame) -> None:
    handles = _active_handles(frame)
    if any(handle.is_async for handle in handles):
        raise _policy_error(
            "Cannot close an asynchronous scroll from a synchronous policy pop",
            axis="scroll_gate",
            mode=ScrollGate.force_close,
            endpoint_id=_CLEAR_SCROLL_ENDPOINT,
            frame_id=frame.frame_id,
        )
    if any(handle.closing for handle in handles):
        raise _pending_cleanup_error(frame)

    for handle in handles:
        try:
            handle.client.options(ignore_status=404).clear_scroll(
                scroll_id=handle.scroll_id
            )
        except Exception as exc:
            raise _cleanup_error(frame, handle.scroll_id) from exc


async def _close_active_scrolls_async(frame: RuntimePolicyFrame) -> None:
    handles = _active_handles(frame)
    if any(not handle.is_async for handle in handles):
        raise _policy_error(
            "Cannot close a synchronous scroll from an asynchronous policy pop",
            axis="scroll_gate",
            mode=ScrollGate.force_close,
            endpoint_id=_CLEAR_SCROLL_ENDPOINT,
            frame_id=frame.frame_id,
        )
    if any(handle.closing for handle in handles):
        raise _pending_cleanup_error(frame)

    for handle in handles:
        try:
            await handle.client.options(ignore_status=404).clear_scroll(
                scroll_id=handle.scroll_id
            )
        except Exception as exc:
            raise _cleanup_error(frame, handle.scroll_id) from exc
