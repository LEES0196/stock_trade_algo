"""
Asynchronous Event Bus for pub/sub between components.

Design
------
We implement a simple type-based pub/sub. Subscribers register an async
callable per event type. Publishers put events on an asyncio queue, and a
single dispatcher task fans out to all handlers subscribed to that event's
class.

Mathematical/Architecture Note
------------------------------
This is the backbone of the Event-Driven Architecture. At discrete time t, a
MarketEvent m_t arrives. Strategy maps m_t -> signal s_t. Risk maps
s_t -> allowed/blocked. Executor maps allowed signal -> order a_t. Fill f_t
updates the portfolio state, used by the next step.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Type, TypeVar

from .events import Event

Handler = Callable[[Event], Awaitable[None]]
E = TypeVar("E", bound=Event)


class EventBus:
    """Async pub/sub event bus.

    - subscribe(event_type, handler): register a coroutine for an event type
    - publish(event): enqueue an event for dispatch
    - run(): run the dispatcher loop (until stop())
    """

    def __init__(self) -> None:
        self._subs: Dict[Type[Event], List[Handler]] = defaultdict(list)
        self._queue: "asyncio.Queue[Event]" = asyncio.Queue()
        self._stop = asyncio.Event()

    def subscribe(self, event_type: Type[E], handler: Handler) -> None:
        """Register a coroutine handler for a specific event type."""

        self._subs[event_type].append(handler)

    async def publish(self, event: Event) -> None:
        """Enqueue an event for processing."""

        await self._queue.put(event)

    async def _dispatch(self, event: Event) -> None:
        """Dispatch a single event to all matching subscribers.

        We match on exact class type; extendable to isinstance checks if we
        later add hierarchies.
        """

        handlers = self._subs.get(type(event), [])
        await asyncio.gather(*(h(event) for h in handlers))

    async def run(self) -> None:
        """Run the dispatch loop until stop() is called."""

        while not self._stop.is_set():
            event = await self._queue.get()
            try:
                await self._dispatch(event)
            finally:
                self._queue.task_done()

    def stop(self) -> None:
        """Signal the dispatcher to stop after current tasks."""

        self._stop.set()

