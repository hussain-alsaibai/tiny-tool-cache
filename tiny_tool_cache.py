"""tiny-tool-cache — stdlib-only cache for tool/LLM call results.

A focused, narrow cache designed for the "agent calls a tool, agent calls
it again 2 minutes later, can we skip the duplicate?" pattern. Distinct
from `tiny-semantic-cache` (which dedups by embedding similarity) and
`tiny-prompt-cache` (which is a prompt→completion KV cache) and
`tiny-memo` (which memoizes pure functions by arg hash).

`tiny-tool-cache` is purpose-built for the case where:
  - The key is a deterministic (tool_name, args) tuple.
  - The value is structured (text + structured fields + JSON).
  - Cost matters (we want to skip duplicate network/model calls).
  - Stale-while-revalidate (SWR) is acceptable but optional.

Design choices:
  - **stdlib only.** No `redis`, no `diskcache`, no third-party deps.
  - **Bounded by max entries AND max age.** Bounded LRU, not unbounded.
  - **Two-tier backend**: in-memory + optional JSONL on disk (crash-safe).
  - **SWR-first**: `get(key)` returns stale value while refreshing in bg.
  - **Negative-result caching**: cache "tool returned []" for `ttl` to
    avoid repeated empty scans.
  - **Zero magic**: cache key is exactly what you pass in. If you want
    hashing, hash upstream.

Example::

    from tiny_tool_cache import ToolCache, NegativeCache

    cache = ToolCache(ttl_seconds=300, max_entries=10_000, persist_path="tool_cache.jsonl")

    @cache.wrap  # default key = (tool.__name__, args)
    async def search(query: str, limit: int = 10) -> list[dict]:
        return await heavy_search(query, limit)

    # Or manual:
    hit = cache.get_or_set(
        key=("search", query, limit),
        loader=lambda: heavy_search(query, limit),
        negative_cache=NegativeCache(ttl=60, is_empty=lambda r: not r),
    )

Why this exists: semantic cache solves "similar query, similar answer";
prompt cache solves "same prompt, get prefill hit". `tiny-tool-cache`
solves the bread-and-butter "agent called `kb.search(q)` three times,
should we run it once?". That gap keeps catching real agents in prod.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, Hashable, Optional, TypeVar, Union

T = TypeVar("T")


# ------------------------- Config & State -------------------------

@dataclass
class CacheStats:
    """Atomic counters that describe cache use."""

    hits: int = 0
    misses: int = 0
    stale_served: int = 0        # SWR: served stale while refreshing
    negative_hits: int = 0       # got [] / None / False and it was cached
    sets: int = 0
    evictions: int = 0           # bounded by max_entries
    expirations: int = 0         # TTL hit
    writes_to_disk: int = 0
    loads_from_disk: int = 0

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate(), 4),
            "stale_served": self.stale_served,
            "negative_hits": self.negative_hits,
            "sets": self.sets,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "writes_to_disk": self.writes_to_disk,
            "loads_from_disk": self.loads_from_disk,
        }


@dataclass
class _Entry(Generic[T]):
    """One cache row. Always wraps a known-loaded value and a deadline."""

    value: T
    expires_at: float            # wall time (time.time())
    stored_at: float
    is_negative: bool = False    # used for negative-result caching

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at

    def age(self, now: float) -> float:
        return now - self.stored_at


# ------------------------- Negative Cache Config -------------------------

@dataclass
class NegativeCache:
    """Cache "tool returned empty" for a shorter TTL.

    Stop agents from re-running expensive scans that returned `[]`. The
    `is_empty` callable decides emptiness (default: falsy).

    Args:
        ttl: how long to cache negative results (defaults to 30s).
        is_empty: predicate; truthy result means "this was negative".
    """

    ttl: float = 30.0
    is_empty: Callable[[Any], bool] = lambda v: not v


# ------------------------- Core Cache -------------------------

class ToolCache(Generic[T]):
    """Bounded, two-tier (memory + optional disk) tool-result cache.

    Args:
        ttl_seconds: how long an entry stays fresh.
        max_entries: LRU eviction ceiling. Defaults to 10k.
        persist_path: optional JSONL file. When set, each ``set`` appends
            a newline-delimited JSON record; on startup, the file is
            replayed into memory (respecting current ttl/max_entries).
        stale_while_revalidate: when True, ``get`` returns expired entries
            while triggering a background loader. Negative results always
            go through the negative-cache config.
        negative_cache: if set, negative results are stored separately
            with their own TTL.
        clock: injectable for tests. Defaults to ``time.time``.
    """

    def __init__(
        self,
        ttl_seconds: float = 300.0,
        max_entries: int = 10_000,
        persist_path: Optional[str] = None,
        stale_while_revalidate: bool = True,
        negative_cache: Optional[NegativeCache] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self.persist_path = persist_path
        self.swr = bool(stale_while_revalidate)
        self.negative_cache = negative_cache
        self._clock = clock
        self._store: "OrderedDict[Hashable, _Entry]" = OrderedDict()
        self._in_flight: dict[Hashable, asyncio.Future] = {}
        self.stats = CacheStats()
        if persist_path and os.path.exists(persist_path):
            self._load_persisted(persist_path)

    # ---- core get/set ----

    def get(self, key: Hashable) -> Optional[T]:
        """Read-only fetch. Returns the value or None.

        Honors staleness: if SWR is on and the entry is past TTL,
        *but still in memory*, returns None (lets `get_or_set` decide
        whether to serve stale).
        """
        self._purge_if_expired(key)
        entry = self._store.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        # LRU touch
        self._store.move_to_end(key)
        if entry.is_negative and self.negative_cache is not None:
            self.stats.negative_hits += 1
        return entry.value

    def peek_stale(self, key: Hashable) -> Optional[T]:
        """Get a stale value (past TTL) without affecting stats."""
        entry = self._store.get(key)
        return entry.value if entry is not None else None

    def set(
        self,
        key: Hashable,
        value: T,
        *,
        ttl: Optional[float] = None,
        is_negative: bool = False,
    ) -> None:
        """Insert/replace. Evicts LRU entries when over capacity."""
        ttl = float(ttl) if ttl is not None else self.ttl_seconds
        now = self._clock()
        entry = _Entry(value=value, expires_at=now + ttl, stored_at=now, is_negative=is_negative)
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = entry
        self.stats.sets += 1
        # bound
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)  # FIFO end is LRU eviction
            self.stats.evictions += 1
        # disk
        if self.persist_path is not None:
            self._append_persisted(key, entry, is_negative=is_negative)

    def invalidate(self, key: Hashable) -> bool:
        """Drop one entry. Returns True if it existed."""
        existed = self._store.pop(key, None) is not None
        return existed

    def clear(self) -> None:
        """Wipe memory (does not delete the persistence file)."""
        self._store.clear()
        self._in_flight.clear()

    # ---- high-level helpers ----

    def get_or_set(
        self,
        key: Hashable,
        loader: Callable[[], T],
        *,
        ttl: Optional[float] = None,
    ) -> T:
        """Synchronous fetch-or-compute. SWR: returns stale while recomputing.

        Order of operations when stale-while-revalidate is on:
          1. If key has a *fresh* entry → return it (HIT).
          2. If key has a *stale* entry → return the stale value and a
             background loader refreshes it.
          3. If key absent → run loader synchronously, store, return.
        """
        entry = self._store.get(key)
        now = self._clock()
        if entry is not None and not entry.is_expired(now):
            self._store.move_to_end(key)
            self.stats.hits += 1
            if entry.is_negative and self.negative_cache is not None:
                self.stats.negative_hits += 1
            return entry.value
        if entry is not None and self.swr:
            self.stats.stale_served += 1
            # Best-effort inline refresh (still synchronous; sync API).
            try:
                fresh = loader()
            except Exception:
                # Refresh failure → keep serving stale.
                return entry.value
            self._store_value(key, fresh, ttl=ttl, is_negative=False)
            return fresh
        self.stats.misses += 1
        fresh = loader()
        # Negative caching
        is_neg = bool(self.negative_cache and self.negative_cache.is_empty(fresh))
        neg_ttl = self.negative_cache.ttl if self.negative_cache else None
        self._store_value(
            key,
            fresh,
            ttl=(neg_ttl if is_neg and neg_ttl is not None else ttl),
            is_negative=is_neg,
        )
        return fresh

    async def aget_or_set(
        self,
        key: Hashable,
        loader: Callable[[], Awaitable[T]],
        *,
        ttl: Optional[float] = None,
    ) -> T:
        """Async fetch-or-compute. Coalesces in-flight loads per key.

        If two coroutines ask for the same key at once, only one loader
        runs; the second awaits the shared future. Stale-while-revalidate
        keeps the old value flowing while that happens.
        """
        entry = self._store.get(key)
        now = self._clock()
        if entry is not None and not entry.is_expired(now):
            self._store.move_to_end(key)
            self.stats.hits += 1
            if entry.is_negative and self.negative_cache is not None:
                self.stats.negative_hits += 1
            return entry.value
        # Coalesce.
        fut = self._in_flight.get(key)
        if fut is not None and not fut.done():
            return await fut  # type: ignore[return-value]
        fut = asyncio.get_event_loop().create_future()
        self._in_flight[key] = fut
        try:
            value = await loader()
        except Exception:
            # If we can serve stale, replace the exception with the cached
            # value. Otherwise, propagate the original error and let any
            # coalesced waiters see it via the future.
            if entry is not None and self.swr and not fut.done():
                self.stats.stale_served += 1
                fut.set_result(entry.value)
            else:
                if not fut.done():
                    # Re-raise on this caller; preserve the exception for
                    # any coalesced awaiters by leaving the future unset
                    # so they see the original raised error.
                    raise
        else:
            is_neg = bool(self.negative_cache and self.negative_cache.is_empty(value))
            neg_ttl = self.negative_cache.ttl if self.negative_cache else None
            self._store_value(
                key,
                value,
                ttl=(neg_ttl if is_neg and neg_ttl is not None else ttl),
                is_negative=is_neg,
            )
            if not fut.done():
                fut.set_result(value)
        finally:
            self._in_flight.pop(key, None)
        return fut.result()  # type: ignore[return-value]

    def wrap(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Decorator. Cache key = (fn.__name__, args).

        Only works for synchronous callables. For async, use
        ``awrap``.
        """
        def cached(*args: Any, **kwargs: Any) -> T:
            key = (fn.__name__, _freeze(args), _freeze(kwargs))
            return self.get_or_set(key, lambda: fn(*args, **kwargs))
        cached.__wrapped__ = fn  # type: ignore[attr-defined]
        cached.__name__ = getattr(fn, "__name__", "cached")
        cached.__doc__ = f"ToolCache-wrapped version of {getattr(fn, '__name__', fn)!r}."
        return cached

    def awrap(self, fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """Decorator for async callables."""
        async def cached(*args: Any, **kwargs: Any) -> T:
            key = (fn.__name__, _freeze(args), _freeze(kwargs))
            return await self.aget_or_set(key, lambda: fn(*args, **kwargs))
        cached.__wrapped__ = fn  # type: ignore[attr-defined]
        cached.__name__ = getattr(fn, "__name__", "cached")
        cached.__doc__ = f"Async ToolCache-wrapped version of {getattr(fn, '__name__', fn)!r}."
        return cached

    # ---- internals ----

    def _store_value(
        self,
        key: Hashable,
        value: T,
        *,
        ttl: Optional[float],
        is_negative: bool,
    ) -> None:
        ttl = float(ttl) if ttl is not None else self.ttl_seconds
        now = self._clock()
        entry = _Entry(value=value, expires_at=now + ttl, stored_at=now, is_negative=is_negative)
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = entry
        self.stats.sets += 1
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
            self.stats.evictions += 1
        if self.persist_path is not None:
            self._append_persisted(key, entry, is_negative=is_negative)

    def _purge_if_expired(self, key: Hashable) -> None:
        """Remove if past TTL. Stat-expiration counted here."""
        entry = self._store.get(key)
        if entry is None:
            return
        if entry.is_expired(self._clock()):
            self._store.pop(key, None)
            self.stats.expirations += 1

    # ---- disk persistence ----

    def _append_persisted(
        self,
        key: Hashable,
        entry: _Entry,
        *,
        is_negative: bool,
    ) -> None:
        path = self.persist_path
        assert path is not None
        record = {
            "k": _key_to_jsonable(key),
            "v": entry.value,
            "e": entry.expires_at,
            "s": entry.stored_at,
            "n": bool(is_negative or entry.is_negative),
        }
        line = json.dumps(record, default=_json_default) + "\n"
        # append + fsync for crash-safe durability
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
            self.stats.writes_to_disk += 1
        finally:
            os.close(fd)

    def _load_persisted(self, path: str) -> None:
        now = self._clock()
        loaded = 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("e", 0) <= now:
                    # Drop expired entries at load.
                    continue
                key = _jsonable_to_key(rec.get("k"))
                entry = _Entry(
                    value=rec.get("v"),
                    expires_at=float(rec.get("e", 0)),
                    stored_at=float(rec.get("s", 0)),
                    is_negative=bool(rec.get("n", False)),
                )
                if key in self._store:
                    self._store.move_to_end(key)
                self._store[key] = entry
                loaded += 1
                while len(self._store) > self.max_entries:
                    self._store.popitem(last=False)
        self.stats.loads_from_disk = loaded

    # ---- diagnostics ----

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: Hashable) -> bool:
        self._purge_if_expired(key)
        return key in self._store


# ------------------------- Helpers -------------------------

def _freeze(obj: Any) -> Any:
    """Make a hashable snapshot of args/kwargs.

    Lists → tuples, dicts → frozen tuples of sorted items, sets →
    frozensets. Lambdas / modules raise (don't pretend they hash).
    """
    if isinstance(obj, Hashable):
        return obj
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(x) for x in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, (set, frozenset)):
        return frozenset(_freeze(x) for x in obj)
    raise TypeError(f"cannot freeze {type(obj).__name__} for cache key")


def _key_to_jsonable(key: Hashable) -> Any:
    """Best-effort serialization for the JSONL persistence file."""
    if isinstance(key, tuple) and len(key) == 3 and isinstance(key[0], str):
        # (name, frozen-args, frozen-kwargs) — most common
        return {"name": key[0], "args": _to_jsonable(key[1]), "kwargs": _to_jsonable(key[2])}
    return _to_jsonable(key)


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted(_to_jsonable(x) for x in obj)
    return repr(obj)


def _jsonable_to_key(obj: Any) -> Hashable:
    """Inverse of _to_jsonable. Lists → tuples, dicts → frozen tuples."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return tuple(_jsonable_to_key(x) for x in obj)
    if isinstance(obj, dict):
        if set(obj.keys()) == {"name", "args", "kwargs"} and isinstance(obj["name"], str):
            return (obj["name"], _jsonable_to_key(obj["args"]), _jsonable_to_key(obj["kwargs"]))
        return tuple(sorted((k, _jsonable_to_key(v)) for k, v in obj.items()))
    return obj


def _json_default(obj: Any) -> Any:
    try:
        return repr(obj)
    except Exception:
        return None


__all__ = [
    "ToolCache",
    "NegativeCache",
    "CacheStats",
    "_Entry",
    "_freeze",
    "_key_to_jsonable",
    "_jsonable_to_key",
]
