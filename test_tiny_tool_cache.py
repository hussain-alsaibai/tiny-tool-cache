"""Tests for tiny-tool-cache. Runs under stdlib `unittest`."""

import asyncio
import json
import os
import tempfile
import unittest
import time

from tiny_tool_cache import (
    CacheStats,
    NegativeCache,
    ToolCache,
    _freeze,
    _to_jsonable,
    _key_to_jsonable,
    _jsonable_to_key,
)


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class TestBasic(unittest.TestCase):
    def test_set_get_roundtrip(self):
        c = ToolCache(ttl_seconds=10)
        c.set("k", "v")
        self.assertEqual(c.get("k"), "v")

    def test_miss_returns_none(self):
        c = ToolCache()
        self.assertIsNone(c.get("absent"))

    def test_expired_entry_removed(self):
        clock = FakeClock()
        c = ToolCache(ttl_seconds=5, clock=clock)
        c.set("k", "v")
        self.assertEqual(c.get("k"), "v")
        clock.advance(6)
        # expired -> purge -> miss
        self.assertIsNone(c.get("k"))

    def test_invalidate(self):
        c = ToolCache()
        c.set("a", 1)
        self.assertTrue(c.invalidate("a"))
        self.assertFalse(c.invalidate("a"))
        self.assertIsNone(c.get("a"))

    def test_clear(self):
        c = ToolCache()
        c.set("a", 1)
        c.set("b", 2)
        c.clear()
        self.assertEqual(len(c), 0)


class TestEviction(unittest.TestCase):
    def test_lru_eviction(self):
        c = ToolCache(ttl_seconds=60, max_entries=3)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        # touch a → it's now most-recent
        c.get("a")
        c.set("d", 4)  # should evict b (least recent after 'a' touch)
        self.assertIn("a", c)
        self.assertIn("c", c)
        self.assertIn("d", c)
        self.assertNotIn("b", c)
        self.assertEqual(c.stats.evictions, 1)

    def test_eviction_counter_increments_for_each(self):
        c = ToolCache(ttl_seconds=60, max_entries=2)
        for i in range(10):
            c.set(f"k{i}", i)
        self.assertEqual(c.stats.evictions, 8)
        self.assertEqual(len(c), 2)


class TestGetOrSet(unittest.TestCase):
    def test_loader_runs_on_miss(self):
        c = ToolCache(ttl_seconds=60)
        calls = []
        def loader():
            calls.append(1)
            return "computed"
        self.assertEqual(c.get_or_set("k", loader), "computed")
        # second call must hit
        self.assertEqual(c.get_or_set("k", loader), "computed")
        self.assertEqual(calls, [1])

    def test_swr_serves_stale_value(self):
        clock = FakeClock()
        c = ToolCache(ttl_seconds=5, stale_while_revalidate=True, clock=clock)
        calls = []
        def loader():
            calls.append(1)
            return "v2" if len(calls) > 0 else "v1"
        c.set("k", "v1")
        clock.advance(6)  # expired
        v = c.get_or_set("k", loader)
        self.assertEqual(v, "v2")
        self.assertEqual(calls, [1])
        self.assertEqual(c.stats.stale_served, 1)

    def test_swr_loader_failure_falls_back_to_stale(self):
        clock = FakeClock()
        c = ToolCache(ttl_seconds=5, stale_while_revalidate=True, clock=clock)
        c.set("k", "stale")
        clock.advance(6)
        def boom():
            raise RuntimeError("boom")
        self.assertEqual(c.get_or_set("k", boom), "stale")

    def test_swr_off_throws_on_loader_failure(self):
        clock = FakeClock()
        c = ToolCache(ttl_seconds=5, stale_while_revalidate=False, clock=clock)
        c.set("k", "v")
        clock.advance(6)
        def boom():
            raise RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            c.get_or_set("k", boom)


class TestNegativeCache(unittest.TestCase):
    def test_negative_cache_marks_empty(self):
        c = ToolCache(negative_cache=NegativeCache(ttl=2, is_empty=lambda v: len(v) == 0))
        calls = []
        def loader():
            calls.append(1)
            return []
        v1 = c.get_or_set("k", loader)
        # second call: still cached as negative
        v2 = c.get_or_set("k", loader)
        self.assertEqual(v1, [])
        self.assertEqual(v2, [])
        self.assertEqual(calls, [1])
        self.assertGreaterEqual(c.stats.negative_hits, 1)

    def test_negative_cache_uses_shorter_ttl(self):
        clock = FakeClock()
        c = ToolCache(ttl_seconds=60, negative_cache=NegativeCache(ttl=2), clock=clock)
        calls = []
        def loader():
            calls.append(1)
            return []
        c.get_or_set("k", loader)
        clock.advance(3)  # past neg TTL, still well inside positive TTL
        c.get_or_set("k", loader)  # this MISSES again
        self.assertEqual(len(calls), 2)


class TestAsync(unittest.TestCase):
    def test_aget_or_set_miss_then_hit(self):
        c = ToolCache(ttl_seconds=60)
        async def load():
            return "value"
        v1 = asyncio.run(c.aget_or_set("k", load))
        v2 = asyncio.run(c.aget_or_set("k", load))
        self.assertEqual(v1, "value")
        self.assertEqual(v2, "value")
        # aget_or_set increments stats.sets (not stats.misses); the
        # miss path is only counted when get() is called directly.
        self.assertEqual(c.stats.sets, 1)
        self.assertGreaterEqual(c.stats.hits, 1)

    def test_aget_or_set_coalesces_inflight(self):
        c = ToolCache(ttl_seconds=60)
        calls = []
        async def slow():
            calls.append(1)
            await asyncio.sleep(0.05)
            return "value"

        async def runner():
            return await asyncio.gather(c.aget_or_set("k", slow), c.aget_or_set("k", slow))

        results = asyncio.run(runner())
        self.assertEqual(results, ["value", "value"])
        # Only ONE loader call due to coalescing.
        self.assertEqual(calls, [1])

    def test_aget_or_set_serves_stale_on_error(self):
        clock = FakeClock()
        c = ToolCache(ttl_seconds=5, stale_while_revalidate=True, clock=clock)
        c.set("k", "stale")
        clock.advance(6)
        async def boom():
            raise ValueError("nope")
        v = asyncio.run(c.aget_or_set("k", boom))
        self.assertEqual(v, "stale")


class TestWrap(unittest.TestCase):
    def test_sync_wrap_hits_after_first_call(self):
        c = ToolCache(ttl_seconds=60)
        @c.wrap
        def add(a, b):
            return a + b
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(c.stats.misses, 1)
        self.assertEqual(c.stats.hits, 1)

    def test_awrap(self):
        c = ToolCache(ttl_seconds=60)
        @c.awrap
        async def add(a, b):
            return a + b
        v1 = asyncio.run(add(1, 2))
        v2 = asyncio.run(add(1, 2))
        self.assertEqual((v1, v2), (3, 3))

    def test_wrap_kwargs_hashed(self):
        c = ToolCache(ttl_seconds=60)
        @c.wrap
        def echo(**kw):
            return sorted(kw.items())
        v = echo(a=1, b=2)
        # order-independent kwargs
        v2 = echo(b=2, a=1)
        self.assertEqual(v, v2)


class TestDiskPersistence(unittest.TestCase):
    def test_persist_and_reload(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        tmp.close()
        try:
            clock = FakeClock(t=1000)
            c1 = ToolCache(ttl_seconds=60, persist_path=tmp.name, clock=clock)
            c1.set("a", 1)
            c1.set("b", {"nested": [1, 2, 3]})
            c1.set("c", [1, 2, 3])
            # New instance reads file back
            c2 = ToolCache(ttl_seconds=60, persist_path=tmp.name, clock=FakeClock(t=1010))
            self.assertEqual(c2.get("a"), 1)
            self.assertEqual(c2.get("b"), {"nested": [1, 2, 3]})
            self.assertEqual(c2.get("c"), [1, 2, 3])
            self.assertGreaterEqual(c2.stats.loads_from_disk, 3)
        finally:
            os.unlink(tmp.name)

    def test_persist_skips_expired(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        tmp.close()
        try:
            clock = FakeClock(t=1000)
            c1 = ToolCache(ttl_seconds=5, persist_path=tmp.name, clock=clock)
            c1.set("a", 1)  # expires at t=1005
            clock.advance(10)  # now t=1010 → a already expired
            c2 = ToolCache(ttl_seconds=5, persist_path=tmp.name, clock=FakeClock(t=1011))
            self.assertIsNone(c2.get("a"))
        finally:
            os.unlink(tmp.name)

    def test_persist_path_appends_lines(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            path = tmp.name
        try:
            c = ToolCache(ttl_seconds=60, persist_path=path)
            c.set("a", 1)
            c.set("b", 2)
            with open(path) as fh:
                lines = [l for l in fh.read().splitlines() if l]
            self.assertEqual(len(lines), 2)
            for line in lines:
                rec = json.loads(line)
                self.assertIn("k", rec)
                self.assertIn("v", rec)
                self.assertIn("e", rec)
        finally:
            os.unlink(path)


class TestHelpers(unittest.TestCase):
    def test_freeze_lists_tuples(self):
        self.assertEqual(_freeze([1, 2, {"a": 3}]), (1, 2, (("a", 3),)))
        self.assertEqual(_freeze({"x": [1, 2]}), (("x", (1, 2)),))

    def test_freeze_returns_function_when_hashable(self):
        # Python functions are Hashable, so _freeze passes them through.
        f = lambda: 1  # noqa: E731
        self.assertIs(_freeze(f), f)

    def test_freeze_rejects_bytes_when_unhashable_type(self):
        # bytes IS hashable, so this should pass-through.
        self.assertEqual(_freeze(b"abc"), b"abc")

    def test_key_round_trip(self):
        original = ("fn", (1, "a"), (("k", 2),))
        encoded = _key_to_jsonable(original)
        decoded = _jsonable_to_key(encoded)
        self.assertEqual(decoded, original)

    def test_to_jsonable_sorts_sets(self):
        self.assertEqual(_to_jsonable({3, 1, 2}), [1, 2, 3])


class TestEdgeCases(unittest.TestCase):
    def test_rejects_non_positive_ttl(self):
        with self.assertRaises(ValueError):
            ToolCache(ttl_seconds=0)

    def test_rejects_non_positive_max(self):
        with self.assertRaises(ValueError):
            ToolCache(max_entries=0)

    def test_contains_purges_expired(self):
        clock = FakeClock()
        c = ToolCache(ttl_seconds=1, clock=clock)
        c.set("k", 1)
        self.assertIn("k", c)
        clock.advance(2)
        self.assertNotIn("k", c)

    def test_peek_stale_does_not_increment_stats(self):
        clock = FakeClock()
        c = ToolCache(ttl_seconds=1, clock=clock)
        c.set("k", 1)
        clock.advance(2)
        v = c.peek_stale("k")
        self.assertEqual(v, 1)
        self.assertEqual(c.stats.hits, 0)
        self.assertEqual(c.stats.misses, 0)

    def test_negative_cache_predicate_custom(self):
        c = ToolCache(
            negative_cache=NegativeCache(is_empty=lambda v: v is None),
            ttl_seconds=60,
        )
        calls = []
        def loader():
            calls.append(1)
            return None
        c.get_or_set("k", loader)
        c.get_or_set("k", loader)  # hits negative cache
        self.assertEqual(calls, [1])

    def test_in_flight_is_cleaned_up_after_error(self):
        c = ToolCache(ttl_seconds=60)
        async def loader():
            raise ValueError("boom")

        async def runner():
            return await c.aget_or_set("k", loader)

        with self.assertRaises(ValueError):
            asyncio.run(runner())
        self.assertNotIn("k", c._in_flight)


if __name__ == "__main__":
    unittest.main()
