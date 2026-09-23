# tiny-tool-cache

> **stdlib-only cache for tool / LLM call results.**
> Bounded LRU, TTL, stale-while-revalidate, negative-result caching,
> crash-safe JSONL disk persistence, async-coalesced loads.
> Zero third-party dependencies. Single file. MIT.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#)
[![stdlib only](https://img.shields.io/badge/stdlib-only-success.svg)](#)
[![tests](https://img.shields.io/badge/tests-33%2F33-success.svg)](#)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](#)

## Why does this exist?

If you build autonomous agents, you've hit *the* pattern:

```
agent: tool.kb_search("What is X?")
   → 800ms network call, expensive embed, returns [{...}, {...}]

agent (10 seconds later): tool.kb_search("What is X?")
   → 800ms network call, again
```

Three problems at once:

1. **Duplicate compute.** Same key, same answer.
2. **Stale-not-served.** When a 5-minute-old result still answers the question,
   agents should keep using it while a background refresh kicks off
   (stale-while-revalidate).
3. **Negative-result storms.** Agent asks → tool returns `[]` → agent retries
   → still `[]` → retry storm. The empty result is itself worth caching.

`tiny-tool-cache` solves those three. It is **not**:

- a `redis` client (use `redis`),
- a process-shared cache (in-process only; for cross-process use shared memory),
- a `semantic` cache (use [`tiny-semantic-cache`](https://github.com/hussain-alsaibai/tiny-semantic-cache)),
- a `prompt → completion` cache (use [`tiny-prompt-cache`](https://github.com/hussain-alsaibai/tiny-prompt-cache)),
- a `memoize-pure-functions` library (use [`tiny-memo`](https://github.com/hussain-alsaibai/tiny-memo)).

Those three sibling repos live in the **tiny-\*** family and solve adjacent problems.
This one is the missing middle: deterministic-key, bounded, TTL-aware, agent-shape.

## Install

```bash
# Copy one file
curl -O https://raw.githubusercontent.com/hussain-alsaibai/tiny-tool-cache/main/tiny_tool_cache.py
```

or

```bash
pip install tiny-tool-cache   # (when published; see pyproject.toml)
```

## Quickstart

### Decorator style (smallest)

```python
from tiny_tool_cache import ToolCache, NegativeCache

cache = ToolCache(
    ttl_seconds=300,
    max_entries=10_000,
    negative_cache=NegativeCache(ttl=60, is_empty=lambda v: not v),
)

@cache.wrap
def search_kb(query: str, limit: int = 10):
    return kb_client.search(query, limit=limit)
```

First call: runs `search_kb(...)`, caches the result. Calls within 300s
return cached. Calls after 300s trigger SWR — return the stale value
while recomputing in the background.

Async works the same: `@cache.awrap`.

### Manual style (more control)

```python
from tiny_tool_cache import ToolCache

cache = ToolCache(ttl_seconds=300, max_entries=50_000)

def load_user(uid: int):
    return db.users.find(uid)

uid = 42
user = cache.get_or_set(("users.find", uid), lambda: load_user(uid))
```

### Crash-safe persistence

```python
cache = ToolCache(
    ttl_seconds=900,
    max_entries=20_000,
    persist_path="/var/tmp/tool-cache.jsonl",   # append-only JSONL
)
```

On startup, the file is replayed into memory (TTL-bounded). On every
`set`, a new line is appended with `fsync()` so a hard kill mid-flight
won't corrupt the cache — at worst, the last few writes are lost.

### Async coalescing

Two concurrent calls to the same key share **one** loader:

```python
import asyncio
from tiny_tool_cache import ToolCache

cache = ToolCache(ttl_seconds=60)

calls = []
async def fetch_one():
    calls.append(1)
    await asyncio.sleep(0.05)
    return "value"

async def runner():
    return await asyncio.gather(
        cache.aget_or_set("k", fetch_one),
        cache.aget_or_set("k", fetch_one),
    )

results = asyncio.run(runner())    # ONE loader call, two results
```

### Negative-result caching (stop empty-result retry storms)

```python
from tiny_tool_cache import ToolCache, NegativeCache

nc = NegativeCache(ttl=60, is_empty=lambda v: len(v) == 0)
cache = ToolCache(negative_cache=nc)

@cache.wrap
def list_open_tickets(since_minutes: int):
    return support.list_tickets(since=since_minutes)
```

When `support.list_tickets(...)` returns `[]`, that empty result is cached
for 60s (vs 5 minutes for non-empty results). Three retries during a
ticket storm collapse into one upstream call.

### Stats

```python
print(cache.stats.to_dict())
# {'hits': 487, 'misses': 21, 'hit_rate': 0.9586, 'stale_served': 4,
#  'negative_hits': 9, 'sets': 21, 'evictions': 0, 'expirations': 0,
#  'writes_to_disk': 21, 'loads_from_disk': 21}
```

### Sync vs async exception behavior

| Mode | Loader raises + stale exists | Loader raises + no stale |
|------|------------------------------|--------------------------|
| `get_or_set`, SWR on  | return stale | raise |
| `get_or_set`, SWR off | raise | raise |
| `aget_or_set`, SWR on | return stale (coalesced waiters see stale too) | raise |
| `aget_or_set`, SWR off| raise | raise |

## API surface

```python
class ToolCache:
    def __init__(
        self,
        ttl_seconds: float = 300.0,
        max_entries: int = 10_000,
        persist_path: Optional[str] = None,
        stale_while_revalidate: bool = True,
        negative_cache: Optional[NegativeCache] = None,
        clock: Callable[[], float] = time.time,
    ): ...

    def get(self, key) -> Any: ...
    def peek_stale(self, key) -> Any: ...   # like get, but no stats impact, returns even-stale
    def set(self, key, value, ttl=None, is_negative=False) -> None: ...
    def invalidate(self, key) -> bool: ...
    def clear(self) -> None: ...
    def __contains__(self, key) -> bool: ...
    def __len__(self) -> int: ...

    def get_or_set(self, key, loader, *, ttl=None) -> Any: ...
    async def aget_or_set(self, key, loader, *, ttl=None) -> Any: ...

    def wrap(self, fn) -> Callable: ...     # sync decorator
    def awrap(self, fn) -> Callable: ...    # async decorator

    @property
    def stats(self) -> CacheStats: ...

@dataclass
class NegativeCache:
    ttl: float = 30.0
    is_empty: Callable[[Any], bool] = lambda v: not v

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    stale_served: int = 0
    negative_hits: int = 0
    sets: int = 0
    evictions: int = 0
    expirations: int = 0
    writes_to_disk: int = 0
    loads_from_disk: int = 0
    def hit_rate(self) -> float: ...
    def to_dict(self) -> dict: ...
```

## How key freezing works

The cache key is whatever you pass in — but `wrap`/`awrap` freeze args
into a tuple so `(1, 2)` and `(2, 1)` hash to the same key, and kwargs
become order-independent. Lambdas and modules are rejected with
`TypeError` (don't pretend they hash).

For manual usage, use a tuple: `("kb.search", query, limit)`.

## Composition with other tiny-\* libraries

| Pair | What you get |
|------|--------------|
| `tiny-tool-cache` + `tiny-budget-ledger` | every cached call auto-counts toward the budget; if the budget throws, no work is done |
| `tiny-tool-cache` + `tiny-trace-otel` | every cached call carries a span; hits show `cache.hit`, misses show `cache.miss` |
| `tiny-tool-cache` + `tiny-rate-limiter` | the limiter gates the loader, the cache short-circuits when the limiter would have throttled |
| `tiny-tool-cache` + `tiny-cli-table` | render `cache.stats.to_dict()` as a CLI table |

## Why stdlib-only?

- You can read all of it in five minutes.
- No `pip install` in the agent's bootstrap path.
- No supply-chain surprise at 03:00.
- No transitive version-conflict drama with the agent's other deps.

The whole library is one file, ~470 LOC including docstrings, 33 unit tests, zero required imports beyond `time`, `json`, `os`, `asyncio`, `dataclasses`, `collections`, `typing`.

## Tests

```bash
cd repos/tiny-tool-cache
python3 -m unittest test_tiny_tool_cache       # 33/33 OK
```

## License

MIT — see [`LICENSE`](LICENSE).
