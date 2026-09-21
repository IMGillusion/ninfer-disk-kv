# diskkv — L3 (disk) tier for the NInfer KV cache

A cold, file-backed third level for the NInfer KV-cache hierarchy, modeled on
SGLang's HiCache L3 ("host full → demote to disk, evict only when the disk
budget is also full").

## Why

NInfer's context cache is two levels today:

```
L1  device page pool   --kv-capacity tokens   (~13.9 GiB on a 5090)
L2  pinned host arena  --host-kv-mib           (e.g. 6 GiB ≈ 250K tokens)
```

When L2 is full, the pressure planner drops the affected checkpoint
**permanently**, and the next turn of that session pays full prefill
recompute (37–80 s for a 150–200K-token context). On a production agent fleet
(9 agents, 5×40K + 4×100–200K tokens, peak concurrent working set 0.8–1.2M
tokens vs ~0.87M tokens of cache budget) this causes chronic eviction waves:
thousands of dropped checkpoints per day and repeated "expired while waiting
for admission" timeouts.

`diskkv` adds a **content-addressed L3** under the host arena:

```
L1  device pages ──demote──▶ L2  pinned host ──demote──▶ L3  NVMe file (this repo)
   (13.9 GiB)                  (e.g. 6 GiB)                (e.g. 64 GiB ≈ 2.8M tokens)
```

A "host full" event demotes the checkpoint's page run to disk instead of
dropping it; a later turn restores it (disk read + H2D, ≈10 s) instead of
recomputing (37–80 s). A miss (never spilt, LRU-evicted from disk, or CRC
mismatch) falls back to exactly what the engine does today — strictly
better-or-equal, never worse.

## Components

| File | Role |
|---|---|
| `include/diskkv/disk_kv_store.h` / `src/disk_kv_store.cpp` | The store. One mmap'd file per KV family; fixed-size page slots; per-group CRC32C; LRU eviction; coarse lock (cold path, never on the decode hot loop). Self-contained: no CUDA, no engine types. |
| `include/diskkv/disk_kv_bridge.h` / `src/disk_kv_bridge.cpp` | Engine-facing facade. Identity-keyed groups — `spill(id, kind, pages, bytes)` / `restore_page(id, kind, page, dst)` / `contains` / `touch` — with per-family files (`main.diskkv`, `backend.diskkv`, `state.diskkv`), automatic LRU make-room, miss→recompute fallback, and counters. |
| `tests/disk_kv_store_test.cpp` | Store logic: round-trip, capacity boundary, LRU order, CRC corruption rejection, size guards. |
| `tests/disk_kv_bridge_test.cpp` | Bridge round-trip: spill → free host replica → restore with byte-identity, LRU evicts the coldest identity, unknown identity misses, family isolation, stats. |

### Key design decisions

- **Content-derived identity key.** A released KV page's engine descriptor
  (and its `content_epoch`) is reclaimed under pressure, so the disk key
  cannot reference the descriptor. Keys are
  `DiskKVIdentity{lo, hi, kind, frontier}` where `(lo, hi)` is the 128-bit
  rolling content digest NInfer already computes per token frontier
  (`PrefixShortlistDigests`). Same content ⇒ same digest at spill time and
  at restore time. A key that never matches is a miss, not corruption.
- **CRC32C on every group.** Publish computes the CRC; restore verifies it.
  A torn/corrupted group is treated as a miss (recompute), never fed to the
  model.
- **mmap'd backing file, LRU inside it.** The kernel page cache gives
  write-back persistence for free; eviction is in-file, so the disk budget
  is a hard ceiling that never asks for more space.
- **Per-family stores.** Main KV, backend KV (MTP/DFlash) and state images
  have different page strides; each family owns its own file, stride and
  LRU clock. A family whose stride is 0 is simply disabled.
- **Best-effort crash semantics.** The file is truncated on open; a
  process restart starts with an empty L3 (a recompute happens, as today).
  L3 is a cache tier, not a WAL — durability across engine crashes is out of
  scope by design.

### Measured performance (RTX 5090 host, NVMe, g++ 11.4 -O2, 1.51 MiB/page)

| op | rate |
|---|---|
| spill (host→disk write) | ~245 MB/s |
| restore (disk read, page cache warm) | ~613 MB/s |
| full restore of a 0.95 GB run | ~1.6 s |
| a 200K-token session (≈4.7 GB) restore | ≈ 8 s read + H2D, vs 37–80 s recompute |

## Build & test

Requires C++20. Two options:

```bash
# standalone
cmake -S . -B build -G Ninja && cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

or the minimal manual build the engine uses for quick checks:

```bash
g++ -std=c++20 -O2 -Wall -Wextra -I include \
    tests/disk_kv_store_test.cpp src/disk_kv_store.cpp -o dks_test
g++ -std=c++20 -O2 -Wall -Wextra -I include \
    tests/disk_kv_bridge_test.cpp src/disk_kv_bridge.cpp src/disk_kv_store.cpp -o dkb_test
./dks_test /tmp/l.bin 1589632 150
./dkb_test /tmp/dkb
```

Both suites exit 0 with `ALL TESTS PASSED`.

## Integrating into NInfer

The store and bridge are engine-type-free on purpose. Integration into
NInfer (branch `pre-disktier-20260921` worktree) touches two seams:

1. **Write seam** — where a host KV run is released (all four
   `release_page_replicas` paths funnel through the extent partitioner):
   copy the run's bytes out before they die and call
   `bridge.spill(id, MainKV/BackendKV, page_count, bytes)`.
2. **Read seam** — where a materialization restore throws
   `checkpoint KV page has no restorable replica`: look the identity up in
   the bridge; on a hit, re-hydrate a fresh host extent from the restored
   bytes so the engine's *existing* H2D restore path runs unchanged; on a
   miss, fall through to recompute exactly as today.

Both seams are gated behind `ContextCacheOptions` flags
(`--disk-kv-path`, `--disk-kv-gib`), default-off; with the flags unset the
engine behaves bit-for-bit as before.

## Status

- [x] Store + bridge implemented, unit-tested, measured
- [x] Standalone repo, standalone CMake, tests green
- [ ] Engine seams (write/read) — in progress in the NInfer worktree
- [ ] Isolated-container validation under real load (eviction waves absorbed
      by L3 hits instead of drops)

## License

Apache-2.0 (see `LICENSE`). Derived from / integrates with [NInfer](https://github.com/Neroued/ninfer)
(Apache-2.0) and is subject to its terms.
