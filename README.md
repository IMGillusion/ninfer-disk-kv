# diskkv — L3 (disk) tier for the NInfer KV cache

A cold, file-backed third level for the NInfer KV-cache hierarchy, modeled on
SGLang's HiCache L3 ("host full → demote to disk, evict only when the disk
budget is also full").

## Why

NInfer's cache is two levels:

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

## Layout

- `include/diskkv/disk_kv_store.h` — identity-keyed slot store. Self-describing
  48-byte slot headers (magic + content identity + CRC32C + LRU stamp), atomic
  index rewrite (`<path>.idx`, tmp+fsync+rename), index rebuild by slot scan,
  true LRU eviction. Self-contained: no CUDA, no engine types.
- `include/diskkv/disk_kv_bridge.h`, `src/disk_kv_bridge.cpp` — engine-facing
  facade. One store per KV family (main / backend / state, different page
  strides), a bounded async spill queue (`spill_page`, 512 pages in flight),
  a **backpressure wait variant** (`spill_page_wait`, used by the owner-death
  sweep so a chain never develops a silent hole), a synchronous variant for
  large state blobs, and the read path (`restore_page`, `probe_prefix`).
- `src/disk_kv_store.cpp` — the store implementation.
- `tests/` — self-contained suites (`disk_kv_store_test`, `disk_kv_bridge_test`,
  incl. a 900-page queue-backpressure regression).
- `patches/ninfer-disk-kv-l3-full.patch` — the complete integration diff against
  the NInfer tree (29 files, +2351 lines): the runtime write points (owner
  death, checkpoint drop, seam tails incl. the MTP draft family), the admission
  probe, the seed path, and the accounting contract.

## Integration summary (see the patch)

Write side, three points:
1. owner-death sweep: pages `[0, frontier)` read via host replica or a one-shot
   D2H; `spill_page_wait` backpressure; prefix-contiguous semantics.
2. checkpoint-drop: `[retained, dropped)` runs, capped at 512 pages.
3. seam tails: main at `digest(F)`, backend (MTP draft) at `digest(F-1)`.

Read side:
1. admission probe (`inspect_lane`): for Root requests, enumerate live state
   frontiers, verify the full-page chain + tail + state (+ MTP backend chain at
   `E-1`), pick the largest restorable `E`. Raising `reuse_base` re-filters
   capture groups / shared candidates and drops an in-prefix rewrite
   checkpoint (L3-Root invariants).
2. seed: device KV `[0, E)` from disk (main + backend), prefill recomputes
   `[E, prompt)`; account `E` as reused, recompute as an unreported replay on
   a seed miss (engine contract preserved).

MTP-compatible: gates accept `None || Mtp`; draft frontier = target `- 1`.

## Measured (2026-09-23, 5090, Qwen3.8-27B NVFP4)

- None path: restore 4/4, cache 92,958/93,056 (99.9%), TTFT 18 s → 4.8 s;
  4/4 bit-exact vs the memory-continuation reference.
- MTP path: restore 4/4 (`frontier=92958/93049`), 3/3 bit-exact vs the
  memory-continuation reference (the 4th session's state had been LRU-evicted
  from a small test-scale state pool; falls back to recompute — correctness
  preserved). Cold first read ≈ 15 s, warm 4.5–4.8 s.
- Standalone suites: store 20/20, bridge 13/13, write ≈ 450 MB/s, read ≈ 580 MB/s
  (256 KiB pages, single thread).

## Build & test

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
ctest --test-dir build --output-on-failure
# or standalone:
g++ -std=c++20 -O2 -Iinclude tests/disk_kv_store_test.cpp src/disk_kv_store.cpp -o dks_test -pthread
g++ -std=c++20 -O2 -Iinclude tests/disk_kv_bridge_test.cpp src/disk_kv_bridge.cpp src/disk_kv_store.cpp -o dkb_test -pthread
```

## Operations (in production)

```
--disk-kv-path <dir> --disk-kv-gib 64 --disk-kv-restore
```

Budget split (backend family enabled): main 65% / backend 25% / state 10%.
State slots = 10% / ~146 MiB (64 GiB → ~45 slots), so size the budget for the
number of sessions you want restorable. A fresh store preallocates the full
budget (ftruncate) — check free space first.
