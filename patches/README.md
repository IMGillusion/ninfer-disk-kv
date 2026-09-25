# Engine-side patches for the disk KV tier

`ninfer-disk-kv-l3-full.patch` is the whole feature against upstream (the module plus
its engine integration points). The two numbered patches below are the *reviewable
delta* for the restore-path work, and their baseline is the author's L3 work tree
(`proactive-evict`), NOT upstream - apply them on top of the full patch.

| patch | what it does | changed lines |
|---|---|---|
| 0001-seed-batching-and-read-pipeline.patch | Restores the request's KV chain in 128-page batches instead of one page per scheduler visit, keeps a batch open until it is full (the shared bridge queue truncates a naive submission to ~37 pages), and submits the next batch's reads in the same visit as the copy. | 204 |
| 0002-store-read-path-outside-lock.patch | Moves the page CRC + memcpy out of `DiskKVStore`'s store-wide lock (only locate/verify-identity/pin stay inside, eviction skips pinned-while-reading slots) and raises the bridge's read workers 3 -> 8. | 59 |

Measured on a 93,481-token prompt, cold VM page cache (see the sibling repo's
`L3_STEP1_PIPELINE.md` for the full tables):

- restore of 2,922 pages (3.4GB): 11,000-13,000ms -> 4,597-5,277ms, i.e. ~700MB/s
  against a ~2.3GB/s path floor, and 4-5x faster than recomputing (19-26s)
- batch statistics: 64-95 short batches -> 25-27 full 128-page batches
- concurrent stream's worst inter-token gap: 76s -> 5.7-7.7s
- acceptance: 9/9 restores, zero HTTP 500/503

Known traps recorded in the patches: raising the shared `kL3ScratchPages` also raised
the owner spill's chain-probe batch, and `try_probe_batch` hard-throws above 32
identities (`disk kv probe batch must contain 1..32 identities`); and draining several
batches per visit was measured to give nothing, because a batch's reads (~66ms) outlast
the copy (~21ms) so the immediate re-poll is still Pending.
