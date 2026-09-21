// Bridge round-trip test: spill a host KV run to disk, free the host replica,
// restore it back, verify byte-identity, and prove LRU eviction + miss fallback.
//
// This exercises DiskKVBridge directly (identity-keyed groups). It uses a real
// HostKVPageLayout-derived page_stride value so the byte geometry matches the
// engine's. Build with g++ -std=c++20 -O2 against core/disk_kv_bridge.cpp and
// core/disk_kv_store.cpp, then run: ./dkb_test <tmpdir>

#include "diskkv/disk_kv_bridge.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace ninfer;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s (line %d)\n", msg, __LINE__); ++failures; } \
    else { std::printf("ok:   %s\n", msg); } \
} while (0)

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/tmp/dkb";
    // Real main-KV page stride from a typical 28-layer GQA geometry (MiB-scale).
    const std::size_t stride = 1589632; // 1.516 MiB

    DiskKVBridge b(DiskKVBridge::Options{
        .base_path            = dir,
        .main_page_stride     = stride,
        .backend_page_stride  = stride,
        .state_page_stride    = 0,             // state family disabled here
        .capacity_bytes       = 0,
        .max_pages            = 30,            // ~46 MB of MainKV on disk (3 runs of 10)
        .verify_crc           = true});
    CHECK(b.enabled(), "bridge enabled");
    CHECK(b.path(DiskKVKind::MainKV).find("main.diskkv") != std::string::npos, "main family path");

    // ---- 1. spill + restore one page-run, verify identity ----
    DiskKVIdentity id{.lo = 0xA11CE, .hi = 0xBEEF, .kind = 0, .frontier = 4096};
    const std::uint32_t pages = 10;
    std::vector<std::byte> src(static_cast<std::size_t>(pages) * stride);
    for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<std::byte>((i * 131 + 17) & 0xFF);
    bool spilled = b.spill(id, DiskKVKind::MainKV, pages, std::span<const std::byte>(src.data(), src.size()));
    CHECK(spilled, "spill 10-page run");
    CHECK(b.contains(id, DiskKVKind::MainKV), "identity now restorable");
    CHECK(b.used_bytes(DiskKVKind::MainKV) == static_cast<std::size_t>(pages) * stride, "used bytes == run");

    std::vector<std::byte> dst(static_cast<std::size_t>(pages) * stride, std::byte{0});
    bool any_read = true;
    for (std::uint32_t p = 0; p < pages && any_read; ++p) {
        std::span<std::byte> one(dst.data() + static_cast<std::size_t>(p) * stride, stride);
        any_read = b.restore_page(id, DiskKVKind::MainKV, p, one);
    }
    CHECK(any_read, "restore_page for all pages");
    CHECK(std::memcmp(src.data(), dst.data(), src.size()) == 0, "round-trip bytes identical");

    // ---- 2. miss fallback: an identity never spilt must miss (recompute path) ----
    DiskKVIdentity cold{.lo = 999, .hi = 1000, .kind = 0, .frontier = 777};
    CHECK(!b.contains(cold, DiskKVKind::MainKV), "unknown identity absent");
    std::vector<std::byte> missbuf(stride);
    CHECK(!b.restore_page(cold, DiskKVKind::MainKV, 0, std::span<std::byte>(missbuf.data(), missbuf.size())),
          "unknown identity restore misses");

    // ---- 3. LRU eviction: fill past capacity, coldest identity must be evicted ----
    // Capacity is 40 pages; id uses 10. Spill 3 more runs (30 pages) to hit 40, then a
    // 4th run must evict the coldest (our id, untouched since spill).
    DiskKVIdentity id2{.lo = 2, .hi = 2, .kind = 0, .frontier = 8192};
    DiskKVIdentity id3{.lo = 3, .hi = 3, .kind = 0, .frontier = 12288};
    DiskKVIdentity id4{.lo = 4, .hi = 4, .kind = 0, .frontier = 16384};
    std::vector<std::byte> run2(static_cast<std::size_t>(pages) * stride, std::byte{0x11});
    std::vector<std::byte> run3(static_cast<std::size_t>(pages) * stride, std::byte{0x22});
    std::vector<std::byte> run4(static_cast<std::size_t>(pages) * stride, std::byte{0x33});
    CHECK(b.spill(id2, DiskKVKind::MainKV, pages, std::span<const std::byte>(run2.data(), run2.size())), "spill id2");
    CHECK(b.spill(id3, DiskKVKind::MainKV, pages, std::span<const std::byte>(run3.data(), run3.size())), "spill id3");
    // touch id2 so it becomes the warmest
    b.touch(id2, DiskKVKind::MainKV);
    CHECK(b.spill(id4, DiskKVKind::MainKV, pages, std::span<const std::byte>(run4.data(), run4.size())), "spill id4 (triggers eviction)");
    CHECK(!b.contains(id, DiskKVKind::MainKV), "coldest (id) evicted under pressure");
    CHECK(b.contains(id2, DiskKVKind::MainKV), "warmest (id2) survives");

    // ---- 4. restore still works for survivors ----
    std::vector<std::byte> dst2(stride, std::byte{0});
    CHECK(b.restore_page(id2, DiskKVKind::MainKV, 0, std::span<std::byte>(dst2.data(), dst2.size())), "restore survivor id2");
    CHECK(dst2[0] == std::byte{0x11}, "survivor byte intact");

    // ---- 5. backend family is isolated from main ----
    DiskKVIdentity bid{.lo = 5, .hi = 5, .kind = 1, .frontier = 4096};
    CHECK(b.spill(bid, DiskKVKind::BackendKV, 2, std::span<const std::byte>(run2.data(), 2 * stride)), "spill backend");
    CHECK(b.used_bytes(DiskKVKind::BackendKV) == 2 * stride, "backend family counts separately");
    CHECK(b.used_bytes(DiskKVKind::MainKV) != 0, "main family unaffected by backend spill");

    // ---- 6. stats sanity ----
    const DiskKVBridgeStats s = b.stats();
    CHECK(s.spills >= 4, "stats.spills counted");
    CHECK(s.spill_bytes >= 4 * static_cast<std::size_t>(pages) * stride, "stats.spill_bytes counted");
    CHECK(s.restores == pages + 1, "stats.restores counted (10 + 1)");
    CHECK(s.evicted_groups >= 1, "stats.evicted_groups counted");

    std::printf("\n%s: %d failure(s)\n", failures ? "TEST FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
