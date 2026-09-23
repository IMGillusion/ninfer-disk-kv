// Suite for DiskKVBridge (v2): round-trip, per-family isolation, dedupe,
// bounded-queue backpressure (spill_page_wait regression), stats.
// Build: g++ -std=c++20 -O2 -Iinclude tests/disk_kv_bridge_test.cpp src/disk_kv_bridge.cpp src/disk_kv_store.cpp -o dkb_test -pthread
// Run:   ./dkb_test <dir>
#include "diskkv/disk_kv_bridge.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace ninfer;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s (line %d)\n", msg, __LINE__); ++failures; } \
    else { std::printf("ok:   %s\n", msg); } \
} while (0)

static DiskKVIdentity make_id(std::uint64_t seed, std::uint32_t frontier) {
    return DiskKVIdentity{.lo       = 0x9e3779b97f4a7c15ULL * (seed + 1),
                          .hi       = 0xbf58476d1ce4e5b9ULL ^ (seed * 0x94d049bb133111ebULL),
                          .tag      = 0x30101,
                          .frontier = frontier};
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/tmp/dkb_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    const std::size_t main_stride    = 65536;
    const std::size_t backend_stride = 32768;
    const std::size_t state_stride   = 131072;
    // Budget split with all three families on: main 65% / backend 25% / state 10%.
    // 160 MiB -> main ~104 MiB (1600+ pages) — room for the 900-page queue test.
    DiskKVBridge b(DiskKVBridge::Options{.base_path           = dir,
                                         .main_page_stride    = main_stride,
                                         .backend_page_stride = backend_stride,
                                         .state_page_stride   = state_stride,
                                         .capacity_bytes      = 160ULL << 20,
                                         .verify_crc          = true});
    CHECK(b.enabled(), "bridge enabled");

    // ---------- 1. spill + wait + contains + restore round-trip ----------
    const auto id = make_id(1, 64);
    std::vector<std::byte> page(main_stride);
    for (std::size_t i = 0; i < main_stride; ++i) page[i] = std::byte((i * 17 + 3) & 0xFF);
    CHECK(b.spill_page(id, DiskKVKind::MainKV, page), "spill queued");
    b.wait_idle();
    CHECK(b.contains(id, DiskKVKind::MainKV), "contains after wait_idle");
    std::vector<std::byte> out(main_stride);
    CHECK(b.restore_page(id, DiskKVKind::MainKV, out) &&
              std::memcmp(out.data(), page.data(), main_stride) == 0,
          "restore round-trip identical");

    // ---------- 2. per-family isolation ----------
    std::vector<std::byte> bpage(backend_stride);
    for (std::size_t i = 0; i < backend_stride; ++i) bpage[i] = std::byte((i * 29 + 5) & 0xFF);
    CHECK(b.spill_page(id, DiskKVKind::BackendKV, bpage), "spill backend queued");
    CHECK(b.spill_page_sync(id, DiskKVKind::StateImage,
                            std::vector<std::byte>(state_stride, std::byte(0x77))),
          "spill state (sync)");
    b.wait_idle();
    CHECK(b.contains(id, DiskKVKind::BackendKV) && b.contains(id, DiskKVKind::StateImage),
          "backend/state present independently");
    std::vector<std::byte> bout(backend_stride);
    CHECK(b.restore_page(id, DiskKVKind::BackendKV, bout) &&
              std::memcmp(bout.data(), bpage.data(), backend_stride) == 0,
          "backend bytes independent of main");

    // ---------- 3. dedupe hit accounting ----------
    const auto st0 = b.stats();
    CHECK(b.spill_page(id, DiskKVKind::MainKV, page), "re-spill (dedupe path)");
    b.wait_idle();
    CHECK(b.stats().spill_dups == st0.spill_dups + 1, "dedupe hit counted");

    // ---------- 4. bounded-queue backpressure (spill_page_wait regression) ----------
    // kQueueCap is 512; push 900 pages through the wait-variant: every page must
    // land (no silent drops) even though the queue overflows repeatedly.
    constexpr int kN = 900;
    for (int i = 0; i < kN; ++i) {
        std::vector<std::byte> p(main_stride, std::byte(i & 0xFF));
        if (!b.spill_page_wait(make_id(10000 + i, 64), DiskKVKind::MainKV, p,
                               std::chrono::milliseconds(5000))) {
            CHECK(false, "spill_page_wait queued every page");
            break;
        }
    }
    b.wait_idle();
    int present = 0;
    for (int i = 0; i < kN; ++i) {
        if (b.contains(make_id(10000 + i, 64), DiskKVKind::MainKV)) { ++present; }
    }
    CHECK(present == kN, "all 900 backpressured pages restorable");
    CHECK(b.stats().queue_drops == 0, "queue_drops stayed 0 (no silent drops)");
    CHECK(b.stats().spills >= static_cast<std::uint64_t>(kN) + 3, "spill counter advanced");

    std::printf(failures == 0 ? "ALL PASS\n" : "%d FAILURES\n", failures);
    return failures == 0 ? 0 : 1;
}
