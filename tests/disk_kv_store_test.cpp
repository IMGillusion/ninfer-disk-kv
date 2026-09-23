// Self-test / micro-benchmark for DiskKVStore (identity-keyed, self-describing slots).
// Build: g++ -std=c++20 -O2 -Iinclude tests/disk_kv_store_test.cpp src/disk_kv_store.cpp -o dks_test -pthread
// Run:   ./dks_test <path> [slot_stride] [max_slots]
//
// Verifies: round-trip integrity, dedupe/refresh, capacity boundary + LRU
// eviction order, CRC corruption rejection, index rebuild from slot scan,
// persistence across reopen. Benchmarks write/read (restore) throughput.
#include "diskkv/disk_kv_store.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
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
    const std::string path   = argc > 1 ? argv[1] : "/tmp/dks_test.bin";
    const std::size_t stride = argc > 2 ? std::stoul(argv[2]) : 262144;  // 256 KiB
    const std::uint32_t slots = argc > 3 ? std::stoul(argv[3]) : 64;

    DiskKVStore::Options opts{.path           = path,
                              .slot_size      = stride,
                              .capacity_bytes = 0,
                              .max_slots      = slots,
                              .verify_crc     = true};

    std::vector<std::byte> page(stride);
    for (std::size_t i = 0; i < stride; ++i) page[i] = std::byte((i * 31 + 7) & 0xFF);

    // ---------- 1. round-trip + dedupe ----------
    {
        std::remove(path.c_str());
        std::remove((path + ".idx").c_str());
        DiskKVStore s(opts);
        const auto id = make_id(1, 64);
        CHECK(s.upsert_page(id, page, nullptr), "upsert page");
        CHECK(s.contains(id), "contains after upsert");
        std::vector<std::byte> out(stride);
        CHECK(s.read_page(id, out), "read_page");
        CHECK(std::memcmp(out.data(), page.data(), stride) == 0, "round-trip bytes identical");
        CHECK(s.upsert_page(id, page, nullptr), "re-upsert same id");
        CHECK(s.live_slots() == 1, "dedupe: one live slot");
    }

    // ---------- 2. persistence across reopen ----------
    {
        DiskKVStore s(opts);
        const auto id = make_id(1, 64);
        CHECK(s.contains(id), "persisted page found after reopen");
        std::vector<std::byte> out(stride);
        CHECK(s.read_page(id, out) && std::memcmp(out.data(), page.data(), stride) == 0,
              "persisted bytes identical");
        // Fast path: the packed index loads directly (all structural changes
        // persist eagerly). Data pages stay CRC-verified at read time: a bad
        // page is a restore miss, not a startup cost, and the restore path
        // self-heals by dropping it (see the bridge/engine integration).
        CHECK(!s.index_rebuilt_from_scan(), "reopen uses the packed index fast path");
    }

    // ---------- 3. LRU eviction order ----------
    {
        DiskKVStore::Options small = opts;
        small.path      = path + ".lru";
        small.max_slots = 3;
        std::remove(small.path.c_str());
        std::remove((small.path + ".idx").c_str());
        DiskKVStore s(small);
        const auto a = make_id(10, 64), b = make_id(11, 64), c = make_id(12, 64), d = make_id(13, 64);
        CHECK(s.upsert_page(a, page, nullptr), "lru: insert a");
        CHECK(s.upsert_page(b, page, nullptr), "lru: insert b");
        CHECK(s.upsert_page(c, page, nullptr), "lru: insert c");
        CHECK(s.touch(a), "lru: touch a (most recent)");
        std::vector<std::uint32_t> evicted;
        CHECK(s.upsert_page(d, page, &evicted), "lru: insert d (must evict)");
        CHECK(!s.contains(b), "lru: b (least recent) evicted");
        CHECK(s.contains(a) && s.contains(c) && s.contains(d), "lru: a/c/d present");
        CHECK(!s.upsert_page(make_id(99, 64), page, nullptr) || s.live_slots() == 3,
              "capacity never exceeded");
    }

    // ---------- 4. CRC corruption rejection ----------
    {
        DiskKVStore::Options crc = opts;
        crc.path = path + ".crc";
        std::remove(crc.path.c_str());
        std::remove((crc.path + ".idx").c_str());
        DiskKVStore s(crc);
        const auto id = make_id(20, 64);
        CHECK(s.upsert_page(id, page, nullptr), "crc: upsert");
        s.flush_index();
        // locate the payload in the data file and corrupt one byte inside it
        std::fstream f(crc.path, std::ios::in | std::ios::out | std::ios::binary);
        std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const std::string needle(reinterpret_cast<const char*>(page.data()), 64);
        const auto pos = data.find(needle);
        CHECK(pos != std::string::npos, "crc: payload pattern found in data file");
        if (pos != std::string::npos) {
            data[pos + 100] ^= 0x5A;
            f.seekp(0);
            f.write(data.data(), static_cast<std::streamsize>(data.size()));
            f.flush();
        }
        std::vector<std::byte> out(stride);
        CHECK(!s.read_page(id, out), "crc: corrupted payload rejected");
    }

    // ---------- 5. index rebuild from scan ----------
    {
        std::remove((path + ".idx").c_str());
        DiskKVStore s(opts);
        CHECK(s.index_rebuilt_from_scan(), "index rebuilt from slot scan");
        CHECK(s.contains(make_id(1, 64)), "data intact after rebuild");
    }

    // ---------- 6. write / read benchmark ----------
    {
        DiskKVStore::Options bench = opts;
        bench.path      = path + ".bench";
        const std::uint32_t n = std::min<std::uint32_t>(2000, std::max<std::uint32_t>(64, slots));
        bench.max_slots = n + 8;
        std::remove(bench.path.c_str());
        std::remove((bench.path + ".idx").c_str());
        DiskKVStore s(bench);
        std::vector<std::byte> out(stride);
        const auto t0 = std::chrono::steady_clock::now();
        std::uint32_t written = 0;
        for (std::uint32_t i = 0; i < n; ++i) {
            if (s.upsert_page(make_id(1000 + i, 64), page, nullptr)) { ++written; }
        }
        s.flush_index();
        const auto t1 = std::chrono::steady_clock::now();
        std::uint32_t read = 0;
        for (std::uint32_t i = 0; i < n; ++i) {
            if (s.read_page(make_id(1000 + i, 64), out)) { ++read; }
        }
        const auto t2 = std::chrono::steady_clock::now();
        const double wsec = std::chrono::duration<double>(t1 - t0).count();
        const double rsec = std::chrono::duration<double>(t2 - t1).count();
        const double mb   = static_cast<double>(n) * static_cast<double>(stride) / (1024.0 * 1024.0);
        std::printf("bench: %u pages x %zu B | write %.0f MB/s (%u) | read %.0f MB/s (%u)\n", n,
                    stride, mb / wsec, written, mb / rsec, read);
        CHECK(written == n && read == n, "bench: all pages written and read back");
    }

    std::printf(failures == 0 ? "ALL PASS\n" : "%d FAILURES\n", failures);
    return failures == 0 ? 0 : 1;
}
