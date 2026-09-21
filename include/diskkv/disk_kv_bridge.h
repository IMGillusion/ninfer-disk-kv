#pragma once
// Engine-facing facade over DiskKVStore (the L3 cold tier).
//
// The engine drives this with plain (kind, identity, page_index) keys and host
// byte spans. It stays engine-type-free on purpose: `identity` is the 128-bit
// content digest the engine already computes per checkpoint frontier
// (PrefixShortlistDigests::at), so a page's KV bytes are addressable on disk
// even after the logical page descriptor (and its content_epoch) has been
// reclaimed under pressure. That is the whole reason this tier works where a
// "just hold the host page" tier would silently lose the key.
//
//   spill  : host run is about to be freed  -> reserve+write+publish a group
//   restore: host page is gone, recompute  -> read_page back into a fresh
//           extent (the engine then runs its normal H2D restore unchanged)
//
// A miss (evicted, corrupt, or never spilt) returns false and the engine falls
// back to exactly what it does today (full recompute). Strictly better-or-equal.

#include "diskkv/disk_kv_store.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>

namespace ninfer {

/** One KV plane family. Text and backend KV have different page strides, so
 *  each family owns its own store (and its own page_stride). */
enum class DiskKVKind : std::uint8_t {
    MainKV      = 0,
    BackendKV   = 1,
    StateImage  = 2,   // state image bytes, spilt the same way (small groups)
};

/** 128-bit content identity for one checkpoint frontier + one KV family.
 *  The engine fills this from its prefix shortlist digest (or a per-block
 *  token hash). Must be identical at spill time and restore time. */
struct DiskKVIdentity {
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;
    std::uint32_t kind = 0;   // DiskKVKind as int (family)
    std::uint32_t frontier = 0;  // token frontier the checkpoint sits at

    [[nodiscard]] bool operator==(const DiskKVIdentity&) const noexcept = default;
};

struct DiskKVIdentityHash {
    [[nodiscard]] std::size_t operator()(const DiskKVIdentity& k) const noexcept {
        std::size_t h = std::hash<std::uint64_t>{}(k.lo);
        h ^= std::hash<std::uint64_t>{}(k.hi) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= static_cast<std::size_t>(k.kind) * 0xff51afd7ed558ccdULL;
        h ^= static_cast<std::size_t>(k.frontier) * 0xc4ceb9fe1a85ec53ULL;
        return h;
    }
};

struct DiskKVBridgeStats {
    std::uint64_t spills          = 0;  // groups written to disk
    std::uint64_t spill_bytes      = 0;
    std::uint64_t restores         = 0;  // pages read back
    std::uint64_t restore_bytes    = 0;
    std::uint64_t restore_misses   = 0;  // fell back to recompute
    std::uint64_t evicted_groups   = 0;
};

class DiskKVBridge {
public:
    struct Options {
        std::string base_path    = "";  // parent dir; bridge appends /<kind>.diskkv
        std::size_t main_page_stride   = 0;   // from HostKVPageLayout(MainKV)
        std::size_t backend_page_stride = 0;  // 0 = family disabled
        std::size_t state_page_stride  = 0;   // 0 = family disabled
        std::size_t capacity_bytes     = 0;   // per-family cap; 0 = all of max_pages
        std::uint32_t max_pages        = 0;   // per-family page budget
        bool verify_crc        = true;
    };

    /** Open (create) the per-family backing files. Throws on failure. */
    explicit DiskKVBridge(Options opts);
    ~DiskKVBridge();

    DiskKVBridge(const DiskKVBridge&)            = delete;
    DiskKVBridge& operator=(const DiskKVBridge&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    /**
     * Spill a contiguous host run (page_count consecutive pages, one family) to
     * disk. Returns true if the run is now restorable (or already was). The
     * caller's bytes are copied; it may free them after this returns. If there
     * is no room the bridge evicts LRU cold groups first (never a live one) and
     * still returns false only if the family truly cannot hold the run.
     */
    bool spill(const DiskKVIdentity& id, DiskKVKind kind, std::uint32_t page_count,
               std::span<const std::byte> bytes);

    /** Read one page of a previously spilt run into `dst` (page_stride bytes).
     *  Returns false on miss/evict/corrupt -> caller recomputes. */
    [[nodiscard]] bool restore_page(const DiskKVIdentity& id, DiskKVKind kind,
                                    std::uint32_t page_index, std::span<std::byte> dst) const;

    /** True if at least one page of this identity is restorable on disk. Cheap. */
    [[nodiscard]] bool contains(const DiskKVIdentity& id, DiskKVKind kind) const;

    /** Refresh LRU position after a successful restore. */
    void touch(const DiskKVIdentity& id, DiskKVKind kind);

    [[nodiscard]] DiskKVBridgeStats stats() const;
    [[nodiscard]] std::size_t used_bytes(DiskKVKind kind) const;
    [[nodiscard]] const std::string& path(DiskKVKind kind) const;

private:
    struct Family {
        std::unique_ptr<DiskKVStore> store;
        std::size_t stride = 0;
        std::string path;
    };

    Family& family(DiskKVKind kind);
    const Family& family(DiskKVKind kind) const;
    [[nodiscard]] std::optional<std::uint32_t>
    group_for(const DiskKVIdentity& id, DiskKVKind kind, bool for_write) const;

    Options opts_;
    bool    enabled_ = false;
    Family families_[3];
    mutable std::mutex mu_;   // guards group cache + stats (store is itself locked)
    mutable std::unordered_map<DiskKVIdentity, std::uint32_t, DiskKVIdentityHash> group_cache_;
    mutable DiskKVBridgeStats stats_;
};

} // namespace ninfer
