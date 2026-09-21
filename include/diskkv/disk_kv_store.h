#pragma once
// L3 (disk) tier for the KV cache hierarchy.
//
// NInfer's cache is today two levels: a device page pool (kv-capacity tokens)
// and a pinned host arena (--host-kv-mib). When the host arena is full, the
// pressure planner drops the affected checkpoint PERMANENTLY, and the next
// turn of that session pays full prefill recompute. This module is a cold,
// file-backed third level so that a "host full" event demotes a checkpoint's
// page run to disk instead of dropping it, and a later turn can restore it
// (disk read + H2D) instead of recomputing.
//
// The store is deliberately self-contained (no CUDA, no engine types) so it
// builds and unit-tests in seconds and can be dropped behind the host arena
// as a transparent cold backing. The engine drives it with plain (page_stride,
// page_count) groups and host byte buffers; it never touches device memory.

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer {

/**
 * A "group" is a contiguous run of KV pages belonging to one checkpoint
 * (one continuation). `page_stride` is the byte size of ONE page (the engine's
 * HostKVPageLayout.page_stride), so a group is page_count * page_stride bytes.
 *
 * Lifecycle:
 *   reserve(group) -> write pages -> publish()          (now restorable)
 *   ... later, possibly under pressure ...
 *   release(group)                                (now evictable)
 *
 * A published group stays addressable (read() returns its bytes) until it is
 * evicted by LRU pressure or the store is cleared. Evicted groups are LOST
 * (the engine then falls back to recompute, exactly as today) — so this tier is
 * strictly "better than today", never worse.
 */
class DiskKVStore {
public:
    struct Options {
        std::string path;          // data file (created/truncated on open)
        std::size_t page_stride    = 0;   // bytes per page (from HostKVPageLayout)
        std::size_t capacity_bytes = 0;   // 0 = auto from page_stride & max_pages
        std::uint32_t max_pages    = 0;   // 0 = derived from capacity_bytes/page_stride
        bool verify_crc            = true; // recompute+check crc on every read
    };

    /** Per-group metadata surfaced to the engine for its cost/planner model. */
    struct GroupInfo {
        std::uint32_t group_id   = 0;
        std::uint32_t page_count = 0;
        bool          published  = false;
        bool          evictable  = false; // published && released
        std::uint64_t crc32      = 0;
    };

    /** Open (create/truncate) the backing file. Throws on failure. */
    explicit DiskKVStore(Options opts);
    ~DiskKVStore();

    DiskKVStore(const DiskKVStore&)            = delete;
    DiskKVStore& operator=(const DiskKVStore&) = delete;

    [[nodiscard]] std::size_t page_stride()   const noexcept { return opts_.page_stride; }
    [[nodiscard]] std::uint32_t page_capacity() const noexcept { return max_pages_; }
    [[nodiscard]] std::uint32_t used_pages()   const noexcept;
    [[nodiscard]] std::size_t  free_bytes()   const noexcept;
    [[nodiscard]] std::size_t  used_bytes()   const noexcept;
    [[nodiscard]] const std::string& path()   const noexcept { return opts_.path; }

    /**
     * Reserve a group of `page_count` pages. Returns a group id, or nullopt if
     * there is no room (the caller may evict LRU groups and retry). Reserved
     * groups are NOT yet restorable; they must be filled with write_page()
     * and then publish().
     */
    [[nodiscard]] std::optional<std::uint32_t> reserve(std::uint32_t page_count);

    /** Write one page of a reserved/published group (0-based within the group). */
    bool write_page(std::uint32_t group_id, std::uint32_t page_index,
                    const std::byte* src);
    /** Bulk-write a whole group from a contiguous buffer of page_count*stride. */
    bool write_group(std::uint32_t group_id, std::span<const std::byte> bytes);

    /** Make the group restorable. Recomputes its crc. Returns false if not all
     *  pages were written. */
    bool publish(std::uint32_t group_id);

    /** Mark a group no longer needed; it becomes LRU-evictable. */
    bool release(std::uint32_t group_id);

    /** Refresh the group's LRU position (call after a successful restore/read). */
    bool touch(std::uint32_t group_id);

    /** Read one page of a published (not yet evicted) group. */
    [[nodiscard]] std::optional<std::span<const std::byte>>
    read_page(std::uint32_t group_id, std::uint32_t page_index) const;

    /** Read the whole group; copies into `dst` (page_count*stride bytes). */
    [[nodiscard]] bool read_group(std::uint32_t group_id, std::span<std::byte> dst) const;

    /** True if the group is currently restorable (published, not yet evicted). */
    [[nodiscard]] bool contains(std::uint32_t group_id);

    /**
     * Evict LRU groups until at least `free_pages` pages are free. Returns
     *  the ids of the groups evicted (empty if none were needed). Never
     *  evicts non-evictable groups. All public methods take an internal lock,
     *  so a read_page() span stays valid until the NEXT call on this store;
     *  copy it out immediately (the restore path does, into the host arena).
     */
    [[nodiscard]] std::vector<std::uint32_t> evict_until_free(std::uint32_t free_pages);

    /** Evict a specific group (if evictable). */
    bool evict(std::uint32_t group_id);

    /** Drop everything (keeps the file, frees all groups). */
    void clear() noexcept;

    /** Snapshot of group metadata (for logging / planner cost model). */
    [[nodiscard]] std::vector<GroupInfo> groups() const;

    /** Total pages evicted since open (for observability). */
    [[nodiscard]] std::uint64_t evicted_pages_total() const noexcept { return evicted_pages_total_; }
    [[nodiscard]] std::uint64_t published_groups_total() const noexcept { return published_groups_total_; }

    static std::uint64_t crc32c(std::span<const std::byte> data);

private:
    struct Group {
        std::uint32_t first_page = 0;   // first slot index in the file
        std::uint32_t page_count = 0;
        std::uint32_t written    = 0;   // pages written so far
        bool          published  = false;
        bool          released   = false;
        std::uint64_t crc        = 0;
        std::uint64_t last_used  = 0;  // LRU clock
    };

    std::uint32_t find_free_run(std::uint32_t pages) const;
    void          reclaim_run(std::uint32_t first, std::uint32_t pages);
    bool          valid_group(std::uint32_t id) const noexcept;
    bool          evict_unlocked(std::uint32_t group_id);   // caller holds mu_
    std::uint64_t bump_clock() noexcept;

    mutable std::mutex mu_;    // guards all state below (coarse lock; cold path)
    Options       opts_;
    int           fd_          = -1;
    std::byte*    base_        = nullptr;   // mmap of the data file
    std::size_t   file_bytes_  = 0;
    std::uint32_t max_pages_   = 0;

    std::vector<Group> groups_;             // index == group id
    std::vector<bool>  slot_used_;          // per-slot in-use flag
    std::uint32_t      used_pages_   = 0;
    std::uint64_t      clock_        = 0;
    std::uint64_t      evicted_pages_total_      = 0;
    std::uint64_t      published_groups_total_   = 0;
    std::vector<std::uint32_t> free_list_;     // free first-slots (coalesced not required)
};

} // namespace ninfer
