#include "diskkv/disk_kv_bridge.h"

#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>

namespace ninfer {

namespace {

std::string family_path(const std::string& base, DiskKVKind kind) {
    const char* name = kind == DiskKVKind::MainKV ? "main"
                     : kind == DiskKVKind::BackendKV ? "backend" : "state";
    return base + "/" + name + ".diskkv";
}

std::uint32_t kind_index(DiskKVKind kind) {
    return static_cast<std::uint32_t>(kind);
}

} // namespace

DiskKVBridge::DiskKVBridge(Options opts) : opts_(std::move(opts)) {
    if (opts_.base_path.empty() || (opts_.max_pages == 0 && opts_.capacity_bytes == 0)) {
        enabled_ = false; // explicit off: no path or no budget
        return;
    }
    const std::size_t stride = opts_.main_page_stride;
    if (stride == 0) { throw std::invalid_argument("disk kv bridge needs main_page_stride"); }

    auto make_family = [&](DiskKVKind kind, std::size_t family_stride) {
        Family& f = families_[kind_index(kind)];
        f.stride  = family_stride;
        f.path    = family_path(opts_.base_path, kind);
        if (family_stride == 0) { return; } // family disabled
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(f.path).parent_path(), ec);
        DiskKVStore::Options sopts;
        sopts.path        = f.path;
        sopts.page_stride = family_stride;
        sopts.max_pages   = opts_.max_pages;
        sopts.capacity_bytes = opts_.capacity_bytes;
        sopts.verify_crc = opts_.verify_crc;
        f.store          = std::make_unique<DiskKVStore>(sopts);
    };
    make_family(DiskKVKind::MainKV, opts_.main_page_stride);
    make_family(DiskKVKind::BackendKV, opts_.backend_page_stride);
    make_family(DiskKVKind::StateImage, opts_.state_page_stride);
    enabled_ = families_[0].store != nullptr;
}

DiskKVBridge::~DiskKVBridge() = default;

DiskKVBridge::Family& DiskKVBridge::family(DiskKVKind kind) {
    return families_[kind_index(kind)];
}
const DiskKVBridge::Family& DiskKVBridge::family(DiskKVKind kind) const {
    return families_[kind_index(kind)];
}

std::optional<std::uint32_t>
DiskKVBridge::group_for(const DiskKVIdentity& id, DiskKVKind kind, bool for_write) const {
    auto it = group_cache_.find(id);
    if (it != group_cache_.end()) {
        if (for_write || family(kind).store->contains(it->second)) { return it->second; }
        group_cache_.erase(it);
    }
    return std::nullopt;
}

bool DiskKVBridge::spill(const DiskKVIdentity& id, DiskKVKind kind, std::uint32_t page_count,
                         std::span<const std::byte> bytes) {
    if (!enabled_) { return false; }
    Family& f = family(kind);
    if (f.store == nullptr || page_count == 0) { return false; }
    const std::size_t expect = static_cast<std::size_t>(page_count) * f.stride;
    if (bytes.size() != expect) {
        throw std::invalid_argument("disk kv spill byte count does not match page_stride*pages");
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto existing = group_for(id, kind, /*for_write*/ true);
        if (existing.has_value()) {
            // Already on disk (re-spill of a live group is idempotent at the
            // identity level; refresh bytes so a partial overwrite cannot win).
            return true;
        }
    }
    DiskKVStore* store = f.store.get();
    auto group = store->reserve(page_count);
    if (!group.has_value()) {
        // LRU-evict cold groups, then retry once.
        auto evicted = store->evict_until_free(page_count);
        {
            std::lock_guard<std::mutex> lock(mu_);
            for (const std::uint32_t dead : evicted) {
                // Drop cache entries that pointed at the evicted group id.
                for (auto it = group_cache_.begin(); it != group_cache_.end();) {
                    it = (it->second == dead) ? group_cache_.erase(it) : std::next(it);
                }
                stats_.evicted_groups += 1;
            }
        }
        group = store->reserve(page_count);
        if (!group.has_value()) { return false; } // family genuinely full of live groups
    }
    const std::uint32_t gid = *group;
    if (!store->write_group(gid, bytes)) { return false; }
    if (!store->publish(gid)) { return false; }
    if (!store->release(gid)) { return false; } // immediately cold: evictable by LRU
    std::lock_guard<std::mutex> lock(mu_);
    group_cache_[id] = gid;
    ++stats_.spills;
    stats_.spill_bytes += bytes.size();
    return true;
}

bool DiskKVBridge::restore_page(const DiskKVIdentity& id, DiskKVKind kind,
                                std::uint32_t page_index, std::span<std::byte> dst) const {
    if (!enabled_) { return false; }
    const Family& f = family(kind);
    if (f.store == nullptr) { return false; }
    std::uint32_t gid;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = group_cache_.find(id);
        if (it == group_cache_.end() || !f.store->contains(it->second)) {
            stats_.restore_misses += 1;
            return false;
        }
        gid = it->second;
    }
    const std::size_t stride = f.stride;
    if (dst.size() != stride) { return false; }
    // read_page returns a zero-copy span into the mmap (the store's lock is released on
    // return, but the mapping is stable for the store's lifetime), so copy it out now.
    auto src = f.store->read_page(gid, page_index);
    if (!src.has_value() || src->size() != stride) {
        std::lock_guard<std::mutex> lock(mu_);
        group_cache_.erase(id);
        stats_.restore_misses += 1;
        return false;
    }
    std::memcpy(dst.data(), src->data(), stride);
    std::lock_guard<std::mutex> lock(mu_);
    stats_.restores += 1;
    stats_.restore_bytes += stride;
    return true;
}

bool DiskKVBridge::contains(const DiskKVIdentity& id, DiskKVKind kind) const {
    if (!enabled_) { return false; }
    const Family& f = family(kind);
    if (f.store == nullptr) { return false; }
    std::lock_guard<std::mutex> lock(mu_);
    auto it = group_cache_.find(id);
    if (it == group_cache_.end()) { return false; }
    return f.store->contains(it->second);
}

void DiskKVBridge::touch(const DiskKVIdentity& id, DiskKVKind kind) {
    if (!enabled_) { return; }
    const Family& f = family(kind);
    if (f.store == nullptr) { return; }
    std::lock_guard<std::mutex> lock(mu_);
    auto it = group_cache_.find(id);
    if (it == group_cache_.end()) { return; }
    f.store->touch(it->second);
}

DiskKVBridgeStats DiskKVBridge::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
}

std::size_t DiskKVBridge::used_bytes(DiskKVKind kind) const {
    const Family& f = family(kind);
    return f.store ? f.store->used_bytes() : 0;
}

const std::string& DiskKVBridge::path(DiskKVKind kind) const {
    static const std::string kEmpty;
    const Family& f = family(kind);
    return f.store ? f.path : kEmpty;
}

} // namespace ninfer
