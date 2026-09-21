#include "diskkv/disk_kv_store.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <system_error>

namespace ninfer {
namespace {

constexpr std::size_t kAlignment = 4096;

std::size_t round_up(std::size_t v, std::size_t a) { return (v + (a - 1)) & ~(a - 1); }

struct CrcTable {
    std::array<std::uint32_t, 256> t{};
    CrcTable() {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            t[i] = c;
        }
    }
};
const CrcTable& crc_table() {
    static const CrcTable tbl;
    return tbl;
}

} // namespace

std::uint64_t DiskKVStore::crc32c(std::span<const std::byte> data) {
    const auto& tbl = crc_table();
    std::uint32_t c = 0xFFFFFFFFu;
    for (const std::byte b : data) {
        const std::uint8_t v = static_cast<std::uint8_t>(b);
        c = tbl.t[(c ^ v) & 0xFFu] ^ (c >> 8);
    }
    return static_cast<std::uint64_t>(c ^ 0xFFFFFFFFu);
}

DiskKVStore::DiskKVStore(Options opts) : opts_(std::move(opts)) {
    if (opts_.path.empty()) { throw std::invalid_argument("disk kv store path is empty"); }
    if (opts_.page_stride == 0) { throw std::invalid_argument("disk kv store page_stride is zero"); }
    if (opts_.max_pages > 0) {
        max_pages_ = opts_.max_pages;
        file_bytes_ = round_up(static_cast<std::size_t>(max_pages_) * opts_.page_stride, kAlignment);
    } else if (opts_.capacity_bytes > 0) {
        max_pages_ = static_cast<std::uint32_t>(opts_.capacity_bytes / opts_.page_stride);
        if (max_pages_ == 0) { throw std::invalid_argument("disk kv store capacity too small"); }
        file_bytes_ = round_up(static_cast<std::size_t>(max_pages_) * opts_.page_stride, kAlignment);
    } else {
        max_pages_ = 65536; // sane default ~ 65536 pages
        file_bytes_ = round_up(static_cast<std::size_t>(max_pages_) * opts_.page_stride, kAlignment);
    }
    if (file_bytes_ == 0) { throw std::invalid_argument("disk kv store file size is zero"); }

    fd_ = ::open(opts_.path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd_ < 0) {
        throw std::runtime_error(std::string("open failed for ") + opts_.path + ": " + std::strerror(errno));
    }
    if (ftruncate(fd_, static_cast<off_t>(file_bytes_)) != 0) {
        const int e = errno;
        ::close(fd_); fd_ = -1;
        throw std::runtime_error(std::string("ftruncate failed: ") + std::strerror(e));
    }
    void* mapped = ::mmap(nullptr, file_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (mapped == MAP_FAILED) {
        const int e = errno;
        ::close(fd_); fd_ = -1;
        throw std::runtime_error(std::string("mmap failed: ") + std::strerror(e));
    }
    base_ = static_cast<std::byte*>(mapped);
    slot_used_.assign(max_pages_, false);
    groups_.reserve(std::min<std::size_t>(max_pages_, 4096));
}

DiskKVStore::~DiskKVStore() {
    if (base_ != nullptr && file_bytes_ > 0) {
        // MAP_SHARED: the kernel flushes dirty pages back on its own schedule, so
        // the data already lives on disk (survives a process crash). Just unmap.
        ::munmap(base_, file_bytes_);
        base_ = nullptr;
    }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

std::uint64_t DiskKVStore::bump_clock() noexcept {
    return ++clock_;
}

bool DiskKVStore::valid_group(std::uint32_t id) const noexcept {
    return id < groups_.size() && groups_[id].page_count > 0;
}

std::size_t DiskKVStore::used_bytes() const noexcept {
    return static_cast<std::size_t>(used_pages_) * opts_.page_stride;
}
std::size_t DiskKVStore::free_bytes() const noexcept {
    return (static_cast<std::size_t>(max_pages_) - used_pages_) * opts_.page_stride;
}
std::uint32_t DiskKVStore::used_pages() const noexcept { return used_pages_; }

std::uint32_t DiskKVStore::find_free_run(std::uint32_t pages) const {
    if (pages == 0 || pages > max_pages_) { return static_cast<std::uint32_t>(-1); }
    std::uint32_t free_run = 0;
    std::uint32_t run_start = 0;
    for (std::uint32_t i = 0; i < max_pages_; ++i) {
        if (slot_used_[i]) { free_run = 0; continue; }
        if (free_run == 0) { run_start = i; } // candidate run start
        ++free_run;
        if (free_run == pages) { return run_start; } // first-fit start index
    }
    return static_cast<std::uint32_t>(-1);
}

void DiskKVStore::reclaim_run(std::uint32_t first, std::uint32_t pages) {
    for (std::uint32_t i = 0; i < pages; ++i) { slot_used_[first + i] = false; }
    used_pages_ -= pages;
}

std::optional<std::uint32_t> DiskKVStore::reserve(std::uint32_t page_count) {
    std::lock_guard<std::mutex> lock(mu_);
    if (page_count == 0 || page_count > max_pages_) { return std::nullopt; }
    free_list_.clear();
    const std::uint32_t first = find_free_run(page_count);
    if (first == static_cast<std::uint32_t>(-1)) { return std::nullopt; }
    for (std::uint32_t i = 0; i < page_count; ++i) { slot_used_[first + i] = true; }
    used_pages_ += page_count;
    groups_.push_back(Group{.first_page = first, .page_count = page_count,
                           .written = 0, .published = false, .released = false,
                           .crc = 0, .last_used = bump_clock()});
    return static_cast<std::uint32_t>(groups_.size() - 1);
}

bool DiskKVStore::write_page(std::uint32_t group_id, std::uint32_t page_index, const std::byte* src) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!valid_group(group_id) || page_index >= groups_[group_id].page_count) { return false; }
    Group& g = groups_[group_id];
    if (g.published) { return false; }
    const std::size_t off =
        (static_cast<std::size_t>(g.first_page) + page_index) * opts_.page_stride;
    std::memcpy(base_ + off, src, opts_.page_stride);
    if (page_index + 1 > g.written) { g.written = page_index + 1; }
    return true;
}

bool DiskKVStore::write_group(std::uint32_t group_id, std::span<const std::byte> bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!valid_group(group_id)) { return false; }
    Group& g = groups_[group_id];
    if (g.published || bytes.size() != static_cast<std::size_t>(g.page_count) * opts_.page_stride) {
        return false;
    }
    const std::size_t off = static_cast<std::size_t>(g.first_page) * opts_.page_stride;
    std::memcpy(base_ + off, bytes.data(), bytes.size());
    g.written = g.page_count;
    return true;
}

bool DiskKVStore::publish(std::uint32_t group_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!valid_group(group_id)) { return false; }
    Group& g = groups_[group_id];
    if (g.published || g.written != g.page_count) { return false; }
    const std::size_t off = static_cast<std::size_t>(g.first_page) * opts_.page_stride;
    const std::size_t bytes = static_cast<std::size_t>(g.page_count) * opts_.page_stride;
    g.crc = crc32c(std::span<const std::byte>(base_ + off, bytes));
    g.published = true;
    g.last_used = bump_clock();
    ++published_groups_total_;
    return true;
}

bool DiskKVStore::release(std::uint32_t group_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!valid_group(group_id) || !groups_[group_id].published) { return false; }
    groups_[group_id].released = true;
    return true;
}

bool DiskKVStore::touch(std::uint32_t group_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!valid_group(group_id) || !groups_[group_id].published) { return false; }
    groups_[group_id].last_used = bump_clock();
    return true;
}

std::optional<std::span<const std::byte>>
DiskKVStore::read_page(std::uint32_t group_id, std::uint32_t page_index) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!valid_group(group_id)) { return std::nullopt; }
    const Group& g = groups_[group_id];
    if (!g.published || g.page_count == 0 || page_index >= g.page_count) { return std::nullopt; }
    const std::size_t off =
        (static_cast<std::size_t>(g.first_page) + page_index) * opts_.page_stride;
    return std::span<const std::byte>(base_ + off, opts_.page_stride);
}

bool DiskKVStore::read_group(std::uint32_t group_id, std::span<std::byte> dst) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!valid_group(group_id)) { return false; }
    const Group& g = groups_[group_id];
    if (!g.published || g.page_count == 0) { return false; }
    if (dst.size() != static_cast<std::size_t>(g.page_count) * opts_.page_stride) { return false; }
    const std::size_t off = static_cast<std::size_t>(g.first_page) * opts_.page_stride;
    if (opts_.verify_crc) {
        const std::size_t bytes = dst.size();
        if (crc32c(std::span<const std::byte>(base_ + off, bytes)) != g.crc) {
            return false; // corrupt -> caller treats as a miss (recompute)
        }
    }
    std::memcpy(dst.data(), base_ + off, dst.size());
    return true;
}

bool DiskKVStore::contains(std::uint32_t group_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!valid_group(group_id)) { return false; }
    const Group& g = groups_[group_id];
    return g.published && g.page_count > 0;
}

std::vector<std::uint32_t> DiskKVStore::evict_until_free(std::uint32_t free_pages) {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::uint32_t> evicted;
    if (free_bytes() >= static_cast<std::size_t>(free_pages) * opts_.page_stride) { return evicted; }
    // LRU: lowest last_used among (published && released) groups.
    for (;;) {
        std::optional<std::uint32_t> victim;
        std::uint64_t best = ~std::uint64_t(0);
        for (std::uint32_t id = 0; id < groups_.size(); ++id) {
            const Group& g = groups_[id];
            if (!g.published || !g.released || g.page_count == 0) { continue; }
            if (g.last_used < best) { best = g.last_used; victim = id; }
        }
        if (!victim.has_value()) { break; } // nothing evictable
        if (!evict_unlocked(*victim)) { break; } // defensive: should not happen
        evicted.push_back(*victim);
        if (free_bytes() >= static_cast<std::size_t>(free_pages) * opts_.page_stride) { break; }
    }
    return evicted;
}

bool DiskKVStore::evict_unlocked(std::uint32_t group_id) {
    if (!valid_group(group_id)) { return false; }
    Group& g = groups_[group_id];
    if (!g.published || !g.released || g.page_count == 0) { return false; }
    reclaim_run(g.first_page, g.page_count);
    evicted_pages_total_ += g.page_count;
    g.page_count = 0; // mark slot dead (keeps id space stable)
    return true;
}

bool DiskKVStore::evict(std::uint32_t group_id) {
    std::lock_guard<std::mutex> lock(mu_);
    return evict_unlocked(group_id);
}

void DiskKVStore::clear() noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    for (std::uint32_t id = 0; id < groups_.size(); ++id) {
        if (groups_[id].page_count > 0) {
            reclaim_run(groups_[id].first_page, groups_[id].page_count);
            groups_[id].page_count = 0;
        }
    }
    free_list_.clear();
}

std::vector<DiskKVStore::GroupInfo> DiskKVStore::groups() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<GroupInfo> out;
    out.reserve(groups_.size());
    for (std::uint32_t id = 0; id < groups_.size(); ++id) {
        const Group& g = groups_[id];
        if (g.page_count == 0) { continue; }
        out.push_back(GroupInfo{.group_id = id, .page_count = g.page_count,
                                .published = g.published,
                                .evictable = g.published && g.released,
                                .crc32 = g.crc});
    }
    return out;
}

} // namespace ninfer
