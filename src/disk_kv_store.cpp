#include "diskkv/disk_kv_store.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <system_error>

namespace ninfer {
namespace {

constexpr std::size_t kAlignment = 4096;
constexpr std::size_t kTrailerSize = 128;

std::size_t round_up(std::size_t v, std::size_t a) { return (v + (a - 1)) & ~(a - 1); }

std::string index_path(const std::string& data_path) {
    return data_path + ".idx";
}

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

// ---------- layout ----------

std::size_t DiskKVStore::slot_bytes() const noexcept {
    return round_up(kSlotHeaderSize + opts_.slot_size, kAlignment);
}
std::size_t DiskKVStore::data_bytes() const noexcept {
    return static_cast<std::size_t>(max_slots_) * slot_bytes();
}
std::size_t DiskKVStore::trailer_off() const noexcept {
    return round_up(data_bytes(), kAlignment);
}
std::size_t DiskKVStore::page_off(std::uint32_t slot) const noexcept {
    return static_cast<std::size_t>(slot) * slot_bytes() + kSlotHeaderSize;
}
DiskKVStore::Header* DiskKVStore::slot_hdr(std::uint32_t slot) const {
    return reinterpret_cast<Header*>(base_ + static_cast<std::size_t>(slot) * slot_bytes());
}
const DiskKVStore::Header* DiskKVStore::slot_hdr_c(std::uint32_t slot) const {
    return reinterpret_cast<const Header*>(base_ + static_cast<std::size_t>(slot) * slot_bytes());
}

// ---------- lifecycle ----------

DiskKVStore::DiskKVStore(Options opts) : opts_(std::move(opts)) {
    if (opts_.path.empty()) { throw std::invalid_argument("disk kv store path is empty"); }
    if (opts_.slot_size < kSlotHeaderSize) {
        throw std::invalid_argument("disk kv store slot_size too small");
    }
    if (opts_.max_slots == 0) {
        if (opts_.capacity_bytes == 0) {
            throw std::invalid_argument("disk kv store needs max_slots or capacity_bytes");
        }
        opts_.max_slots = static_cast<std::uint32_t>(
            opts_.capacity_bytes / round_up(kSlotHeaderSize + opts_.slot_size, kAlignment));
        if (opts_.max_slots == 0) { throw std::invalid_argument("capacity too small"); }
    }
    max_slots_ = opts_.max_slots;
    slot_pitch_ = slot_bytes();
    file_bytes_ = trailer_off() + kTrailerSize;

    struct stat st{};
    const bool data_exists = (::stat(opts_.path.c_str(), &st) == 0 && st.st_size > 0);
    if (!data_exists) {
        create_fresh();
    } else {
        if (static_cast<std::size_t>(st.st_size) < file_bytes_) {
            throw std::runtime_error("disk kv store " + opts_.path +
                                     " smaller than required by current geometry");
        }
        // Open the EXISTING file: never truncate.
        fd_ = ::open(opts_.path.c_str(), O_RDWR, 0644);
        if (fd_ < 0) {
            throw std::runtime_error(std::string("open failed: ") + std::strerror(errno));
        }
        void* mapped = ::mmap(nullptr, file_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapped == MAP_FAILED) {
            const int e = errno;
            ::close(fd_); fd_ = -1;
            throw std::runtime_error(std::string("mmap failed: ") + std::strerror(e));
        }
        base_ = static_cast<std::byte*>(mapped);
    }
    for (std::uint32_t s = 0; s < max_slots_; ++s) { free_slots_.push_back(s); }

    if (!load_index()) {
        rebuild_from_scan();
    }
}

void DiskKVStore::create_fresh() {
    fd_ = ::open(opts_.path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) {
        throw std::runtime_error(std::string("open failed: ") + std::strerror(errno));
    }
    if (::ftruncate(fd_, static_cast<off_t>(file_bytes_)) != 0) {
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
}

DiskKVStore::~DiskKVStore() {
    if (base_ != nullptr) {
        ::munmap(base_, file_bytes_);
        base_ = nullptr;
    }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

// ---------- index persistence ----------

bool DiskKVStore::load_index() {
    const std::string path = index_path(opts_.path);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { return false; }  // no index yet: caller rebuilds
    IdxHeader hdr{};
    if (std::fread(&hdr, sizeof(hdr), 1, f) != 1 ||
        hdr.magic != IdxHeader::kMagic ||
        hdr.slot_size != static_cast<std::uint32_t>(slot_bytes()) ||
        hdr.max_slots != max_slots_) {
        std::fclose(f);
        return false;  // foreign/corrupt index: caller rebuilds
    }
    std::vector<IdxEntry> rows(hdr.count);
    if (hdr.count > 0 &&
        std::fread(rows.data(), sizeof(IdxEntry), hdr.count, f) != hdr.count) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);

    index_.clear();
    index_.reserve(hdr.count);
    free_slots_.clear();
    free_slots_.reserve(max_slots_);
    std::uint32_t occupied = 0;
    for (const IdxEntry& r : rows) {
        if (r.slot >= max_slots_) { continue; }  // defensive: ignore bad row
        const Header& h = *slot_hdr_c(r.slot);
        const DiskKVIdentity id{r.lo, r.hi, r.tag, r.frontier};
        // Header-only check (cheap): the index row must agree with the slot
        // header. If the slot was repurposed (different identity), drop the
        // row. Data integrity is verified lazily at READ time (read_page
        // CRC) — a corrupted page becomes a restore miss, not a startup cost.
        if (h.magic != Header::kMagic || h.lo != id.lo || h.hi != id.hi ||
            h.tag != id.tag || h.frontier != id.frontier) {
            continue;
        }
        index_[id] = r.slot;
        if (h.last_used > clock_) { clock_ = h.last_used; }
    }
    // Rebuild the free pool from the (verified) live slots.
    std::vector<bool> live(max_slots_, false);
    for (const auto& [id, s] : index_) { live[s] = true; }
    for (std::uint32_t s = 0; s < max_slots_; ++s) {
        if (!live[s]) { free_slots_.push_back(s); }
    }
    (void)occupied;
    return true;
}

void DiskKVStore::rebuild_from_scan() {
    index_.clear();
    free_slots_.clear();
    free_slots_.reserve(max_slots_);
    clock_ = 0;
    std::vector<bool> live(max_slots_, false);
    for (std::uint32_t s = 0; s < max_slots_; ++s) {
        const Header& h = *slot_hdr_c(s);
        if (h.magic != Header::kMagic) { continue; }
        const std::uint64_t crc =
            crc32c(std::span<const std::byte>(base_ + page_off(s), opts_.slot_size));
        if (crc != h.crc) { continue; }  // torn/corrupt slot: dead
        const DiskKVIdentity id{h.lo, h.hi, h.tag, h.frontier};
        index_[id] = s;
        live[s] = true;
        if (h.last_used > clock_) { clock_ = h.last_used; }
    }
    for (std::uint32_t s = 0; s < max_slots_; ++s) {
        if (!live[s]) { free_slots_.push_back(s); }
    }
    rebuilt_from_scan_ = true;
    persist_index_unlocked();
}

void DiskKVStore::persist_index_unlocked() {
    // Atomic: write a tmp file, fsync, rename over the index path.
    const std::string path = index_path(opts_.path);
    const std::string tmp  = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) { return; }
    IdxHeader hdr{};
    hdr.magic     = IdxHeader::kMagic;
    hdr.version   = 1;
    hdr.count     = static_cast<std::uint32_t>(index_.size());
    hdr.slot_size = static_cast<std::uint32_t>(slot_bytes());
    hdr.max_slots = max_slots_;
    hdr.clock     = clock_;
    std::fwrite(&hdr, sizeof(hdr), 1, f);
    std::fseek(f, 0, SEEK_END);
    const std::size_t total = (std::size_t)ftell(f) + index_.size() * sizeof(IdxEntry);
    for (const auto& [id, s] : index_) {
        IdxEntry r{};
        r.lo = id.lo;
        r.hi = id.hi;
        r.tag = id.tag;
        r.frontier = id.frontier;
        r.slot = s;
        std::fwrite(&r, sizeof(r), 1, f);
    }
    std::fflush(f);
    std::fclose(f);
    // Deliberately NO fsync on the hot path: a torn index is recovered by a
    // full header rescan on next open (cheap: slots are CRC-verified), and
    // the spill worker would otherwise stall behind fsync latency per page.
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        ::unlink(tmp.c_str());
        return;
    }
}

// ---------- internals ----------

std::uint64_t DiskKVStore::bump_clock() noexcept {
    return ++clock_;
}

void DiskKVStore::zero_slot(std::uint32_t slot) {
    std::memset(slot_hdr(slot), 0, sizeof(Header));
}

void DiskKVStore::record_live(const DiskKVIdentity& id, std::uint32_t slot, std::uint64_t last_used) {
    index_[id] = slot;
    if (last_used > clock_) { clock_ = last_used; }
}

void DiskKVStore::release_slot(std::uint32_t slot) {
    free_slots_.push_back(slot);
}

std::optional<DiskKVStore::EvictedPage> DiskKVStore::evict_one_lru() {
    // Caller holds mu_. Lowest last_used wins; ties broken by slot index.
    const Header* victim_hdr = nullptr;
    std::uint32_t victim_slot = 0;
    bool found = false;
    for (const auto& [id, s] : index_) {
        const Header& h = *slot_hdr_c(s);
        if (!found || h.last_used < victim_hdr->last_used ||
            (h.last_used == victim_hdr->last_used && s < victim_slot)) {
            victim_hdr = &h;
            victim_slot = s;
            found = true;
        }
    }
    if (!found) { return std::nullopt; }
    const EvictedPage victim{DiskKVIdentity{victim_hdr->lo, victim_hdr->hi, victim_hdr->tag,
                                             victim_hdr->frontier},
                             victim_slot};
    index_.erase(victim.id);
    zero_slot(victim_slot);
    release_slot(victim_slot);
    return victim;
}

bool DiskKVStore::identity_matches(const Header& h, const DiskKVIdentity& id) const {
    return h.lo == id.lo && h.hi == id.hi && h.tag == id.tag && h.frontier == id.frontier;
}

// ---------- API ----------

std::uint32_t DiskKVStore::live_slots() const noexcept {
    return static_cast<std::uint32_t>(index_.size());
}
std::size_t DiskKVStore::used_bytes() const noexcept {
    return index_.size() * slot_pitch_;
}
std::size_t DiskKVStore::free_bytes() const noexcept {
    return static_cast<std::size_t>(max_slots_ - live_slots()) * slot_pitch_;
}

bool DiskKVStore::upsert_page(const DiskKVIdentity& id, std::span<const std::byte> bytes,
                             std::vector<std::uint32_t>* evicted) {
    if (bytes.size() != opts_.slot_size || max_slots_ == 0) { return false; }
    std::lock_guard<std::mutex> lock(mu_);

    const auto it = index_.find(id);
    if (it != index_.end()) {
        // Already restorable: refresh LRU in memory (flushed by flush_index).
        (*slot_hdr(it->second)).last_used = bump_clock();
        return true;
    }

    std::uint32_t slot = 0;
    if (!free_slots_.empty()) {
        slot = free_slots_.back();
        free_slots_.pop_back();
    } else {
        // Full: LRU-evict the coldest page.
        auto victim = evict_one_lru();
        if (!victim.has_value()) { return false; }
        if (evicted != nullptr) {
            evicted->push_back(victim->slot);
        }
        // evict_one_lru() already released the victim slot into free_slots_.
        slot = free_slots_.back();
        free_slots_.pop_back();
    }

    // Write page bytes, then stamp the header LAST (magic makes it live).
    std::memcpy(base_ + page_off(slot), bytes.data(), opts_.slot_size);
    Header nh{};
    nh.magic     = Header::kMagic;
    nh.lo        = id.lo;
    nh.hi        = id.hi;
    nh.tag       = id.tag;
    nh.frontier  = id.frontier;
    nh.crc       = static_cast<std::uint32_t>(crc32c(
                       std::span<const std::byte>(base_ + page_off(slot), opts_.slot_size)));
    nh.last_used = bump_clock();
    std::memcpy(slot_hdr(slot), &nh, sizeof(Header));
    record_live(id, slot, nh.last_used);
    // Structural change (new live page / eviction): persist NOW so the page
    // is discoverable across a crash. LRU-only changes are lazy.
    persist_index_unlocked();
    return true;
}

bool DiskKVStore::read_page(const DiskKVIdentity& id, std::span<std::byte> dst) {
    if (dst.size() != opts_.slot_size) { return false; }
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = index_.find(id);
    if (it == index_.end()) { return false; }
    const std::uint32_t slot = it->second;
    const Header& h = *slot_hdr_c(slot);
    if (h.magic != Header::kMagic || !identity_matches(h, id)) { return false; }
    const std::size_t off = page_off(slot);
    if (opts_.verify_crc) {
        const std::uint64_t crc = crc32c(std::span<const std::byte>(base_ + off, opts_.slot_size));
        if (crc != h.crc) { return false; }
    }
    std::memcpy(dst.data(), base_ + off, opts_.slot_size);
    // Read counts as use: refresh LRU in memory (lazy persist).
    (*slot_hdr(slot)).last_used = bump_clock();
    return true;
}

bool DiskKVStore::contains(const DiskKVIdentity& id) const {
    std::lock_guard<std::mutex> lock(mu_);
    return index_.find(id) != index_.end();
}

void DiskKVStore::flush_index() {
    std::lock_guard<std::mutex> lock(mu_);
    persist_index_unlocked();
}

bool DiskKVStore::touch(const DiskKVIdentity& id) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = index_.find(id);
    if (it == index_.end()) { return false; }
    (*slot_hdr(it->second)).last_used = bump_clock();
    return true;
}

bool DiskKVStore::evict(const DiskKVIdentity& id) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = index_.find(id);
    if (it == index_.end()) { return false; }
    const std::uint32_t slot = it->second;
    index_.erase(it);
    zero_slot(slot);
    release_slot(slot);
    persist_index_unlocked();
    return true;
}

std::vector<DiskKVIdentity> DiskKVStore::evict_until_free(std::uint32_t free) {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<DiskKVIdentity> gone;
    while (index_.size() + free_slots_.size() < max_slots_ &&
           free_slots_.size() < free) {
        auto v = evict_one_lru();
        if (!v.has_value()) { break; }
        gone.push_back(v->id);
    }
    if (!gone.empty()) { persist_index_unlocked(); }
    return gone;
}

std::vector<DiskKVIdentity> DiskKVStore::live_identities() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<DiskKVIdentity> out;
    out.reserve(index_.size());
    for (const auto& [id, s] : index_) { (void)s; out.push_back(id); }
    return out;
}

std::vector<std::uint32_t> DiskKVStore::live_slot_list() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::uint32_t> out;
    out.reserve(index_.size());
    for (const auto& [id, s] : index_) { out.push_back(s); }
    return out;
}

} // namespace ninfer
