#include "diskkv/disk_kv_bridge.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace ninfer {

namespace {

std::string family_path(const std::string& base, DiskKVKind kind) {
    const char* name = kind == DiskKVKind::MainKV ? "main"
                     : kind == DiskKVKind::BackendKV ? "backend" : "state";
    return base + "/diskkv_" + name;
}

// Split the total budget across enabled families: 60% main / 30% backend /
// 10% state (main KV dominates the working set in the 9-agent profile).
std::size_t family_budget(std::size_t total, std::size_t stride, std::uint32_t share,
                         std::uint32_t denominator) {
    if (stride == 0) { return 0; }
    const std::size_t bytes = total * share / denominator;
    return (bytes / stride) * stride;
}

} // namespace

DiskKVBridge::DiskKVBridge(Options opts) : opts_(std::move(opts)) {
    if (opts_.base_path.empty() || opts_.capacity_bytes == 0) {
        enabled_ = false;  // explicit off
        return;
    }
    if (opts_.main_page_stride == 0) {
        throw std::invalid_argument("disk kv bridge needs main_page_stride");
    }
    auto make_family = [&](DiskKVKind kind, std::size_t family_stride, std::size_t budget) {
        Family& f       = families_[kind_index(kind)];
        f.stride        = family_stride;
        f.path          = family_path(opts_.base_path, kind);
        if (family_stride == 0 || budget == 0) { return; }  // family disabled
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(f.path).parent_path(), ec);
        DiskKVStore::Options sopts;
        sopts.path           = f.path;
        sopts.slot_size      = family_stride;
        sopts.capacity_bytes = budget;
        sopts.max_slots      = 0;  // derived from capacity_bytes
        sopts.verify_crc     = opts_.verify_crc;
        f.store = std::make_unique<DiskKVStore>(sopts);
    };
    const std::size_t total = opts_.capacity_bytes;
    const std::uint32_t main_share    = opts_.backend_page_stride ? 65U : 85U;
    const std::uint32_t backend_share = opts_.backend_page_stride ? 25U : 0U;
    const std::uint32_t main_denom    = 100U;
    make_family(DiskKVKind::MainKV, opts_.main_page_stride,
                family_budget(total, opts_.main_page_stride, main_share, main_denom));
    make_family(DiskKVKind::BackendKV, opts_.backend_page_stride,
                family_budget(total, opts_.backend_page_stride, backend_share, 100U));
    // State images are large single blobs (~146 MiB each for the 48-layer GDN
    // state). Give them a slice big enough to hold several, so a restarted
    // multi-turn session can restore its recurrent state at a shared frontier.
    make_family(DiskKVKind::StateImage, opts_.state_page_stride,
                family_budget(total, opts_.state_page_stride, 100U - main_share - backend_share, 100U));
    enabled_ = families_[0].store != nullptr;
    if (!enabled_) {
        throw std::invalid_argument("disk kv bridge budget too small for one main page");
    }
    worker_ = std::thread([this] { worker_loop(); });
}

DiskKVBridge::~DiskKVBridge() {
    {
        std::lock_guard<std::mutex> lock(qmu_);
        quit_ = true;
    }
    qcv_.notify_all();
    if (worker_.joinable()) { worker_.join(); }
    // Drain whatever the worker already pulled off the queue so no page is
    // lost between join and destruction (best effort; store is still alive).
    while (true) {
        SpillJob job;
        {
            std::lock_guard<std::mutex> lock(qmu_);
            if (queue_.empty()) { break; }
            job = std::move(queue_.front());
            queue_.pop_front();
        }
        spill_queued(job);
    }
}

void DiskKVBridge::wait_idle() {
    std::unique_lock<std::mutex> lock(qmu_);
    qcv_.wait(lock, [this] { return queue_.empty() && in_flight_ == 0; });
}

void DiskKVBridge::worker_loop() {
    while (true) {
        SpillJob job;
        bool have = false;
        {
            std::unique_lock<std::mutex> lock(qmu_);
            qcv_.wait(lock, [this] { return quit_ || !queue_.empty(); });
            if (quit_ && queue_.empty()) { return; }
            if (queue_.empty()) { continue; }
            job = std::move(queue_.front());
            queue_.pop_front();
            in_flight_ += 1;
            have = true;
        }
        if (!have) {
            std::lock_guard<std::mutex> lock(qmu_);
            if (quit_) { return; }
            continue;
        }
        spill_queued(job);
        std::lock_guard<std::mutex> lock(qmu_);
        in_flight_ -= 1;
        qcv_.notify_all();  // wait_idle may be waiting on the queue draining
    }
}

bool DiskKVBridge::spill_page(const DiskKVIdentity& id, DiskKVKind kind,
                              std::span<const std::byte> bytes) {
    if (!enabled_) { return false; }
    const Family& f = family(kind);
    if (f.store == nullptr || bytes.size() != f.stride) { return false; }
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (f.store->contains(id)) {
            stats_.spill_dups += 1;
            return true;  // already restorable; shared-prefix dedupe
        }
    }
    std::lock_guard<std::mutex> lock(qmu_);
    if (queue_.size() >= kQueueCap) {
        std::lock_guard<std::mutex> stats_lock(mu_);
        stats_.queue_drops += 1;
        return false;  // backpressure: page won't be restorable (recompute later)
    }
    queue_.emplace_back(SpillJob{id, kind, {bytes.begin(), bytes.end()}});
    qcv_.notify_all();
    return true;
}

bool DiskKVBridge::spill_page_wait(const DiskKVIdentity& id, DiskKVKind kind,
                                   std::span<const std::byte> bytes,
                                   std::chrono::milliseconds timeout) {
    if (!enabled_) { return false; }
    const Family& f = family(kind);
    if (f.store == nullptr || bytes.size() != f.stride) { return false; }
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (f.store->contains(id)) {
            stats_.spill_dups += 1;
            return true;  // already restorable; shared-prefix dedupe
        }
    }
    std::unique_lock<std::mutex> lock(qmu_);
    // Bounded backpressure: the owner-death sweep must land a CONTIGUOUS
    // chain, so wait for a free slot instead of dropping the page. The
    // worker notifies qcv_ after every pop; on shutdown we bail out.
    if (!qcv_.wait_for(lock, timeout, [this] { return quit_ || queue_.size() < kQueueCap; })) {
        std::lock_guard<std::mutex> stats_lock(mu_);
        stats_.queue_drops += 1;
        return false;
    }
    if (quit_) { return false; }
    queue_.emplace_back(SpillJob{id, kind, {bytes.begin(), bytes.end()}});
    qcv_.notify_all();
    return true;
}

void DiskKVBridge::spill_queued(SpillJob& job) {
    Family& f = family(job.kind);
    if (f.store == nullptr || job.bytes.size() != f.stride) { return; }
    std::vector<std::uint32_t> evicted;
    std::span<const std::byte> bytes(job.bytes);
    if (!f.store->upsert_page(job.id, bytes, &evicted)) { return; }
    f.store->flush_index();  // durability point for this page
    std::lock_guard<std::mutex> lock(mu_);
    stats_.spills += 1;
    stats_.spill_bytes += job.bytes.size();
    stats_.evicted_slots += evicted.size();
}

bool DiskKVBridge::spill_page_sync(const DiskKVIdentity& id, DiskKVKind kind,
                                   std::span<const std::byte> bytes) {
    if (!enabled_) { return false; }
    const Family& f = family(kind);
    if (f.store == nullptr || bytes.size() != f.stride) { return false; }
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (f.store->contains(id)) {
            stats_.spill_dups += 1;
            return true;  // already restorable (dedupe)
        }
    }
    std::vector<std::uint32_t> evicted;
    if (!f.store->upsert_page(id, bytes, &evicted)) { return false; }
    f.store->flush_index();
    std::lock_guard<std::mutex> lock(mu_);
    stats_.spills += 1;
    stats_.spill_bytes += bytes.size();
    stats_.evicted_slots += evicted.size();
    return true;
}

bool DiskKVBridge::restore_page(const DiskKVIdentity& id, DiskKVKind kind,
                                std::span<std::byte> dst) const {
    if (!enabled_) { return false; }
    const Family& f = family(kind);
    if (f.store == nullptr || dst.size() != f.stride) { return false; }
    if (!f.store->read_page(id, dst)) {
        std::lock_guard<std::mutex> lock(mu_);
        stats_.restore_misses += 1;
        return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    stats_.restores += 1;
    stats_.restore_bytes += f.stride;
    return true;
}

bool DiskKVBridge::contains(const DiskKVIdentity& id, DiskKVKind kind) const {
    if (!enabled_) { return false; }
    const Family& f = family(kind);
    if (f.store == nullptr) { return false; }
    return f.store->contains(id);
}

std::size_t DiskKVBridge::probe_prefix(const std::vector<DiskKVIdentity>& page_ids,
                                       DiskKVKind kind) const {
    if (!enabled_) { return 0; }
    const Family& f = family(kind);
    if (f.store == nullptr) { return 0; }
    // Walk from page 0; the first miss stops the prefix.
    for (std::size_t p = 0; p < page_ids.size(); ++p) {
        if (!f.store->contains(page_ids[p])) { return p; }
    }
    return page_ids.size();
}

void DiskKVBridge::record_probe(std::uint32_t prompt_tokens,
                                std::uint32_t restorable_tokens) const {
    if (!enabled_ || restorable_tokens == 0 || opts_.base_path.empty()) { return; }
    try {
        const std::string path = opts_.base_path + "/diskkv_probe.jsonl";
        std::ofstream out(path, std::ios::app);
        if (!out) { return; }
        out << "{\"event\":\"diskkv_probe\",\"prompt_tokens\":" << prompt_tokens
            << ",\"restorable_tokens\":" << restorable_tokens << "}\n";
    } catch (...) {
    }
}

void DiskKVBridge::touch(const DiskKVIdentity& id, DiskKVKind kind) {
    if (!enabled_) { return; }
    const Family& f = family(kind);
    if (f.store == nullptr) { return; }
    f.store->touch(id);
}

std::vector<std::uint32_t> DiskKVBridge::live_state_frontiers() const {
    std::vector<std::uint32_t> out;
    if (!enabled_) { return out; }
    const Family& f = family(DiskKVKind::StateImage);
    if (f.store == nullptr) { return out; }
    for (const DiskKVIdentity& id : f.store->live_identities()) {
        if (id.frontier != 0) { out.push_back(id.frontier); }
    }
    return out;
}

DiskKVBridgeStats DiskKVBridge::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
}

std::size_t DiskKVBridge::used_bytes(DiskKVKind kind) const {
    const Family& f = family(kind);
    return f.store ? f.store->used_bytes() : 0;
}

std::uint32_t DiskKVBridge::slot_count(DiskKVKind kind) const {
    const Family& f = family(kind);
    return f.store ? f.store->slot_count() : 0;
}

const std::string& DiskKVBridge::path(DiskKVKind kind) const {
    static const std::string kEmpty;
    const Family& f = family(kind);
    return f.store ? f.path : kEmpty;
}

DiskKVBridge::Family& DiskKVBridge::family(DiskKVKind kind) {
    return families_[kind_index(kind)];
}
const DiskKVBridge::Family& DiskKVBridge::family(DiskKVKind kind) const {
    return families_[kind_index(kind)];
}

} // namespace ninfer
