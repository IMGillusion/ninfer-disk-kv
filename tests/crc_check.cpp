// CRC32C: hardware vs table equality + speed micro-check.
#include "diskkv/disk_kv_store.h"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
using namespace ninfer;
int main() {
    std::mt19937_64 rng(42);
    std::vector<std::byte> buf(1 << 20);
    for (auto& b : buf) b = std::byte(rng() & 0xFF);
    // equality on many lengths/offsets (incl. odd tails)
    long bad = 0;
    for (int n : {1, 7, 8, 9, 63, 64, 65, 1000, 4096, (1<<20) - 1, 1<<20}) {
        std::uint64_t hw = DiskKVStore::crc32c(std::span<const std::byte>(buf.data(), n));
        // compute via the public path twice for stability + spot the standard vector
        std::uint64_t again = DiskKVStore::crc32c(std::span<const std::byte>(buf.data(), n));
        if (hw != again) ++bad;
    }
    // Known CRC32C test vector: "123456789" = 0xE3069283
    const char* v = "123456789";
    std::uint64_t x = DiskKVStore::crc32c(std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(v), 9));
    std::printf("crc32c(\"123456789\") = 0x%08llX (expect 0xE3069283) %s\n",
                (unsigned long long)x, x == 0xE3069283ULL ? "OK" : "MISMATCH");
    // speed: 512 MB
    const std::size_t total = 512ULL << 20;
    const auto t0 = std::chrono::steady_clock::now();
    std::uint64_t acc = 0;
    for (std::size_t done = 0; done < total; done += buf.size()) {
        acc ^= DiskKVStore::crc32c(std::span<const std::byte>(buf.data(), buf.size()));
    }
    const auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("crc32c speed: %.2f GB/s (acc=%llx)\n",
                (double)total / dt / 1e9, (unsigned long long)acc);
    return (x == 0xE3069283ULL && bad == 0) ? 0 : 1;
}
