// CRC-32 (stored-format) compatibility check: the accelerated implementation
// must be bit-identical to the original byte-wise table loop (0xEDB88320).
#include "diskkv/disk_kv_store.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <random>
#include <span>
#include <vector>
using namespace ninfer;

// The ORIGINAL byte-wise reference (kept here verbatim as the compatibility oracle).
static std::uint64_t crc32_reference(std::span<const std::byte> data) {
    static std::array<std::uint32_t, 256> tbl = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    std::uint32_t c = 0xFFFFFFFFu;
    for (const std::byte b : data) {
        const std::uint8_t v = static_cast<std::uint8_t>(b);
        c = tbl[(c ^ v) & 0xFFu] ^ (c >> 8);
    }
    return static_cast<std::uint64_t>(c ^ 0xFFFFFFFFu);
}

int main() {
    std::mt19937_64 rng(1234);
    std::vector<std::byte> buf(1 << 20);
    for (auto& b : buf) b = std::byte(rng() & 0xFF);
    long bad = 0, checked = 0;
    // every length 0..600 plus large sizes at odd alignments
    std::vector<std::size_t> lens;
    for (std::size_t n = 0; n <= 600; ++n) lens.push_back(n);
    for (std::size_t n : std::vector<std::size_t>{1025, 4095, 4099, 65535, (1u<<20)-1, 1u<<20}) lens.push_back(n);
    for (std::size_t off = 0; off < 16; ++off) {
        for (std::size_t n : lens) {
            if (off + n > buf.size()) continue;
            const std::span<const std::byte> s(buf.data() + off, n);
            const auto acc = DiskKVStore::crc32c(s), ref = crc32_reference(s);
            ++checked;
            if (acc != ref) {
                if (++bad <= 3)
                    std::printf("MISMATCH off=%zu n=%zu acc=0x%08llX ref=0x%08llX\n", off, n,
                                (unsigned long long)acc, (unsigned long long)ref);
            }
        }
    }
    const char* v = "123456789";
    const auto x = DiskKVStore::crc32c(std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(v), 9));
    std::printf("crc32(\"123456789\") = 0x%08llX (expect 0xCBF43926 / CRC-32) %s\n",
                (unsigned long long)x, x == 0xCBF43926ULL ? "OK" : "MISMATCH");
    std::printf("compat: %ld/%ld identical to byte-wise reference\n", checked - bad, checked);
    const std::size_t total = 512ULL << 20;
    const auto t0 = std::chrono::steady_clock::now();
    std::uint64_t acc_bench = 0;
    for (std::size_t done = 0; done < total; done += buf.size())
        acc_bench ^= DiskKVStore::crc32c(std::span<const std::byte>(buf.data(), buf.size()));
    const auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("speed: %.2f GB/s (acc=%llx)\n", (double)total / dt / 1e9,
                (unsigned long long)acc_bench);
    return (bad == 0 && x == 0xCBF43926ULL) ? 0 : 1;
}
