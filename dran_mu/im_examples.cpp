// im_examples.cpp — basic InfiniteMemory usage, one small example per pattern.
//
// InfiniteMemory = the card's DRAM acts as a cache in front of the card's SSD. The host sees
// one big address space (e.g. 1.8 TB) and uses it like ordinary memory; the device moves
// 512 KB pages between DRAM and SSD on its own. The card DRAM is split in two:
//   Cache Region   - automatic cache for host loads/stores (SSD misses handled by the device)
//   Preload Region - data you explicitly preloadMemory(); the ONLY memory MU kernels may read
//
// Examples (run one or several):  ./im_examples [--device N] [--ex 123456]
//   1  plain memory: allocate, write, read back
//   2  write dirty pages to the SSD: commitSyncMemory (data still ends with the process)
//   3  sparse reads: prefetchMemory first, then read
//   4  keep hot data in DRAM: pinMemory / unpinMemory
//   5  compute on the card: preload -> MU kernel -> check -> unload
//   6  reuse a preloaded slot: overwrite it from the host, never unload (no SSD write-back)
//
// None of these write filler data or evict anything, so they are safe on a shared card.
// Each step prints the change in the card's SSD byte counters (from `xcena_cli im get-smart`):
// that delta, not the time, tells you whether the SSD was touched.
//
// Rules learned the hard way (all measured on an MX1P with an SSD attached):
//   - never call unmapMemory to "evict" data: it discards the data (reads back all zero)
//   - call pxl::flushHostCache after the host writes and before the host re-reads device output
//   - an MU kernel on a buffer that is NOT preloaded returns Success with wrong results
//
// Build: listed in CMakeLists.txt; examples 5 and 6 need mu_kernel/mu_kernel.mubin (dran_mix_values).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "pxl/pxl.hpp"

using half_t = uint16_t;

namespace
{
constexpr size_t GB = 1ull << 30;
constexpr size_t MB = 1ull << 20;
int gDev = 0;
pxl::Context* gCtx = nullptr;

double msSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

const char* st(pxl::MemoryStatus s)
{
    switch (s)
    {
    case pxl::MemoryStatus::Success: return "Success";
    case pxl::MemoryStatus::ResourceExhaust: return "ResourceExhaust";
    case pxl::MemoryStatus::InvalidRequest: return "InvalidRequest";
    case pxl::MemoryStatus::NotSupported: return "NotSupported";
    default: return "?";
    }
}

// SSD traffic so far: the "nvme_read" / "nvme_write" lines of the Bytes section of get-smart.
struct Ssd
{
    long long rd = -1, wr = -1;
};
Ssd readSsd()
{
    Ssd s;
    const std::string cmd = "xcena_cli im get-smart " + std::to_string(gDev) + " 2>/dev/null";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return s;
    char line[256];
    bool inBytes = false;
    while (fgets(line, sizeof(line), p))
    {
        if (strstr(line, "Bytes:")) { inBytes = true; continue; }
        if (!inBytes) continue;
        const char* colon = strchr(line, ':');
        if (!colon) continue;
        if (strstr(line, "nvme_read")) s.rd = std::strtoll(colon + 1, nullptr, 10);
        if (strstr(line, "nvme_write")) s.wr = std::strtoll(colon + 1, nullptr, 10);
    }
    pclose(p);
    return s;
}

// Run one step, print status, time and SSD bytes moved.
template <class F>
void step(const char* name, F call)
{
    const Ssd a = readSsd();
    const auto t0 = std::chrono::steady_clock::now();
    const pxl::MemoryStatus s = call();
    const double ms = msSince(t0);
    const Ssd b = readSsd();
    if (a.rd < 0 || b.rd < 0)
        printf("  %-40s %-15s %8.1f ms | SSD counters n/a\n", name, st(s), ms);
    else
        printf("  %-40s %-15s %8.1f ms | SSD read %7.1f MB, write %7.1f MB\n", name, st(s), ms,
               (b.rd - a.rd) / 1e6, (b.wr - a.wr) / 1e6);
}

// Deterministic content so every read can be checked.
uint64_t word(size_t i, uint64_t seed) { return (i + 1) * 0x9E3779B97F4A7C15ull ^ seed; }
void fill(void* p, size_t bytes, uint64_t seed)
{
    auto* w = static_cast<uint64_t*>(p);
    for (size_t i = 0; i < bytes / 8; i++) w[i] = word(i, seed);
    pxl::flushHostCache(p, bytes);  // push the host CPU's cached copy out to the device
}
size_t countBad(const void* p, size_t bytes, uint64_t seed)
{
    pxl::flushHostCache(const_cast<void*>(p), bytes);  // drop stale CPU-cached lines first
    const auto* w = static_cast<const uint64_t*>(p);
    size_t bad = 0;
    for (size_t i = 0; i < bytes / 8; i++) bad += w[i] != word(i, seed);
    return bad;
}

half_t f2h(float f)
{
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = static_cast<int32_t>((x >> 23) & 0xFF) - 112;
    uint32_t m = x & 0x7FFFFFu;
    if (e <= 0) return static_cast<half_t>(sign);
    if (e >= 31) return static_cast<half_t>(sign | 0x7C00u);
    return static_cast<half_t>(sign | (e << 10) | ((m + 0x1000u) >> 13));
}

// ---------------------------------------------------------------------------------------------
// 1. Plain memory. allocateMemory returns a host pointer into the card's address space.
//    Ordinary loads and stores work; the device caches pages in DRAM and writes them to the SSD
//    when it needs room. Nothing else is required to use InfiniteMemory as big, slow memory.
void ex1()
{
    printf("\n== 1  plain memory: allocate, write, read back ==\n");
    void* p = pxl::allocateMemory(gDev, GB);
    if (!p) { printf("  allocateMemory failed\n"); return; }

    auto t0 = std::chrono::steady_clock::now();
    fill(p, GB, 1);
    printf("  write 1 GB from the host: %.0f ms\n", msSince(t0));

    t0 = std::chrono::steady_clock::now();
    const size_t bad = countBad(p, GB, 1);
    printf("  read 1 GB back: %.0f ms, %zu wrong words\n", msSince(t0), bad);

    pxl::releaseMemory(p);  // release discards the data (it unmaps the range)
}

// 2. Commit. Writes first land in the DRAM cache ("dirty"). commitSyncMemory writes the
//    dirty pages of a range to the SSD and waits. The data stays cached and readable.
//    This is NOT persistence across processes: releaseMemory and process exit drop the
//    mapping, and the data cannot be reached again. Keep a copy in a file if it must survive.
//    (commitMemory is the asynchronous form.)
void ex2()
{
    printf("\n== 2  write dirty pages to the SSD: commitSyncMemory ==\n");
    void* p = pxl::allocateMemory(gDev, GB);
    if (!p) { printf("  allocateMemory failed\n"); return; }
    fill(p, GB, 2);
    step("commitSyncMemory(1 GB)", [&] { return pxl::commitSyncMemory(p, GB); });
    step("commitSyncMemory again (nothing dirty)", [&] { return pxl::commitSyncMemory(p, GB); });
    printf("  read back after commit: %zu wrong words\n", countBad(p, GB, 2));
    // To make data cold for a test, commit it and then let other data push it out of the cache
    // (or shrink the cache with `xcena_cli im set-dram-capacity`). Do NOT use unmapMemory.
    pxl::releaseMemory(p);
}

// 3. Sparse reads. The device moves whole 512 KB pages, so reading a 6 KB column that is on the
//    SSD costs a 512 KB SSD read. If you know which pieces you need, issue prefetchMemory for
//    all of them first (asynchronous), then read: the SSD reads overlap instead of queueing.
//    Here the data is still cached, so the SSD delta is ~0; with evicted data (im_probe --part B
//    --fill N) this pattern took 64 x 6 KB columns from 33.7 ms to 9.8 ms.
void ex3()
{
    printf("\n== 3  sparse reads: prefetchMemory, then read ==\n");
    const size_t colBytes = 6 * 1024, numCols = GB / colBytes, want = 64;
    auto* base = static_cast<uint8_t*>(pxl::allocateMemory(gDev, GB));
    if (!base) { printf("  allocateMemory failed\n"); return; }
    fill(base, GB, 3);

    size_t cols[want];
    uint64_t x = 12345;
    for (size_t i = 0; i < want; i++)
    {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        cols[i] = x % numCols;
    }

    step("prefetchMemory x64 (6 KB each)", [&] {
        pxl::MemoryStatus s = pxl::MemoryStatus::Success;
        for (size_t i = 0; i < want; i++)
        {
            const pxl::MemoryStatus r = pxl::prefetchMemory(base + cols[i] * colBytes, colBytes);
            if (r != pxl::MemoryStatus::Success) s = r;
        }
        return s;
    });

    static uint8_t staging[want][6 * 1024];
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < want; i++)
    {
        pxl::flushHostCache(base + cols[i] * colBytes, colBytes);
        std::memcpy(staging[i], base + cols[i] * colBytes, colBytes);
    }
    printf("  gather 64 columns: %.2f ms\n", msSince(t0));
    pxl::releaseMemory(base);
}

// 4. Pinning. pinMemory keeps a range in the DRAM cache so it is never evicted (useful for a
//    small, hot index while a large table streams through the cache). Unpin when done.
void ex4()
{
    printf("\n== 4  keep hot data in DRAM: pinMemory ==\n");
    const size_t bytes = 64 * MB;
    void* p = pxl::allocateMemory(gDev, bytes);
    if (!p) { printf("  allocateMemory failed\n"); return; }
    fill(p, bytes, 4);
    step("pinMemory(64 MB)", [&] { return pxl::pinMemory(p, bytes); });
    printf("  ... host reads of this range now never go to the SSD ...\n");
    step("unpinMemory(64 MB)", [&] { return pxl::unpinMemory(p, bytes); });
    pxl::releaseMemory(p);
}

// 5. Compute on the card. MU kernels can only read the Preload Region, so:
//      write from host -> flushHostCache -> preloadMemory -> run kernel -> check -> unloadMemory
//    preloadMemory always reads the range from the SSD (even if it is cached), and unloadMemory
//    always writes it back (even if unchanged): ~0.2 s + ~0.35 s per GB measured. Preload in
//    1 GB units. On a card without an SSD, preload returns NotSupported and is not needed.
//    Kernel used: dran_mix_values(values, indices, weights, outDim, topK, out):
//      out[r] = sum_i weights[i] * values[indices[i] * outDim + r]
void ex5()
{
    printf("\n== 5  compute on the card: preload -> MU kernel -> unload ==\n");
    auto* mod = pxl::createModule("mu_kernel/mu_kernel.mubin");
    auto* job = gCtx->createJob();
    if (!mod || !job || job->load(mod) != pxl::Result::Success) { printf("  module load failed\n"); return; }
    auto* fn = mod->createFunction("dran_mix_values");
    auto exec = job->buildMap(fn, 64);  // 64 tasks

    const int outDim = 3072, topK = 2;
    // Two 1 GB regions: the value table, and a small arena for indices / weights / output.
    auto* V = static_cast<half_t*>(pxl::allocateMemory(gDev, GB));
    auto* arena = static_cast<uint8_t*>(pxl::allocateMemory(gDev, GB));
    if (!V || !arena) { printf("  allocateMemory failed\n"); return; }
    auto* idx = reinterpret_cast<uint32_t*>(arena);
    auto* w = reinterpret_cast<float*>(arena + 4096);
    auto* out = reinterpret_cast<float*>(arena + 8192);

    // Column 10 holds r, column 20 holds 2r (small integers are exact in fp16).
    for (int r = 0; r < outDim; r++)
    {
        V[10 * outDim + r] = f2h(static_cast<float>(r % 1000));
        V[20 * outDim + r] = f2h(static_cast<float>(2 * (r % 1000)));
    }
    idx[0] = 10; idx[1] = 20;
    w[0] = 1.0f; w[1] = 0.5f;  // expected out[r] = (r % 1000) + (r % 1000) = 2 * (r % 1000)
    for (int r = 0; r < outDim; r++) out[r] = -1.0f;
    pxl::flushHostCache(V, GB);
    pxl::flushHostCache(arena, 16384);

    step("preloadMemory(V, 1 GB)", [&] { return pxl::preloadMemory(V, GB); });
    step("preloadMemory(arena, 1 GB)", [&] { return pxl::preloadMemory(arena, GB); });

    const auto t0 = std::chrono::steady_clock::now();
    pxl::Result e = exec->execute(V, idx, w, outDim, topK, out);
    if (e == pxl::Result::Success) e = exec->synchronize();
    printf("  kernel: %s, %.2f ms\n", e == pxl::Result::Success ? "Success" : "FAILED", msSince(t0));

    pxl::flushHostCache(out, outDim * sizeof(float));  // read the device's result, not a stale copy
    int good = 0;
    for (int r = 0; r < outDim; r++) good += std::fabs(out[r] - 2.0f * (r % 1000)) < 1e-3f;
    printf("  correct output rows: %d / %d\n", good, outDim);

    step("unloadMemory(V, 1 GB)", [&] { return pxl::unloadMemory(V, GB); });
    step("unloadMemory(arena, 1 GB)", [&] { return pxl::unloadMemory(arena, GB); });
    pxl::releaseMemory(V);
    pxl::releaseMemory(arena);
}

// 6. Reuse a preloaded slot instead of unload/preload. unloadMemory always writes the region
//    back to the SSD, even if nothing changed. To switch the data the MU sees without that
//    write-back: preload a "slot" once, keep the master copies elsewhere in InfiniteMemory, and
//    overwrite the slot from the host. Questions this answers (nvme_write tells):
//      a. do host writes into a preloaded range stay in device DRAM (no SSD write)?
//      b. does the MU see the new data after flushHostCache?
//      c. does an idle preloaded, dirty range get written back in the background?
//      d. does releaseMemory without unloadMemory write back, and does it free the slot?
//    Masters are plain InfiniteMemory (cached here, so the copy is DRAM -> DRAM; a cold master
//    adds SSD reads). Needs mu_kernel/mu_kernel.mubin. Uses 2 GB of the Preload Region.
constexpr int kOut = 3072;  // column length (fp16 elements), like a V column
half_t gPat[1024];          // fp16 of 0..1023, all exact
inline half_t pat(size_t col, int row, uint64_t seed) { return gPat[(col * 7 + row + seed * 13) % 1024]; }

void fillPattern(half_t* p, size_t bytes, uint64_t seed)
{
    const size_t cols = bytes / (kOut * sizeof(half_t));
    for (size_t c = 0; c < cols; c++)
        for (int r = 0; r < kOut; r++) p[c * kOut + r] = pat(c, r, seed);
    pxl::flushHostCache(p, bytes);
}

void parCopy(void* dst, const void* src, size_t bytes, int threads = 8)
{
    std::vector<std::thread> t;
    const size_t chunk = (bytes / threads + 4095) & ~size_t(4095);
    for (int i = 0; i < threads; i++)
    {
        const size_t off = i * chunk;
        if (off >= bytes) break;
        const size_t n = std::min(chunk, bytes - off);
        t.emplace_back([=] { std::memcpy(static_cast<uint8_t*>(dst) + off, static_cast<const uint8_t*>(src) + off, n); });
    }
    for (auto& x : t) x.join();
}

void ex6()
{
    printf("\n== 6  reuse a preloaded slot: overwrite from the host, never unload ==\n");
    for (int i = 0; i < 1024; i++) gPat[i] = f2h(static_cast<float>(i));
    auto* mod = pxl::createModule("mu_kernel/mu_kernel.mubin");
    auto* job = gCtx->createJob();
    if (!mod || !job || job->load(mod) != pxl::Result::Success) { printf("  module load failed\n"); return; }
    auto exec = job->buildMap(mod->createFunction("dran_mix_values"), 64);

    auto* mA = static_cast<half_t*>(pxl::allocateMemory(gDev, GB));
    auto* mB = static_cast<half_t*>(pxl::allocateMemory(gDev, GB));
    auto* slot = static_cast<half_t*>(pxl::allocateMemory(gDev, GB));
    auto* arena = static_cast<uint8_t*>(pxl::allocateMemory(gDev, GB));
    if (!mA || !mB || !slot || !arena) { printf("  allocateMemory failed\n"); return; }
    auto* idx = reinterpret_cast<uint32_t*>(arena);
    auto* w = reinterpret_cast<float*>(arena + 4096);
    auto* out = reinterpret_cast<float*>(arena + 8192);
    const size_t cols = GB / (kOut * sizeof(half_t));

    printf("  masters A and B: 1 GB each, %zu columns of %d fp16\n", cols, kOut);
    fillPattern(mA, GB, 1);
    fillPattern(mB, GB, 2);
    step("commitSyncMemory(masters, 2 GB)", [&] {
        const pxl::MemoryStatus s = pxl::commitSyncMemory(mA, GB);
        return s != pxl::MemoryStatus::Success ? s : pxl::commitSyncMemory(mB, GB);
    });
    parCopy(slot, mA, GB);
    pxl::flushHostCache(slot, GB);
    step("preloadMemory(slot = A, 1 GB)", [&] { return pxl::preloadMemory(slot, GB); });
    step("preloadMemory(arena, 1 GB)", [&] { return pxl::preloadMemory(arena, GB); });

    // MU reads 8 columns spread over the slot (first and last included) and they are compared
    // with the pattern of `seed`. A stale slot shows the other master's pattern.
    auto muCheck = [&](uint64_t seed) {
        int ok = 0;
        for (int j = 0; j < 8; j++)
        {
            const size_t c = j == 7 ? cols - 1 : j * (cols / 7);
            idx[0] = static_cast<uint32_t>(c);
            w[0] = 1.0f;
            pxl::flushHostCache(arena, 8192);
            if (exec->execute(slot, idx, w, kOut, 1, out) != pxl::Result::Success || exec->synchronize() != pxl::Result::Success)
                return -1;
            pxl::flushHostCache(out, kOut * sizeof(float));
            int good = 0;
            for (int r = 0; r < kOut; r++) good += out[r] == static_cast<float>((c * 7 + r + seed * 13) % 1024);
            ok += good == kOut;
        }
        return ok;
    };
    printf("  MU check after preload: %d / 8 columns match A\n", muCheck(1));

    const int rounds = 6;
    const Ssd s0 = readSsd();
    for (int i = 0; i < rounds; i++)
    {
        const uint64_t seed = i % 2 ? 1 : 2;  // B, A, B, ...
        auto t0 = std::chrono::steady_clock::now();
        parCopy(slot, seed == 1 ? mA : mB, GB);
        const double cp = msSince(t0);
        t0 = std::chrono::steady_clock::now();
        pxl::flushHostCache(slot, GB);
        const double fl = msSince(t0);
        printf("  overwrite %d with %c: copy %6.1f ms (%.2f GB/s), flush %6.1f ms | MU sees new data: %d / 8\n",
               i + 1, seed == 1 ? 'A' : 'B', cp, GB / cp / 1e6, fl, muCheck(seed));
    }
    const Ssd s1 = readSsd();
    if (s0.rd >= 0 && s1.rd >= 0)
        printf("  a. %d overwrites (%d GB written into the slot): SSD read %.1f MB, write %.1f MB  (want ~0)\n",
               rounds, rounds, (s1.rd - s0.rd) / 1e6, (s1.wr - s0.wr) / 1e6);

    const int idle = 20;
    printf("  c. idle %d s with a dirty preloaded slot ...\n", idle);
    std::this_thread::sleep_for(std::chrono::seconds(idle));
    const Ssd s2 = readSsd();
    if (s1.rd >= 0 && s2.rd >= 0)
        printf("     SSD read %.1f MB, write %.1f MB  (want ~0)\n", (s2.rd - s1.rd) / 1e6, (s2.wr - s1.wr) / 1e6);

    step("d. releaseMemory(slot), no unload", [&] { pxl::releaseMemory(slot); return pxl::MemoryStatus::Success; });
    // If the release freed the slot's Preload Region space, a new 1 GB preload succeeds.
    auto* probe = pxl::allocateMemory(gDev, GB);
    if (probe)
    {
        step("   preloadMemory(new 1 GB) after release", [&] { return pxl::preloadMemory(probe, GB); });
        step("   releaseMemory(new 1 GB), no unload", [&] { pxl::releaseMemory(probe); return pxl::MemoryStatus::Success; });
    }
    pxl::releaseMemory(arena);
    pxl::releaseMemory(mA);
    pxl::releaseMemory(mB);
}
}  // namespace

int main(int argc, char* argv[])
{
    std::string ex = "12345";  // 6 is opt-in: --ex 6
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) gDev = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ex") && i + 1 < argc) ex = argv[++i];
    }
    printf("im_examples  device=%d  examples=%s\n", gDev, ex.c_str());
    gCtx = pxl::createContext(gDev);
    if (!gCtx) { printf("createContext failed\n"); return 1; }
    if (ex.find('1') != std::string::npos) ex1();
    if (ex.find('2') != std::string::npos) ex2();
    if (ex.find('3') != std::string::npos) ex3();
    if (ex.find('4') != std::string::npos) ex4();
    if (ex.find('5') != std::string::npos) ex5();
    if (ex.find('6') != std::string::npos) ex6();
    return 0;
}
