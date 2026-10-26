// im_probe.cpp — map out how InfiniteMemory behaves, one primitive at a time.
//
// Every step prints: the API status, wall time, the card's InfiniteMemory counter deltas
// (from `xcena_cli im get-smart`, device-wide) and, where data is read back, whether the
// bytes are the ones written. Counter deltas are the ground truth for "did this touch
// the SSD": a step that claims to fetch from SSD but moves 0 nvme_read bytes did not.
//
// Parts
//   A  primitives on 1 GB regions: what unmap does (A1, A3), the correct commit/preload
//      path (A2), flushMemory, pin, prefetch granularity (A4)
//   B  real eviction (needs --fill N, N GB > the DRAM cache): host read of an evicted
//      region, 64 scattered 6 KB "V columns", prefetchSync and preload of evicted data
//   C  MU access: dran_mix_values over a buffer that is preloaded / resident but not
//      preloaded / (with --mu-cold) committed or evicted. Output checked on the host.
//
//   D  unload write-back off the critical path: releaseMemory without unload (D1),
//      unload and preload of two regions at the same time (D2)
//
// Run:  ./im_probe [--device N] [--part ABCD] [--fill GB] [--mu-cold]
// Build: listed in CMakeLists.txt next to dran_retrieve; needs mu_kernel/mu_kernel.mubin.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "pxl/pxl.hpp"

using half_t = uint16_t;

namespace
{
constexpr size_t GB = 1ull << 30;
constexpr size_t PAGE = 4096;
int gDev = 0;

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
    default: return "Unknown";
    }
}

// ---- InfiniteMemory counters: every "name : number" line, keyed by section ----
using Counters = std::map<std::string, long long>;
Counters readCounters()
{
    Counters c;
    const std::string cmd = "xcena_cli im get-smart " + std::to_string(gDev) + " 2>/dev/null";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return c;
    char line[256];
    std::string sec = "?";
    bool inStats = false;
    while (fgets(line, sizeof(line), p))
    {
        if (strstr(line, "InfiniteMemory Statistics")) inStats = true;
        if (!inStats) continue;
        if (strstr(line, "Config:")) { sec = "cfg"; continue; }
        if (strstr(line, "Status:")) { sec = "st"; continue; }
        if (strstr(line, "Counters:")) { sec = "ctr"; continue; }
        if (strstr(line, "Commands:")) { sec = "cmd"; continue; }
        if (strstr(line, "Bytes:")) { sec = "B"; continue; }
        const char* colon = strchr(line, ':');
        if (!colon) continue;
        std::string key(line, static_cast<size_t>(colon - line));
        while (!key.empty() && key.front() == ' ') key.erase(0, 1);
        while (!key.empty() && key.back() == ' ') key.pop_back();
        char* end = nullptr;
        const long long v = std::strtoll(colon + 1, &end, 10);
        if (end != colon + 1) c[sec + "." + key] = v;
    }
    pclose(p);
    return c;
}

std::string deltas(const Counters& a, const Counters& b)
{
    std::string out;
    char buf[96];
    for (const auto& [k, v] : b)
    {
        auto it = a.find(k);
        const long long d = v - (it == a.end() ? 0 : it->second);
        if (d == 0) continue;
        if (k.rfind("cfg.", 0) == 0) continue;  // configuration, not activity
        const std::string name = k.substr(k.find('.') + 1);
        if (k.rfind("B.", 0) == 0) snprintf(buf, sizeof(buf), " %s=%.1fMB", name.c_str(), d / 1e6);
        else if (k.rfind("st.", 0) == 0) snprintf(buf, sizeof(buf), " %s%+.1fMB", name.c_str(), d / 1e6);
        else snprintf(buf, sizeof(buf), " %s=%lld", name.c_str(), d);
        out += buf;
    }
    return out.empty() ? " (no counter moved)" : out;
}

// ---- deterministic pattern, checked per 4 KB page ----
inline uint64_t pat(uint64_t seed, uint64_t i)
{
    uint64_t x = i * 0x9E3779B97F4A7C15ull ^ seed;
    x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
    return x;
}
void fill(void* p, size_t bytes, uint64_t seed)
{
    auto* w = static_cast<uint64_t*>(p);
    for (size_t i = 0; i < bytes / 8; i++) w[i] = pat(seed, i);
}
// Returns a short verdict; also reports how many bad pages are all-zero.
std::string verify(const void* p, size_t bytes, uint64_t seed, size_t wordOffset = 0)
{
    const auto* w = static_cast<const volatile uint64_t*>(p);
    size_t bad = 0, zero = 0, pages = bytes / PAGE;
    for (size_t pg = 0; pg < pages; pg++)
    {
        bool ok = true, allZero = true;
        for (size_t j = 0; j < PAGE / 8; j++)
        {
            const uint64_t v = w[pg * (PAGE / 8) + j];
            if (v != pat(seed, wordOffset + pg * (PAGE / 8) + j)) ok = false;
            if (v != 0) allZero = false;
        }
        if (!ok) { bad++; zero += allZero; }
    }
    char buf[128];
    if (!bad) snprintf(buf, sizeof(buf), "intact (%zu pages)", pages);
    else snprintf(buf, sizeof(buf), "BAD %zu/%zu pages (%zu all-zero), first word 0x%016llx",
                  bad, pages, zero, static_cast<unsigned long long>(w[0]));
    return buf;
}

// One measured step: status (optional), time, counter deltas, verdict.
template <typename F>
void stepM(const char* name, F&& call)
{
    const Counters a = readCounters();
    auto t0 = std::chrono::steady_clock::now();
    const pxl::MemoryStatus s = call();
    const double ms = msSince(t0);
    const Counters b = readCounters();
    printf("  %-44s %-15s %9.1f ms |%s\n", name, st(s), ms, deltas(a, b).c_str());
}
void checkM(const char* name, const void* p, size_t bytes, uint64_t seed, size_t wordOffset = 0)
{
    pxl::flushHostCache(const_cast<void*>(p), bytes);  // drop CPU-cached copies first
    const Counters a = readCounters();
    auto t0 = std::chrono::steady_clock::now();
    const std::string v = verify(p, bytes, seed, wordOffset);
    const double ms = msSince(t0);
    const Counters b = readCounters();
    printf("  %-44s %-15s %9.1f ms (%.2f GB/s) |%s\n      -> %s\n", name, "host read", ms,
           bytes / 1e6 / ms, deltas(a, b).c_str(), v.c_str());
}

void* alloc1(size_t bytes, const char* what)
{
    void* p = pxl::allocateMemory(gDev, bytes);
    if (!p) printf("  allocateMemory(%s, %zu GB) FAILED\n", what, bytes / GB);
    return p;
}
void freshRegion(void* p, size_t bytes, uint64_t seed, const char* label)
{
    auto t0 = std::chrono::steady_clock::now();
    fill(p, bytes, seed);
    pxl::flushHostCache(p, bytes);
    printf("  [%s] filled %zu GB + flushHostCache: %.1f ms\n", label, bytes / GB, msSince(t0));
}
void touchFiller(void* f, size_t bytes, int round)
{
    if (!f) return;
    const Counters a = readCounters();
    auto t0 = std::chrono::steady_clock::now();
    memset(f, 0x33 + round, bytes);
    pxl::flushHostCache(f, bytes);
    const double ms = msSince(t0);
    printf("  filler %zu GB written: %.0f ms (%.2f GB/s) |%s\n", bytes / GB, ms, bytes / 1e6 / ms,
           deltas(a, readCounters()).c_str());
}

// ---- fp16 helpers for part C ----
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
float h2f(half_t h)
{
    const uint32_t sign = (h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FFu;
    uint32_t x = (e == 0) ? sign : (e == 31 ? (sign | 0x7F800000u | (m << 13)) : (sign | ((e + 112) << 23) | (m << 13)));
    float f; std::memcpy(&f, &x, 4);
    return f;
}
}  // namespace

int main(int argc, char* argv[])
{
    std::string parts = "ABC";
    size_t fillGB = 0;
    bool muCold = false;
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) gDev = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--part") && i + 1 < argc) parts = argv[++i];
        else if (!strcmp(argv[i], "--fill") && i + 1 < argc) fillGB = std::strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--mu-cold")) muCold = true;
    }
    const bool doA = parts.find('A') != std::string::npos;
    const bool doB = parts.find('B') != std::string::npos;
    const bool doC = parts.find('C') != std::string::npos;

    printf("im_probe  device=%d  parts=%s  fill=%zu GB  mu-cold=%d\n", gDev, parts.c_str(), fillGB, muCold);
    printf("\n== raw get-smart (once, to see every field) ==\n");
    {
        const std::string cmd = "xcena_cli im get-smart " + std::to_string(gDev) + " 2>&1";
        FILE* p = popen(cmd.c_str(), "r");
        char line[256];
        while (p && fgets(line, sizeof(line), p)) printf("  | %s", line);
        if (p) pclose(p);
    }

    auto* ctx = pxl::createContext(gDev);
    if (!ctx) { printf("createContext failed\n"); return 1; }

    // ================= A: primitives =================
    if (doA)
    {
        printf("\n== A1  what unmap does: commitSync -> unmap -> host read -> prefetchSync -> host read ==\n");
        if (void* r = alloc1(GB, "A1"))
        {
            freshRegion(r, GB, 101, "A1");
            checkM("A1.0 read right after write", r, GB, 101);
            stepM("A1.1 commitSyncMemory(1 GB)", [&] { return pxl::commitSyncMemory(r, GB); });
            checkM("A1.2 read after commitSync", r, GB, 101);
            stepM("A1.3 unmapMemory(1 GB)", [&] { return pxl::unmapMemory(r, GB); });
            checkM("A1.4 read after unmap", r, GB, 101);
            checkM("A1.5 read again", r, GB, 101);
            stepM("A1.6 unmapMemory again", [&] { return pxl::unmapMemory(r, GB); });
            stepM("A1.7 prefetchSyncMemory(1 GB)", [&] { return pxl::prefetchSyncMemory(r, GB); });
            checkM("A1.8 read after prefetchSync", r, GB, 101);
            pxl::releaseMemory(r);
        }

        printf("\n== A2  commitSync -> preload -> host read -> unload  (correct path, no unmap) ==\n");
        if (void* r = alloc1(GB, "A2"))
        {
            freshRegion(r, GB, 202, "A2");
            stepM("A2.1 commitSyncMemory(1 GB)", [&] { return pxl::commitSyncMemory(r, GB); });
            stepM("A2.3 preloadMemory(1 GB)", [&] { return pxl::preloadMemory(r, GB); });
            checkM("A2.4 read after preload", r, GB, 202);
            stepM("A2.5 unloadMemory(1 GB)", [&] { return pxl::unloadMemory(r, GB); });
            checkM("A2.6 read after unload", r, GB, 202);
            pxl::releaseMemory(r);
        }

        printf("\n== A3  unmap WITHOUT commit (is dirty data written back or lost?) ==\n");
        if (void* r = alloc1(GB, "A3"))
        {
            freshRegion(r, GB, 303, "A3");
            stepM("A3.1 unmapMemory(1 GB), no commit", [&] { return pxl::unmapMemory(r, GB); });
            checkM("A3.2 read after unmap", r, GB, 303);
            pxl::releaseMemory(r);
        }

        printf("\n== A4  flushMemory and granularity of prefetchSync ==\n");
        if (void* r = alloc1(GB, "A4"))
        {
            freshRegion(r, GB, 404, "A4");
            stepM("A4.1 flushMemory(1 GB)", [&] { return pxl::flushMemory(r, GB); });
            checkM("A4.2 read after flushMemory", r, GB, 404);
            stepM("A4.3 commitSyncMemory(1 GB)", [&] { return pxl::commitSyncMemory(r, GB); });
            auto* b = static_cast<uint8_t*>(r);
            stepM("A4.4 prefetchSync 4 KB (resident)", [&] { return pxl::prefetchSyncMemory(b, PAGE); });
            stepM("A4.4b pinMemory 64 MB", [&] { return pxl::pinMemory(b + (512u << 20), 64u << 20); });
            stepM("A4.4c unpinMemory 64 MB", [&] { return pxl::unpinMemory(b + (512u << 20), 64u << 20); });
            stepM("A4.5 prefetchSync 2 MB", [&] { return pxl::prefetchSyncMemory(b + 64 * PAGE, 2 << 20); });
            checkM("A4.6 read first 4 KB", b, PAGE, 404);
            checkM("A4.7 read 2 MB at +256 KB", b + 64 * PAGE, 2 << 20, 404, 64 * PAGE / 8);
            checkM("A4.8 read a 4 KB page never prefetched", b + 512 * PAGE * 2, PAGE, 404, 512 * PAGE * 2 / 8);
            pxl::releaseMemory(r);
        }
    }

    // ================= B: real eviction =================
    void* filler = nullptr;
    const size_t fillBytes = fillGB * GB;
    if ((doB || (doC && muCold)) && fillGB > 0)
    {
        filler = alloc1(fillBytes, "filler");
        if (!filler) fillGB = 0;
    }
    if (doB)
    {
        if (!filler) printf("\n== B skipped: pass --fill <GB larger than the DRAM cache> ==\n");
        else
        {
            const size_t colB = 3072 * sizeof(half_t);  // one V column, 6 KB
            printf("\n== B1  committed region, evicted by %zu GB of filler, read sequentially ==\n", fillGB);
            if (void* r = alloc1(GB, "B1"))
            {
                freshRegion(r, GB, 505, "B1");
                stepM("B1.1 commitSyncMemory(1 GB)", [&] { return pxl::commitSyncMemory(r, GB); });
                touchFiller(filler, fillBytes, 0);
                checkM("B1.2 host read of evicted 1 GB", r, GB, 505);
                checkM("B1.3 host read again (now cached?)", r, GB, 505);

                printf("\n== B2  64 scattered 6 KB columns from evicted data ==\n");
                touchFiller(filler, fillBytes, 1);
                std::vector<uint8_t> stage(64 * colB);
                const Counters a = readCounters();
                auto t0 = std::chrono::steady_clock::now();
                uint64_t x = 12345;
                std::vector<size_t> offs;
                for (int j = 0; j < 64; j++)
                {
                    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                    const size_t off = (x % (GB / colB)) * colB;
                    offs.push_back(off);
                    std::memcpy(stage.data() + j * colB, static_cast<uint8_t*>(r) + off, colB);
                }
                const double ms = msSince(t0);
                int ok = 0;
                for (int j = 0; j < 64; j++)
                {
                    bool good = true;
                    const auto* w = reinterpret_cast<const uint64_t*>(stage.data() + j * colB);
                    for (size_t q = 0; q < colB / 8; q++) good &= w[q] == pat(505, offs[j] / 8 + q);
                    ok += good;
                }
                printf("  B2.1 gather 64 x 6 KB (plain loads)          %9.2f ms |%s\n      -> %d/64 intact\n",
                       ms, deltas(a, readCounters()).c_str(), ok);

                printf("\n== B3  prefetchSync / preload of evicted data ==\n");
                touchFiller(filler, fillBytes, 2);
                stepM("B3.1 prefetchSyncMemory(1 GB) evicted", [&] { return pxl::prefetchSyncMemory(r, GB); });
                checkM("B3.2 read after prefetchSync", r, GB, 505);
                touchFiller(filler, fillBytes, 3);
                stepM("B3.3 preloadMemory(1 GB) evicted", [&] { return pxl::preloadMemory(r, GB); });
                checkM("B3.4 read after preload", r, GB, 505);
                stepM("B3.5 unloadMemory(1 GB)", [&] { return pxl::unloadMemory(r, GB); });
                pxl::releaseMemory(r);
            }
        }
    }

    // ================= C: MU access =================
    if (doC)
    {
        printf("\n== C  MU kernel dran_mix_values over V in different states ==\n");
        const int outDim = 3072, topK = 64, tasks = 64;
        const size_t colB = outDim * sizeof(half_t);
        const size_t numCols = GB / colB;
        auto* mod = pxl::createModule("mu_kernel/mu_kernel.mubin");
        auto* job = ctx->createJob();
        if (!mod || job->load(mod) != pxl::Result::Success) { printf("  module load failed\n"); return 1; }
        auto* fn = mod->createFunction("dran_mix_values");
        auto exec = job->buildMap(fn, tasks);

        // small arena: indices, weights, out — always preloaded
        auto* small = static_cast<uint8_t*>(alloc1(GB, "small"));
        if (!small) return 1;
        auto* idx = reinterpret_cast<uint32_t*>(small);
        auto* w = reinterpret_cast<float*>(small + 4096);
        auto* out = reinterpret_cast<float*>(small + 8192);
        printf("  small arena preload: %s\n", st(pxl::preloadMemory(small, GB)));

        auto* V = static_cast<half_t*>(alloc1(GB, "V"));
        if (!V) return 1;
        // V column c, row r = deterministic value in [-1, 1]
        auto val = [](size_t c, int r) { return static_cast<float>(pat(7, c * 3072 + r) % 2001) / 1000.0f - 1.0f; };
        auto writeV = [&] {
            for (size_t c = 0; c < numCols; c++)
                for (int r = 0; r < outDim; r++) V[c * outDim + r] = f2h(val(c, r));
            pxl::flushHostCache(V, GB);
        };
        uint64_t x = 999;
        for (int i = 0; i < topK; i++)
        {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            idx[i] = static_cast<uint32_t>(x % numCols);
            w[i] = 1.0f / topK;
        }
        pxl::flushHostCache(small, 16384);

        auto runMix = [&](const char* label) {
            for (int r = 0; r < outDim; r++) out[r] = -12345.0f;
            pxl::flushHostCache(out, outDim * sizeof(float));
            const Counters a = readCounters();
            auto t0 = std::chrono::steady_clock::now();
            const pxl::Result e = exec->execute(V, idx, w, outDim, topK, out);
            const pxl::Result s = e == pxl::Result::Success ? exec->synchronize() : e;
            const double ms = msSince(t0);
            const Counters b = readCounters();
            pxl::flushHostCache(out, outDim * sizeof(float));
            int good = 0, untouched = 0;
            for (int r = 0; r < outDim; r++)
            {
                float ref = 0.0f;
                for (int i = 0; i < topK; i++) ref += w[i] * h2f(f2h(val(idx[i], r)));
                good += std::fabs(out[r] - ref) < 1e-3f;
                untouched += out[r] == -12345.0f;
            }
            printf("  %-44s exec=%d sync=%d %8.2f ms |%s\n      -> out rows correct %d/%d (untouched %d)\n",
                   label, static_cast<int>(e), static_cast<int>(s), ms, deltas(a, b).c_str(), good, outDim, untouched);
        };

        writeV();
        printf("  V preload: %s\n", st(pxl::preloadMemory(V, GB)));
        runMix("C1 V preloaded");
        printf("  V unload: %s\n", st(pxl::unloadMemory(V, GB)));
        runMix("C2 V resident, NOT preloaded");
        if (muCold)
        {
            printf("  V commitSync: %s\n", st(pxl::commitSyncMemory(V, GB)));
            runMix("C3 V committed, not preloaded");
            if (filler)
            {
                writeV();
                printf("  V commitSync: %s\n", st(pxl::commitSyncMemory(V, GB)));
                touchFiller(filler, fillBytes, 9);
                runMix("C4 V evicted by filler");
            }
        }
        else printf("  (C3/C4 skipped: pass --mu-cold to let the MU touch committed/evicted memory)\n");

        (void)pxl::unloadMemory(small, GB);
        pxl::releaseMemory(small);
        pxl::releaseMemory(V);
        ctx->destroyJob(job);
    }

    // ================= D: getting unload's write-back off the critical path =================
    if (parts.find('D') != std::string::npos)
    {
        printf("\n== D1  releaseMemory on a preloaded region, without unload: write-back or discard? ==\n");
        if (void* r = alloc1(GB, "D1"))
        {
            freshRegion(r, GB, 606, "D1");
            stepM("D1.1 commitSyncMemory(1 GB)", [&] { return pxl::commitSyncMemory(r, GB); });
            stepM("D1.2 preloadMemory(1 GB)", [&] { return pxl::preloadMemory(r, GB); });
            stepM("D1.3 releaseMemory (no unload)", [&] { pxl::releaseMemory(r); return pxl::MemoryStatus::Success; });
        }

        printf("\n== D2  unload one region while preloading another: do they overlap? ==\n");
        void* x = alloc1(GB, "D2-x");
        void* y = alloc1(GB, "D2-y");
        void* x2 = alloc1(GB, "D2-x2");
        void* y2 = alloc1(GB, "D2-y2");
        if (x && y && x2 && y2)
        {
            freshRegion(x, GB, 701, "x");  freshRegion(y, GB, 702, "y");
            freshRegion(x2, GB, 703, "x2"); freshRegion(y2, GB, 704, "y2");
            for (void* p : {x, y, x2, y2}) (void)pxl::commitSyncMemory(p, GB);
            (void)pxl::preloadMemory(x, GB);
            (void)pxl::preloadMemory(x2, GB);
            // one after the other
            stepM("D2.1 unload x  (alone)", [&] { return pxl::unloadMemory(x, GB); });
            stepM("D2.2 preload y (alone)", [&] { return pxl::preloadMemory(y, GB); });
            // at the same time, from two threads
            const Counters a = readCounters();
            pxl::MemoryStatus su = pxl::MemoryStatus::Success, sp = pxl::MemoryStatus::Success;
            double mu = 0, mp = 0;
            auto t0 = std::chrono::steady_clock::now();
            std::thread tu([&] { auto t = std::chrono::steady_clock::now(); su = pxl::unloadMemory(x2, GB); mu = msSince(t); });
            std::thread tp([&] { auto t = std::chrono::steady_clock::now(); sp = pxl::preloadMemory(y2, GB); mp = msSince(t); });
            tu.join(); tp.join();
            const double both = msSince(t0);
            printf("  D2.3 unload x2 || preload y2               %s/%s  unload %.1f ms, preload %.1f ms, both done %.1f ms |%s\n",
                   st(su), st(sp), mu, mp, both, deltas(a, readCounters()).c_str());
            checkM("D2.4 y2 correct after concurrent preload", y2, GB, 704);
            (void)pxl::unloadMemory(y, GB);
            (void)pxl::unloadMemory(y2, GB);
        }
        for (void* p : {x, y, x2, y2}) if (p) pxl::releaseMemory(p);
    }

    if (filler) pxl::releaseMemory(filler);
    pxl::destroyContext(ctx);
    printf("\nim_probe done\n");
    return 0;
}
