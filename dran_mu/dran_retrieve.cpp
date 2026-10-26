// dran_retrieve.cpp — host driver for Case C: DRAN retrieval on XCENA MX1P.
//
// Flow per retrieval (r attractor iterations):
//   [device] dran_scan_topk      : stripe K across tasks, local top-k per task
//   [device] dran_merge_feedback : global top-k, softmax, xi <- normalize(K(:,topk) . w)
//   ... repeat ...
//   [device] dran_mix_values     : out = V(:,topk) . w
//
// Placement (--place) decides where K and V live between retrievals:
//   dram    K and V preloaded in device DRAM for the whole run (the original setup)
//   hybrid  K preloaded; V committed to the card's SSD and pushed out of the DRAM cache
//           (--evict filler, or a cache shrunk with `xcena_cli im set-dram-capacity`).
//           Per retrieval the host reads only the k winning V columns — through
//           InfiniteMemory's miss path — into a small preloaded staging buffer,
//           and the device mixes from there.
//   ssd     K and V both on the card's SSD and cold in DRAM at the start of every
//           retrieval. K is staged in with preloadMemory, then the same iterations
//           run from DRAM; V is handled as in hybrid.
//
// Why V goes through a staging buffer: in Computing mode the MU cores only see the
// Preload Region (the runtime refuses a Map over non-preloaded memory), and V is
// read so sparsely — k of M columns — that preloading all of it would move ~6 GB
// to use ~0.4 MB.
//
// Memory: preload works in 1 GB-aligned, 1 GB-multiple units, so K, V and
// the small buffers each get their own 1 GB-aligned allocation and can be placed
// independently.
//
// Each placement step prints its MemoryStatus, so a run shows which step, if any,
// the firmware refused. The probe check (the noised probe key must come back as the
// winner) is the end-to-end proof that data survived the trip to SSD and back.
//
// Build: see build.sh (needs XCENA SDK v1.4.9, PXL, MU LLVM toolchain).
// Run:   ./dran_retrieve [--place dram|hybrid|ssd] [-r reps] [-n M] [-d dim] [-o outDim]
//                        [-k topK] [-i iters] [-t tasks] [-s numSub] [-b batch] [--spread]
//                        [--device N]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "pxl/pxl.hpp"

using half_t = uint16_t;  // storage-only fp16 on the host side

namespace
{
constexpr size_t ALIGN_GB = 1ull << 30;
inline size_t alignUpGB(size_t x) { return (x + ALIGN_GB - 1) & ~(ALIGN_GB - 1); }

// Minimal IEEE-754 binary16 <-> float conversion (host side only).
half_t floatToHalf(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0)
    {
        return static_cast<half_t>(sign);
    }
    if (exp >= 31)
    {
        return static_cast<half_t>(sign | 0x7C00u);
    }
    return static_cast<half_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float halfToFloat(half_t h)
{
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t out;
    if (exp == 0)
    {
        out = sign;
    }
    else if (exp == 31)
    {
        out = sign | 0x7F800000u | (mant << 13);
    }
    else
    {
        out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

double msSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

const char* statusName(pxl::MemoryStatus s)
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

// Log one InfiniteMemory call. Returns true on Success.
bool step(const char* what, pxl::MemoryStatus s, double ms = -1.0)
{
    if (ms >= 0.0)
    {
        printf("  [step] %-34s %-15s %9.1f ms\n", what, statusName(s), ms);
    }
    else
    {
        printf("  [step] %-34s %s\n", what, statusName(s));
    }
    return s == pxl::MemoryStatus::Success;
}

// Print the InfiniteMemory counters that show how much moved between DRAM and SSD.
void imCounters(int deviceId, const char* tag)
{
    const std::string cmd = "xcena_cli im get-smart " + std::to_string(deviceId) + " 2>/dev/null";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p)
    {
        printf("  [im %s] could not run xcena_cli\n", tag);
        return;
    }
    char line[256];
    bool inStats = false;
    std::string section;
    while (fgets(line, sizeof(line), p))
    {
        if (strstr(line, "InfiniteMemory Statistics")) inStats = true;
        if (!inStats) continue;
        if (strstr(line, "Commands:")) section = "cmds ";
        if (strstr(line, "Bytes:")) section = "bytes";
        if (strstr(line, "preload") || strstr(line, "nvme_read") || strstr(line, "nvme_write") ||
            strstr(line, "prefetch") || strstr(line, "unmap") || strstr(line, "page_"))
        {
            line[strcspn(line, "\n")] = 0;
            const char* s = line;
            while (*s == ' ') s++;
            printf("  [im %s %s] %s\n", tag, section.c_str(), s);
        }
    }
    pclose(p);
}

// Device-wide InfiniteMemory "nvme_read" byte counter, or -1 if unreadable. The delta
// across a retrieval is the proof that data really came from the card's SSD.
long long nvmeReadBytes(int deviceId)
{
    const std::string cmd = "xcena_cli im get-smart " + std::to_string(deviceId) + " 2>/dev/null";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return -1;
    char line[256];
    bool inBytes = false;
    long long v = -1;
    while (fgets(line, sizeof(line), p))
    {
        if (strstr(line, "Bytes:")) inBytes = true;
        const char* c = strchr(line, ':');
        if (inBytes && strstr(line, "nvme_read") && c) v = std::atoll(c + 1);
    }
    pclose(p);
    return v;
}

double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
}  // namespace

int main(int argc, char* argv[])
{
    size_t numCols = 1000000;  // M — memories stored
    int dim = 1024;            // n — key dimension
    int outDim = 3072;         // o — value dimension
    int topK = 64;             // k
    int iters = 3;             // attractor iterations
    int taskCount = 2816;      // MU tasks for the scan (one per MU core)
    // These three default to "do not call" so the baseline is exactly the
    // configuration that was known to produce a correct HIT. Turn them on one at a
    // time to attribute any change in speed or correctness.
    int numSub = 0;            // 0 = createJob() with no argument
    int batchSize = 0;         // 0 = leave the Map default (16)
    int spread = 0;            // 1 = setLocalityMode(SpreadMode)
    int deviceId = 0;
    int reps = 3;
    std::string place = "dram";
    int prefetch = 0;          // 1 = prefetchSync each winning V column before reading it, 2 = async prefetch all 64 first
    size_t evictGB = 0;        // >0 = touch this many GB of filler before every rep to push K/V out of the DRAM cache

    for (int i = 1; i < argc; i++)
    {
        auto next = [&](int& idx) { return std::stoll(argv[++idx]); };
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numCols = static_cast<size_t>(next(i));
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) dim = static_cast<int>(next(i));
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) outDim = static_cast<int>(next(i));
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) topK = static_cast<int>(next(i));
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iters = static_cast<int>(next(i));
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) taskCount = static_cast<int>(next(i));
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) numSub = static_cast<int>(next(i));
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) batchSize = static_cast<int>(next(i));
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) reps = static_cast<int>(next(i));
        else if (!strcmp(argv[i], "--spread")) spread = 1;
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) deviceId = static_cast<int>(next(i));
        else if (!strcmp(argv[i], "--place") && i + 1 < argc) place = argv[++i];
        else if (!strcmp(argv[i], "--prefetch")) prefetch = 1;
        else if (!strcmp(argv[i], "--prefetch-async")) prefetch = 2;
        else if (!strcmp(argv[i], "--evict") && i + 1 < argc) evictGB = static_cast<size_t>(next(i));
    }
    if (place != "dram" && place != "hybrid" && place != "ssd")
    {
        printf("--place must be dram, hybrid or ssd\n");
        return 1;
    }
    const bool vOnSsd = (place != "dram");
    const bool kOnSsd = (place == "ssd");

    const size_t keyBytes = static_cast<size_t>(dim) * numCols * sizeof(half_t);
    const size_t valBytes = static_cast<size_t>(outDim) * numCols * sizeof(half_t);
    const size_t colBytesV = static_cast<size_t>(outDim) * sizeof(half_t);
    printf("DRAN retrieval on device %d   place=%s   reps=%d\n", deviceId, place.c_str(), reps);
    printf("  M=%zu  n=%d  o=%d  k=%d  iters=%d\n", numCols, dim, outDim, topK, iters);
    printf("  tasks=%d  numSub=%s  batchSize=%s  locality=%s\n", taskCount,
           numSub ? std::to_string(numSub).c_str() : "default",
           batchSize ? std::to_string(batchSize).c_str() : "default(16)",
           spread ? "Spread" : "default(Compact)");
    printf("  K = %.3f GB (%s)   V = %.3f GB (%s)\n", keyBytes / 1e9, kOnSsd ? "SSD" : "DRAM",
           valBytes / 1e9, vOnSsd ? "SSD" : "DRAM");

    auto context = pxl::createContext(deviceId);
    if (!context)
    {
        printf("createContext failed\n");
        return 1;
    }

    // ---- three 1 GB-aligned allocations so K and V can be placed independently ----
    const size_t kAlloc = alignUpGB(keyBytes);
    const size_t vAlloc = alignUpGB(valBytes);

    size_t off = 0;
    auto reserve = [&](size_t bytes, size_t align = 4096) {
        off = (off + align - 1) & ~(align - 1);
        const size_t here = off;
        off += bytes;
        return here;
    };
    const size_t offQ   = reserve(static_cast<size_t>(dim) * sizeof(half_t));
    const size_t offPS  = reserve(static_cast<size_t>(taskCount) * topK * sizeof(float));
    const size_t offPI  = reserve(static_cast<size_t>(taskCount) * topK * sizeof(uint32_t));
    // winners: k uint32 indices immediately followed by k float weights, written
    // by dran_merge_feedback and consumed by dran_mix_values without a host round trip
    const size_t offWin = reserve(static_cast<size_t>(topK) * 2 * sizeof(uint32_t));
    const size_t offOut = reserve(static_cast<size_t>(outDim) * sizeof(float));
    const size_t offVS  = reserve(static_cast<size_t>(topK) * colBytesV);         // staged V columns
    const size_t offSI  = reserve(static_cast<size_t>(topK) * sizeof(uint32_t));  // 0..k-1
    const size_t sAlloc = alignUpGB(off);
    printf("  allocations: K %zu GB + V %zu GB + small %zu GB\n",
           kAlloc / ALIGN_GB, vAlloc / ALIGN_GB, sAlloc / ALIGN_GB);

    auto* kArena = reinterpret_cast<uint8_t*>(pxl::allocateMemory(deviceId, kAlloc));
    auto* vArena = reinterpret_cast<uint8_t*>(pxl::allocateMemory(deviceId, vAlloc));
    auto* sArena = reinterpret_cast<uint8_t*>(pxl::allocateMemory(deviceId, sAlloc));
    if (!kArena || !vArena || !sArena)
    {
        printf("allocateMemory failed (K %p, V %p, small %p)\n", kArena, vArena, sArena);
        return 1;
    }
    auto* keys        = reinterpret_cast<half_t*>(kArena);
    auto* values      = reinterpret_cast<half_t*>(vArena);
    auto* query       = reinterpret_cast<half_t*>(sArena + offQ);
    auto* partScores  = reinterpret_cast<float*>(sArena + offPS);
    auto* partIndices = reinterpret_cast<uint32_t*>(sArena + offPI);
    auto* winnerIdx   = reinterpret_cast<uint32_t*>(sArena + offWin);
    auto* winnerW     = reinterpret_cast<float*>(sArena + offWin + topK * sizeof(uint32_t));
    auto* outVec      = reinterpret_cast<float*>(sArena + offOut);
    auto* vStage      = reinterpret_cast<half_t*>(sArena + offVS);
    auto* stageIdx    = reinterpret_cast<uint32_t*>(sArena + offSI);

    // ---- fill K and V with unit-norm-ish synthetic data ----
    printf("filling K/V ...\n");
    auto tFill = std::chrono::steady_clock::now();
    uint64_t seed = 88172645463325252ull;
    auto rnd = [&]() {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        return static_cast<float>(static_cast<double>(seed >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
    };
    std::vector<float> col(std::max(dim, outDim));
    for (size_t c = 0; c < numCols; c++)
    {
        float norm = 0.0f;
        for (int r = 0; r < dim; r++) { col[r] = rnd(); norm += col[r] * col[r]; }
        norm = std::sqrt(norm) + 1e-9f;
        half_t* kcol = keys + c * static_cast<size_t>(dim);
        for (int r = 0; r < dim; r++) kcol[r] = floatToHalf(col[r] / norm);
        half_t* vcol = values + c * static_cast<size_t>(outDim);
        for (int r = 0; r < outDim; r++) vcol[r] = floatToHalf(rnd());
    }
    for (int i = 0; i < topK; i++) stageIdx[i] = static_cast<uint32_t>(i);
    printf("  fill: %.1f ms\n", msSince(tFill));

    // ---- everything that reads K on the host happens now, while K is still hot ----
    const size_t probe = numCols / 3;
    std::vector<float> xi(dim);
    {
        const half_t* kcol = keys + probe * static_cast<size_t>(dim);
        float norm = 0.0f;
        for (int r = 0; r < dim; r++) { xi[r] = halfToFloat(kcol[r]) + 0.05f * rnd(); norm += xi[r] * xi[r]; }
        norm = std::sqrt(norm) + 1e-9f;
        for (int r = 0; r < dim; r++) xi[r] /= norm;
    }
    // Copies of one K column and one V column, compared again after the SSD round trip.
    const size_t checkV = numCols / 2;
    std::vector<half_t> probeK(keys + probe * dim, keys + (probe + 1) * dim);
    std::vector<half_t> probeV(values + checkV * outDim, values + (checkV + 1) * outDim);
    // Full host copy of V (6 GB at M = 1e6) so every staged winner column can be checked;
    // the probe HIT only proves K, not the V bytes that reach the mix.
    std::vector<half_t> hostV;
    if (vOnSsd) hostV.assign(values, values + static_cast<size_t>(outDim) * numCols);

    // Publish the host's writes to device memory. Without this the MU cores read
    // whatever was in device DRAM before, not what we just wrote.
    auto tFlush = std::chrono::steady_clock::now();
    pxl::flushHostCache(kArena, kAlloc);
    pxl::flushHostCache(vArena, vAlloc);
    pxl::flushHostCache(sArena, sAlloc);
    printf("  flushHostCache(%.2f GB): %.1f ms\n", (kAlloc + vAlloc + sAlloc) / 1e9, msSince(tFlush));

    // ---- placement ----
    printf("\nplacement: %s\n", place.c_str());
    // A card without InfiniteMemory (no SSD tier) has no Preload Region: all of its DRAM
    // is directly addressable by the MU cores, so preload is NotSupported and unneeded.
    const pxl::MemoryStatus sPre = pxl::preloadMemory(sArena, sAlloc);
    const bool plainDram = sPre == pxl::MemoryStatus::NotSupported;
    if (plainDram)
    {
        printf("  [step] preload small buffers              NotSupported -> no InfiniteMemory,"
               " DRAM is directly mapped, preload skipped\n");
        if (place != "dram")
        {
            printf("place=%s needs the SSD tier (InfiniteMemory); this device has none\n", place.c_str());
            return 1;
        }
    }
    else if (!step("preload small buffers", sPre)) return 1;

    auto timed = [&](const char* what, auto&& call) {
        auto t0 = std::chrono::steady_clock::now();
        const pxl::MemoryStatus s = call();
        return step(what, s, msSince(t0));
    };
    // Write a region back to SSD so it is clean. Never unmap: unmapMemory clears the
    // region's mapping (the data is gone), it is not an eviction. Coldness comes from
    // --evict (filler beyond the Cache Region) or `xcena_cli im set-dram-capacity`.
    auto toSsd = [&](const char* name, void* p, size_t bytes) {
        const std::string c = std::string("commitSync ") + name;
        return timed(c.c_str(), [&] { return pxl::commitSyncMemory(p, bytes); });
    };

    if (!plainDram && !kOnSsd && !timed("preload K", [&] { return pxl::preloadMemory(kArena, kAlloc); })) return 1;
    if (!plainDram && !vOnSsd && !timed("preload V", [&] { return pxl::preloadMemory(vArena, vAlloc); })) return 1;
    if (vOnSsd && evictGB == 0)
        printf("  note: no --evict given, so committed data may still sit in the DRAM cache\n");
    if (kOnSsd && !toSsd("K", kArena, kAlloc)) return 1;
    if (vOnSsd && !toSsd("V", vArena, vAlloc)) return 1;

    // Did the data survive? Read one column of each back through the host path.
    if (kOnSsd || vOnSsd)
    {
        if (kOnSsd)
        {
            pxl::flushHostCache(keys + probe * dim, dim * sizeof(half_t));
            const bool ok = !std::memcmp(probeK.data(), keys + probe * dim, dim * sizeof(half_t));
            printf("  [step] K column %zu after SSD round trip    %s\n", probe, ok ? "intact" : "CHANGED");
        }
        pxl::flushHostCache(values + checkV * outDim, colBytesV);
        const bool ok = !std::memcmp(probeV.data(), values + checkV * outDim, colBytesV);
        printf("  [step] V column %zu after SSD round trip    %s\n", checkV, ok ? "intact" : "CHANGED");
    }

    // ---- optional filler that pushes K/V out of the DRAM cache before every rep ----
    // Touching more than the Cache Region holds forces K/V out under the normal policy, and the nvme_read delta per rep shows whether it worked.
    uint8_t* evictArena = nullptr;
    const size_t evictBytes = evictGB * ALIGN_GB;
    if (evictGB > 0)
    {
        evictArena = reinterpret_cast<uint8_t*>(pxl::allocateMemory(deviceId, evictBytes));
        if (!evictArena)
        {
            printf("allocateMemory for %zu GB filler failed\n", evictGB);
            return 1;
        }
        printf("  filler: %zu GB allocated, touched before every rep (not timed)\n", evictGB);
    }
    auto touchFiller = [&](int rep) {
        if (!evictArena) return;
        auto te = std::chrono::steady_clock::now();
        std::memset(evictArena, 0x5A ^ rep, evictBytes);
        pxl::flushHostCache(evictArena, evictBytes);
        printf("  filler touched: %.1f ms\n", msSince(te));
    };

    // ---- load the MU module and build the three executors ----
    auto module = pxl::createModule("mu_kernel/mu_kernel.mubin");
    auto job = numSub > 0 ? context->createJob(numSub) : context->createJob();
    if (job->load(module) != pxl::Result::Success)
    {
        printf("job load failed\n");
        return 1;
    }
    auto* scanFunc = module->createFunction("dran_scan_topk");
    auto* mfFunc = module->createFunction("dran_merge_feedback");
    auto* mixFunc = module->createFunction("dran_mix_values");
    auto scanExec = job->buildMap(scanFunc, taskCount);
    auto mfExec = job->buildMap(mfFunc, 1);   // single task: it is a reduction
    auto mixExec = job->buildMap(mixFunc, taskCount);
    if (batchSize > 0)
    {
        scanExec->setBatchSize(batchSize);
        mixExec->setBatchSize(batchSize);
    }
    if (spread)
    {
        scanExec->setLocalityMode(pxl::LocalityMode::SpreadMode);
        mixExec->setLocalityMode(pxl::LocalityMode::SpreadMode);
    }

    const int numCand = taskCount * topK;
    std::vector<double> tFetchK, tScan, tMf, tGatherV, tMix, tTotal;
    int hits = 0;

    imCounters(deviceId, "before");

    for (int rep = 0; rep < reps; rep++)
    {
        printf("\nretrieval %d/%d\n", rep + 1, reps);
        // every rep starts from the same query; feedback overwrites it in place
        for (int r = 0; r < dim; r++) query[r] = floatToHalf(xi[r]);
        pxl::flushHostCache(query, static_cast<size_t>(dim) * sizeof(half_t));
        touchFiller(rep);
        const long long nvmeBefore = nvmeReadBytes(deviceId);

        auto t0 = std::chrono::steady_clock::now();

        // --- K: stage in from SSD (ssd only) ---
        double fetchK = 0.0;
        if (kOnSsd)
        {
            auto tk = std::chrono::steady_clock::now();
            if (!step("preload K from SSD", pxl::preloadMemory(kArena, kAlloc))) return 1;
            fetchK = msSince(tk);
            printf("  fetch K: %.2f ms  (%.2f GB/s)\n", fetchK, keyBytes / 1e6 / fetchK);
        }

        // --- attractor iterations, all on the device ---
        double scanMs = 0.0, mfMs = 0.0;
        for (int it = 0; it < iters; it++)
        {
            auto ts = std::chrono::steady_clock::now();
            if (scanExec->execute(keys, query, dim, static_cast<uint64_t>(numCols), topK, partScores, partIndices)
                    != pxl::Result::Success ||
                scanExec->synchronize() != pxl::Result::Success)
            {
                printf("scan failed\n");
                return 1;
            }
            scanMs += msSince(ts);
            auto tm = std::chrono::steady_clock::now();
            if (mfExec->execute(partScores, partIndices, numCand, topK, keys, dim, topK, winnerIdx, query)
                    != pxl::Result::Success ||
                mfExec->synchronize() != pxl::Result::Success)
            {
                printf("merge_feedback failed\n");
                return 1;
            }
            mfMs += msSince(tm);
        }

        // --- V: mix from DRAM directly, or gather the k winners from SSD first ---
        double gatherMs = 0.0;
        const half_t* mixSrc = values;
        const uint32_t* mixIdx = winnerIdx;
        if (vOnSsd)
        {
            auto tg = std::chrono::steady_clock::now();
            pxl::flushHostCache(winnerIdx, static_cast<size_t>(topK) * sizeof(uint32_t));
            if (prefetch == 2)
            {
                // issue all 64 page fetches before touching any, so the SSD works on them in parallel
                for (int j = 0; j < topK; j++)
                {
                    const uintptr_t a = reinterpret_cast<uintptr_t>(values + static_cast<size_t>(winnerIdx[j]) * outDim);
                    const uintptr_t lo = a & ~static_cast<uintptr_t>(4095);
                    const uintptr_t hi = (a + colBytesV + 4095) & ~static_cast<uintptr_t>(4095);
                    const pxl::MemoryStatus ps = pxl::prefetchMemory(reinterpret_cast<void*>(lo), hi - lo);
                    if (ps != pxl::MemoryStatus::Success && j == 0) printf("  prefetch V col: %s\n", statusName(ps));
                }
            }
            for (int j = 0; j < topK; j++)
            {
                const size_t c = winnerIdx[j];
                if (prefetch == 1)
                {
                    // page-aligned range covering the column
                    const uintptr_t a = reinterpret_cast<uintptr_t>(values + c * outDim);
                    const uintptr_t lo = a & ~static_cast<uintptr_t>(4095);
                    const uintptr_t hi = (a + colBytesV + 4095) & ~static_cast<uintptr_t>(4095);
                    const pxl::MemoryStatus ps = pxl::prefetchSyncMemory(reinterpret_cast<void*>(lo), hi - lo);
                    if (ps != pxl::MemoryStatus::Success && j == 0) printf("  prefetchSync V col: %s\n", statusName(ps));
                }
                std::memcpy(vStage + static_cast<size_t>(j) * outDim, values + c * outDim, colBytesV);
            }
            pxl::flushHostCache(vStage, static_cast<size_t>(topK) * colBytesV);
            gatherMs = msSince(tg);
            mixSrc = vStage;
            mixIdx = stageIdx;
        }
        auto tx = std::chrono::steady_clock::now();
        if (mixExec->execute(mixSrc, mixIdx, winnerW, outDim, topK, outVec) != pxl::Result::Success ||
            mixExec->synchronize() != pxl::Result::Success)
        {
            printf("mix failed\n");
            return 1;
        }
        const double mixMs = msSince(tx);
        const double total = msSince(t0);
        const long long nvmeAfter = nvmeReadBytes(deviceId);
        if (nvmeBefore >= 0 && nvmeAfter >= 0)
            printf("  card SSD read during this retrieval: %.3f MB\n", (nvmeAfter - nvmeBefore) / 1e6);

        // Were the V bytes that reached the mix the ones we wrote?
        if (vOnSsd)
        {
            int intact = 0;
            for (int j = 0; j < topK; j++)
            {
                const size_t c = winnerIdx[j];
                intact += !std::memcmp(vStage + static_cast<size_t>(j) * outDim, hostV.data() + c * outDim, colBytesV);
            }
            printf("  staged V columns intact: %d/%d\n", intact, topK);
        }

        pxl::flushHostCache(winnerIdx, static_cast<size_t>(topK) * 2 * sizeof(uint32_t));
        pxl::flushHostCache(outVec, static_cast<size_t>(outDim) * sizeof(float));
        const bool hit = (winnerIdx[0] == probe);
        hits += hit;
        printf("  scan %.2f  merge %.2f  gatherV %.2f  mix %.2f  total %.2f ms   probe %s (w %.3f)\n",
               scanMs, mfMs, gatherMs, mixMs, total, hit ? "HIT" : "MISS", winnerW[0]);

        tFetchK.push_back(fetchK); tScan.push_back(scanMs); tMf.push_back(mfMs);
        tGatherV.push_back(gatherMs); tMix.push_back(mixMs); tTotal.push_back(total);

        // --- send K and V back to cold before the next rep ---
        if (rep + 1 < reps)
        {
            if (kOnSsd)
            {
                if (!step("unload K", pxl::unloadMemory(kArena, kAlloc))) return 1;
            }
        }
    }

    imCounters(deviceId, "after ");

    printf("\n--- timing, median of %d (place=%s) ---\n", reps, place.c_str());
    printf("  fetch K from card SSD          : %8.2f ms\n", median(tFetchK));
    printf("  scan (%d iters)                 : %8.2f ms\n", iters, median(tScan));
    printf("  merge+softmax+feedback         : %8.2f ms\n", median(tMf));
    printf("  gather %d V cols from card SSD : %8.2f ms\n", topK, median(tGatherV));
    printf("  value mix                      : %8.2f ms\n", median(tMix));
    printf("  total retrieval                : %8.2f ms\n", median(tTotal));
    printf("\n--- correctness ---\n");
    printf("  probe column #%zu -> HIT in %d/%d retrievals\n", probe, hits, reps);
    printf("\n--- link traffic per retrieval ---\n");
    printf("  in  : query %zu B%s\n", static_cast<size_t>(dim) * sizeof(half_t),
           vOnSsd ? "  +  staged V columns (host -> device)" : "");
    printf("  out : winners %zu B  +  result %zu B\n",
           static_cast<size_t>(topK) * 2 * sizeof(uint32_t), static_cast<size_t>(outDim) * sizeof(float));

    // K is preloaded at this point in every mode (ssd staged it in on the last rep)
    if (!plainDram)
    {
        step("unload K", pxl::unloadMemory(kArena, kAlloc));
        if (!vOnSsd) step("unload V", pxl::unloadMemory(vArena, vAlloc));
        step("unload small buffers", pxl::unloadMemory(sArena, sAlloc));
    }
    pxl::releaseMemory(kArena);
    pxl::releaseMemory(vArena);
    pxl::releaseMemory(sArena);
    if (evictArena) pxl::releaseMemory(evictArena);
    context->destroyJob(job);
    pxl::destroyContext(context);
    return 0;
}
