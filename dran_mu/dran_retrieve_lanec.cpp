// dran_retrieve.cpp — host driver for Case C: DRAN retrieval on XCENA MX1P.
//
// Flow per retrieval (r attractor iterations):
//   [device] dran_scan_topk : stripe K across tasks, local top-k per task
//   [host]   merge the per-task partials into a global top-k, softmax the scores
//   [host]   feedback  xi <- normalize(K(:,topk) . w)   (k columns only)
//   ... repeat ...
//   [device] dran_mix_values : out = V(:,topk) . w
//
// Memory: everything the MU cores touch must live in the Preload Region, and
// preloadMemory works in 1 GB-aligned, 1 GB-multiple units. So this allocates ONE
// arena, preloads it once, and carves K, V and every small buffer out of it — a
// per-buffer preload would burn a whole 1 GB slot on a 2 KB query vector.
//
// Build: see build.sh (needs XCENA SDK v1.4.9, PXL, MU LLVM toolchain).
// Run:   ./dran_retrieve [-n M] [-d dim] [-o outDim] [-k topK] [-i iters] [-t tasks] [--device N]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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
        else if (!strcmp(argv[i], "--spread")) spread = 1;
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) deviceId = static_cast<int>(next(i));
    }

    const size_t keyBytes = static_cast<size_t>(dim) * numCols * sizeof(half_t);
    const size_t valBytes = static_cast<size_t>(outDim) * numCols * sizeof(half_t);
    printf("DRAN retrieval on device %d\n", deviceId);
    printf("  M=%zu  n=%d  o=%d  k=%d  iters=%d\n", numCols, dim, outDim, topK, iters);
    printf("  tasks=%d  numSub=%s  batchSize=%s  locality=%s\n", taskCount,
           numSub ? std::to_string(numSub).c_str() : "default",
           batchSize ? std::to_string(batchSize).c_str() : "default(16)",
           spread ? "Spread" : "default(Compact)");
    printf("  K = %.3f GB   V = %.3f GB\n", keyBytes / 1e9, valBytes / 1e9);

    auto context = pxl::createContext(deviceId);
    if (!context)
    {
        printf("createContext failed\n");
        return 1;
    }

    // ---- one arena: everything the MU cores touch lives here ----
    size_t off = 0;
    auto reserve = [&](size_t bytes, size_t align = 4096) {
        off = (off + align - 1) & ~(align - 1);
        const size_t here = off;
        off += bytes;
        return here;
    };
    const size_t offK   = reserve(keyBytes);
    const size_t offV   = reserve(valBytes);
    const size_t offQ   = reserve(static_cast<size_t>(dim) * sizeof(half_t));
    const size_t offPS  = reserve(static_cast<size_t>(taskCount) * topK * sizeof(float));
    const size_t offPI  = reserve(static_cast<size_t>(taskCount) * topK * sizeof(uint32_t));
    // winners: k uint32 indices immediately followed by k float weights, written
    // by dran_merge_feedback and consumed by dran_mix_values without a host round trip
    const size_t offWin = reserve(static_cast<size_t>(topK) * 2 * sizeof(uint32_t));
    const size_t offOut = reserve(static_cast<size_t>(outDim) * sizeof(float));
    const size_t arenaBytes = alignUpGB(off);
    printf("  arena = %.3f GB used -> %.0f GB preloaded (%zu x 1 GB slots)\n",
           off / 1e9, arenaBytes / 1e9, arenaBytes / ALIGN_GB);

    auto* arena = reinterpret_cast<uint8_t*>(pxl::allocateMemory(deviceId, arenaBytes));
    if (!arena)
    {
        printf("allocateMemory(%.0f GB) failed\n", arenaBytes / 1e9);
        return 1;
    }
    auto* keys        = reinterpret_cast<half_t*>(arena + offK);
    auto* values      = reinterpret_cast<half_t*>(arena + offV);
    auto* query       = reinterpret_cast<half_t*>(arena + offQ);
    auto* partScores  = reinterpret_cast<float*>(arena + offPS);
    auto* partIndices = reinterpret_cast<uint32_t*>(arena + offPI);
    auto* winnerIdx   = reinterpret_cast<uint32_t*>(arena + offWin);
    auto* winnerW     = reinterpret_cast<float*>(arena + offWin + topK * sizeof(uint32_t));
    auto* outVec      = reinterpret_cast<float*>(arena + offOut);

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
    printf("  fill: %.1f ms\n", msSince(tFill));

    // Publish the host's writes to device memory. Without this the MU cores read
    // whatever was in device DRAM before, not what we just wrote: the host's own
    // reads hit its cache and look correct, so the mismatch only shows up on the
    // device side. (InfiniteMemory doc: flush before kernel execute.)
    auto tFlush = std::chrono::steady_clock::now();
    pxl::flushHostCache(arena, arenaBytes);
    printf("  flushHostCache(%.2f GB): %.1f ms\n", arenaBytes / 1e9, msSince(tFlush));

    // ---- pin the whole arena into the Preload Region (Computing mode) ----
    // A card without InfiniteMemory has no Preload Region: its DRAM is directly mapped,
    // preload returns NotSupported, and the MU cores can use the arena as is.
    auto tPre = std::chrono::steady_clock::now();
    const pxl::MemoryStatus sPre = pxl::preloadMemory(arena, arenaBytes);
    const bool plainDram = sPre == pxl::MemoryStatus::NotSupported;
    if (plainDram)
    {
        printf("preload: NotSupported -> no InfiniteMemory, DRAM directly mapped, skipped\n");
    }
    else if (sPre != pxl::MemoryStatus::Success)
    {
        printf("preloadMemory failed (status %d) - device in Computing mode? enough free 1 GB slots?\n",
               static_cast<int>(sPre));
        pxl::releaseMemory(arena);
        return 1;
    }
    else printf("preload arena: %.1f ms\n", msSince(tPre));

    // ---- load the MU module and build the two executors ----
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

    // ---- initial query: a noised copy of one stored key ----
    const size_t probe = numCols / 3;
    std::vector<float> xi(dim);
    {
        const half_t* kcol = keys + probe * static_cast<size_t>(dim);
        float norm = 0.0f;
        for (int r = 0; r < dim; r++) { xi[r] = halfToFloat(kcol[r]) + 0.05f * rnd(); norm += xi[r] * xi[r]; }
        norm = std::sqrt(norm) + 1e-9f;
        for (int r = 0; r < dim; r++) xi[r] /= norm;
    }

    const int numCand = taskCount * topK;
    double scanMs = 0.0, mfMs = 0.0;

    // The only host write into device memory: the initial query. Everything after
    // this stays on the device until the final result is read back.
    for (int r = 0; r < dim; r++) query[r] = floatToHalf(xi[r]);
    pxl::flushHostCache(query, static_cast<size_t>(dim) * sizeof(half_t));

    {   // the same three things the kernel reports for task 0, computed here in fp32
        // from the fp16 the device sees, so the comparison is apples to apples
        const size_t perTask = (numCols + taskCount - 1) / taskCount;
        std::vector<float> qf(dim);
        for (int r = 0; r < dim; r++) qf[r] = halfToFloat(query[r]);

        auto dotCol = [&](size_t c) {
            const half_t* kc = keys + c * static_cast<size_t>(dim);
            float d = 0.0f;
            for (int r = 0; r < dim; r++) d += halfToFloat(kc[r]) * qf[r];
            return d;
        };
        const float d0 = dotCol(0);
        float best = -3.0e38f;
        size_t bestCol = 0;
        for (size_t c = 0; c < perTask && c < numCols; c++)
        {
            const float d = dotCol(c);
            if (d > best) { best = d; bestCol = c; }
        }
        const float probeScore = dotCol(probe);
        uint32_t b0, bb;
        std::memcpy(&b0, &d0, 4);
        std::memcpy(&bb, &best, 4);
        printf("[host]   perTask=%zu range=[0,%zu)\n", perTask, perTask);
        printf("[host]   dot(col 0) = 0x%08x (%.6f)\n", b0, d0);
        printf("[host]   task0 best = col %zu  score 0x%08x (%.6f)\n", bestCol, bb, best);
        printf("[host]   dot(probe col %zu) = %.6f   <- this should win globally\n",
               probe, probeScore);
    }

    printf("\nretrieval: %d attractor iterations (merge/softmax/feedback on device)\n", iters);
    auto tTotal = std::chrono::steady_clock::now();

    for (int it = 0; it < iters; it++)
    {
        // --- device: scan all of K, per-task local top-k ---
        auto tScan = std::chrono::steady_clock::now();
        if (scanExec->execute(keys, query, dim, static_cast<uint64_t>(numCols), topK, partScores, partIndices)
            != pxl::Result::Success)
        {
            printf("scan execute failed\n");
            return 1;
        }
        if (scanExec->synchronize() != pxl::Result::Success)
        {
            printf("scan synchronize failed\n");
            return 1;
        }
        const double thisScan = msSince(tScan);
        scanMs += thisScan;

        // --- device: merge partials, softmax, write the next query in place ---
        auto tMf = std::chrono::steady_clock::now();
        if (mfExec->execute(partScores, partIndices, numCand, topK, keys, dim, topK, winnerIdx, query)
            != pxl::Result::Success)
        {
            printf("merge_feedback execute failed\n");
            return 1;
        }
        if (mfExec->synchronize() != pxl::Result::Success)
        {
            printf("merge_feedback synchronize failed\n");
            return 1;
        }
        const double thisMf = msSince(tMf);
        mfMs += thisMf;

        printf("  iter %d: scan %.2f ms   merge+feedback %.2f ms\n", it, thisScan, thisMf);
    }

    // ---- device: mix the winning value columns (winners already in device memory) ----
    auto tMix = std::chrono::steady_clock::now();
    if (mixExec->execute(values, winnerIdx, winnerW, outDim, topK, outVec) != pxl::Result::Success)
    {
        printf("mix execute failed\n");
        return 1;
    }
    if (mixExec->synchronize() != pxl::Result::Success)
    {
        printf("mix synchronize failed\n");
        return 1;
    }
    const double mixMs = msSince(tMix);
    const double totalMs = msSince(tTotal);

    // Read back only the result and the winner list.
    pxl::flushHostCache(outVec, static_cast<size_t>(outDim) * sizeof(float));
    pxl::flushHostCache(winnerIdx, static_cast<size_t>(topK) * 2 * sizeof(uint32_t));

    printf("\n--- timing ---\n");
    printf("  scan (%d iters, %.3f GB each)  : %8.2f ms   (%.2f ms/iter)\n",
           iters, keyBytes / 1e9, scanMs, scanMs / iters);
    printf("  merge+softmax+feedback (device): %8.2f ms   (%.2f ms/iter)\n", mfMs, mfMs / iters);
    printf("  value mix (device)             : %8.2f ms\n", mixMs);
    printf("  total retrieval                : %8.2f ms\n", totalMs);
    printf("\n--- correctness ---\n");
    printf("  probe column #%zu -> winner #%u  (%s)  weight %.3f\n",
           probe, winnerIdx[0], (winnerIdx[0] == probe ? "HIT" : "miss"), winnerW[0]);
    printf("\n--- link traffic per retrieval ---\n");
    printf("  in  : query      %6zu B\n", static_cast<size_t>(dim) * sizeof(half_t));
    printf("  out : winners    %6zu B  +  result %6zu B\n",
           static_cast<size_t>(topK) * 2 * sizeof(uint32_t),
           static_cast<size_t>(outDim) * sizeof(float));
    printf("  scanned inside the pool       : %.3f GB\n", keyBytes * iters / 1e9);

    if (!plainDram && pxl::unloadMemory(arena, arenaBytes) != pxl::MemoryStatus::Success)
    {
        printf("warning: unloadMemory failed\n");
    }
    pxl::releaseMemory(arena);
    context->destroyJob(job);
    pxl::destroyContext(context);
    return 0;
}
