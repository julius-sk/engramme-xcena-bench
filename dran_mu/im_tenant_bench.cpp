// im_tenant_bench.cpp - many users' DRAN memories in InfiniteMemory, a few computable at once.
//
// Each user u has K_u (dim x M fp16, column-major) and V_u (outDim x M fp16). All users' K and V
// live in plain InfiniteMemory (the "masters": Cache Region in front of the SSD). Only S users'
// K can be computed on at a time, because MU kernels read only the Preload Region. A stream of
// queries picks users with a Zipf law (a few users get most queries); a user whose K is not
// computable is "cold" and must be brought in, evicting the least recently used one.
//
// Two ways to bring a user in (--swap):
//   copy    (default) S slots are preloaded ONCE; the host copies K_u from its master into a
//           slot. Never unloads, so no SSD write-back (see im_examples example 6).
//   preload each user's master K is its own 1 GB-multiple range; swap = unloadMemory(victim)
//           + preloadMemory(K_u). unload always writes back to the SSD.
//
// Per query: host writes the query -> [MU] scan + merge/softmax/feedback x iters -> host reads
// the top-k winners -> host prefetches and gathers the k V columns from V_u (InfiniteMemory) ->
// host mixes them. Every query is a noised copy of one stored key, so "HIT" (winner == that key)
// also proves the slot held the right user's K.
//
// Run (check the card is idle first; with more data than the Cache Region the fill evicts
// whatever else is cached):
//   ./im_tenant_bench [--users 32] [-n 500000] [--slots 8] [--queries 300] [--zipf 1.0]
//                     [--swap copy|preload] [-d 1024] [-o 3072] [-k 64] [-i 3]
//                     [-t 2816] [-b 4] [--threads 8] [--csv out.csv] [--device N]
//                     [--warmup 200] [--policy lru|lfu] [--merge split|single] [--fb-tasks 256] [--local-k 64]
//   --local-k   per-task top-k kept by the scan (<= k); the merge still returns the global top k
//   --warmup W  the first W queries run but are not counted (cold start of every user)
//   --policy    victim choice when a cold user needs a slot: lru = least recently used;
//               lfu = fewest queries so far (ties: least recent). Empty slots are used first.
//   --merge     split (default) = 1-task merge/softmax + row-parallel feedback kernel;
//               single = the original 1-task dran_merge_feedback (for comparison)
// Defaults: K 0.95 GiB + V 2.9 GiB per user, 32 users = 128 GB total (> the ~57 GB Cache
// Region, so V and cold K reads reach the SSD), 8 computable users.
// Limits from the kernels: k <= 64, dim <= 1024.

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
constexpr size_t GiB = 1ull << 30;
inline size_t upGiB(size_t x) { return (x + GiB - 1) & ~(GiB - 1); }
inline size_t up4K(size_t x) { return (x + 4095) & ~size_t(4095); }
using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }
int gDev = 0;

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

// ---- SSD byte counters (xcena_cli im get-smart, "Bytes:" section) ----
struct Ssd { long long rd = -1, wr = -1; };
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
void printSsd(const char* what, const Ssd& a, const Ssd& b)
{
    if (a.rd < 0 || b.rd < 0) printf("  SSD %-28s counters n/a\n", what);
    else printf("  SSD %-28s read %9.1f MB  write %9.1f MB\n", what, (b.rd - a.rd) / 1e6, (b.wr - a.wr) / 1e6);
}

// ---- fp16 helpers and deterministic data ----
float h2f(half_t h)
{
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    uint32_t o = e == 0 ? sign : e == 31 ? (sign | 0x7F800000u | (m << 13)) : (sign | ((e - 15 + 127) << 23) | (m << 13));
    float f; std::memcpy(&f, &o, 4); return f;
}
half_t f2h(float f)
{
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = static_cast<int32_t>((x >> 23) & 0xFF) - 112;
    if (e <= 0) return static_cast<half_t>(sign);
    if (e >= 31) return static_cast<half_t>(sign | 0x7C00u);
    return static_cast<half_t>(sign | (e << 10) | (((x & 0x7FFFFFu) + 0x1000u) >> 13));
}
inline uint64_t mix64(uint64_t z)
{
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
// Four fp16 values in +-[0.5, 1): random sign and mantissa, fixed exponent. Never NaN/Inf.
inline uint64_t word4(uint64_t key)
{
    return (mix64(key) & 0x83FF83FF83FF83FFull) | 0x3800380038003800ull;
}
// Element-wise generators (word w of column c of user u; salt separates K from V).
inline uint64_t kWord(int u, size_t c, size_t w) { return word4((uint64_t(u) << 52) ^ (uint64_t(c) << 12) ^ w); }
inline uint64_t vWord(int u, size_t c, size_t w) { return word4(~((uint64_t(u) << 52) ^ (uint64_t(c) << 12) ^ w)); }
// V column tag: rows 0..3 hold (u, c) so a gathered column can be checked. Rows 4.. are data.
inline uint64_t vTag(int u, size_t c) { return (uint64_t(u) << 32) | uint64_t(c); }

template <class F>
void parFor(size_t n, int threads, F f)  // f(begin, end)
{
    std::vector<std::thread> t;
    const size_t chunk = (n + threads - 1) / threads;
    for (int i = 0; i < threads; i++)
    {
        const size_t b = i * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        t.emplace_back([=] { f(b, e); });
    }
    for (auto& x : t) x.join();
}
void parCopy(void* dst, const void* src, size_t bytes, int threads)
{
    const size_t pages = (bytes + 4095) / 4096;
    parFor(pages, threads, [=](size_t b, size_t e) {
        const size_t off = b * 4096, n = std::min(e * 4096, bytes) - off;
        std::memcpy(static_cast<uint8_t*>(dst) + off, static_cast<const uint8_t*>(src) + off, n);
    });
}

double pct(std::vector<double> v, double p)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(p * (v.size() - 1) + 0.5))];
}
}  // namespace

int main(int argc, char* argv[])
{
    int users = 32, slots = 8, queries = 600, warmup = 200, fbTasks = 256, localK = 0, dim = 1024, outDim = 3072, topK = 64, iters = 3;
    int tasks = 2816, batch = 4, threads = 8;
    size_t M = 500000;
    double zipf = 1.0;
    std::string swap = "copy", csv, policy = "lru", merge = "split";
    for (int i = 1; i < argc; i++)
    {
        auto nx = [&] { return std::string(argv[++i]); };
        const std::string a = argv[i];
        if (i + 1 >= argc) break;
        if (a == "--users") users = std::stoi(nx());
        else if (a == "-n") M = std::stoull(nx());
        else if (a == "--slots") slots = std::stoi(nx());
        else if (a == "--queries") queries = std::stoi(nx());
        else if (a == "--zipf") zipf = std::stod(nx());
        else if (a == "--swap") swap = nx();
        else if (a == "-d") dim = std::stoi(nx());
        else if (a == "-o") outDim = std::stoi(nx());
        else if (a == "-k") topK = std::stoi(nx());
        else if (a == "-i") iters = std::stoi(nx());
        else if (a == "-t") tasks = std::stoi(nx());
        else if (a == "-b") batch = std::stoi(nx());
        else if (a == "--threads") threads = std::stoi(nx());
        else if (a == "--csv") csv = nx();
        else if (a == "--warmup") warmup = std::stoi(nx());
        else if (a == "--policy") policy = nx();
        else if (a == "--merge") merge = nx();
        else if (a == "--fb-tasks") fbTasks = std::stoi(nx());
        else if (a == "--local-k") localK = std::stoi(nx());
        else if (a == "--device") gDev = std::stoi(nx());
    }
    const bool copyMode = swap == "copy";
    const bool lfu = policy == "lfu", split = merge == "split";
    if (localK <= 0 || localK > topK) localK = topK;
    if ((policy != "lru" && !lfu) || (merge != "split" && merge != "single") || warmup >= queries)
    {
        printf("bad arguments: --policy lru|lfu, --merge split|single, warmup < queries\n");
        return 1;
    }
    if ((swap != "copy" && swap != "preload") || topK > 64 || dim > 1024 || dim % 4 || outDim % 4 || slots > users)
    {
        printf("bad arguments: --swap copy|preload, k <= 64, dim <= 1024, dim/outDim multiples of 4, slots <= users\n");
        return 1;
    }
    const size_t kBytes = size_t(dim) * M * 2, vBytes = size_t(outDim) * M * 2;
    const size_t kAlloc = upGiB(kBytes), vAlloc = upGiB(vBytes);
    const size_t dW = dim / 4, oW = outDim / 4;  // 64-bit words per column
    printf("im_tenant_bench  device=%d  swap=%s\n", gDev, swap.c_str());
    printf("  users=%d  M=%zu  dim=%d  outDim=%d  k=%d  iters=%d  slots=%d  queries=%d (warmup %d)  zipf=%.2f\n",
           users, M, dim, outDim, topK, iters, slots, queries, warmup, zipf);
    printf("  policy=%s  merge=%s", policy.c_str(), merge.c_str());
    if (split) printf(" (feedback tasks %d)", fbTasks);
    printf("  local-k %d", localK);
    printf("\n");
    printf("  per user: K %.2f GiB (alloc %zu GiB), V %.2f GiB (alloc %zu GiB); all users %.1f GiB\n",
           kBytes / double(GiB), kAlloc / GiB, vBytes / double(GiB), vAlloc / GiB,
           users * double(kAlloc + vAlloc) / GiB);

    auto* ctx = pxl::createContext(gDev);
    if (!ctx) { printf("createContext failed\n"); return 1; }

    // ---- masters: every user's K and V in plain InfiniteMemory ----
    std::vector<half_t*> K(users), V(users);
    for (int u = 0; u < users; u++)
    {
        K[u] = static_cast<half_t*>(pxl::allocateMemory(gDev, kAlloc));
        V[u] = static_cast<half_t*>(pxl::allocateMemory(gDev, vAlloc));
        if (!K[u] || !V[u]) { printf("allocateMemory failed at user %d\n", u); return 1; }
    }
    const Ssd sFill0 = readSsd();
    auto tFill = Clock::now();
    for (int u = 0; u < users; u++)
    {
        auto* k = reinterpret_cast<uint64_t*>(K[u]);
        auto* v = reinterpret_cast<uint64_t*>(V[u]);
        parFor(M, threads, [&](size_t b, size_t e) {
            for (size_t c = b; c < e; c++)
            {
                for (size_t w = 0; w < dW; w++) k[c * dW + w] = kWord(u, c, w);
                v[c * oW] = vTag(u, c);
                for (size_t w = 1; w < oW; w++) v[c * oW + w] = vWord(u, c, w);
            }
            pxl::flushHostCache(k + b * dW, (e - b) * dW * 8);
            pxl::flushHostCache(v + b * oW, (e - b) * oW * 8);
        });
        // Commit so the cache holds only clean pages: later evictions then need no write-back.
        if (pxl::commitSyncMemory(K[u], kAlloc) != pxl::MemoryStatus::Success ||
            pxl::commitSyncMemory(V[u], vAlloc) != pxl::MemoryStatus::Success)
            printf("  warning: commitSyncMemory failed for user %d\n", u);
        printf("\r  filled + committed %d / %d users (%.0f s)", u + 1, users, msSince(tFill) / 1e3);
        fflush(stdout);
    }
    printf("\n");
    printSsd("fill + commit", sFill0, readSsd());

    // ---- the computable area (Preload Region) ----
    // copy:    one arena = S K-slots + small buffers, preloaded once.
    // preload: one arena of small buffers; masters K[u] are preloaded / unloaded on demand.
    const size_t slotBytes = up4K(kBytes);
    size_t off = copyMode ? slotBytes * slots : 0;
    auto take = [&](size_t bytes) { const size_t here = off; off = up4K(off + bytes); return here; };
    const size_t offQ = take(size_t(dim) * 2), offPS = take(size_t(tasks) * topK * 4),
                 offPI = take(size_t(tasks) * topK * 4), offWin = take(size_t(topK) * 8);
    const size_t arenaBytes = upGiB(off);
    auto* arena = static_cast<uint8_t*>(pxl::allocateMemory(gDev, arenaBytes));
    if (!arena) { printf("allocateMemory(arena) failed\n"); return 1; }
    auto t0 = Clock::now();
    pxl::MemoryStatus ps = pxl::preloadMemory(arena, arenaBytes);
    printf("  preload arena %zu GiB: %s, %.0f ms\n", arenaBytes / GiB, st(ps), msSince(t0));
    if (ps != pxl::MemoryStatus::Success) { printf("  (Computing mode on? enough Preload Region?)\n"); return 1; }
    auto* query = reinterpret_cast<half_t*>(arena + offQ);
    auto* partS = reinterpret_cast<float*>(arena + offPS);
    auto* partI = reinterpret_cast<uint32_t*>(arena + offPI);
    auto* winI = reinterpret_cast<uint32_t*>(arena + offWin);
    auto* winW = reinterpret_cast<float*>(arena + offWin + topK * 4);
    auto slotPtr = [&](int s) { return reinterpret_cast<half_t*>(arena + size_t(s) * slotBytes); };

    // ---- MU kernels ----
    auto* mod = pxl::createModule("mu_kernel/mu_kernel.mubin");
    auto* job = ctx->createJob();
    if (!mod || !job || job->load(mod) != pxl::Result::Success) { printf("module load failed\n"); return 1; }
    auto scanX = job->buildMap(mod->createFunction("dran_scan_topk_q"), tasks);
    auto mfX = job->buildMap(mod->createFunction(split ? "dran_merge_softmax" : "dran_merge_feedback"), 1);
    auto fbX = job->buildMap(mod->createFunction("dran_feedback_rows"), fbTasks);
    if (batch > 0) scanX->setBatchSize(batch);
    fbX->setBatchSize(1);  // tiny tasks: one per core

    // ---- Zipf user stream (rank r -> user r-1; fixed seed) ----
    std::vector<double> cdf(users);
    double z = 0.0;
    for (int r = 0; r < users; r++) cdf[r] = (z += 1.0 / std::pow(r + 1.0, zipf));
    uint64_t rs = 0x2545F4914F6CDD1Dull;
    auto urand = [&] { rs = mix64(rs); return (rs >> 11) / 9007199254740992.0; };

    std::vector<int> slotUser(slots, -1), userSlot(users, -1);
    std::vector<long> lastUse(slots, -1), freq(users, 0);
    struct Row { int q, u; bool cold; bool hit; bool vok; double swap, scan, mf, vget, mix, total; };
    std::vector<Row> rows;
    std::vector<float> xi(dim), outV(outDim);
    std::vector<half_t> vbuf(size_t(topK) * outDim);
    long swaps = 0;

    printf("\nrunning %d queries (%d warmup) ...\n", queries, warmup);
    Ssd sRun0;
    for (int q = 0; q < queries; q++)
    {
        if (q == warmup) sRun0 = readSsd();
        const double x = urand() * z;
        const int u = int(std::lower_bound(cdf.begin(), cdf.end(), x) - cdf.begin());
        freq[u]++;
        Row row{q, u, false, false, false, 0, 0, 0, 0, 0, 0};
        const auto tq = Clock::now();

        // -- bring the user in if needed (empty slot first, then the policy's victim) --
        int s = userSlot[u];
        if (s < 0)
        {
            row.cold = true;
            if (q >= warmup) swaps++;
            s = 0;
            for (int t = 1; t < slots; t++)
            {
                const bool empty = slotUser[t] < 0, emptyS = slotUser[s] < 0;
                if (empty != emptyS) { if (empty) s = t; continue; }
                if (empty) continue;
                const long ft = freq[slotUser[t]], fs = freq[slotUser[s]];
                if (lfu ? (ft < fs || (ft == fs && lastUse[t] < lastUse[s])) : lastUse[t] < lastUse[s]) s = t;
            }
            const int victim = slotUser[s];
            const auto ts = Clock::now();
            if (copyMode)
            {
                parCopy(slotPtr(s), K[u], kBytes, threads);
                pxl::flushHostCache(slotPtr(s), kBytes);
            }
            else
            {
                if (victim >= 0 && pxl::unloadMemory(K[victim], kAlloc) != pxl::MemoryStatus::Success)
                    printf("  warning: unload user %d failed\n", victim);
                if ((ps = pxl::preloadMemory(K[u], kAlloc)) != pxl::MemoryStatus::Success)
                {
                    printf("preload user %d: %s\n", u, st(ps));
                    return 1;
                }
            }
            row.swap = msSince(ts);
            if (victim >= 0) userSlot[victim] = -1;
            slotUser[s] = u;
            userSlot[u] = s;
        }
        lastUse[s] = q;
        const half_t* keys = copyMode ? slotPtr(s) : K[u];

        // -- the query: a noised copy of one stored key (stands in for the GPU encoder output) --
        const size_t probe = size_t(urand() * M) % M;
        float nrm = 0.0f;
        for (size_t w = 0; w < dW; w++)
        {
            const uint64_t v = kWord(u, probe, w);
            for (int j = 0; j < 4; j++) xi[w * 4 + j] = h2f(half_t(v >> (16 * j)));
        }
        for (int r = 0; r < dim; r++) nrm += xi[r] * xi[r];
        nrm = std::sqrt(nrm);
        float n2 = 0.0f;
        for (int r = 0; r < dim; r++) { xi[r] = xi[r] / nrm + 0.05f * float(urand() * 2 - 1); n2 += xi[r] * xi[r]; }
        n2 = std::sqrt(n2);
        for (int r = 0; r < dim; r++) query[r] = f2h(xi[r] / n2);
        pxl::flushHostCache(query, size_t(dim) * 2);

        // -- MU: scan + merge/softmax/feedback, iters times --
        for (int it = 0; it < iters; it++)
        {
            auto ti = Clock::now();
            if (scanX->execute(keys, query, dim, uint64_t(M), localK, partS, partI) != pxl::Result::Success ||
                scanX->synchronize() != pxl::Result::Success) { printf("scan failed\n"); return 1; }
            row.scan += msSince(ti);
            ti = Clock::now();
            const pxl::Result e = split
                ? mfX->execute(partS, partI, tasks * localK, localK, query, dim, topK, winI)
                : mfX->execute(partS, partI, tasks * localK, localK, keys, dim, topK, winI, query);
            if (e != pxl::Result::Success || mfX->synchronize() != pxl::Result::Success) { printf("merge failed\n"); return 1; }
            // split form: next query = K(:, winners) . w, unnormalized (the next merge divides by |q|)
            if (split && it + 1 < iters &&
                (fbX->execute(keys, winI, dim, topK, query) != pxl::Result::Success ||
                 fbX->synchronize() != pxl::Result::Success)) { printf("feedback failed\n"); return 1; }
            row.mf += msSince(ti);
        }
        pxl::flushHostCache(winI, size_t(topK) * 8);
        row.hit = winI[0] == probe;

        // -- host: prefetch then gather the k winning V columns from InfiniteMemory --
        auto tv = Clock::now();
        const size_t colB = size_t(outDim) * 2;
        for (int i = 0; i < topK; i++) (void)pxl::prefetchMemory(V[u] + size_t(winI[i]) * outDim, colB);
        bool vok = true;
        for (int i = 0; i < topK; i++)
        {
            const half_t* src = V[u] + size_t(winI[i]) * outDim;
            std::memcpy(vbuf.data() + size_t(i) * outDim, src, colB);
            uint64_t tag;
            std::memcpy(&tag, vbuf.data() + size_t(i) * outDim, 8);
            vok &= tag == vTag(u, winI[i]);
        }
        row.vget = msSince(tv);
        row.vok = vok;

        // -- host mix (on a GPU this is a 3 MB H2D copy + one small GEMV) --
        tv = Clock::now();
        std::fill(outV.begin(), outV.end(), 0.0f);
        for (int i = 0; i < topK; i++)
            for (int r = 4; r < outDim; r++) outV[r] += winW[i] * h2f(vbuf[size_t(i) * outDim + r]);
        row.mix = msSince(tv);
        row.total = msSince(tq);
        rows.push_back(row);
        if (q < 5 || q % 50 == 0)
            printf("  q%-4d user %2d %-4s swap %7.1f  scan %7.1f  merge %5.1f  V %6.1f  total %7.1f ms  %s%s\n", q, u,
                   row.cold ? "cold" : "hot", row.swap, row.scan, row.mf, row.vget, row.total,
                   row.hit ? "HIT" : "miss", row.vok ? "" : "  V-TAG-BAD");
    }
    const Ssd sRun1 = readSsd();

    // ---- summary ----
    auto summary = [&](const char* name, bool want, bool all) {
        std::vector<double> tot, sw, sc, vg;
        int hit = 0, vok = 0;
        for (auto& r : rows)
            if (r.q >= warmup && (all || r.cold == want))
            {
                tot.push_back(r.total); sw.push_back(r.swap); sc.push_back(r.scan); vg.push_back(r.vget);
                hit += r.hit; vok += r.vok;
            }
        if (tot.empty()) { printf("  %-5s none\n", name); return; }
        printf("  %-5s n=%4zu  total p50 %7.1f p99 %7.1f ms | swap p50 %7.1f | scan p50 %6.1f | V gather p50 %6.1f p99 %6.1f | HIT %d/%zu  V ok %d/%zu\n",
               name, tot.size(), pct(tot, .5), pct(tot, .99), pct(sw, .5), pct(sc, .5), pct(vg, .5), pct(vg, .99),
               hit, tot.size(), vok, tot.size());
    };
    printf("\n--- summary (%s, %d slots for %d users) ---\n", swap.c_str(), slots, users);
    const int measured = queries - warmup;
    printf("  measured queries %d..%d\n", warmup, queries - 1);
    printf("  slot hit rate: %.1f%%  (%ld swaps in %d queries); static top-%d users get %.1f%% of queries\n",
           100.0 * (measured - swaps) / measured, swaps, measured, slots, 100.0 * cdf[slots - 1] / z);
    summary("hot", false, false);
    summary("cold", true, false);
    summary("all", false, true);
    printSsd("during queries", sRun0, sRun1);
    if (sRun0.rd >= 0 && sRun1.rd > sRun0.rd)
        printf("  SSD write / read: %.3f   SSD read per query: %.1f MB   write per swap: %.1f MB\n",
               double(sRun1.wr - sRun0.wr) / double(sRun1.rd - sRun0.rd), (sRun1.rd - sRun0.rd) / 1e6 / measured,
               swaps ? (sRun1.wr - sRun0.wr) / 1e6 / swaps : 0.0);

    if (!csv.empty())
    {
        if (FILE* f = fopen(csv.c_str(), "w"))
        {
            fprintf(f, "q,user,cold,hit,v_ok,swap_ms,scan_ms,merge_ms,vgather_ms,mix_ms,total_ms\n");
            for (size_t i = 0; i < rows.size(); i++)
            {
                const Row& r = rows[i];
                fprintf(f, "%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", r.q, r.u, r.cold, r.hit, r.vok, r.swap,
                        r.scan, r.mf, r.vget, r.mix, r.total);
            }
            fclose(f);
            printf("  per-query rows -> %s\n", csv.c_str());
        }
    }

    // ---- cleanup. Unloading the arena writes it back once (slots x K); example 6 (d) shows
    //      whether a plain release would avoid that. ----
    if (!copyMode)
        for (int u = 0; u < users; u++)
            if (userSlot[u] >= 0) (void)pxl::unloadMemory(K[u], kAlloc);
    (void)pxl::unloadMemory(arena, arenaBytes);
    pxl::releaseMemory(arena);
    for (int u = 0; u < users; u++) { pxl::releaseMemory(K[u]); pxl::releaseMemory(V[u]); }
    ctx->destroyJob(job);
    pxl::destroyContext(ctx);
    return 0;
}
