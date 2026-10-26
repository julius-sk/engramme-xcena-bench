// mu_dran.cpp — DRAN retrieval kernels for XCENA MX1P (Case C).
//
// Workload: one attractor iteration of modern-Hopfield recall over a memory store
//   s = K^T xi        (K: n x M, column-major; one column = one memory's key)
//   w = softmax(beta*s), truncated to the top-k columns
//   out = V(:, topk) . w
//
// The scan is the only O(M) stage and is what belongs in computational memory.
// Each MU task owns a contiguous stripe of K's columns, computes the dot product
// of each column against the query on the FP16 vector engine, and keeps a local
// top-k. The host merges the per-task partials.
//
// Layout note: K is stored column-major (a column is contiguous, n elements),
// so each dot product reads a sequential run — the access pattern DRAM serves best.
//
// Kernel memory limits (SDK): heap 3 MB, stack 64 KB per task, <=9 kernel params.

#include <cmath>

#include "mu/mu.hpp"
#include "mu/vector.hpp"        // mu::vector::dotProduct
#include "mu/vector/half.hpp"   // complete half_float::half (mu.hpp does not pull it in)

namespace
{
using half = half_float::half;

constexpr int K_MAX = 64;   // max top-k per task; keeps the stack well under 64 KB
constexpr float BETA = 8.0f;  // softmax inverse temperature (kernel arg budget is 9)

// Insert (score, index) into a descending-by-score local top-k list.
inline void topkInsert(float* scores, uint32_t* indices, int k, float score, uint32_t index)
{
    if (score <= scores[k - 1])
    {
        return;
    }
    int pos = k - 1;
    while (pos > 0 && scores[pos - 1] < score)
    {
        scores[pos] = scores[pos - 1];
        indices[pos] = indices[pos - 1];
        --pos;
    }
    scores[pos] = score;
    indices[pos] = index;
}
}  // namespace

// ---------------------------------------------------------------------------
// Stage 1 — the scan. One task per stripe of K's columns.
//
//   keys      : K, n x M, column-major, fp16          (device memory, preloaded)
//   query     : xi, n elements, fp16                   (device memory, preloaded)
//   dim       : n
//   numCols   : M (total columns across all tasks)
//   topK      : k (<= K_MAX)
//   outScores : taskCount * topK floats  — per-task local top-k scores
//   outIndices: taskCount * topK uint32  — matching global column indices
//
// Each task writes its own slice, so no cross-task synchronization is needed.
// ---------------------------------------------------------------------------
void dran_scan_topk(const half* keys,
                    const half* query,
                    int dim,
                    uint64_t numCols,
                    int topK,
                    float* outScores,
                    uint32_t* outIndices)
{
    const uint32_t taskIdx = mu::getTaskIdx();
    const uint32_t taskCount = mu::getTaskCount();

    if (topK > K_MAX)
    {
        topK = K_MAX;
    }

    // Even split of columns; the last task absorbs the remainder.
    const uint64_t perTask = (numCols + taskCount - 1) / taskCount;
    const uint64_t colBegin = static_cast<uint64_t>(taskIdx) * perTask;
    uint64_t colEnd = colBegin + perTask;
    if (colEnd > numCols)
    {
        colEnd = numCols;
    }

    float localScores[K_MAX];
    uint32_t localIndices[K_MAX];
    for (int i = 0; i < topK; i++)
    {
        localScores[i] = -3.0e38f;
        localIndices[i] = 0;
    }

    if (taskIdx == 0)
    {
        mu::hostPrintf("[kernel] taskCount=%u perTask=%u range=[%u,%u)\n",
                       (unsigned)taskCount, (unsigned)perTask,
                       (unsigned)colBegin, (unsigned)colEnd);
    }

    // FP16 vector engine, one call per column. A hand-written scalar loop was tried
    // and produced wrong dot products on this toolchain at -O3 (host 0.001360 vs
    // device -0.245 on the same bytes), at identical speed, so the vector engine
    // stays: it is both correct and no slower.
    const uint64_t vectorDim = static_cast<uint64_t>(dim);
    for (uint64_t col = colBegin; col < colEnd; col++)
    {
        const half* kcol = keys + col * vectorDim;
        float acc = 0.0f;
        auto vErr = mu::vector::dotProduct(kcol, query, &acc, vectorDim);
        if (taskIdx == 0 && col == colBegin)
        {
            uint32_t ab = 0;
            __builtin_memcpy(&ab, &acc, 4);
            mu::hostPrintf("[kernel] dot(col %u) = 0x%08x  vdmaOk=%d\n",
                           (unsigned)col, (unsigned)ab, (int)vErr.isSuccessful());
        }
        topkInsert(localScores, localIndices, topK, acc, static_cast<uint32_t>(col));
    }

    if (taskIdx == 0)
    {
        uint32_t sb = 0;
        __builtin_memcpy(&sb, &localScores[0], 4);
        mu::hostPrintf("[kernel] task0 best = col %u  score 0x%08x\n",
                       (unsigned)localIndices[0], (unsigned)sb);
    }

    float* myScores = outScores + static_cast<uint64_t>(taskIdx) * topK;
    uint32_t* myIndices = outIndices + static_cast<uint64_t>(taskIdx) * topK;
    for (int i = 0; i < topK; i++)
    {
        myScores[i] = localScores[i];
        myIndices[i] = localIndices[i];
    }
}

// ---------------------------------------------------------------------------
// Stage 1b — merge, softmax and feedback, entirely on the device.
//
// Closes the attractor loop without the host touching data: it reduces the
// per-task partials to a global top-k, softmaxes them, and writes the next query
// back into the same buffer the scan kernel reads.
//
//   partScores  : numCand floats   — laid out as (scan tasks) x candStride
//   partIndices : numCand uint32   — matching global column indices
//   numCand     : total candidates = scanTasks * candStride
//   candStride  : per-scan-task list length; each list is sorted descending
//   keys        : K, n x M, column-major, fp16
//   dim         : n
//   topK        : k (<= K_MAX)
//   winnersOut  : k uint32 indices, immediately followed by k floats of weight
//   queryOut    : n fp16 — overwritten with normalize(K(:,topk) . w)
//
// Runs on a single task (launch this Map with a task count of 1). The merge is
// cheap because each partial list is descending: once a list's first entry falls
// below the running threshold, the rest of that list is skipped. That turns
// scanTasks*candStride comparisons into roughly scanTasks of them.
// ---------------------------------------------------------------------------
void dran_merge_feedback(const float* partScores,
                         const uint32_t* partIndices,
                         int numCand,
                         int candStride,
                         const half* keys,
                         int dim,
                         int topK,
                         uint32_t* winnersOut,
                         half* queryOut)
{
    if (mu::getTaskIdx() != 0)
    {
        return;
    }
    if (topK > K_MAX)
    {
        topK = K_MAX;
    }

    float scores[K_MAX];
    uint32_t indices[K_MAX];
    for (int i = 0; i < topK; i++)
    {
        scores[i] = -3.0e38f;
        indices[i] = 0;
    }

    // --- merge: one descending list per scan task, skip a list once it drops below
    //     the running k-th best ---
    for (int base = 0; base < numCand; base += candStride)
    {
        const int end = (base + candStride < numCand) ? base + candStride : numCand;
        for (int j = base; j < end; j++)
        {
            if (partScores[j] <= scores[topK - 1])
            {
                break;  // list is descending: nothing later in it can qualify
            }
            topkInsert(scores, indices, topK, partScores[j], partIndices[j]);
        }
    }

    // --- softmax over the k survivors, max-subtracted for numerical stability ---
    float* weights = reinterpret_cast<float*>(winnersOut + topK);
    const float top = scores[0];  // topkInsert keeps the list descending
    float wsum = 0.0f;
    for (int i = 0; i < topK; i++)
    {
        const float e = expf(BETA * (scores[i] - top));
        weights[i] = e;
        wsum += e;
    }
    const float inv = 1.0f / wsum;
    for (int i = 0; i < topK; i++)
    {
        weights[i] *= inv;
        winnersOut[i] = indices[i];
    }

    // --- feedback: xi <- normalize(K(:,topk) . w), touching only k columns ---
    float acc[1024];
    const int n = (dim <= 1024) ? dim : 1024;
    for (int r = 0; r < n; r++)
    {
        acc[r] = 0.0f;
    }
    for (int i = 0; i < topK; i++)
    {
        const half* kcol = keys + static_cast<uint64_t>(indices[i]) * dim;
        const float w = weights[i];
        for (int r = 0; r < n; r++)
        {
            acc[r] += w * static_cast<float>(kcol[r]);
        }
    }
    float norm = 0.0f;
    for (int r = 0; r < n; r++)
    {
        norm += acc[r] * acc[r];
    }
    norm = sqrtf(norm) + 1e-9f;
    const float invNorm = 1.0f / norm;
    for (int r = 0; r < n; r++)
    {
        queryOut[r] = static_cast<half>(acc[r] * invNorm);
    }
}

// ---------------------------------------------------------------------------
// Stage 2 — weighted mix of the winning value columns.
//
//   values  : V, o x M, column-major, fp16
//   indices : k global column indices chosen by the host merge
//   weights : k softmax weights (fp32)
//   outDim  : o
//   topK    : k
//   out     : o floats — sum_i weights[i] * V(:, indices[i])
//
// Parallel over output rows: each task reduces its own row slice, so the k
// column reads are shared but the accumulation is race-free.
// ---------------------------------------------------------------------------
void dran_mix_values(const half* values,
                     const uint32_t* indices,
                     const float* weights,
                     int outDim,
                     int topK,
                     float* out)
{
    const uint32_t taskIdx = mu::getTaskIdx();
    const uint32_t taskCount = mu::getTaskCount();

    const int perTask = (outDim + static_cast<int>(taskCount) - 1) / static_cast<int>(taskCount);
    const int rowBegin = static_cast<int>(taskIdx) * perTask;
    int rowEnd = rowBegin + perTask;
    if (rowEnd > outDim)
    {
        rowEnd = outDim;
    }

    for (int r = rowBegin; r < rowEnd; r++)
    {
        float acc = 0.0f;
        for (int i = 0; i < topK; i++)
        {
            const half* vcol = values + static_cast<uint64_t>(indices[i]) * outDim;
            acc += weights[i] * static_cast<float>(vcol[r]);
        }
        out[r] = acc;
    }
}

MU_KERNEL_ADD(dran_scan_topk);
MU_KERNEL_ADD(dran_merge_feedback);
MU_KERNEL_ADD(dran_mix_values);
