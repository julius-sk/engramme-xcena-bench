// im_store.cpp — a tiny C API over pxl so Python (ctypes) can put arrays in InfiniteMemory.
//
// The pointer returned by im_alloc is an ordinary host virtual address into the card's address
// space: numpy / torch can read and write it directly. The device caches 512 KB pages in its
// DRAM and moves them to and from its SSD on its own.
//
// IMPORTANT: the memory belongs to this process. When the process exits (or im_free is called)
// pxl releases it, and release unmaps the range, which DISCARDS the data. InfiniteMemory is
// not a persistent store across runs: load the dataset in the same process that uses it.

#include <cstddef>
#include <cstdint>

#include "pxl/pxl.hpp"

namespace
{
pxl::Context* gCtx = nullptr;
int gDev = 0;
}  // namespace

extern "C"
{
// 0 on success.
int im_init(int deviceId)
{
    gDev = deviceId;
    gCtx = pxl::createContext(static_cast<uint32_t>(deviceId));
    return gCtx ? 0 : -1;
}

// Returns nullptr on failure.
void* im_alloc(size_t bytes)
{
    return pxl::allocateMemory(static_cast<uint32_t>(gDev), bytes);
}

void im_free(void* p)
{
    pxl::releaseMemory(p);
}

// Push the host CPU's cached copy of [p, p+bytes) out to the device. Call after host writes
// and before commit / preload / an MU kernel; call before the host reads device-written data.
void im_flush(void* p, size_t bytes)
{
    pxl::flushHostCache(p, bytes);
}

// The functions below return pxl::MemoryStatus as an int:
// 0 Success, 1 ResourceExhaust, 2 InvalidRequest, 3 NotSupported.

// Write dirty pages to the SSD and wait. Data stays cached and readable.
int im_commit(const void* p, size_t bytes) { return static_cast<int>(pxl::commitSyncMemory(p, bytes)); }

// Asynchronous page fetch from the SSD (512 KB granularity). Use before scattered reads.
int im_prefetch(const void* p, size_t bytes) { return static_cast<int>(pxl::prefetchMemory(p, bytes)); }

// Prefetch the pages that hold base + offs[i] for i < n. Sort offs first so that offsets on the
// same page are adjacent: each distinct page is issued once. Returns the number of pages issued.
int64_t im_prefetch_offsets(const void* base, const int64_t* offs, int64_t n, uint64_t page)
{
    uintptr_t last = UINTPTR_MAX;
    int64_t issued = 0;
    for (int64_t i = 0; i < n; i++)
    {
        const uintptr_t a = (reinterpret_cast<uintptr_t>(base) + static_cast<uintptr_t>(offs[i])) & ~(page - 1);
        if (a == last) continue;
        (void)pxl::prefetchMemory(reinterpret_cast<const void*>(a), page);
        last = a;
        issued++;
    }
    return issued;
}

// Keep a range in the DRAM cache.
int im_pin(const void* p, size_t bytes) { return static_cast<int>(pxl::pinMemory(p, bytes)); }
int im_unpin(const void* p, size_t bytes) { return static_cast<int>(pxl::unpinMemory(p, bytes)); }

// Move a range into the Preload Region (needed before an MU kernel reads it; 1 GB units).
// Always re-reads the range from the SSD. Unload always writes it back.
int im_preload(const void* p, size_t bytes) { return static_cast<int>(pxl::preloadMemory(p, bytes)); }
int im_unload(const void* p, size_t bytes) { return static_cast<int>(pxl::unloadMemory(p, bytes)); }
}
