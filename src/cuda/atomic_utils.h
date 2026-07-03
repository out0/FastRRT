#ifndef __GRAPH_CPU_ATOMIC_UTILS_H
#define __GRAPH_CPU_ATOMIC_UTILS_H

#include <cstring> // memcpy
#include <atomic>

inline float atomicAddFloat(float *addr, float val)
{
    int *iaddr = reinterpret_cast<int*>(addr);
    int old_i, new_i;
    float old_f;

    do {
        old_i = __atomic_load_n(iaddr, __ATOMIC_SEQ_CST);
        std::memcpy(&old_f, &old_i, sizeof(float));
        float new_f = old_f + val;
        std::memcpy(&new_i, &new_f, sizeof(int));
    } while (!__atomic_compare_exchange_n(
                 iaddr, &old_i, new_i,
                 false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));

    return old_f; // previous value, matches CUDA atomicAdd's return
}


inline long long atomicMin(std::atomic<long long>& val, long long new_val) {
    long long old = val.load(std::memory_order_relaxed);
    while (new_val < old &&
           !val.compare_exchange_weak(old, new_val, std::memory_order_relaxed))
    {}
    return old;
}


#endif