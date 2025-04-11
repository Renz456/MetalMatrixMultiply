/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
A shader that multiplies two matrices using block tiling for better performance.
*/

#include <metal_stdlib>
#include <metal_logging>


using namespace metal;

// Define tile sizes to match CUDA implementation
constant int BM = 2;  // Block size for M dimension
constant int BN = 2;  // Block size for N dimension
constant int BK = 1;   // Block size for K dimension
constant int TM = 2;   // Number of results per thread

/// This is a Metal Shading Language (MSL) function that performs matrix multiplication on a GPU
/// using shared memory for better performance with optimized memory coalescing.
/// Each thread computes multiple results stored in registers.
kernel void add_arrays(device const float* A,
                      device const float* B,
                      device float* result,
                      device const int& M [[buffer(3)]],
                      device const int& N [[buffer(4)]],
                      device const int& K [[buffer(5)]],
                      uint gid [[thread_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint bid [[threadgroup_position_in_grid]])
{
    // Calculate 2D indices from 1D grid position
    uint row = bid / (M/BM);
    uint col = bid % (N/BN);
    
    // Check if we're within bounds
    if (row >= M/BM || col >= N/BN) {
        return;
    }
    
    // Calculate block indices
    const uint blockRow = row;  // Block row
    const uint blockCol = col;  // Block column
    
    // Calculate thread indices within block
    const uint threadRow = tid / BN;
    const uint threadCol = tid % BN;
    
    // Create shared memory tiles
    threadgroup float As[BM * BK];
    threadgroup float Bs[BK * BN];
    
    // Allocate thread-local cache for results
    float threadResults[TM] = {0.0};
    
    os_log_default.log_error("hello!!! tid: %d blockRow: %d blockCol: %d threadRow: %d threadCol: %d, gid: %d", tid, blockRow, blockCol, threadRow, threadCol, gid);
    
    // Calculate the number of threads needed to load all elements
    // (BM * BN)/TM = BM * BK = BK * BN
    const uint numThreadsNeeded = (BM * BN)/TM;
    
    // Calculate which element this thread is responsible for loading
    const uint innerRowA = tid / BK;
    const uint innerColA = tid % BK;
    const uint innerRowB = tid / BN;
    const uint innerColB = tid % BN;
    
    // Loop over blocks in K dimension
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Populate the shared memory caches
        if (tid < numThreadsNeeded) {
            // Load one value from A into shared memory
            if (innerRowA < BM && (bkIdx + innerColA) < K) {
                As[innerRowA * BK + innerColA] = A[(blockRow * BM + innerRowA) * K + (bkIdx + innerColA)];
            } else {
                As[innerRowA * BK + innerColA] = 0.0f;
            }
            
            // Load one value from B into shared memory
            if ((bkIdx + innerRowB) < K && innerColB < BN) {
                Bs[innerRowB * BN + innerColB] = B[(bkIdx + innerRowB) * N + (blockCol * BN + innerColB)];
            } else {
                Bs[innerRowB * BN + innerColB] = 0.0f;
            }
        }
        
        // Synchronize threads
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // We make the dot product loop the outside loop, which facilitates
            // reuse of the Bs entry, which we can cache in a tmp var.
            float tmpB = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        
        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write results back to global memory
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        uint effectiveThreadRow = threadRow * TM + resIdx;
        if (effectiveThreadRow < BM && threadCol < BN) {
            result[(blockRow * BM + effectiveThreadRow) * N + (blockCol * BN + threadCol)] = threadResults[resIdx];
        }
    }
}
