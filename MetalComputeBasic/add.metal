/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
A shader that multiplies two matrices using block tiling for better performance.
*/

#include <metal_stdlib>
using namespace metal;

// Define tile sizes to match CUDA implementation
constant int BM = 128;  // Block size for M dimension
constant int BN = 128;  // Block size for N dimension
constant int BK = 16;   // Block size for K dimension
constant int TM = 16;   // Number of results per thread

/// This is a Metal Shading Language (MSL) function that performs matrix multiplication on a GPU
/// using shared memory for better performance with optimized memory coalescing.
/// Each thread computes multiple results stored in registers.
kernel void add_arrays(device const float* A,
                      device const float* B,
                      device float* result,
                      device const int& M [[buffer(3)]],
                      device const int& N [[buffer(4)]],
                      device const int& K [[buffer(5)]],
                      uint2 gid [[thread_position_in_grid]],
                      uint2 tid [[thread_position_in_threadgroup]])
{
    // Check if we're within bounds
    if (gid.x >= M || gid.y >= N) {
        return;
    }
    
    // Calculate block indices
    const uint cRow = gid.y / BM;  // Block row
    const uint cCol = gid.x / BN;  // Block column
    
    // Calculate thread indices within block
    const uint threadCol = tid.x % BN;
    const uint threadRow = tid.x / BN;
    
    // Create shared memory tiles
    threadgroup float As[BM * BK];
    threadgroup float Bs[BK * BN];
    
    // Calculate global indices for A and B matrices
    const uint globalRowA = cRow * BM + threadRow;
    const uint globalColA = threadCol;
    const uint globalRowB = threadRow;
    const uint globalColB = cCol * BN + threadCol;
    
    // Move pointers to the start of the current block
    A += cRow * BM * K;
    B += cCol * BN;
    result += cRow * BM * N + cCol * BN;
    
    // Allocate thread-local cache for results
    float threadResults[TM] = {0.0};
    
    // Loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load data into shared memory
        if (globalRowA < M && (bkIdx + threadCol) < K) {
            As[threadRow * BK + threadCol] = A[threadRow * K + threadCol];
        } else {
            As[threadRow * BK + threadCol] = 0.0f;
        }
        
        if ((bkIdx + threadRow) < K && globalColB < N) {
            Bs[threadRow * BN + threadCol] = B[threadRow * N + threadCol];
        } else {
            Bs[threadRow * BN + threadCol] = 0.0f;
        }
        
        // Synchronize threads
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Calculate dot products
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float tmpB = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                if ((threadRow * TM + resIdx) < BM) {
                    threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
                }
            }
        }
        
        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Advance pointers
        A += BK;
        B += BK * N;
    }
    
    // Write results back to global memory
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        if ((threadRow * TM + resIdx) < BM && threadCol < BN) {
            result[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
        }
    }
}
