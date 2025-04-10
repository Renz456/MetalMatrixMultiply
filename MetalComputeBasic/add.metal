/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
A shader that multiplies two matrices using 2D block tiling for better performance.
*/

#include <metal_stdlib>
using namespace metal;

// Define tile sizes to match CUDA implementation
constant int BM = 64;  // Block size for M dimension
constant int BN = 64;  // Block size for N dimension
constant int BK = 16;   // Block size for K dimension
constant int TM = 16;   // Number of results per thread in M dimension
constant int TN = 16;   // Number of results per thread in N dimension

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
    
    // Calculate total results and threads per block
    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
    
    // Calculate thread indices within block
    const uint threadCol = tid.x % (BN / TN);
    const uint threadRow = tid.x / (BN / TN);
    
    // Create shared memory tiles
    threadgroup float As[BM * BK];
    threadgroup float Bs[BK * BN];
    
    // Move pointers to the start of the current block
    A += cRow * BM * K;
    B += cCol * BN;
    result += cRow * BM * N + cCol * BN;
    
    // Calculate indices for loading into shared memory
    const uint innerRowA = tid.x / BK;
    const uint innerColA = tid.x % BK;
    const uint strideA = numThreadsBlocktile / BK;
    const uint innerRowB = tid.x / BN;
    const uint innerColB = tid.x % BN;
    const uint strideB = numThreadsBlocktile / BN;
    
    // Allocate thread-local cache for results and register caches
    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};
    
    // Loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load data into shared memory with stride for better coalescing
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            if ((innerRowA + loadOffset) < BM && (bkIdx + innerColA) < K) {
                As[(innerRowA + loadOffset) * BK + innerColA] = 
                    A[(innerRowA + loadOffset) * K + innerColA];
            } else {
                As[(innerRowA + loadOffset) * BK + innerColA] = 0.0f;
            }
        }
        
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            if ((innerRowB + loadOffset) < BK && (threadCol * TN + innerColB) < BN) {
                Bs[(innerRowB + loadOffset) * BN + innerColB] = 
                    B[(innerRowB + loadOffset) * N + innerColB];
            } else {
                Bs[(innerRowB + loadOffset) * BN + innerColB] = 0.0f;
            }
        }
        
        // Synchronize threads
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Advance pointers
        A += BK;
        B += BK * N;
        
        // Calculate dot products using register caches
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load into registers
            for (uint i = 0; i < TM; ++i) {
                if ((threadRow * TM + i) < BM) {
                    regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
                } else {
                    regM[i] = 0.0f;
                }
            }
            
            for (uint i = 0; i < TN; ++i) {
                if ((threadCol * TN + i) < BN) {
                    regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
                } else {
                    regN[i] = 0.0f;
                }
            }
            
            // Compute results using register caches
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        
        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write results back to global memory
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            if ((threadRow * TM + resIdxM) < BM && (threadCol * TN + resIdxN) < BN) {
                result[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = 
                    threadResults[resIdxM * TN + resIdxN];
            }
        }
    }
}
