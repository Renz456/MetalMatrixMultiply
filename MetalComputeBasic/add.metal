/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
A shader that multiplies two matrices using block tiling for better performance.
*/

#include <metal_stdlib>
#include <metal_logging>


using namespace metal;

// Define tile sizes to match CUDA implementation
constant int BM = 32;  // Block size for M dimension
constant int BN = 32;  // Block size for N dimension
constant int BK = 8;   // Block size for K dimension
constant int TM = 4;   // Number of results per thread in M dimension
constant int TN = 4;   // Number of results per thread in N dimension

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
    const uint threadRow = tid / (BN/TN);
    const uint threadCol = tid % (BN/TN);
    
    // Create double-buffered shared memory tiles
    threadgroup float As[2][BM * BK];  // Double buffer for A
    threadgroup float Bs[2][BK * BN];  // Double buffer for B
    
    // Allocate thread-local cache for results
    float threadResults[TM][TN] = {0.0};
    
    // Allocate register memory for blocking
    float regM[TM];
    float regN[TN];
    
    // Calculate the number of threads needed to load all elements
    // (BM * BN)/(TM * TN) = BK * BK
    const uint numThreadsNeeded = (BM * BN)/(TM * TN);
    
    // Calculate which element this thread is responsible for loading
    const uint innerRowA = tid / BK;
    const uint innerColA = tid % BK;
    const uint innerRowB = tid / (BN);
    const uint innerColB = tid % (BN);

    const uint strideA = numThreadsNeeded / BK;
    const uint strideB = numThreadsNeeded / BN;
    // os_log_default.log_error("hello!!! tid: %d blockRow: %d blockCol: %d threadRow: %d threadCol: %d, gid: %d", tid, blockRow, blockCol, threadRow, threadCol, gid);
    
    // Current buffer being computed on (0 or 1)
    uint currentBuffer = 0;
    // Next buffer to load into (1 or 0)
    uint nextBuffer = 1;
    
    // Load first tile
    if (tid < numThreadsNeeded) {
        // Load values from A into shared memory with stride
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            if ((innerRowA + loadOffset) < BM && innerColA < K) {
                As[currentBuffer][(innerRowA + loadOffset) * BK + innerColA] = 
                    A[(blockRow * BM + innerRowA + loadOffset) * K + innerColA];
            } else {
                As[currentBuffer][(innerRowA + loadOffset) * BK + innerColA] = 0.0f;
            }
        }
        
        // Load values from B into shared memory with stride
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            if ((innerRowB + loadOffset) < K && innerColB < BN) {
                Bs[currentBuffer][(innerRowB + loadOffset) * BN + innerColB] = 
                    B[(innerRowB + loadOffset) * N + (blockCol * BN + innerColB)];
            } else {
                Bs[currentBuffer][(innerRowB + loadOffset) * BN + innerColB] = 0.0f;
            }
        }
    }
    
    // // Synchronize threads for first load
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Loop over blocks in K dimension
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {

        // Calculate per-thread results using register blocking for current tile
        // Calculate per-thread results using register blocking for current tile
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Block into registers
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[currentBuffer][(threadRow * TM + i) * BK + dotIdx];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[currentBuffer][dotIdx * BN + threadCol * TN + i];
            }
            
            // Compute dot products using register values
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM][resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }


        
        // Start loading next tile while computing current tile
        if (tid < numThreadsNeeded && (bkIdx + BK) < K) {
            // Load values from A into shared memory with stride
            for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
                if ((innerRowA + loadOffset) < BM && (bkIdx + BK + innerColA) < K) {
                    As[nextBuffer][(innerRowA + loadOffset) * BK + innerColA] = 
                        A[(blockRow * BM + innerRowA + loadOffset) * K + (bkIdx + BK + innerColA)];
                } else {
                    As[nextBuffer][(innerRowA + loadOffset) * BK + innerColA] = 0.0f;
                }
            }
            
            // Load values from B into shared memory with stride
            for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
                if ((bkIdx + BK + innerRowB + loadOffset) < K && innerColB < BN) {
                    Bs[nextBuffer][(innerRowB + loadOffset) * BN + innerColB] = 
                        B[(bkIdx + BK + innerRowB + loadOffset) * N + (blockCol * BN + innerColB)];
                } else {
                    Bs[nextBuffer][(innerRowB + loadOffset) * BN + innerColB] = 0.0f;
                }
            }
        }

        // threadgroup_barrier(mem_flags::mem_threadgroup);

        
        
        // Swap buffers
        currentBuffer = nextBuffer;
        nextBuffer = 1 - nextBuffer;
        
        // Synchronize before next iteration
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
    }
    
    // Write results back to global memory
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            uint effectiveThreadRow = threadRow * TM + resIdxM;
            uint effectiveThreadCol = threadCol * TN + resIdxN;
            
            if (effectiveThreadRow < BM && effectiveThreadCol < BN) {
                result[(blockRow * BM + effectiveThreadRow) * N + (blockCol * BN + effectiveThreadCol)] = 
                    threadResults[resIdxM][resIdxN];
            }
        }
    }
}
