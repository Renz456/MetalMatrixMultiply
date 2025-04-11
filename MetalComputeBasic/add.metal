/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
A shader that performs matrix multiplication using shared memory for better performance.
*/

#include <metal_stdlib>
#include <metal_logging>

using namespace metal;

// Define block size
constant int BLOCKSIZE = 2;  // Size of the block (16x16)

/// This is a Metal Shading Language (MSL) function that performs matrix multiplication on a GPU
/// using shared memory for better performance with optimized memory coalescing.
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
    // Calculate block indices
    uint blockRow = bid / ((N + BLOCKSIZE - 1) / BLOCKSIZE);
    uint blockCol = bid % ((N + BLOCKSIZE - 1) / BLOCKSIZE);
    
    // Calculate thread indices within block
    uint threadRow = tid / BLOCKSIZE;
    uint threadCol = tid % BLOCKSIZE;
    
    // Calculate global indices
    uint row = blockRow * BLOCKSIZE + threadRow;
    uint col = blockCol * BLOCKSIZE + threadCol;
    
    // Check if we're within bounds
    if (row >= M || col >= N) {
        return;
    }
    
    // Create shared memory tiles
    threadgroup float As[BLOCKSIZE * BLOCKSIZE];
    threadgroup float Bs[BLOCKSIZE * BLOCKSIZE];
    
    // Initialize accumulator
    float sum = 0.0f;
    
    os_log_default.log_error("tid: %u, bid: %u, gid: %u, threadRow: %u, threadCol: %u, blockRow: %u, blockCol: %u", 
                            tid, bid, gid, threadRow, threadCol, blockRow, blockCol);
    
    // Loop over blocks in K dimension
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // Have each thread load one of the elements in A & B
        // Make the threadCol the consecutive index to allow global memory access coalescing
        if (threadRow < BLOCKSIZE && threadCol < BLOCKSIZE) {
            if (row < M && (bkIdx + threadCol) < K) {
                As[threadRow * BLOCKSIZE + threadCol] = A[row * K + (bkIdx + threadCol)];
            } else {
                As[threadRow * BLOCKSIZE + threadCol] = 0.0f;
            }
            
            if ((bkIdx + threadRow) < K && col < N) {
                Bs[threadRow * BLOCKSIZE + threadCol] = B[(bkIdx + threadRow) * N + col];
            } else {
                Bs[threadRow * BLOCKSIZE + threadCol] = 0.0f;
            }
        }
        
        // Block threads in this block until cache is fully populated
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Execute the dot product on the currently cached block
        for (uint dotIdx = 0; dotIdx < BLOCKSIZE && (bkIdx + dotIdx) < K; ++dotIdx) {
            for (uint bIdx = 0; bIdx < BLOCKSIZE; ++bIdx) {
                sum += As[threadRow * BLOCKSIZE + dotIdx] * Bs[bIdx * BLOCKSIZE + threadCol];
            }
        }
        
        // Need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result back to global memory
    result[row * N + col] = sum;
}
