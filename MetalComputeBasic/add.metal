/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
A shader that multiplies two matrices.
*/

#include <metal_stdlib>
using namespace metal;

// Define tile size for shared memory
constant int TILE_SIZE = 16;
// Define number of results each thread computes
constant int RESULTS_PER_THREAD = 1;

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
    if (gid.x >= M || gid.y * RESULTS_PER_THREAD >= N) {
        return;
    }
    
    // Create shared memory tiles
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];
    
    // Allocate thread-local cache for results in register file
    float threadResults[RESULTS_PER_THREAD] = {0.0};
    
    // Calculate global indices
    int globalRow = gid.x;
    int globalCol = gid.y * RESULTS_PER_THREAD;
    
    // Loop over block tiles
    for (int bkIdx = 0; bkIdx < K; bkIdx += TILE_SIZE) {
        // Calculate indices for loading into shared memory
        int innerRowA = tid.x;
        int innerColA = tid.y;
        int innerRowB = tid.x;
        int innerColB = tid.y;
        
        // Calculate global indices for loading from global memory
        int globalRowA = globalRow;
        int globalColA = bkIdx + innerColA;
        int globalRowB = bkIdx + innerRowB;
        int globalColB = globalCol + innerColB;
        
        // Load data into shared memory
        if (bkIdx + tid.y < K) {
            // Load from matrix A: row gid.x, column bkIdx + tid.y
            // This ensures each thread in a threadgroup loads from the same row of A
            As[tid.x][tid.y] = A[gid.x * K + (bkIdx + tid.y)];

            // Load from matrix B: row bkIdx + tid.x, column gid.y
            // This ensures each thread in a threadgroup loads from the same column of B
            // Bs[tid.x][tid.y] = B[(bkIdx + tid.x) * N + gid.y];
        } else {
            // Pad with zeros if we're beyond the matrix dimensions
            As[tid.x][tid.y] = 0.0f;
            // Bs[tid.x][tid.y] = 0.0f;
        }
        
        // Load from matrix B for each result this thread will compute
        for (int r = 0; r < RESULTS_PER_THREAD; r++) {
            if (globalRowB < K && (globalColB + r) < N) {
                Bs[innerRowB][innerColB] = B[(bkIdx + tid.x + r) * N + gid.y];
            } else {
                Bs[innerRowB][innerColB] = 0.0f;
            }
        }
        
        // Synchronize threads to ensure shared memory is loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Calculate per-thread results
        for (int dotIdx = 0; dotIdx < TILE_SIZE; ++dotIdx) {
            // Cache the B value to reuse it for all results
            float Btmp = Bs[dotIdx][tid.y];
            
            // Compute dot product for each result
            for (int resIdx = 0; resIdx < RESULTS_PER_THREAD; ++resIdx) {
                threadResults[resIdx] += As[tid.x][dotIdx] * Btmp;
            }
        }
        
        // Synchronize threads before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write results back to global memory
    for (int r = 0; r < RESULTS_PER_THREAD; r++) {
        int col = globalCol + r;
        if (col < N) {
            result[globalRow * N + col] = threadResults[r];
        }
    }
}
