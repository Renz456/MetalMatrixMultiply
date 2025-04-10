/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
A shader that multiplies two matrices.
*/

#include <metal_stdlib>
using namespace metal;

// Define tile size for shared memory
constant int TILE_SIZE = 16;

/// This is a Metal Shading Language (MSL) function that performs matrix multiplication on a GPU
/// using shared memory for better performance.
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
    
    // Create shared memory tiles
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Calculate global indices for this tile
        int tileStartK = tile * TILE_SIZE;
        
        // Load data into shared memory
        if (tileStartK + tid.y < K) {
            // Load from matrix A: row gid.x, column tileStartK + tid.y
            As[tid.x][tid.y] = A[gid.x * K + (tileStartK + tid.y)];
            
            // Load from matrix B: row tileStartK + tid.x, column gid.y
            Bs[tid.x][tid.y] = B[(tileStartK + tid.x) * N + gid.y];
        } else {
            // Pad with zeros if we're beyond the matrix dimensions
            As[tid.x][tid.y] = 0.0f;
            Bs[tid.x][tid.y] = 0.0f;
        }
        
        // Synchronize threads to ensure shared memory is loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product for this tile
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[tid.x][i] * Bs[i][tid.y];
        }
        
        // Synchronize threads before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result back to global memory
    result[gid.x * N + gid.y] = sum;
}
