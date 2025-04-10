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

// Define vector load size (2 floats = 64 bits)
constant int VEC_SIZE = 2;

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
    
    // Create double-buffered shared memory tiles
    threadgroup float As[2][BM * BK];  // Double buffer for A
    threadgroup float Bs[2][BK * BN];  // Double buffer for B
    
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
    
    // Initialize current buffer index
    uint currentBuffer = 0;
    uint nextBuffer = 1;
    
    // Load first tile into shared memory
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA * VEC_SIZE) {
        if ((innerRowA + loadOffset) < BM && innerColA < K) {
            // Cast pointer to float2* and load 2 floats at once from global memory
            device const float2* vecPtr = reinterpret_cast<device const float2*>(&A[(innerRowA + loadOffset) * K + innerColA]);
            float2 vecA = *vecPtr;
            
            // Cast pointer to float2* and store 2 floats at once to shared memory
            threadgroup float2* smemPtr = reinterpret_cast<threadgroup float2*>(&As[currentBuffer][(innerRowA + loadOffset) * BK + innerColA]);
            *smemPtr = vecA;
        }
    }
    
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB * VEC_SIZE) {
        if ((innerRowB + loadOffset) < BK && (threadCol * TN + innerColB) < BN) {
            // Cast pointer to float2* and load 2 floats at once from global memory
            device const float2* vecPtr = reinterpret_cast<device const float2*>(&B[(innerRowB + loadOffset) * N + innerColB]);
            float2 vecB = *vecPtr;
            
            // Cast pointer to float2* and store 2 floats at once to shared memory
            threadgroup float2* smemPtr = reinterpret_cast<threadgroup float2*>(&Bs[currentBuffer][(innerRowB + loadOffset) * BN + innerColB]);
            *smemPtr = vecB;
        }
    }
    
    // Synchronize threads after initial load
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Loop over block tiles
    for (uint bkIdx = BK; bkIdx < K; bkIdx += BK) {
        // Start loading next tile while computing current tile
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA * VEC_SIZE) {
            if ((innerRowA + loadOffset) < BM && (bkIdx + innerColA) < K) {
                // Cast pointer to float2* and load 2 floats at once from global memory
                device const float2* vecPtr = reinterpret_cast<device const float2*>(&A[(innerRowA + loadOffset) * K + innerColA]);
                float2 vecA = *vecPtr;
                
                // Cast pointer to float2* and store 2 floats at once to shared memory
                threadgroup float2* smemPtr = reinterpret_cast<threadgroup float2*>(&As[nextBuffer][(innerRowA + loadOffset) * BK + innerColA]);
                *smemPtr = vecA;
            }
        }
        
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB * VEC_SIZE) {
            if ((innerRowB + loadOffset) < BK && (threadCol * TN + innerColB) < BN) {
                // Cast pointer to float2* and load 2 floats at once from global memory
                device const float2* vecPtr = reinterpret_cast<device const float2*>(&B[(innerRowB + loadOffset) * N + innerColB]);
                float2 vecB = *vecPtr;
                
                // Cast pointer to float2* and store 2 floats at once to shared memory
                threadgroup float2* smemPtr = reinterpret_cast<threadgroup float2*>(&Bs[nextBuffer][(innerRowB + loadOffset) * BN + innerColB]);
                *smemPtr = vecB;
            }
        }
        
        // Calculate dot products using current buffer
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load into registers with vectorized loads from shared memory
            for (uint i = 0; i < TM; i += VEC_SIZE) {
                if ((threadRow * TM + i) < BM) {
                    // Cast pointer to float2* and load 2 floats at once from shared memory
                    threadgroup const float2* smemPtr = reinterpret_cast<threadgroup const float2*>(&As[currentBuffer][(threadRow * TM + i) * BK + dotIdx]);
                    float2 vecM = *smemPtr;
                    regM[i] = vecM.x;
                    regM[i + 1] = vecM.y;
                } else {
                    regM[i] = 0.0f;
                    regM[i + 1] = 0.0f;
                }
            }
            
            for (uint i = 0; i < TN; i += VEC_SIZE) {
                if ((threadCol * TN + i) < BN) {
                    // Cast pointer to float2* and load 2 floats at once from shared memory
                    threadgroup const float2* smemPtr = reinterpret_cast<threadgroup const float2*>(&Bs[currentBuffer][dotIdx * BN + threadCol * TN + i]);
                    float2 vecN = *smemPtr;
                    regN[i] = vecN.x;
                    regN[i + 1] = vecN.y;
                } else {
                    regN[i] = 0.0f;
                    regN[i + 1] = 0.0f;
                }
            }
            
            // Compute results using register caches
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        
        // Synchronize before switching buffers
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Swap buffers
        currentBuffer = nextBuffer;
        nextBuffer = 1 - currentBuffer;
        
        // Advance pointers
        A += BK;
        B += BK * N;
    }
    
    // Process the last tile using current buffer
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        // Load into registers with vectorized loads from shared memory
        for (uint i = 0; i < TM; i += VEC_SIZE) {
            if ((threadRow * TM + i) < BM) {
                // Cast pointer to float2* and load 2 floats at once from shared memory
                threadgroup const float2* smemPtr = reinterpret_cast<threadgroup const float2*>(&As[currentBuffer][(threadRow * TM + i) * BK + dotIdx]);
                float2 vecM = *smemPtr;
                regM[i] = vecM.x;
                regM[i + 1] = vecM.y;
            } else {
                regM[i] = 0.0f;
                regM[i + 1] = 0.0f;
            }
        }
        
        for (uint i = 0; i < TN; i += VEC_SIZE) {
            if ((threadCol * TN + i) < BN) {
                // Cast pointer to float2* and load 2 floats at once from shared memory
                threadgroup const float2* smemPtr = reinterpret_cast<threadgroup const float2*>(&Bs[currentBuffer][dotIdx * BN + threadCol * TN + i]);
                float2 vecN = *smemPtr;
                regN[i] = vecN.x;
                regN[i + 1] = vecN.y;
            } else {
                regN[i] = 0.0f;
                regN[i + 1] = 0.0f;
            }
        }
        
        // Compute results using register caches
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
            }
        }
    }
    
    // Write results back to global memory with vectorized stores
    for (uint resIdxM = 0; resIdxM < TM; resIdxM += VEC_SIZE) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            if ((threadRow * TM + resIdxM) < BM && (threadCol * TN + resIdxN) < BN) {
                float2 vecResult;
                vecResult.x = threadResults[resIdxM * TN + resIdxN];
                vecResult.y = (resIdxM + 1 < TM) ? threadResults[(resIdxM + 1) * TN + resIdxN] : 0.0f;
                
                // Cast pointer to float2* and store 2 floats at once
                device float2* vecPtr = reinterpret_cast<device float2*>(&result[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN]);
                *vecPtr = vecResult;
            }
        }
    }
}
