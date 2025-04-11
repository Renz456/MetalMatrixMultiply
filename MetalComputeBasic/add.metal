/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
A shader that performs naive matrix multiplication without any optimizations.
*/

#include <metal_stdlib>
#include <metal_logging>

using namespace metal;

/// This is a Metal Shading Language (MSL) function that performs naive matrix multiplication on a GPU.
/// Each thread computes a single element of the result matrix.
kernel void add_arrays(device const float* A,
                      device const float* B,
                      device float* result,
                      device const int& M [[buffer(3)]],
                      device const int& N [[buffer(4)]],
                      device const int& K [[buffer(5)]],
                      uint gid [[thread_position_in_grid]])
{
    // Calculate row and column indices from the global thread ID
    uint row = gid / N;
    uint col = gid % N;
    
    // Check if we're within bounds
    if (row >= M || col >= N) {
        return;
    }
    
    // Compute the dot product for this element
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    // Store the result
    result[row * N + col] = sum;
}
