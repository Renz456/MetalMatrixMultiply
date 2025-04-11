/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
An app that performs matrix multiplication on a GPU.
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "MetalAdder.h"

// This is the C version of the function that the sample
// implements in Metal Shading Language.
void matrix_multiply_cpu(const float* matrixA,
                        const float* matrixB,
                        float* result,
                        int M, int N, int K)
{
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += matrixA[row * K + k] * matrixB[k * N + col];
            }
            result[row * N + col] = sum;
        }
    }
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            NSLog(@"Failed to create Metal device");
            return 1;
        }
        
        NSLog(@"Using device: %@", device.name);
        
        // Create the custom object used to encapsulate the Metal code.
        MetalAdder* adder = [[MetalAdder alloc] initWithDevice:device];
        if (adder == nil) {
            NSLog(@"Failed to create MetalAdder object");
            return 1;
        }
        
        // Set matrix dimensions (M x K) * (K x N) = (M x N)
        const int M = 4;  // rows of A
        const int K = 4;  // cols of A, rows of B
        const int N = 4;  // cols of B
        
        // Create buffers to hold data
        [adder prepareDataWithSizeM:M sizeN:N sizeK:K];
        
        // Send a command to the GPU to perform the calculation.
        [adder sendComputeCommand];
        
        NSLog(@"Matrix multiplication completed");
    }
    return 0;
}
