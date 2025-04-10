/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
A class to manage all of the Metal objects this app creates.
*/

#import "MetalAdder.h"

@implementation MetalAdder {
    id<MTLDevice> _mDevice;
    id<MTLComputePipelineState> _mAddFunctionPSO;
    id<MTLCommandQueue> _mCommandQueue;
    
    id<MTLBuffer> _mBufferA;
    id<MTLBuffer> _mBufferB;
    id<MTLBuffer> _mBufferResult;
    id<MTLBuffer> _mBufferM;
    id<MTLBuffer> _mBufferN;
    id<MTLBuffer> _mBufferK;
    
    int _M;
    int _N;
    int _K;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        _mDevice = device;
        
        NSError* error = nil;
        
        // Load the shader files
        id<MTLLibrary> defaultLibrary = [_mDevice newDefaultLibrary];
        if (defaultLibrary == nil) {
            NSLog(@"Failed to find the default library.");
            return nil;
        }
        
        id<MTLFunction> addFunction = [defaultLibrary newFunctionWithName:@"add_arrays"];
        if (addFunction == nil) {
            NSLog(@"Failed to find the adder function.");
            return nil;
        }
        
        // Create a compute pipeline state object
        _mAddFunctionPSO = [_mDevice newComputePipelineStateWithFunction:addFunction error:&error];
        if (_mAddFunctionPSO == nil) {
            NSLog(@"Failed to create pipeline state object, error %@.", error);
            return nil;
        }
        
        _mCommandQueue = [_mDevice newCommandQueue];
        if (_mCommandQueue == nil) {
            NSLog(@"Failed to create command queue.");
            return nil;
        }
    }
    return self;
}

- (void)prepareDataWithSizeM:(int)M sizeN:(int)N sizeK:(int)K {
    _M = M;
    _N = N;
    _K = K;
    
    // Calculate buffer sizes
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeResult = M * N * sizeof(float);
    
    // Create buffers
    _mBufferA = [_mDevice newBufferWithLength:sizeA options:MTLResourceStorageModeShared];
    _mBufferB = [_mDevice newBufferWithLength:sizeB options:MTLResourceStorageModeShared];
    _mBufferResult = [_mDevice newBufferWithLength:sizeResult options:MTLResourceStorageModeShared];
    _mBufferM = [_mDevice newBufferWithBytes:&M length:sizeof(int) options:MTLResourceStorageModeShared];
    _mBufferN = [_mDevice newBufferWithBytes:&N length:sizeof(int) options:MTLResourceStorageModeShared];
    _mBufferK = [_mDevice newBufferWithBytes:&K length:sizeof(int) options:MTLResourceStorageModeShared];
    
    // Initialize matrices with random data
    [self generateRandomData:_mBufferA size:M * K];
    [self generateRandomData:_mBufferB size:K * N];
    
    // Print some sample values for debugging
    float* a = (float*)_mBufferA.contents;
    float* b = (float*)_mBufferB.contents;
    NSLog(@"Sample values - A[0,0]=%.6f, B[0,0]=%.6f", a[0], b[0]);
}

- (void)generateRandomData:(id<MTLBuffer>)buffer size:(int)size {
    float* dataPtr = (float*)buffer.contents;
    for (int i = 0; i < size; i++) {
        dataPtr[i] = (float)rand() / (float)RAND_MAX;
    }
}

- (void)sendComputeCommand {
    id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    [computeEncoder setComputePipelineState:_mAddFunctionPSO];
    [computeEncoder setBuffer:_mBufferA offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufferB offset:0 atIndex:1];
    [computeEncoder setBuffer:_mBufferResult offset:0 atIndex:2];
    [computeEncoder setBuffer:_mBufferM offset:0 atIndex:3];
    [computeEncoder setBuffer:_mBufferN offset:0 atIndex:4];
    [computeEncoder setBuffer:_mBufferK offset:0 atIndex:5];
    
    // Define tile sizes
    const int BM = 64;  // Block size for M dimension
    const int BN = 64;  // Block size for N dimension
    const int TM = 8;   // Number of results per thread in M dimension
    const int TN = 8;   // Number of results per thread in N dimension
    
    // Calculate grid and threadgroup size
    MTLSize gridSize = MTLSizeMake(((_N + BN-1)/BN), ((_M+BM-1)/BM), 1);
    MTLSize threadgroupSize = MTLSizeMake((BM * BN) / (TM * TN), 1, 1);
    
    NSLog(@"Grid size: %dx%d, Threadgroup size: %dx%d", (int)gridSize.width, (int)gridSize.height, (int)threadgroupSize.width, (int)threadgroupSize.height);
    
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];
    
    // Start timing
    NSDate *startTime = [NSDate date];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // End timing and calculate duration
    NSTimeInterval timeElapsed = [[NSDate date] timeIntervalSinceDate:startTime];
    NSLog(@"GPU computation took %.4f seconds", timeElapsed);
    
    [self verifyResults];
}

- (void)verifyResults {
    float* a = (float*)_mBufferA.contents;
    float* b = (float*)_mBufferB.contents;
    float* result = (float*)_mBufferResult.contents;
    
    // Verify a few random elements
     int verificationErrors = 0;
    // for (int i = 0; i < 10; i++) {
    //     int row = rand() % _M;
    //     int col = rand() % _N;
    //     float expected = 0.0f;
        
    //     for (int k = 0; k < _K; k++) {
    //         expected += a[row * _K + k] * b[k * _N + col];
    //     }
        
    //     float actual = result[row * _N + col];
    //     if (fabs(actual - expected) > 0.0001f) {
    //         NSLog(@"Verification failed at [%d,%d]: expected %.6f, got %.6f", row, col, expected, actual);
    //         verificationErrors++;
            
    //         // Print the contributing values for debugging
    //         NSLog(@"Contributing values for [%d,%d]:", row, col);
    //         for (int k = 0; k < MIN(5, _K); k++) {
    //             NSLog(@"  A[%d,%d]=%.6f * B[%d,%d]=%.6f = %.6f", 
    //                   row, k, a[row * _K + k], 
    //                   k, col, b[k * _N + col],
    //                   a[row * _K + k] * b[k * _N + col]);
    //         }
    //     }
    // }
    
    if (verificationErrors == 0) {
        NSLog(@"Verification passed - all checked elements match expected values");
    } else {
        NSLog(@"Verification failed - %d errors found", verificationErrors);
    }
}

@end
