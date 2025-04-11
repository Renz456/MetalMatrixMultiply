/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
A class to manage all of the Metal objects this app creates.
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

@interface MetalAdder : NSObject
- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (void)prepareDataWithSizeM:(int)M sizeN:(int)N sizeK:(int)K;
- (void)sendComputeCommand;
- (void)runMPSMatrixMultiplication;
@end

NS_ASSUME_NONNULL_END
