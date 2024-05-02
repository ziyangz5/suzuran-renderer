#include <cuda_runtime_api.h>
#include "cutil.h"
#include "cutil_inline_runtime.h"
#include "Postprocess.h"
#include <fstream>

#define USE_SHARED_MEM 0

#define FILTER_SIZE (5*5) // 5x5 kernel filter
#define BLOCK_SIZE 16     // block size

__constant__ float kernelFilter_device[FILTER_SIZE];
__constant__ int indexOffsetsU_device[25];
__constant__ int indexOffsetsV_device[25];
__constant__ float invScale_device;
__constant__ float offset_device;

texture<float4, cudaTextureType2D, cudaReadModeElementType> texRef;
texture<float1, cudaTextureType2D, cudaReadModeElementType> texDepthRef;
template< typename R, typename T >
__device__ R Clamp( T value, T min, T max )
{
    if ( value < min )
    {
        return (R)min;
    }
    else if ( value > max )
    {
        return (R)max;
    }
    else
    {
        return (R)value;
    }
}

__global__ void cuConvert8uC4To32fC4Kernel(const uchar4* src, size_t src_stride,
                                           float4* dst, size_t dst_stride, int width, int height)
{
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bw = blockDim.x;
    unsigned int bh = blockDim.y;
    // Non-normalized U, V coordinates of input texture for current thread.
    unsigned int u = ( bw * blockIdx.x ) + tx;
    unsigned int v = ( bh * blockIdx.y ) + ty;
    if ( u > width || v > height ) return;
    unsigned int index = ( v * width ) + u;


    {
        uchar4 val = src[index];
        float4 res;
        res.x = (float)val.x / 255.0f;
        res.y = (float)val.y / 255.0f;
        res.z = (float)val.z / 255.0f;
        res.w = (float)val.w;
        dst[index] = res;
    }
}

float4* g_dstBuffer = NULL;
float1* g_dstDepthBuffer = NULL;
size_t g_BufferSize = 0;
size_t g_DepthBufferSize = 0;
void deleter(void *arg){};


torch::Tensor CreateTensorFromGLViaCUDA( cudaGraphicsResource_t& src, unsigned int width, unsigned int height)
{
    cudaGraphicsResource_t resources[1] = { src };

    // Map the resources so they can be used in the kernel.
    cutilSafeCall( cudaGraphicsMapResources( 1, resources ) );

    cudaArray* srcArray;

    // Get a device pointer to the OpenGL buffers
    cutilSafeCall( cudaGraphicsSubResourceGetMappedArray( &srcArray, src, 0, 0 ) );

    // Map the source texture to a texture reference.
    cutilSafeCall( cudaBindTextureToArray( texRef, srcArray ) );

    // Destination buffer to store the result of the postprocess effect.
    size_t bufferSize = width * height * sizeof(float4);
    if ( g_BufferSize != bufferSize )
    {
        if ( g_dstBuffer != NULL )
        {
            cudaFree( g_dstBuffer );
        }
        // Only re-allocate the global memory buffer if the screen size changes,
        // or it has never been allocated before (g_BufferSize is still 0)
        g_BufferSize = bufferSize;
        cutilSafeCall( cudaMalloc( &g_dstBuffer, g_BufferSize ) );
    }

    // Copy the destination back to the source array
    cutilSafeCall( cudaMemcpy2DFromArray(g_dstBuffer,sizeof(float4)*width,srcArray,0,0,sizeof(float4)*width,height,cudaMemcpyDeviceToDevice));

    std::vector<int64_t> dims = {width, height, 4};
    long long step = width * sizeof(float4) / sizeof(float);
    std::vector<int64_t> strides = {step, 4, 1};
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    torch::Tensor result = torch::from_blob(g_dstBuffer, dims,strides, deleter,options).clone();

    // Unbind the texture reference
    cutilSafeCall( cudaUnbindTexture( texRef ) );

    // Unmap the resources again so the texture can be rendered in OpenGL
    cutilSafeCall( cudaGraphicsUnmapResources( 1, resources ) );

    return result;
}

torch::Tensor CreateDepthTensorFromGLViaCUDA( cudaGraphicsResource_t& src, unsigned int width, unsigned int height)
{
    cudaGraphicsResource_t resources[1] = { src };

    // Map the resources so they can be used in the kernel.
    cutilSafeCall( cudaGraphicsMapResources( 1, resources ) );

    cudaArray* srcArray;

    // Get a device pointer to the OpenGL buffers
    cutilSafeCall( cudaGraphicsSubResourceGetMappedArray( &srcArray, src, 0, 0 ) );

    // Map the source texture to a texture reference.
    cutilSafeCall( cudaBindTextureToArray( texDepthRef, srcArray ) );

    // Destination buffer to store the result of the postprocess effect.
    size_t bufferSize = width * height * sizeof(float1);
    if ( g_DepthBufferSize != bufferSize )
    {
        if ( g_dstDepthBuffer != NULL )
        {
            cudaFree( g_dstDepthBuffer );
        }
        // Only re-allocate the global memory buffer if the screen size changes,
        // or it has never been allocated before (g_dstDepthBuffer is still 0)
        g_DepthBufferSize = bufferSize;
        cutilSafeCall( cudaMalloc( &g_dstDepthBuffer, g_DepthBufferSize ) );
    }

    // Copy the destination back to the source array
    cutilSafeCall( cudaMemcpy2DFromArray(g_dstDepthBuffer,sizeof(float1)*width,srcArray,0,0,sizeof(float1)*width,height,cudaMemcpyDeviceToDevice));

    std::vector<int64_t> dims = {width, height, 1};
    long long step = width * sizeof(float1) / sizeof(float);
    std::vector<int64_t> strides = {step, 1, 1};
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    torch::Tensor result = torch::from_blob(g_dstDepthBuffer, dims,strides, deleter,options).clone();

    // Unbind the texture reference
    cutilSafeCall( cudaUnbindTexture( texDepthRef ) );

    // Unmap the resources again so the texture can be rendered in OpenGL
    cutilSafeCall( cudaGraphicsUnmapResources( 1, resources ) );

    return result;
}

float* MoveDataFromTensorToGLTexture( cudaGraphicsResource_t& dst,torch::Tensor& data, unsigned int width, unsigned int height) {
    float *resultBuffer;
    size_t bufferSize = width * height * sizeof(float);
    cutilSafeCall(cudaMalloc(&resultBuffer, bufferSize));


    resultBuffer = data.data_ptr<float>();

    cudaGraphicsResource_t resources[1] = {dst};

    // Map the resources so they can be used in the kernel.
    cutilSafeCall(cudaGraphicsMapResources(1, resources));

    cudaArray *dstArray;

    // Get a device pointer to the OpenGL buffers
    cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&dstArray, dst, 0, 0));

    // Destination buffer to store the result of the postprocess effect.

    cudaThreadSynchronize();
    cutilSafeCall( cudaMemcpyToArray( dstArray, 0, 0, resultBuffer, bufferSize, cudaMemcpyDeviceToDevice ) );

    cudaMemcpyToArray(dstArray, 0, 0, resultBuffer, bufferSize, cudaMemcpyDeviceToDevice);
    cudaFree(resultBuffer);
    cutilSafeCall(cudaGraphicsUnmapResources(1, resources));
    return resultBuffer;
}

