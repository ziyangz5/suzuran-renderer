//
// Created by MatheusFaria from https://github.com/MatheusFaria/cuda_opengl_cmake/blob/master/src/cudahelperlib.cu
//

#ifndef SUZURANRENDERER_CUDAHELPERLIB_H
#define SUZURANRENDERER_CUDAHELPERLIB_H

#include <curand.h>
#include <curand_kernel.h>
#include <vector_types.h>

extern __host__ void cudaErrorCheck(cudaError_t err);
extern __device__ float4 pickRandomFloat4(curandState * randState);

#endif //SUZURANRENDERER_CUDAHELPERLIB_H
