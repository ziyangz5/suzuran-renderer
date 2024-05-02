#include <torch/torch.h>
torch::Tensor CreateTensorFromGLViaCUDA( cudaGraphicsResource_t& src, unsigned int width, unsigned int height);
torch::Tensor CreateDepthTensorFromGLViaCUDA( cudaGraphicsResource_t& src, unsigned int width, unsigned int height);
float * MoveDataFromTensorToGLTexture( cudaGraphicsResource_t& dst,torch::Tensor& data, unsigned int width, unsigned int height);
、