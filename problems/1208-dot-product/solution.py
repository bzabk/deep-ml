#include <cuda_runtime.h>
#include <vector>

__global__ void dot_kernel(const float* a, const float* b, float* out, int n) {
    float local_out = 0;
    int t = blockDim.x*blockIdx.x+threadIdx.x;
    if(t<n){
        local_out += a[t]*b[t];
    }
    
    atomicAdd(out,local_out);
}

float dot_product(const std::vector<float>& a, const std::vector<float>& b) {


    float* src_vector_a = nullptr;
    float* src_vector_b = nullptr;
    float* res = nullptr;
    size_t s = a.size()*sizeof(float);
    
    cudaMalloc(&src_vector_a, s);
    cudaMalloc(&src_vector_b, s);
    cudaMalloc(&res,s);

    
    cudaMemset(res,0,sizeof(float));
    cudaMemcpy(src_vector_a, a.data(), s, cudaMemcpyHostToDevice);
    cudaMemcpy(src_vector_b, b.data(), s, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (a.size() + threads_per_block - 1) / threads_per_block;

    dot_kernel<<<blocks_per_grid,threads_per_block>>>(src_vector_a,src_vector_b,res,a.size());
    float res_host = 0.0f;
    cudaMemcpy(&res_host, res, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(src_vector_a);
    cudaFree(src_vector_b);
    cudaFree(res);


    return res_host;
}
