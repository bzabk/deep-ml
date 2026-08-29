#include <cuda_runtime.h>
#include <vector>

__global__ void scale_kernel(const float* x, float a, float* y, int n) {
    int t = blockDim.x*blockIdx.x+threadIdx.x;
    if(t<n){
        y[t]=x[t]*a;
    }
}

std::vector<float> scalar_multiply(const std::vector<float>& x, float a) {
    int n = x.size();
    std::vector<float> dest(n);
    size_t sizev = n*sizeof(float);

    float* xcuda;
    float* ycuda;
    cudaMalloc(&ycuda, sizev);
    cudaMalloc(&xcuda, sizev);
    cudaMemcpy(xcuda, x.data(), sizev, cudaMemcpyHostToDevice);    
    
    int blocks = (n+256-1)/256;
    scale_kernel<<<blocks,256>>>(xcuda,a,ycuda,n);
    cudaMemcpy(dest.data(), ycuda, sizev, cudaMemcpyDeviceToHost);
    return dest;
}
