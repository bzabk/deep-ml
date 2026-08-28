#include <cuda_runtime.h>
#include <vector>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    // c[i] = a[i] + b[i], guarded by i < n
    int t = blockDim.x*blockIdx.x+threadIdx.x;
    if(t<n){
        c[t]=a[t]+b[t];
    }
}

std::vector<float> vector_add(const std::vector<float>& a, const std::vector<float>& b) {
    // allocate a, b, c on the device; copy a and b over (Host -> Device);
    // launch the kernel; copy c back; free memory
    int n = a.size();
    std::vector<float> dest(n);
    size_t vector_mem_size = n*sizeof(float);
    float* acuda;
    float* bcuda;
    float* ccuda;
    cudaMalloc(&acuda, vector_mem_size);
    cudaMalloc(&bcuda, vector_mem_size);
    cudaMalloc(&ccuda, vector_mem_size);

    cudaMemcpy(acuda, a.data(), vector_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(bcuda, b.data(), vector_mem_size, cudaMemcpyHostToDevice);

    int tpb = (n+256-1)/256;
    add_kernel<<<tpb,256>>>(acuda,bcuda,ccuda,n);

    cudaMemcpy(dest.data(), ccuda, vector_mem_size, cudaMemcpyDeviceToHost);
    return dest;
}
