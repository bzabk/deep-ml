#include <cuda_runtime.h>
#include <vector>

__global__ void relu_kernel(const float* x, float* out, int n) {
    int t = blockDim.x*blockIdx.x+threadIdx.x;
    if(t<n){
        out[t] = fmaxf(0.0f,x[t]);
    }
}

std::vector<float> relu(const std::vector<float>& x) {
    int s = x.size()*sizeof(float);
    std::vector<float> result(x.size());
    float* input;
    float* output;

    cudaMalloc(&input, s);
    cudaMalloc(&output, s);

    cudaMemcpy(input, x.data(), s, cudaMemcpyHostToDevice);
    int blocks = (x.size() +255)/256;
    relu_kernel<<<blocks,256>>>(input,output,x.size());
    cudaMemcpy(result.data(), output, s, cudaMemcpyDeviceToHost);
    cudaFree(input);
    cudaFree(output);
    return result;
}
