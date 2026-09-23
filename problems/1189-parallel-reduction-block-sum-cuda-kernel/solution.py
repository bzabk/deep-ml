#include <cuda_runtime.h>
#define BLOCK_SIZE 256
__global__ void kernel(const float* input, float* out, int N) {
    // TODO: block-wise reduction using shared memory
    __shared__ float A[BLOCK_SIZE];
    int tx = blockDim.x*blockIdx.x+threadIdx.x;
    if(tx < N){
        A[threadIdx.x] = input[tx];
    }else{
        A[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for(int stride = blockDim.x /2; stride>0;stride>>=1){
        if(threadIdx.x < stride){
            A[threadIdx.x] += A[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x==0){
        atomicAdd(out, A[0]);
    }
}

void solve(const float* input, float* output, int N) {
    // TODO: allocate device memory, copy in, launch kernel, copy out, free

    float* cuda_input = nullptr;
    float* cuda_res = nullptr;
    size_t s = N*sizeof(float);
    cudaMalloc(&cuda_input, s);
    cudaMalloc(&cuda_res, sizeof(float));
    cudaMemcpy(cuda_input, input, s, cudaMemcpyHostToDevice);
    cudaMemset(cuda_res, 0, sizeof(float));

    int threadsperblock = 256;

    kernel<<<(N+255)/256,threadsperblock>>>(cuda_input,cuda_res,N);

    cudaMemcpy(output, cuda_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(cuda_input);
    cudaFree(cuda_res);
}