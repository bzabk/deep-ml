#include <cuda_runtime.h>
#define BLOCK_SIZE 256
__global__ void histogram_kernel(const int* input, int* hist,int n, int num_bins){
    __shared__ int localhist[BLOCK_SIZE];
    if (threadIdx.x < BLOCK_SIZE) {
        localhist[threadIdx.x] = 0;
    }
    __syncthreads();

    int globalidx = blockDim.x*blockIdx.x+threadIdx.x;
    if(globalidx < n){
        atomicAdd(&localhist[input[globalidx]], 1);
    }
    __syncthreads();
    if(threadIdx.x < num_bins){
        atomicAdd(&hist[threadIdx.x],localhist[threadIdx.x]);
    }

    
}

void solve(const int* input, int* hist, int n, int num_bins) {
    int* cuda_hist = nullptr;
    int* cuda_input = nullptr;

    cudaMalloc(&cuda_hist, sizeof(int)*num_bins);
    cudaMalloc(&cuda_input, sizeof(int)*n);

    cudaMemcpy(cuda_hist, hist, num_bins*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_input, input, n*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(cuda_hist, 0, num_bins*sizeof(int));
    int threadsperblock = 256;

    int blocks = (n+255)/256;
    histogram_kernel<<<blocks,threadsperblock>>>(cuda_input,cuda_hist,n,num_bins);

    cudaMemcpy(hist, cuda_hist, sizeof(int)*num_bins, cudaMemcpyDeviceToHost);
    cudaFree(cuda_hist);
    cudaFree(cuda_input);
    
}
