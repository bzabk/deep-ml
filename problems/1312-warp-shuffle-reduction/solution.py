#include <cuda_runtime.h>
#define BLOCK_SIZE 32
#define WARP_MASK 0xFFFFFFFF
__global__ void kernel(const float* x, float* out, int n){

    int laneId = threadIdx.x % 32;
    float localsum=0.0f;
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    if(tid < n){
        localsum = x[tid];
    }



    localsum += __shfl_down_sync(0xffffffff, localsum, 16);
    localsum += __shfl_down_sync(0xffffffff, localsum, 8);
    localsum += __shfl_down_sync(0xffffffff, localsum, 4);
    localsum += __shfl_down_sync(0xffffffff, localsum, 2);
    localsum += __shfl_down_sync(0xffffffff, localsum, 1);


    if(laneId==0){
        atomicAdd(out, localsum);
    }


}

void solve(const float* x, float* out, int n) {
    
    float* cuda_x = nullptr;
    float* cuda_out = nullptr;

    size_t s = n*sizeof(float);

    cudaMalloc(&cuda_out, sizeof(float));
    cudaMalloc(&cuda_x, s);

    cudaMemset(cuda_out, 0, sizeof(float));
    cudaMemcpy(cuda_x, x, s, cudaMemcpyHostToDevice);

    int threads_per_block = BLOCK_SIZE;
    int blocks = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

    kernel<<<blocks,threads_per_block>>>(cuda_x,cuda_out,n);

    cudaMemcpy(out, cuda_out, sizeof(float), cudaMemcpyDeviceToHost);

}
