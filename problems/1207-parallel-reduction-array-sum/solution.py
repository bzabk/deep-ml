#include <cuda_runtime.h>
#include <vector>
#define BLOCK_SIZE 256
__global__ void sum_kernel(const float* x, float* out, int n) {
    int globalidx = blockDim.x*blockIdx.x+threadIdx.x;
    __shared__ float localscore[BLOCK_SIZE];

    if(globalidx < n){
        localscore[threadIdx.x] = x[globalidx];
    }else{
        localscore[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for(int stride = blockDim.x/2;stride>0;stride>>=1){
        if(threadIdx.x < stride){
            localscore[threadIdx.x] +=localscore[threadIdx.x+stride];
        }
        __syncthreads();
    }




    
    if(threadIdx.x==0){
        atomicAdd(out, localscore[0]);
    }




}

float array_sum(const std::vector<float>& x) {


    float* src_cuda = nullptr;
    float* res = nullptr;
    size_t s = x.size()*sizeof(float);
    cudaMalloc(&res,sizeof(float));
    cudaMalloc(&src_cuda, s);
    cudaMemset(res, 0, sizeof(float));

    cudaMemcpy(src_cuda, x.data(), s, cudaMemcpyHostToDevice);
    int threadperblocks = 256;
    int blocks = (x.size()+255)/256;
    sum_kernel<<<blocks,threadperblocks>>>(src_cuda,res,x.size());

    float ans = 0.0f;
    cudaMemcpy(&ans, res, sizeof(float), cudaMemcpyDeviceToHost);
    return ans;
}
