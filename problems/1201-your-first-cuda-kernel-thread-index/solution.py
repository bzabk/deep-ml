#include <cuda_runtime.h>
#include <vector>

__global__ void index_kernel(int* out, int n) {
    // Compute this thread's global index and, if it is < n, write it to out.
    int t = blockDim.x*blockIdx.x+threadIdx.x;
    if(t<n){
        out[t]=t;
    }
}

std::vector<int> global_thread_indices(int n) {
    // 1. allocate device memory for n ints
    // 2. launch the kernel with enough threads to cover n
    // 3. copy the result back to the host and return it
    std::vector<int> dest(n);
    int* dout;
    cudaMalloc((void**)&dout, n*sizeof(n));
    int tpb = (n+256-1)/256;
    index_kernel<<<tpb,256>>>(dout,n);
    cudaMemcpy(dest.data(), dout, n * sizeof(int), cudaMemcpyDeviceToHost);


    return dest;
}
