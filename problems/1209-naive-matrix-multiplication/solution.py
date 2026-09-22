#include <cuda_runtime.h>
#include <vector>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    // thread (row, col): if row<M && col<N, C[row*N+col] = sum_k A[row*K+k]*B[k*N+col]

    int tx = blockDim.x*blockIdx.x+threadIdx.x;
    int ty = blockDim.y*blockIdx.y+threadIdx.y;
    float sum = 0.0f;
    if(tx < N && ty < M){
        for(int k=0;k<K;k++){
            sum+=A[ty*K+k]*B[k*N+tx];
        }
        C[N*ty+tx] = sum;
    } 

}

    
    


std::vector<float> matmul(const std::vector<std::vector<float>>& A,
                          const std::vector<std::vector<float>>& B) {


    int a_height = A.size();
    int common_dim = A[0].size();
    int b_width = B[0].size();

    float* acuda = nullptr;
    float* bcuda = nullptr;
    float* ccuda = nullptr;


    size_t asize = a_height*common_dim*sizeof(float);
    size_t bsize = b_width*common_dim*sizeof(float);
    size_t csize = a_height*b_width*sizeof(float);

    std::vector<float> flatten_a(a_height*common_dim,0.0f);
    std::vector<float> flatten_b(b_width*common_dim,0.0f);
    std::vector<float> flatten_c(a_height*b_width,0.0f);

    for(int i=0;i<a_height;i++){
        for(int j=0;j<common_dim;j++){
            flatten_a[i*common_dim+j] = A[i][j];
        }
    }

    for(int i=0;i<common_dim;i++){
        for(int j=0;j<b_width;j++){
            flatten_b[i*b_width+j] = B[i][j];
        }
    }

    cudaMalloc(&acuda, asize);
    cudaMalloc(&bcuda, bsize);
    cudaMalloc(&ccuda, csize);

    cudaMemcpy(acuda, flatten_a.data(), asize, cudaMemcpyHostToDevice);
    cudaMemcpy(bcuda, flatten_b.data(), bsize, cudaMemcpyHostToDevice);
    dim3 threadpreblock(16,16);
    int blockx = (b_width+15)/16;
    int blocky = (a_height+15)/16;
    dim3 blocksize(blockx,blocky);
    matmul_kernel<<<blocksize,threadpreblock>>>(acuda,bcuda,ccuda,a_height,common_dim,b_width);

    cudaMemcpy(flatten_c.data(), ccuda, csize, cudaMemcpyDeviceToHost);


    return flatten_c;
}
