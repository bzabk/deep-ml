#include <cuda_runtime.h>
#define BLOCK_SIZE 16
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    
    __shared__ float submatrix_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float submatrix_B[BLOCK_SIZE][BLOCK_SIZE];

    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int col = blockDim.x*blockIdx.x+threadIdx.x;
    int row = blockDim.y*blockIdx.y+threadIdx.y;

    float sum = 0.0f;

    for(int k=0;k<K;k+=BLOCK_SIZE){

        if( (k+thread_x)<K && row<M ){
            submatrix_A[thread_y][thread_x] = A[K*row+k+thread_x];
        }else{
            submatrix_A[thread_y][thread_x] = 0.0f;
        }

        if(col<N  && (k+thread_x)<K){
            submatrix_B[thread_y][thread_x] = B[N*(k+thread_y)+col];
        }else{
            submatrix_B[thread_y][thread_x] = 0.0f;
        }

        __syncthreads();

        for(int j=0;j<BLOCK_SIZE;j++){
            sum += submatrix_A[thread_y][j]*submatrix_B[j][thread_x];
        }

        __syncthreads();
    }
    if(row<M && col<N){
        C[N*row+col] = sum;
    }
}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {

    float* cuda_a = nullptr;
    float* cuda_b = nullptr;
    float* cuda_c = nullptr;

    size_t s_a = M*K*sizeof(float);
    size_t s_b = N*K*sizeof(float);
    size_t s_c = N*M*sizeof(float);

    cudaMalloc(&cuda_a, s_a);
    cudaMalloc(&cuda_b, s_b);
    cudaMalloc(&cuda_c, s_c);

    cudaMemcpy(cuda_a, A, s_a, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, B, s_b, cudaMemcpyHostToDevice);
    cudaMemset(cuda_c, 0, s_c);

    dim3 threads(BLOCK_SIZE,BLOCK_SIZE);
    dim3 blocks((N+BLOCK_SIZE-1)/BLOCK_SIZE,(M+BLOCK_SIZE-1)/BLOCK_SIZE);

    matmul_kernel<<<blocks,threads>>>(cuda_a,cuda_b,cuda_c,M,N,K);

    cudaMemcpy(C, cuda_c, s_c, cudaMemcpyDeviceToHost);

    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);

    
}