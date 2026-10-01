#include <cuda_runtime.h>
#define BLOCK_SIZE 16
__global__ void kernel(const float* input, const float* bias, float* output, int rows, int cols) {
    
    int col = blockDim.x*blockIdx.x+threadIdx.x;
    int row = blockDim.y*blockIdx.y+threadIdx.y;

    if(row < rows && col < cols){
        output[cols*row+col] = max(0.0f,input[cols*row+col]+bias[col]);
    }
}

void solve(const float* input, const float* bias, float* output, int rows, int cols) {
    

    float* cuda_input = nullptr;
    float* cuda_bias = nullptr;
    float* cuda_output = nullptr;

    size_t s_input = (size_t)rows*cols*sizeof(float);
    size_t s_bias = (size_t)cols*sizeof(float);

    cudaMalloc(&cuda_input, s_input);
    cudaMalloc(&cuda_bias, s_bias);
    cudaMalloc(&cuda_output, s_input);

    cudaMemcpy(cuda_input, input, s_input, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_bias, bias, s_bias, cudaMemcpyHostToDevice);


    dim3 blocks(BLOCK_SIZE,BLOCK_SIZE);
    dim3 grid((cols+BLOCK_SIZE-1)/BLOCK_SIZE,
                (rows+BLOCK_SIZE-1)/BLOCK_SIZE);

    kernel<<<grid,blocks>>>(cuda_input,cuda_bias,cuda_output,rows,cols);


    cudaMemcpy(output, cuda_output, s_input, cudaMemcpyDeviceToHost);

    cudaFree(cuda_input);
    cudaFree(cuda_bias);
    cudaFree(cuda_output);
}