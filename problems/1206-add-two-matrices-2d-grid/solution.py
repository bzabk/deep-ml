#include <cuda_runtime.h>
#include <vector>

__global__ void matadd_kernel(const float* A, const float* B, float* C, int rows, int cols) {
    int tx = blockDim.x*blockIdx.x+threadIdx.x;
    int ty = blockDim.y*blockIdx.y+threadIdx.y;

    if(tx < cols && ty < rows){
        int index = ty*cols+tx;
        C[index] = A[index] + B[index];
    }
}

std::vector<float> matrix_add(const std::vector<std::vector<float>>& A,
                              const std::vector<std::vector<float>>& B) {
    
    float* src_data_a = nullptr;
    float* src_data_b = nullptr;
    float* res = nullptr;
    int rows = A.size();
    int cols = A[0].size();
    int num_elements = A.size()*A[0].size();
    size_t s = num_elements*sizeof(float);

    std::vector<float> flatten_a(num_elements,0);
    std::vector<float> flatten_b(num_elements,0);
    std::vector<float> flatten_c(num_elements,0);

    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            flatten_a[i*cols+j] = A[i][j];
            flatten_b[i*cols+j] = B[i][j];
        }
    }

    cudaMalloc(&src_data_a, s);
    cudaMalloc(&src_data_b, s);
    cudaMalloc(&res, s);

    cudaMemcpy(src_data_a, flatten_a.data(), s, cudaMemcpyHostToDevice);
    cudaMemcpy(src_data_b, flatten_b.data(), s, cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks((A[0].size()+15)/16,
                (A.size()+15)/16);
    
    matadd_kernel<<<blocks,threads>>>(src_data_a,src_data_b,res,A.size(),A[0].size());


    cudaMemcpy(flatten_c.data(), res, s, cudaMemcpyDeviceToHost);

    cudaFree(src_data_a);
    cudaFree(src_data_b);
    cudaFree(res);

    return flatten_c;
}
