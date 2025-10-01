#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define CUDA_CHECK(call){\
    cudaError_t err = call;\
    if (err != cudaSuccess){\
        printf("%s in %s at line %d!\n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(EXIT_FAILURE);\
    }\
}

__global__ void SumKernel(int n, float* d_A, float *d_B, float* d_C){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n) d_C[i]= d_A[i] + d_B[i];
}

void SumByGPU(int n, float* A, float *B, float* C){
    clock_t begin = clock();

    float *d_A, *d_B, *d_C;
    int N = n*sizeof(float);

    cudaMalloc((void**)&d_A, N); 
    cudaMalloc((void**)&d_B, N);
    cudaMalloc((void**)&d_C, N);
    
    cudaMemcpy(d_A, A, N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    SumKernel<<<ceil(n/256.0), 256>>>(n, d_A, d_B, d_C);
    cudaEventRecord(stop);

    cudaMemcpy(C, d_C, N, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    clock_t end = clock();
    float time = (float)(end -begin)/CLOCKS_PER_SEC;
    printf("Total time: %f\n", time);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(){
    int n = 100;
    float* h_A = (float*)malloc(n* sizeof(*h_A));
    float* h_B = (float*)malloc(n* sizeof(*h_B));
    float* h_C = (float*)malloc(n* sizeof(*h_C));
    for(int i=0; i<n; i++){
        *(h_A + i) = i+1;
        *(h_B + i) = i+1;
    }
    SumByGPU(n, h_A, h_B, h_C);
    free(h_A);
    free(h_B);
    for(int i=0; i<n; i++){
        printf("%8.2f",*(h_C + i));
        if ((i+1) % 10 == 0) printf("\n");
    }
    free(h_C);
    printf("\n");
    return 0;
}