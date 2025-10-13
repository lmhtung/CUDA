#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#define BLOCK_WIDTH 16


float* convert2DTo1D(float **a, int r, int c){
    float *b = (float*)malloc(r*c*sizeof(*b));
    for(int i =0; i<r; i++){
        for(int j = 0; j<c; j++){
            b[i * c + j] = a[i][j];
        }
    }
    return b;
}


__global__ void mulMatKernel(int r1,int c1r2, int c2, float *d_m1, float *d_m2, float *d_m3){
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < r1 && c < c2){
        float result = 0;
        for (int i=0;i < c1r2;i++){
            result += d_m1[r * c1r2 + i] * d_m2[c2 * i + c];
        }
        d_m3[r * c2 + c] = result;
    }
}

void computeMul(int r1,int c1r2, int c2, float *m1, float *m2, float *m3){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float *d_m1, *d_m2, *d_m3;
    cudaMalloc((void**)&d_m1, r1*c1r2* sizeof(*d_m1));
    cudaMalloc((void**)&d_m2, c1r2*c2* sizeof(*d_m2));
    cudaMalloc((void**)&d_m3, r1*c2* sizeof(*d_m3));

    cudaMemcpy(d_m1, m1, r1*c1r2* sizeof(*d_m1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, m2, c1r2*c2* sizeof(*d_m2), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 gridSize((c2 -1)/BLOCK_WIDTH +1 , (r1 -1)/BLOCK_WIDTH +1);
    cudaEventRecord(start);
    mulMatKernel<<<gridSize, blockSize>>>(r1, c1r2, c2, d_m1, d_m2, d_m3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(m3, d_m3 , r1*c2* sizeof(*d_m3), cudaMemcpyDeviceToHost);

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_m3);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %f ms\n", ms);
}

int main(){
    int r = 2;
    clock_t begin = clock(); 
    float h_a[r*r] = {1,2,
                    3,4};
    float h_b[r*r] = {1,2,
                    3,4};
    float h_c[r*r];
    computeMul(r,r,r,h_a, h_b, h_c);
    clock_t end = clock();
    float time = (float)(end - begin)/CLOCKS_PER_SEC;
    printf("Total execution time: %f ms\n", time*1000);
    for (int i=0;i<r*r;i++){
        printf("%.2f\t", h_c[i]);
    }
    printf("\n");
}
