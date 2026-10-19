#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<chrono>

// Kiểm tra mã lỗi CUDA; dừng chương trình và chỉ ra vị trí khi có lỗi.
#define CUDA_CHECK(call){\
    cudaError_t err = call;\
    if (err != cudaSuccess){\
        printf("%s in %s at line %d!\n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(EXIT_FAILURE);\
    }\
}

__global__ void SumKernel(int n, float* d_A, float *d_B, float* d_C){
    // __global__: hàm chạy trên GPU, được CPU gọi bằng cú pháp <<<...>>>.
    // Mỗi thread xử lý một phần tử; ghép chỉ số block và thread thành chỉ số toàn cục.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Block cuối có thể có thread thừa nên phải kiểm tra giới hạn mảng.
    if(i<n) d_C[i]= d_A[i] + d_B[i];
}

void SumByGPU(int n, float* A, float *B, float* C){
    // Đo thời gian thực: gồm chuẩn bị GPU, cấp phát, sao chép, tính và giải phóng.
    // steady_clock là đồng hồ đơn điệu, không bị ảnh hưởng khi chỉnh giờ hệ thống.
    const auto begin = std::chrono::steady_clock::now();

    // d_ là con trỏ bộ nhớ GPU (device); A, B, C nằm trên CPU (host).
    float *d_A, *d_B, *d_C;
    int N = n*sizeof(float);

    // cudaMalloc nhận kích thước theo byte, không phải số phần tử.
    CUDA_CHECK(cudaMalloc((void**)&d_A, N));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N));
    
    // Chuyển hai mảng đầu vào từ RAM sang bộ nhớ GPU.
    CUDA_CHECK(cudaMemcpy(d_A, A, N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, N, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    // CUDA Events đo thời gian trên GPU, riêng phần kernel giữa hai mốc.
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Mỗi block có 256 thread; chia làm tròn lên để đủ thread cho n phần tử.
    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    CUDA_CHECK(cudaEventRecord(start));
    SumKernel<<<blocks, threadsPerBlock>>>(n, d_A, d_B, d_C);
    CUDA_CHECK(cudaGetLastError()); // Kiểm tra lỗi khởi chạy kernel.
    CUDA_CHECK(cudaEventRecord(stop));

    // Kernel chạy bất đồng bộ; chờ xong trước khi đọc thời gian và kết quả.
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpy(C, d_C, N, cudaMemcpyDeviceToHost));
    
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel execution time: %f ms\n", ms);

    // Giải phóng tài nguyên GPU sau khi đã lấy kết quả về CPU.
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    const auto end = std::chrono::steady_clock::now();
    const double totalMs = std::chrono::duration<double, std::milli>(end - begin).count();
    printf("Total execution time: %f ms\n", totalMs);
}

int main(){
    // Cùng dữ liệu với bài Python: A = B = [1, 2, ..., 100].
    int n = 100;
    float* h_A = (float*)malloc(n* sizeof(*h_A));
    float* h_B = (float*)malloc(n* sizeof(*h_B));
    float* h_C = (float*)malloc(n* sizeof(*h_C));
    for(int i=0; i<n; i++){
        *(h_A + i) = i+1;
        *(h_B + i) = i+1;
    }
    // h_ biểu thị bộ nhớ CPU; C sẽ chứa [2, 4, ..., 200] sau lời gọi.
    SumByGPU(n, h_A, h_B, h_C);
    free(h_A);
    free(h_B);
    // In 10 phần tử mỗi dòng. Phần in không nằm trong thời gian xử lý đã đo.
    for(int i=0; i<n; i++){
        printf("%8.2f",*(h_C + i));
        if ((i+1) % 10 == 0) printf("\n");
    }
    free(h_C);
    printf("\n");
    return 0;
}
