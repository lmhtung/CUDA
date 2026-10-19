#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<chrono>
// Block 2 chiều gồm 16 x 16 = 256 thread.
#define BLOCK_WIDTH 16

// Kiểm tra từng lệnh CUDA để tránh in kết quả chưa hợp lệ khi GPU gặp lỗi.
#define CUDA_CHECK(call) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "%s in %s at line %d!\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Trải ma trận thành mảng liên tục theo hàng: a[i][j] -> b[i * c + j].
// Hàm minh họa này chưa được main sử dụng vì dữ liệu đã là mảng 1 chiều.
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
    // A có kích thước r1 x c1r2; B có kích thước c1r2 x c2.
    // Mỗi thread tính một ô C[r, c] của ma trận kết quả r1 x c2.
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    // Bỏ qua thread nằm ngoài ma trận ở các block biên.
    if(r < r1 && c < c2){
        float result = 0;
        // Tích vô hướng hàng r của A và cột c của B.
        for (int i=0;i < c1r2;i++){
            result += d_m1[r * c1r2 + i] * d_m2[c2 * i + c];
        }
        d_m3[r * c2 + c] = result;
    }
}

void computeMul(int r1,int c1r2, int c2, float *m1, float *m2, float *m3){
    // Events bao quanh kernel để chỉ đo phần tính toán trên GPU.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // d_ chỉ bộ nhớ GPU; m1, m2, m3 là các mảng trên CPU.
    float *d_m1, *d_m2, *d_m3;
    CUDA_CHECK(cudaMalloc((void**)&d_m1, r1*c1r2* sizeof(*d_m1)));
    CUDA_CHECK(cudaMalloc((void**)&d_m2, c1r2*c2* sizeof(*d_m2)));
    CUDA_CHECK(cudaMalloc((void**)&d_m3, r1*c2* sizeof(*d_m3)));

    // Sao chép hai ma trận đầu vào lên GPU, kích thước tính bằng byte.
    CUDA_CHECK(cudaMemcpy(d_m1, m1, r1*c1r2* sizeof(*d_m1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m2, m2, c1r2*c2* sizeof(*d_m2), cudaMemcpyHostToDevice));

    // Grid phủ đủ số cột theo trục x và số hàng theo trục y, chia làm tròn lên.
    dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 gridSize((c2 -1)/BLOCK_WIDTH +1 , (r1 -1)/BLOCK_WIDTH +1);
    CUDA_CHECK(cudaEventRecord(start));
    mulMatKernel<<<gridSize, blockSize>>>(r1, c1r2, c2, d_m1, d_m2, d_m3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    // Chờ kernel kết thúc vì lệnh <<<...>>> trả về trước khi GPU tính xong.
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Đưa kết quả về CPU rồi giải phóng ba vùng nhớ GPU.
    CUDA_CHECK(cudaMemcpy(m3, d_m3 , r1*c2* sizeof(*d_m3), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_m1));
    CUDA_CHECK(cudaFree(d_m2));
    CUDA_CHECK(cudaFree(d_m3));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel execution time: %f ms\n", ms);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(){
    // Hằng số giúp kích thước các mảng hợp lệ theo chuẩn C++.
    const int r = 2;
    // A = B = [[1, 2], [3, 4]], lưu liên tiếp theo hàng.
    float h_a[r*r] = {1,2,
                    3,4};
    float h_b[r*r] = {1,2,
                    3,4};
    float h_c[r*r];
    // Thời gian thực gồm chuẩn bị GPU, cấp phát, truyền dữ liệu và tính toán.
    // Không tính việc tạo dữ liệu đầu vào hay in ma trận kết quả.
    const auto begin = std::chrono::steady_clock::now();
    computeMul(r,r,r,h_a, h_b, h_c);
    const auto end = std::chrono::steady_clock::now();
    const double totalMs = std::chrono::duration<double, std::milli>(end - begin).count();
    printf("Total execution time: %f ms\n", totalMs);
    // Kết quả mong đợi: [[7, 10], [15, 22]].
    for (int i=0;i<r*r;i++){
        printf("%.2f\t", h_c[i]);
    }
    printf("\n");
}
