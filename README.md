# CUDA
Programming Parallel

## Chạy các ví dụ

Từ thư mục `CUDA`, biên dịch từng file riêng bằng CUDA Toolkit trên máy có GPU NVIDIA:

```bash
nvcc -std=c++11 -arch=native part1/chapter1.cu -o chapter1.out
./chapter1.out
nvcc -std=c++11 -arch=native part1/chapter2.cu -o chapter2.out
./chapter2.out
```

`chapter1.cu` cộng hai vector 100 phần tử, cho kết quả 2, 4, ..., 200.
`chapter2.cu` nhân hai ma trận `[[1, 2], [3, 4]]`, cho kết quả `[[7, 10], [15, 22]]`.
Các file có chú thích tiếng Việt về thread, block, grid, bộ nhớ và đo thời gian.
`part2/chapter1.cu` hiện đang trống.

## Phiên bản Python và đo thời gian

```bash
python3 part1/python_equivalent.py
python3 part1/python_equivalent.py --repeat 10000
```

File Python dùng thư viện chuẩn, thực hiện cả hai bài trên CPU với cùng dữ liệu.
Thời gian là trung bình mỗi lần gọi sau một lần khởi động, đo bằng `perf_counter_ns`.
Phép đo gồm việc tạo mảng kết quả, gọi hàm và chi phí vòng lặp; không gồm tạo dữ liệu
đầu vào, kiểm tra kết quả hay in ra màn hình. Mặc định đo 1000 lần.

CUDA in hai mốc: `Kernel execution time` đo kernel bằng CUDA Events và
`Total execution time` đo thời gian thực của lời gọi xử lý GPU, gồm khởi tạo,
cấp phát, sao chép dữ liệu và giải phóng. Việc in thời gian kernel cũng nằm trong
phép đo tổng. Lần chạy CUDA đầu tiên có thể chịu chi phí khởi tạo lớn.
Thời gian Python sau khởi động và thời gian CUDA một lần chạy có phạm vi đo khác nhau,
nên chưa phải phép benchmark công bằng để kết luận tốc độ CPU/GPU.
Các mảng CUDA dùng `float` 32-bit; Python dùng `float` 64-bit.
