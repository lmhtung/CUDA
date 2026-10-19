"""Hai bài tương đương part1/chapter1.cu và part1/chapter2.cu, chạy trên CPU.

Chạy từ thư mục CUDA: python3 part1/python_equivalent.py --repeat 1000
Chỉ dùng thư viện chuẩn Python, không cần cài NumPy hoặc CUDA.
"""

import argparse
from time import perf_counter_ns


def sum_vectors(a, b):
    """Cộng từng cặp phần tử, tương đương mỗi thread trong SumKernel."""
    return [a[i] + b[i] for i in range(len(a))]


def multiply_matrices(a, b, rows, inner, cols):
    """Nhân A[rows, inner] với B[inner, cols], lưu theo hàng như code CUDA."""
    result = [0.0] * (rows * cols)
    # Python xử lý tuần tự; GPU giao mỗi cặp (r, c) cho một thread.
    for r in range(rows):
        for c in range(cols):
            value = 0.0
            for i in range(inner):
                value += a[r * inner + i] * b[i * cols + c]
            result[r * cols + c] = value
    return result


def measure(function, args, repeat):
    """Đo thời gian thực trung bình mỗi lần gọi, đơn vị mili giây."""
    # Chạy khởi động ngoài vùng đo để giảm ảnh hưởng của lần gọi đầu tiên.
    function(*args)
    start = perf_counter_ns()
    for _ in range(repeat):
        result = function(*args)
    elapsed_ns = perf_counter_ns() - start
    # Đo cả lô giúp giảm ảnh hưởng của việc đọc đồng hồ với phép tính rất ngắn.
    # Bao gồm gọi hàm, tạo kết quả và chi phí vòng lặp; không gồm tạo input/in.
    return result, elapsed_ns / repeat / 1_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeat", type=int, default=1000, help="Số lần đo mỗi bài (mặc định: 1000)."
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat phải lớn hơn hoặc bằng 1")

    # Dữ liệu đầu vào giống chapter1.cu: hai vector từ 1 đến 100.
    a = [float(i) for i in range(1, 101)]
    b = a.copy()
    vector, vector_ms = measure(sum_vectors, (a, b), args.repeat)
    if vector != [float(2 * i) for i in range(1, 101)]:
        raise RuntimeError("Kết quả cộng vector không đúng")

    print("Bài 1: cộng hai vector trên CPU")
    for i in range(0, len(vector), 10):
        print("".join(f"{value:8.2f}" for value in vector[i : i + 10]))
    print(f"Thời gian trung bình: {vector_ms:.6f} ms/lần ({args.repeat} lần)")

    # Dữ liệu đầu vào giống chapter2.cu: A = B = [[1, 2], [3, 4]].
    matrix_a = [1.0, 2.0, 3.0, 4.0]
    matrix_b = matrix_a.copy()
    matrix, matrix_ms = measure(
        multiply_matrices, (matrix_a, matrix_b, 2, 2, 2), args.repeat
    )
    if matrix != [7.0, 10.0, 15.0, 22.0]:
        raise RuntimeError("Kết quả nhân ma trận không đúng")

    print("\nBài 2: nhân hai ma trận trên CPU")
    for i in range(0, len(matrix), 2):
        print("".join(f"{value:8.2f}" for value in matrix[i : i + 2]))
    print(f"Thời gian trung bình: {matrix_ms:.6f} ms/lần ({args.repeat} lần)")
    print("\nPython đo trên CPU; CUDA Events chỉ đo kernel trên GPU.")
    print("Dữ liệu rất nhỏ, chi phí khởi tạo GPU/truyền dữ liệu có thể chiếm phần lớn.")
    print("Python dùng float 64-bit, còn các mảng float trong CUDA là 32-bit.")


if __name__ == "__main__":
    main()
