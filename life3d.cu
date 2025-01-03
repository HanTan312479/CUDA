/*-----------------------------------------------
 * 请在此处填写你的个人信息
 * 学号:SA24218177
 * 姓名:niujiahao
 * 邮箱:hantan312479@gmail.com
 ------------------------------------------------*/

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <cuda_runtime.h>

// 宏定义用于简化访问3D数组
#define AT_GPU(x, y, z) universe[(x) * N * N + (y) * N + z]
#define NEXT_GPU(x, y, z) next[(x) * N * N + (y) * N + z]
#define AT(x, y, z) universe[(x) * N * N + (y) * N + z]

using namespace std;

// 存活细胞数
int population(int N, char *universe)
{
    int result = 0;
    for (int i = 0; i < N * N * N; i++)
        result += universe[i];
    return result;
}

// 打印世界状态
void print_universe(int N, char *universe)
{
    // 仅在N较小(<= 32)时用于Debug
    if (N > 32)
        return;
    for (int x = 0; x < N; x++)
    {
        for (int y = 0; y < N; y++)
        {
            for (int z = 0; z < N; z++)
            {
                if (AT(x, y, z))
                    cout << "O ";
                else
                    cout << "* ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << "population: " << population(N, universe) << endl;
}

// CUDA 核函数
__global__ void life3d_kernel(int N, char *universe, char *next) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // 仅处理在有效范围内的线程
    if (x >= N || y >= N || z >= N) return;

    // 计算当前细胞的存活邻居数量
    int alive = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;

                // 循环边界处理
                int nx = (x + dx + N) % N;
                int ny = (y + dy + N) % N;
                int nz = (z + dz + N) % N;

                alive += AT_GPU(nx, ny, nz);
            }
        }
    }

    // 状态转移规则
    if (AT_GPU(x, y, z) && (alive < 5 || alive > 7))
        NEXT_GPU(x, y, z) = 0;
    else if (!AT_GPU(x, y, z) && alive == 6)
        NEXT_GPU(x, y, z) = 1;
    else
        NEXT_GPU(x, y, z) = AT_GPU(x, y, z);
}

void life3d_run(int N, char *universe, int T) {
    char *d_universe, *d_next;
    size_t size = N * N * N * sizeof(char);

    // 分配设备内存
    cudaMalloc((void **)&d_universe, size);
    cudaMalloc((void **)&d_next, size);

    // 将数据从主机传输到设备
    cudaMemcpy(d_universe, universe, size, cudaMemcpyHostToDevice);

    // 设置 CUDA 网格和块的大小
    dim3 blockSize(8, 8, 8);  // 每个线程块 8x8x8 = 512 个线程
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y,
                  (N + blockSize.z - 1) / blockSize.z);

    for (int t = 0; t < T; t++) {
        // 调用核函数
        life3d_kernel<<<gridSize, blockSize>>>(N, d_universe, d_next);

        // 交换 universe 和 next 的指针
        char *temp = d_universe;
        d_universe = d_next;
        d_next = temp;
    }

    // 将数据从设备传回主机
    cudaMemcpy(universe, d_universe, size, cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_universe);
    cudaFree(d_next);
}
// 核心计算代码，将世界向前推进T个时刻
// void life3d_run(int N, char *universe, int T)
// {
//     char *next = (char *)malloc(N * N * N);
//     for (int t = 0; t < T; t++)
//     {
//         // outerloop: iter universe
//         for (int x = 0; x < N; x++)
//             for (int y = 0; y < N; y++)
//                 for (int z = 0; z < N; z++)
//                 {
//                     // inner loop: stencil
//                     int alive = 0;
//                     for (int dx = -1; dx <= 1; dx++)
//                         for (int dy = -1; dy <= 1; dy++)
//                             for (int dz = -1; dz <= 1; dz++)
//                             {
//                                 if (dx == 0 && dy == 0 && dz == 0)
//                                     continue;
//                                 int nx = (x + dx + N) % N;
//                                 int ny = (y + dy + N) % N;
//                                 int nz = (z + dz + N) % N;
//                                 alive += AT(nx, ny, nz);
//                             }
//                     if (AT(x, y, z) && (alive < 5 || alive > 7))
//                         next[x * N * N + y * N + z] = 0;
//                     else if (!AT(x, y, z) && alive == 6)
//                         next[x * N * N + y * N + z] = 1;
//                     else
//                         next[x * N * N + y * N + z] = AT(x, y, z);
//                 }
//         memcpy(universe, next, N * N * N);
//     }
//     free(next);
// }

// 读取输入文件
void read_file(char *input_file, char *buffer)
{
    ifstream file(input_file, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        cout << "Error: Could not open file " << input_file << std::endl;
        exit(1);
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer, file_size))
    {
        std::cerr << "Error: Could not read file " << input_file << std::endl;
        exit(1);
    }
    file.close();
}

// 写入输出文件
void write_file(char *output_file, char *buffer, int N)
{
    ofstream file(output_file, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        cout << "Error: Could not open file " << output_file << std::endl;
        exit(1);
    }
    file.write(buffer, N * N * N);
    file.close();
}

int main(int argc, char **argv)
{
    // cmd args
    if (argc < 5)
    {
        cout << "usage: ./life3d N T input output" << endl;
        return 1;
    }
    int N = std::stoi(argv[1]);
    int T = std::stoi(argv[2]);
    char *input_file = argv[3];
    char *output_file = argv[4];

    char *universe = (char *)malloc(N * N * N);
    read_file(input_file, universe);

    int start_pop = population(N, universe);
    auto start_time = std::chrono::high_resolution_clock::now();
    life3d_run(N, universe, T);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    int final_pop = population(N, universe);
    write_file(output_file, universe, N);

    cout << "start population: " << start_pop << endl;
    cout << "final population: " << final_pop << endl;
    double time = duration.count();
    cout << "time: " << time << "s" << endl;
    cout << "cell per sec: " << T / time * N * N * N << endl;

    free(universe);
    return 0;
}
