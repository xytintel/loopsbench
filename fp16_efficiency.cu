#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
using namespace std;

void print_info(bool enable, size_t N, uint64_t total_bytes, double timems) {
    if (enable)
    cout << "{'numel':" << N << ", 'R/W (MBytes)':" << total_bytes/1024/1024 << ", 'time (ms)':" << timems << ", 'bw (GBps)':" << (total_bytes/(timems/1000))/1024/1024/1024 << "}" << endl;
}

template<typename T>
__global__ void sm_copy_kernel(const T *in, T *out, size_t numel) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < numel) out[idx] = in[idx];
}

template<typename T>
void sm_copy(const T *in, T *out, size_t numel, bool verbose = true) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int group_size = 256;
    auto num_groups = (numel + group_size - 1) / group_size;
    dim3 grid(num_groups), block(group_size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEventRecord(start);
    sm_copy_kernel<<<grid, block, 0, stream>>>(in, out, numel);
    cudaEventRecord(stop);
    cudaStreamSynchronize(stream);
    cudaEventSynchronize(stop);
    cudaStreamDestroy(stream);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    print_info(verbose, numel, 2*numel*sizeof(T), ms);
}

int main() {
    cudaSetDevice(0);
    {
        std::cout << "-------------------- using cuda::float output --------------------\n";
        using scalar_t = float;
        int data_bytes = 256*1024*1024; // 256MB
        int numel = data_bytes/sizeof(scalar_t);
        scalar_t *in, *out;
        cudaMalloc((void **)&in, numel * sizeof(scalar_t));
        cudaMalloc((void **)&out, numel * sizeof(scalar_t));
        for (int i=0; i<5; i++)
            sm_copy(in, out, numel);
        cudaFree(in);
        cudaFree(out);
    } 
    {
        std::cout << "-------------------- using cuda::half output --------------------\n";
        using scalar_t = __half;
        int data_bytes = 256*1024*1024; // 256MB
        int numel = data_bytes/sizeof(scalar_t);
        scalar_t *in, *out;
        cudaMalloc((void **)&in, numel * sizeof(scalar_t));
        cudaMalloc((void **)&out, numel * sizeof(scalar_t));
        for (int i=0; i<5; i++)
            sm_copy(in, out, numel);
        cudaFree(in);
        cudaFree(out);
    }

}
