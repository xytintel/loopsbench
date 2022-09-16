#include <iostream>
#include <chrono>
#include "launcher.h"
#include "measure.h"
using namespace std;
using namespace pmkl;
using namespace pmkl::utils;

#define FLOAT_N 4
template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_array {
    T val[vec_size];
    DEVICE T &operator[](int i) {
        return val[i];
    }
    DEVICE T const &operator[](int i) const {
        return val[i];
    }
};
typedef aligned_array<float, FLOAT_N> floatn;

void print_info(bool enable, size_t N, uint64_t total_bytes, double timems) {
    if (enable)
    cout << "{'numel':" << N << ", 'MBytes':" << total_bytes/1024/1024 << ", 'timems':" << timems << ", 'GBps':" << (total_bytes/(timems/1000))/1024/1024/1024 << "}" << endl;
}

template<typename T>
void eu_memset(T *out, size_t N, bool verbose = true) {
    int group_size = 256;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    auto l = GpuLauncher::GetInstance();
    l->stream_begin();
    cudaEventRecord(start);
    l->submit(
        0, {(int)((N + group_size - 1) / group_size)}, {group_size},
        [=] DEVICE(KernelInfo &info) {
            auto idx = info.thread_idx(0) + info.thread_range(0) * info.block_idx(0);
            if(idx < N) {
                floatn temp;
#pragma unroll
                for(int i=0; i<FLOAT_N; i++)
                    temp[i] = 1;
                out[idx] = temp;
            }
        });
    cudaEventRecord(stop);
    l->stream_sync();
    cudaEventSynchronize(stop);
    l->stream_end();
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    print_info(verbose, N, 1*N*sizeof(T), ms);
}

int main() {
    std::cout << "-------------------- output --------------------\n";
    auto l = GpuLauncher::GetInstance();
    auto d = sizeof(floatn);
    int numel = 256*1024*1024/d;
    auto out = l->malloc<floatn>(numel);
    
    for(int numel = 1*1024*1024/d; numel < 256*1024*1024/d; numel += 1*1024*1024/d) {
        eu_memset<floatn>(out, numel);
    }

    l->free(out);
}
