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
    T &operator[](int i) {
        return val[i];
    }
    T const &operator[](int i) const {
        return val[i];
    }
};
typedef aligned_array<float, FLOAT_N> floatn;

void print_info(bool enable, size_t N, uint64_t total_bytes, double timems) {
    if (enable)
    cout << "{'numel':" << N << ", 'MBytes':" << total_bytes/1024/1024 << ", 'timems':" << timems << ", 'GBps':" << (total_bytes/(timems/1000))/1024/1024/1024 << "}" << endl;
}

template<int unroll_size, typename T>
void eu_copy(const T *in0, T *out, size_t N, bool verbose = true) {
    int group_size = 256;
    int group_work_size = group_size * unroll_size;
    int num_groups = (N + group_work_size - 1) / group_work_size;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    auto l = GpuLauncher::GetInstance();
    l->stream_begin();
    cudaEventRecord(start);
    l->submit(
        0, {num_groups}, {group_size},
        [=] DEVICE(KernelInfo &info) {
            auto idx = info.thread_idx(0) + group_work_size * info.block_idx(0);
#pragma unroll
            for(int i=0; i<unroll_size; i++) {
                if(idx < N) {
                    out[idx] = in0[idx];
                    idx += group_size;
                }
            }
        });
    cudaEventRecord(stop);
    l->stream_sync();
    cudaEventSynchronize(stop);
    l->stream_end();
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    print_info(verbose, N, 2*N*sizeof(T), ms);
}

int main() {
    std::cout << "-------------------- output --------------------\n";
    auto l = GpuLauncher::GetInstance();
    typedef short scalar_t;

    auto d = sizeof(scalar_t);
    int numel = 256*1024*1024/d;
    auto in0 = l->malloc<scalar_t>(numel);
    auto out = l->malloc<scalar_t>(numel);
    l->memset((void*)in0, 0, numel * d);
    
    for(int numel = 1*1024*1024/d; numel < 256*1024*1024/d; numel += 1*1024*1024/d) {
        // l->memset((void*)in0, 0, numel * d);
        eu_copy<4>(in0, out, numel);
    }

    l->free(in0);
    l->free(out);
}
