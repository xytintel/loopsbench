#include <iostream>
#include <chrono>
#include <CL/sycl.hpp>
using namespace std;

double timeit(cl::sycl::event &event) {
    auto submit_time = event.get_profiling_info<cl::sycl::info::event_profiling::command_submit>();
    auto start_time = event.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
    auto end_time = event.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
    auto submission_time = (start_time - submit_time) / 1000000.0f;
    auto execution_time = (end_time - start_time) / 1000000.0f;
    return execution_time;
}

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
// typedef sycl::half floatn;
// typedef sycl::ext::oneapi::experimental::bfloat16 floatn;

void print_info(bool enable, size_t N, uint64_t total_bytes, double timems) {
    if (enable)
    cout << "{'numel':" << N << ", 'MBytes':" << total_bytes/1024/1024 << ", 'timems':" << timems << ", 'GBps':" << (total_bytes/(timems/1000))/1024/1024/1024 << "}" << endl;
}

template<typename T, typename queue_t>
void eu_memset(queue_t &q, T *out, size_t N, bool verbose = true) {
    int group_size = 256;
    auto num_groups = (N + group_size - 1) / group_size;
    auto event = q.submit([&](sycl::handler& h){
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_groups * group_size), sycl::range<1>(group_size)),
            [=](sycl::nd_item<1> item) {
            auto idx = item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);
            if(idx < N) {
                floatn temp;
#pragma unroll
                for(int i=0; i<FLOAT_N; i++)
                    temp[i] = 1;
                out[idx] = temp;
            }
        });
    });
    print_info(verbose, N, 1*N*sizeof(T), timeit(event));    
}

int main() {
    sycl::queue q(sycl::gpu_selector{}, cl::sycl::property_list {cl::sycl::property::queue::enable_profiling()});
    std::cout << "-------------------- output --------------------\n";
    auto d = sizeof(floatn);
    int numel = 256*1024*1024/d;
    auto out = sycl::aligned_alloc_device<floatn>(4096, numel, q);
    
    for(int numel = 1*1024*1024/d; numel < 256*1024*1024/d; numel += 1*1024*1024/d) {
        // q.fill<char>(out, 1.0, numel * d);
        eu_memset(q, out, numel);
    }

    sycl::free(out, q);
}
