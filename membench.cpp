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

template<typename T, typename queue_t>
void uniary_group_stride(queue_t &q, const T *in0, T *out, size_t N, bool verbose = true) {
    int group_size = 256;
    auto num_groups = (N + group_size - 1) / group_size;
    auto event = q.submit([&](sycl::handler& h){
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_groups * group_size), sycl::range<1>(group_size)),
            [=](sycl::nd_item<1> item) {
            auto idx = item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);
            if(idx < N) out[idx] = in0[idx];
        });
    });
    auto timems = timeit(event);
    uint64_t total_bytes = 2*N*sizeof(T); //MB
    if (verbose)
    cout << "{'numel':" << N << ", 'MBytes':" << total_bytes/1024/1024 << ", 'timems':" << timems << ", 'GBps':" << (total_bytes/(timems/1000))/1024/1024/1024 << "}" << endl;
}

int main() {
    sycl::queue q(sycl::gpu_selector{}, cl::sycl::property_list {cl::sycl::property::queue::enable_profiling()});
    auto d = sizeof(floatn);
    int numel = 256*1024*1024/d;
    auto in0 = sycl::aligned_alloc_device<floatn>(4096, numel, q);
    auto in1 = sycl::aligned_alloc_device<floatn>(4096, numel, q);
    auto out = sycl::aligned_alloc_device<floatn>(4096, numel, q);
    q.fill<float>(in0, 1.0, numel * FLOAT_N);
    
    for(int numel = 1*1024*1024/d; numel < 256*1024*1024/d; numel += 1*1024*1024/d) {
        q.fill<float>(in0, 1.0, numel * FLOAT_N);
        // uniary_group_stride(q, in0, out, numel);
        uniary_group_stride(q, in0, out, numel);
    }

    sycl::free(in0, q);
    sycl::free(in1, q);
    sycl::free(out, q);
}