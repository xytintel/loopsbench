// export ZE_AFFINITY_MASK=0.0 ; dpcpp fp16_efficency.cpp ; ./a.out
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

void print_info(bool enable, size_t N, uint64_t total_bytes, double timems) {
    if (enable)
    cout << "{'numel':" << N << ", 'R/W (MBytes)':" << total_bytes/1024/1024 << ", 'time (ms)':" << timems << ", 'bw (GBps)':" << (total_bytes/(timems/1000))/1024/1024/1024 << "}" << endl;
}

template<typename T, typename queue_t>
void eu_copy(queue_t &q, const T *in, T *out, size_t numel, bool verbose = true) {
    int group_size = 256;
    auto num_groups = (numel + group_size - 1) / group_size;
    auto event = q.submit([&](sycl::handler& h){
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_groups * group_size), sycl::range<1>(group_size)),
            [=](sycl::nd_item<1> item) {
            auto idx = item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);
            if(idx < numel) out[idx] = in[idx];
        });
    });
    q.wait();
    print_info(verbose, numel, 2*numel*sizeof(T), timeit(event));    
}

int main() {
    sycl::queue q(sycl::gpu_selector{}, cl::sycl::property_list {cl::sycl::property::queue::enable_profiling()});
    
    {
        std::cout << "-------------------- using sycl::float output --------------------\n";
        using scalar_t = float;
        int data_bytes = 256*1024*1024; // 256MB
        int numel = data_bytes/sizeof(scalar_t);
        auto in = sycl::aligned_alloc_device<scalar_t>(4096, numel, q);
        auto out = sycl::aligned_alloc_device<scalar_t>(4096, numel, q);
        for (int i=0; i<5; i++)
            eu_copy(q, in, out, numel);
        sycl::free(in, q);
        sycl::free(out, q);
    }
    {
        std::cout << "-------------------- using sycl::half output --------------------\n";
        using scalar_t = sycl::half;
        int data_bytes = 256*1024*1024; // 256MB
        int numel = data_bytes/sizeof(scalar_t);
        auto in = sycl::aligned_alloc_device<scalar_t>(4096, numel, q);
        auto out = sycl::aligned_alloc_device<scalar_t>(4096, numel, q);
        for (int i=0; i<5; i++)
            eu_copy(q, in, out, numel);
        sycl::free(in, q);
        sycl::free(out, q);
    }

}
