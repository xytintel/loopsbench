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

template<int unroll_factor, typename T, typename item_t>
inline void binary_group_stride_impl(item_t &item, T *in0, T *in1, T *out, int N) {
    int group_size = item.get_local_range(0);
    int group_work_size = group_size * unroll_factor;
    int idx = item.get_local_id(0) + item.get_group(0) * group_work_size;
#pragma unroll
    for(int i = 0; i< unroll_factor; i++) {
        if(idx < N) {
            out[idx] = in0[idx] * in1[idx];
            
        }
        idx += group_size;
    }
}

template<int unroll_factor, typename T, typename item_t>
inline void binary_grid_stride_impl(item_t &item, T *in0, T *in1, T *out, int N) {
    int group_size = item.get_local_range(0);
    int num_groups = item.get_group_range(0);
    int full_tile_work_size = group_size * num_groups * unroll_factor;
    int rounded_size = ((N - 1) / full_tile_work_size + 1) * full_tile_work_size;
    int idx = item.get_group(0) * group_size + item.get_local_id(0);
    for (int linear_index = idx; linear_index < rounded_size;
       linear_index += full_tile_work_size) {
#pragma unroll
    for (int i = 0; i < unroll_factor; i++) {
      int li = linear_index + group_size * num_groups * i;
      if (li < N) {
        out[li] = in0[li] * in1[li];
      }
    }
    // item.barrier(dpcpp_local_fence);
    // item.barrier(sycl::access::fence_space::local_space);
  }
}

template<typename T>
void binary_group_stride(T *in0, T *in1, T *out, int N) {
    sycl::queue q(sycl::gpu_selector{}, cl::sycl::property_list {cl::sycl::property::queue::enable_profiling()});
    int group_size = 1024;
    int num_groups = (N + group_size - 1) / group_size;
    auto event = q.submit([&](sycl::handler& h){
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_groups * group_size), sycl::range<1>(group_size)),
            [=](sycl::nd_item<1> item) {
            binary_group_stride_impl<4>(item, in0, in1, out, N);
        });
    });
    q.wait();
    cout << timeit(event) << endl;
}

template<typename T>
void binary_grid_stride(T *in0, T *in1, T *out, int N) {
    sycl::queue q(sycl::gpu_selector{}, cl::sycl::property_list {cl::sycl::property::queue::enable_profiling()});
    int group_size = 1024;
    int num_groups = (N + group_size - 1) / group_size;
    int hw_max_groups = (512*8*32) / group_size;
    num_groups = num_groups > hw_max_groups ? hw_max_groups : num_groups;
    auto event = q.submit([&](sycl::handler& h){
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_groups * group_size), sycl::range<1>(group_size)),
            [=](sycl::nd_item<1> item) {
            binary_grid_stride_impl<4>(item, in0, in1, out, N);
        });
    });
    q.wait();
    cout << timeit(event) << endl;
}


int main()
{
    const int numel = 8192*8192; // slow
    // const int numel = 347820; // fast

    std::cout << "using binary_group_stride\n";
    for (int it = 0; it < 10; ++it) {
        sycl::queue q(sycl::gpu_selector{});
        float* in0 = sycl::aligned_alloc_device<float>(4096, numel, q);
        float* in1 = sycl::aligned_alloc_device<float>(4096, numel, q);
        float* out = sycl::aligned_alloc_device<float>(4096, numel, q);
        binary_group_stride(in0, in1, out, numel);
    }

    std::cout << "using binary_grid_stride\n";
    for (int it = 0; it < 10; ++it) {
        sycl::queue q(sycl::gpu_selector{});
        float* in0 = sycl::aligned_alloc_device<float>(4096, numel, q);
        float* in1 = sycl::aligned_alloc_device<float>(4096, numel, q);
        float* out = sycl::aligned_alloc_device<float>(4096, numel, q);
        binary_grid_stride(in0, in1, out, numel);
    }

}
