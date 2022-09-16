#include <iostream>
#include <chrono>
#include <CL/sycl.hpp>
using namespace std;

struct alignas(2) MyHalf {
  unsigned short x;
  MyHalf() = default;
  MyHalf(float value) {
      x = sycl::bit_cast<uint16_t>(sycl::half(value));
  }
  inline operator float() const {
      return float(sycl::bit_cast<sycl::half>(x));
  }
};

struct alignas(2) MyBF {
  unsigned short x;
  MyBF() = default;
  MyBF(float value) {
      x = sycl::bit_cast<uint16_t>(sycl::ext::oneapi::experimental::bfloat16(value));
  }
  inline operator float() const {
      return float(*reinterpret_cast<const sycl::ext::oneapi::experimental::bfloat16*>(&x));
  }
};

inline MyBF operator*(const MyBF& a, const MyBF& b) {
    return static_cast<float>(a) * static_cast<float>(b);
}

typedef MyBF scalar_t;

template<typename item_t>
inline void mul_kernel(item_t &item, const scalar_t *in0, const scalar_t *in1, scalar_t *out, int n) {
    int lid = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (lid < n) {
        out[lid] = in0[lid] * in1[lid];
    }
}

void device_mul(const scalar_t *in0, const scalar_t *in1, scalar_t *out, int n) {
    sycl::queue q(sycl::gpu_selector{});
    int group_size = 1024;
    int num_groups = (n + group_size - 1) / group_size;
    auto event = q.submit([&](sycl::handler& h){
        h.parallel_for(
            sycl::nd_range<1>(num_groups * group_size, group_size),
            [=](sycl::nd_item<1> item) {
            mul_kernel(item, in0, in1, out, n);
        });
    });
    q.wait();
}

int main()
{
    const int numel = 8192*8192;
    {
        sycl::queue q(sycl::gpu_selector{});
        scalar_t* in0 = sycl::aligned_alloc_device<scalar_t>(4096, numel, q);
        scalar_t* in1 = sycl::aligned_alloc_device<scalar_t>(4096, numel, q);
        scalar_t* out = sycl::aligned_alloc_device<scalar_t>(4096, numel, q);
        device_mul(in0, in1, out, numel);
    }
}
