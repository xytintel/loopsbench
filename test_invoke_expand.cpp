#include <iostream>
#include <chrono>
#include <array>
#include <CL/sycl.hpp>
using namespace std;

namespace impl {

// ========================== function_traits impl ==========================

template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {
};

template <typename ClassType, typename T>
struct function_traits<T ClassType::*> : public function_traits<T> {
};

// Const class member functions
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const> : public function_traits<ReturnType(Args...)> {
};

// Reference types
template <typename T>
struct function_traits<T&> : public function_traits<T> {};
template <typename T>
struct function_traits<T*> : public function_traits<T> {};

// Free functions
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType(Args...)> {
  // arity is the number of arguments.
  enum { arity = sizeof...(Args) };

  typedef std::tuple<Args...> ArgsTuple;
  typedef ReturnType result_type;

  template <size_t i>
  struct arg
  {
      typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
      // the i-th argument is equivalent to the i-th tuple element of a tuple
      // composed of those arguments.
  };
};

// ========================== invoke impl ==========================

template <typename T>
struct LoadImpl {
  static T apply(const void* src) {
    return *reinterpret_cast<const T*>(src);
  }
};

template <>
struct LoadImpl<bool> {
  static bool apply(const void* src) {
    static_assert(sizeof(bool) == sizeof(char), "");
    // NOTE: [Loading boolean values]
    // Protect against invalid boolean values by loading as a byte
    // first, then converting to bool (see gh-54789).
    return *reinterpret_cast<const unsigned char*>(src);
  }
};

template <typename T>
inline T load(const void* src) {
  return LoadImpl<T>::apply(src);
}

template <typename scalar_t>
inline scalar_t load(const scalar_t* src) {
  return LoadImpl<scalar_t>::apply(src);
}

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
typename traits::result_type invoke_impl(
    const func_t& f,
    char* const __restrict data[],
    const index_t strides[],
    int i,
    std::index_sequence<INDEX...>) {
  (void)strides;
  (void)i;
  return f(
      load<typename traits::template arg<INDEX>::type>(
          data[INDEX] + i * strides[INDEX])...);
}

template <
    typename func_t,
    typename index_t,
    typename traits = function_traits<func_t>>
typename traits::result_type invoke(
    const func_t& f,
    char* const __restrict data[],
    const index_t strides[],
    int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, i, Indices{});
}

// ========================== offset calculator ==========================

// Result of div/mod operation stored together.
template <typename Value>
struct DivMod {
  Value div, mod;

  DivMod(Value div, Value mod) : div(div), mod(mod) {}
};

// Base case: we only have an implementation for uint32_t for now.  For
// everything else, we use plain division.
template <typename Value>
struct IntDivider {
  IntDivider() {} // Dummy constructor for arrays.
  IntDivider(Value d) : divisor(d) {}

  inline Value div(Value n) const {
    return n / divisor;
  }
  inline Value mod(Value n) const {
    return n % divisor;
  }
  inline DivMod<Value> divmod(Value n) const {
    return DivMod<Value>(n / divisor, n % divisor);
  }

  Value divisor;
};

template <typename T, int size>
struct alignas(16) Array {
  T data[size];

  T operator[](int i) const {
    return data[i];
  }
  T& operator[](int i) {
    return data[i];
  }

  Array() = default;
  Array(const Array&) = default;
  Array& operator=(const Array&) = default;

  // Fill the array with x.
  Array(T x) {
    for (int i = 0; i < size; i++) {
      data[i] = x;
    }
  }
};

template <int NARGS, typename index_t = uint32_t, bool signed_strides = false>
struct OffsetCalculator {
  static constexpr int MAX_DIMS = 12;

  // We allow having negative strides to implement some operations like
  // torch.flip
  using stride_t =
      std::conditional_t<signed_strides, std::make_signed_t<index_t>, index_t>;

  // The offset for each argument (in bytes). Wrapper around fixed-size array.
  using offset_type = Array<stride_t, std::max<int>(NARGS, 1)>;

  // if element_sizes is nullptr, then the strides will be in bytes, otherwise
  // the strides will be in # of elements.
  OffsetCalculator(
      int dims,
      const int64_t* sizes,
      const int64_t* const* strides,
      const int64_t* element_sizes = nullptr)
      : dims(dims) {
    for (int i = 0; i < dims; i++) {
      sizes_[i] = IntDivider<index_t>(sizes[i]);
      for (int arg = 0; arg < NARGS; arg++) {
        int64_t element_size =
            (element_sizes == nullptr ? 1LL : element_sizes[arg]);
        strides_[i][arg] = strides[arg][i] / element_size;
      }
    }
  }

  offset_type get(index_t linear_idx) const {
    offset_type offsets;
#pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = 0;
    }

#pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims)
        break;
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

#pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] += divmod.mod * strides_[dim][arg];
      }
    }
    return offsets;
  }

  int dims;
  IntDivider<index_t> sizes_[MAX_DIMS];
  stride_t strides_[MAX_DIMS][std::max<int>(NARGS, 1)];
};

}


template<typename T, typename queue_t>
void eu_fill(queue_t &q, T *in0, size_t N) {
    int group_size = 256;
    auto num_groups = (N + group_size - 1) / group_size;
    auto event = q.submit([&](sycl::handler& h){
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_groups * group_size), sycl::range<1>(group_size)),
            [=](sycl::nd_item<1> item) {
            auto idx = item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);
            if(idx < N) {
                if (idx < N/2)
                  in0[idx] = idx;
                else
                  in0[idx] = 0;
            }
        });
    });
}

template<typename T, typename queue_t>
void eu_mod(queue_t &q, T *in0, T in1, T *out, size_t N) {

    int64_t n2 = N/2;

    int64_t strides_data_0[2] = {1, n2};
    int64_t strides_data_1[2] = {1*sizeof(int), 0};
    std::array<const int64_t*, 2> strides;
    strides[0] = strides_data_0;
    strides[1] = strides_data_1;

    std::array<int64_t, 2> shapes;
    shapes[0] = n2;
    shapes[1] = 2;

    auto offset_calc = impl::OffsetCalculator<2, uint32_t, false>(2, shapes.data(), strides.data());

    auto f = [=](int a) { return a % in1; };
    char *pdatas[1];
    pdatas[0] = reinterpret_cast<char*>(in0);

    int group_size = 256;
    auto num_groups = (N + group_size - 1) / group_size;
    auto event = q.submit([&](sycl::handler& h){
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_groups * group_size), sycl::range<1>(group_size)),
            [=](sycl::nd_item<1> item) {
            auto idx = item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);
            if(idx < N) {
                // out[idx] = in0[idx] % in1; // correct
                auto offsets = offset_calc.get(idx);
                out[offsets[0]] = impl::invoke(f, &pdatas[0], &offsets.data[1], 1);
            }
        });
    });
}

int main() {
    sycl::queue q(sycl::gpu_selector{});
    int numel = 128; // 2,64

    auto in0 = sycl::aligned_alloc_device<int>(4096, numel, q);
    int in1 = 7;
    auto out = sycl::aligned_alloc_device<int>(4096, numel, q);
    q.fill<int>(in0, 1.0, numel);
    eu_fill(q, in0, numel);
    q.wait();

    eu_mod(q, in0, in1, out, numel);
    q.wait();

    auto out_cpu = new int[numel];
    q.memcpy(out_cpu, out, numel * sizeof(int)).wait();

    for(int i=0; i<numel/2; i++)
        cout << out_cpu[i] << " ";
    cout << endl;
    for(int i=0; i<numel/2; i++)
        cout << out_cpu[numel/2 + i] << " ";
    cout << endl;

    auto cpu_ref = new int[numel];
    for (int i=0; i<numel/2; i++)
      cpu_ref[i] = i % in1;
    for (int i=0; i<numel/2; i++)
      cpu_ref[numel/2+i] = i % in1;
    
    // evaluate
    for(int i=0; i<numel; i++) {
      if(cpu_ref[i] != out_cpu[i]) {
        cout << "ERROR\n";
        return -1;
      }
    }

    sycl::free(in0, q);
    sycl::free(out, q);
    delete[]out_cpu;
    delete[]cpu_ref;
}
