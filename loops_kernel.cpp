#include <CL/sycl.hpp>
#include <array>
#include <chrono>
#include <iostream>
#include <limits>
using namespace std;

namespace loops {

namespace memory {

template <typename T, int size> struct alignas(16) Array {
  T data[size];
  T operator[](int i) const { return data[i]; }
  T &operator[](int i) { return data[i]; }
  Array() = default;
  Array(const Array &) = default;
  Array &operator=(const Array &) = default;
};

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
  scalar_t &operator[](int index) { return val[index]; }
  scalar_t const &operator[](int index) const { return val[index]; }
};

template <int vec_size, typename scalar_t>
inline aligned_vector<scalar_t, vec_size> load_vector(const scalar_t *base_ptr,
                                                      uint32_t offset) {
  using vec_t = aligned_vector<scalar_t, vec_size>;
  auto *from = reinterpret_cast<const vec_t *>(base_ptr);
  return from[offset];
}

template <int vec_size>
inline aligned_vector<bool, vec_size> load_vector(const bool *base_ptr,
                                                  uint32_t offset) {
  auto tmp = load_vector<vec_size>(reinterpret_cast<const uint8_t *>(base_ptr),
                                   offset);
  aligned_vector<bool, vec_size> ret;
  for (int i = 0; i < vec_size; ++i) {
    ret.val[i] = bool(tmp.val[i]);
  }
  return ret;
}

struct LoadWithoutCast {
  template <typename scalar_t, typename offset_t>
  scalar_t load(char *base_ptr, offset_t offset, int arg) {
    return *(reinterpret_cast<scalar_t *>(base_ptr) + offset);
  }
};

struct StoreWithoutCast {
  template <typename scalar_t, typename offset_t>
  void store(scalar_t value, char *base_ptr, offset_t offset) {
    *(reinterpret_cast<scalar_t *>(base_ptr) + offset) = value;
  }
};

} // namespace memory

namespace function {

template <template <int i> typename func, int end, int current = 0>
struct static_unroll {
  template <typename... Args> static inline void with_args(Args &&... args) {
    func<current>::apply(std::forward<Args>(args)...);
    static_unroll<func, end, current + 1>::with_args(args...);
  }
};

template <template <int i> typename func, int end>
struct static_unroll<func, end, end> {
  template <typename... Args> static inline void with_args(Args... args) {}
};

template <class F, class Tuple>
inline constexpr decltype(auto) apply(F &&f, Tuple &&t) {
  return std::apply(std::forward<F>(f), std::forward<Tuple>(t));
}

// ========================== function_traits impl ==========================

template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {};

template <typename ClassType, typename T>
struct function_traits<T ClassType::*> : public function_traits<T> {};

// Const class member functions
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...) const>
    : public function_traits<ReturnType(Args...)> {};

// Reference types
template <typename T>
struct function_traits<T &> : public function_traits<T> {};
template <typename T>
struct function_traits<T *> : public function_traits<T> {};

// Free functions
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType(Args...)> {
  // arity is the number of arguments.
  enum { arity = sizeof...(Args) };

  typedef std::tuple<Args...> ArgsTuple;
  typedef ReturnType result_type;

  template <size_t i> struct arg {
    typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
    // the i-th argument is equivalent to the i-th tuple element of a tuple
    // composed of those arguments.
  };
};

// ========================== invoke impl ==========================

template <typename T> struct LoadImpl {
  static T apply(const void *src) { return *reinterpret_cast<const T *>(src); }
};

template <> struct LoadImpl<bool> {
  static bool apply(const void *src) {
    static_assert(sizeof(bool) == sizeof(char), "");
    // NOTE: [Loading boolean values]
    // Protect against invalid boolean values by loading as a byte
    // first, then converting to bool (see gh-54789).
    return *reinterpret_cast<const unsigned char *>(src);
  }
};

template <typename T> inline T load(const void *src) {
  return LoadImpl<T>::apply(src);
}

template <typename scalar_t> inline scalar_t load(const scalar_t *src) {
  return LoadImpl<scalar_t>::apply(src);
}

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
typename traits::result_type
invoke_impl(const func_t &f, char *const __restrict data[],
            const index_t strides[], int i, std::index_sequence<INDEX...>) {
  (void)strides;
  (void)i;
  return f(load<typename traits::template arg<INDEX>::type>(
      data[INDEX] + i * strides[INDEX])...);
}

template <typename func_t, typename index_t,
          typename traits = function_traits<func_t>>
typename traits::result_type invoke(const func_t &f,
                                    char *const __restrict data[],
                                    const index_t strides[], int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, i, Indices{});
}

// ========================== max_scalar_size impl ==========================

constexpr int max_scalar_size_(std::tuple<>) { return 0; }

template <typename scalar_t, typename... types>
constexpr int max_scalar_size_(std::tuple<scalar_t, types...>) {
  return std::max<int>(sizeof(scalar_t),
                       max_scalar_size_(std::tuple<types...>{}));
}

template <typename func_t> constexpr static inline int max_scalar_size() {
  using traits = function_traits<func_t>;
  using args_t = typename traits::ArgsTuple;
  constexpr auto size = max_scalar_size_(args_t{});
  using return_t = typename traits::result_type;
  return std::max<int>(sizeof(return_t), size);
}

} // namespace function

namespace offset {

template <typename Value> struct DivMod {
  Value div, mod;
  DivMod(Value div, Value mod) : div(div), mod(mod) {}
};

// Base case: we only have an implementation for uint32_t for now.  For
// everything else, we use plain division.
template <typename Value> struct IntDivider {
  IntDivider() {} // Dummy constructor for arrays.
  IntDivider(Value d) : divisor(d) {}
  inline Value div(Value n) const { return n / divisor; }
  inline Value mod(Value n) const { return n % divisor; }
  inline DivMod<Value> divmod(Value n) const {
    return DivMod<Value>(n / divisor, n % divisor);
  }
  Value divisor;
};

template <int NARGS, typename index_t = uint32_t, bool signed_strides = false>
struct OffsetCalculator {
  static constexpr int MAX_DIMS = 12;
  using stride_t =
      std::conditional_t<signed_strides, std::make_signed_t<index_t>, index_t>;
  using offset_type = memory::Array<stride_t, std::max<int>(NARGS, 1)>;

  // if element_sizes is nullptr, then the strides will be in bytes, otherwise
  // the strides will be in # of elements.
  OffsetCalculator(int dims, const int64_t *sizes,
                   const int64_t *const *strides,
                   const int64_t *element_sizes = nullptr)
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

template <int NARGS, typename index_t = uint32_t>
struct TrivialOffsetCalculator {
  using offset_type = memory::Array<index_t, std::max<int>(NARGS, 1)>;

  offset_type get(index_t linear_idx) const {
    offset_type offsets;
#pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = linear_idx;
    }
    return offsets;
  }
};

} // namespace offset

namespace policy {

template <int arg_index> struct vectorized_load_helper {
  template <typename args_t, typename policy_t, typename offset_t>
  static void apply(policy_t &self, args_t *args, offset_t offset,
                    int args_vec_base) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    auto ptr =
        reinterpret_cast<arg_t *>(self.data[arg_index + 1]) + offset[arg_index];
    auto args_accessor = [&args,
                          args_vec_base](int thread_unroll_idx) -> arg_t & {
      return std::get<arg_index>(args[args_vec_base + thread_unroll_idx]);
    };
    self.load_single_arg(args_accessor, ptr);
  }
};

template <int ITEM_WORK_SIZE, int vec_size, typename data_t,
          typename inp_calc_t>
struct vectorized {
  static_assert(ITEM_WORK_SIZE % vec_size == 0,
                "The workload per thread must be a multiple of vec_size");
  static constexpr int loop_size = ITEM_WORK_SIZE / vec_size;

  data_t data;
  inp_calc_t input_offset_calculator;
  int thread_idx;
  int group_idx;
  int group_items;
  int group_work_size;

  vectorized(data_t data, inp_calc_t ic, int thread_idx, int group_idx,
             int group_items)
      : data(data), input_offset_calculator(ic), thread_idx(thread_idx),
        group_idx(group_idx), group_items(group_items),
        group_work_size(ITEM_WORK_SIZE * group_items) {}

  inline constexpr bool check_inbounds(int thread_work_elem) const {
    return true;
  }

  template <typename accessor_t, typename scalar_t>
  inline void load_single_arg(accessor_t to, scalar_t *from) {
    auto v = load_vector<vec_size>(from, 0);
#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      to(j) = v.val[j];
    }
  }

  template <typename args_t> inline void load(args_t *args) {
    constexpr int arity = std::tuple_size<args_t>::value;
    int group_offset = group_work_size * group_idx;
#pragma unroll
    for (int i = 0; i < loop_size; i++) {
      auto linear_idx =
          group_offset + (thread_idx + i * group_items) * vec_size;
      auto offset = input_offset_calculator.get(linear_idx);
      function::static_unroll<vectorized_load_helper, arity>::with_args(
          *this, args, offset, vec_size * i);
    }
  }

  template <typename scalar_t> inline void store(scalar_t *from) {
    using vec_t = aligned_vector<scalar_t, vec_size>;
    scalar_t *to =
        reinterpret_cast<scalar_t *>(data[0]) + group_work_size * group_idx;
    vec_t *to_ = reinterpret_cast<vec_t *>(to);
#pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * group_items;
      vec_t v;
#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        v.val[j] = from[vec_size * i + j];
      }
      to_[index] = v;
    }
  }
};

template <int arg_index>
struct unroll_load_helper {
  template <
      typename args_t,
      typename policy_t,
      typename offset_t,
      typename loader_t>
  static void apply(
      policy_t& self,
      args_t* args,
      offset_t offset,
      loader_t loader,
      int j,
      int num_outputs) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    std::get<arg_index>(args[j]) = loader.template load<arg_t>(
        self.data[arg_index + num_outputs], offset[arg_index], arg_index);
  }
};

template <
    int ITEM_WORK_SIZE,
    typename data_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t,
    int num_outputs = 1>
struct unroll {
  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  loader_t loader;
  storer_t storer;
  int thread_idx;
  int group_idx;
  int group_items;
  int group_work_size;

  unroll(
      data_t data,
      int remaining,
      inp_calc_t ic,
      out_calc_t oc,
      loader_t l,
      storer_t s,
      int thread_idx,
      int group_idx,
      int group_items)
      : data(data),
        remaining(remaining),
        input_offset_calculator(ic),
        output_offset_calculator(oc),
        loader(l),
        storer(s),
        thread_idx(thread_idx),
        group_idx(group_idx),
        group_items(group_items),
        group_work_size(ITEM_WORK_SIZE * group_items) {}

  inline bool check_inbounds(int thread_work_elem) const {
    return (thread_idx + thread_work_elem * group_items < remaining);
  }

  template <typename args_t>
  inline void load(args_t* args) {
    constexpr int arity = std::tuple_size<args_t>::value;
    int thread_idx_ = thread_idx;
#pragma unroll
    for (int i = 0; i < ITEM_WORK_SIZE; i++) {
      if (thread_idx_ >= remaining) {
        return;
      }
      int linear_idx = thread_idx_ + group_work_size * group_idx;
      auto offset = input_offset_calculator.get(linear_idx);
      function::static_unroll<unroll_load_helper, arity>::with_args(
          *this, args, offset, loader, i, num_outputs);
      thread_idx_ += group_items;
    }
  }

  template <typename scalar_t>
  inline void store(scalar_t* from) {
    int thread_idx_ = thread_idx;
#pragma unroll
    for (int i = 0; i < ITEM_WORK_SIZE; i++) {
      if (thread_idx_ >= remaining) {
        return;
      }
      int linear_idx = thread_idx_ + group_work_size * group_idx;
      int offset = output_offset_calculator.get(linear_idx)[0];
      storer.store(from[i], data[0], offset);
      thread_idx_ += group_items;
    }
  }
};

} // namespace policy

namespace impl {

using namespace memory;
using namespace function;
using namespace offset;
using namespace policy;

template <int WORK_SIZE, typename func_t, typename policy_t>
inline void elementwise_kernel_helper(func_t f, policy_t policy) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  using args_t = typename traits::ArgsTuple;

  return_t results[WORK_SIZE];
  args_t args[WORK_SIZE];

  // load
  policy.load(args);

  // compute
#pragma unroll
  for (int i = 0; i < WORK_SIZE; i++) {
    if (policy.check_inbounds(i)) {
      results[i] = apply(f, args[i]);
    }
  }

  // store
  policy.store(results);
}

template <int ITEM_WORK_SIZE, int vec_size, typename func_t, typename array_t,
          typename inp_calc_t>
void vectorized_elementwise_kernel(sycl::nd_item<1> &item, int numel, func_t fn,
                                   array_t data, inp_calc_t input_calc) {
  int group_items = item.get_local_range(0);
  int thread_idx = item.get_local_id(0);
  int group_idx = item.get_group(0);
  int group_work_size = ITEM_WORK_SIZE * group_items;
  int remaining = numel - group_idx * group_work_size;
  if (remaining < group_work_size) {
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = LoadWithoutCast();
    auto storer = StoreWithoutCast();
    auto policy =
        unroll<ITEM_WORK_SIZE, array_t, decltype(input_calc),
               decltype(output_calc), LoadWithoutCast, StoreWithoutCast>(
            data, remaining, input_calc, output_calc, loader, storer,
            thread_idx, group_idx, group_items);
    elementwise_kernel_helper<ITEM_WORK_SIZE>(fn, policy);
  } else {
    auto policy = vectorized<ITEM_WORK_SIZE, vec_size, array_t, inp_calc_t>(
        data, input_calc, thread_idx, group_idx, group_items);
    elementwise_kernel_helper<ITEM_WORK_SIZE>(fn, policy);
  }
}

template <typename func_t, typename array_t, typename inp_calc_t,
          typename queue_t>
static inline void
launch_vectorized_kernel(queue_t &q, int64_t N, const func_t &fn, array_t data,
                         inp_calc_t input_calc, int vec_size) {
  constexpr auto max_scalar_bytes = impl::max_scalar_size<func_t>();
  static_assert(N > 0 && N <= std::numeric_limits<int32_t>::max());

  int group_size = 256;

#define VEC_LOOPS_KERNEL(vec_size)                                             \
  {                                                                            \
    static_assert(max_scalar_bytes * vec_size <= 16);                          \
    if constexpr (max_scalar_bytes * vec_size <= 16) {                         \
      int group_work_size = group_size * vec_size;                             \
      int num_groups = (N + group_work_size - 1) / group_work_size;            \
      auto event = q.submit([&](sycl::handler &h) {                            \
        h.parallel_for(                                                        \
            sycl::nd_range<1>(sycl::range<1>(num_groups * group_size),         \
                              sycl::range<1>(group_size)),                     \
            [=](sycl::nd_item<1> item) {                                       \
              auto idx = item.get_local_id(0) +                                \
                         item.get_group(0) * item.get_local_range(0);          \
              vectorized_elementwise_kernel<vec_size, vec_size>(               \
                  itemId, N, fn, data, input_calc);                            \
            });                                                                \
      });                                                                      \
      q.wait();                                                                \
    }                                                                          \
  }

  switch (vec_size) {
  case 16: {
    VEC_LOOPS_KERNEL(16);
    break;
  }
  case 8: {
    VEC_LOOPS_KERNEL(8);
    break;
  }
  case 4: {
    VEC_LOOPS_KERNEL(4);
    break;
  }
  case 2: {
    VEC_LOOPS_KERNEL(2);
    break;
  }
  case 1: {
    VEC_LOOPS_KERNEL(1);
    break;
  }
  default:
    break;
  }

#undef VEC_LOOPS_KERNEL
}

} // namespace impl

template <typename func_t, bool signed_strides = false, bool fast_mode = false>
void loops_kernel(TensorIteratorBase& iter, const func_t f) {
  using traits = function::function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  static_assert(iter.can_use_32bit_indexing());
  static_assert(iter.ninputs() >= traits::arity);
  static_assert(iter.noutputs() == 1);

  memory::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();
  bool dynamic_casting = false;
  auto item_of_tile = dpcppMaxWorkItemsPerTile();
  bool latency_case =
      numel <= item_of_tile * 4; /* on tuning for different data types */

  if (!dynamic_casting) {
    if (contiguous) {
      int vec_size = at::native::Memory::can_vectorize_up_to_loop<func_t>(
          getDeviceIdOfCurrentQueue(), data);
      auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
      launch_vectorized_kernel(
          numel, f, data, input_offset_calculator, vec_size);
    } else {
      if constexpr (fast_mode) {
        int vec_size;
        if (!latency_case &&
            can_use_broadcast_vectorize<func_t>(iter, data, vec_size) &&
            !signed_strides) {
          auto input_offset_calculator =
              make_input_offset_calculator<traits::arity, signed_strides>(iter);
          launch_vectorized_kernel(
              numel, f, data, input_offset_calculator, vec_size);
          return;
        }
      }
      auto offset_calc =
          make_offset_calculator<traits::arity + 1, signed_strides>(iter);
      launch_legacy_kernel(numel, [=](int idx) {
        auto offsets = offset_calc.get(idx);
        arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
        *out = invoke(f, &data.data[1], &offsets.data[1], 1);
      });
    }
  } else {
    xpu::dpcpp::Array<ScalarType, traits::arity> dtypes;
    for (int i = 0; i < traits::arity; i++) {
      dtypes[i] = iter.tensor(i + 1).scalar_type();
    }

#define HANDLE_DYNAMIC_CAST(REMOVE_DOUBLE)                                     \
  {                                                                            \
    if (contiguous) {                                                          \
      auto loader =                                                            \
          at::native::Memory::LoadWithCast<traits::arity, REMOVE_DOUBLE>(      \
              dtypes);                                                         \
      auto storer = at::native::Memory::StoreWithCast<REMOVE_DOUBLE>(          \
          iter.tensor(0).scalar_type());                                       \
      auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>(); \
      auto output_offset_calculator = TrivialOffsetCalculator<1>();            \
      launch_unrolled_kernel<UNROLLED_ELEM_PER_WORK_ITEM>(                     \
          numel,                                                               \
          f,                                                                   \
          data,                                                                \
          input_offset_calculator,                                             \
          output_offset_calculator,                                            \
          loader,                                                              \
          storer);                                                             \
    } else {                                                                   \
      at::detail::Array<ScalarType, ntensors> dtypes;                          \
      for (int i = 0; i < ntensors; i++) {                                     \
        dtypes[i] = iter.dtype(i);                                             \
      }                                                                        \
      auto offset_calc =                                                       \
          make_offset_calculator<traits::arity + 1, signed_strides>(iter);     \
      launch_legacy_kernel<UNROLLED_ELEM_PER_WORK_ITEM>(numel, [=](int idx) {  \
        auto offsets = offset_calc.get(idx);                                   \
        void* out = data[0] + offsets[0];                                      \
        arg0_t result = invoke_with_cast<REMOVE_DOUBLE>(                       \
            f, &data.data[1], &offsets.data[1], &dtypes.data[1], 1);           \
        if constexpr (REMOVE_DOUBLE)                                           \
          at::native::Memory::no_double_cast_and_store<arg0_t>(                \
              dtypes[0], out, result);                                         \
        else                                                                   \
          c10::cast_and_store<arg0_t>(dtypes[0], out, result);                 \
      });                                                                      \
    }                                                                          \
  }

#ifndef USE_SPLIT_FP64_LOOPS
    if constexpr (fast_mode)
#endif
    {
      if (!has_double_arg<func_t>(iter)) {
        HANDLE_DYNAMIC_CAST(true)
        return;
      }
    }

    HANDLE_DYNAMIC_CAST(false)

#undef HANDLE_DYNAMIC_CAST
  }
}

} // namespace loops

int main() {

}
