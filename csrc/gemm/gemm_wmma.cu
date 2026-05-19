#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

#include "../utils/constants.cuh"

using namespace nvcuda;

constexpr int kWmmaMKN = 16;
using femtovllm::kWarpSize;

template <typename T>
struct WmmaTypeMapper {};

template <>
struct WmmaTypeMapper<c10::Half> {
    using type = half;
    __device__ __forceinline__ static type from_float(
        float v
    ) {
        return __float2half(v);
    }
};

template <>
struct WmmaTypeMapper<c10::BFloat16> {
    using type = nv_bfloat16;
    __device__ __forceinline__ static type from_float(
        float v
    ) {
        return __float2bfloat16(v);
    }
};

template <typename scalar_t>
__global__ void GemmWmmaKernel(

    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    int m,
    int k,
    int n

) {
    using wmma_scalar_t =
        typename WmmaTypeMapper<scalar_t>::type;

    wmma::fragment<
        wmma::matrix_a,
        kWmmaMKN,
        kWmmaMKN,
        kWmmaMKN,
        wmma_scalar_t,
        wmma::row_major>
        a_frag;

    wmma::fragment<
        wmma::matrix_b,
        kWmmaMKN,
        kWmmaMKN,
        kWmmaMKN,
        wmma_scalar_t,
        wmma::row_major>
        b_frag;

    wmma::fragment<
        wmma::accumulator,
        kWmmaMKN,
        kWmmaMKN,
        kWmmaMKN,
        float>
        acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    __shared__ wmma_scalar_t a_shm[kWmmaMKN * kWmmaMKN];
    __shared__ wmma_scalar_t b_shm[kWmmaMKN * kWmmaMKN];
    __shared__ float c_shm[kWmmaMKN * kWmmaMKN];

    int m_base = blockIdx.x * kWmmaMKN;
    int n_base = blockIdx.y * kWmmaMKN;

    auto a_ptr = reinterpret_cast<const wmma_scalar_t*>(a);
    auto b_ptr = reinterpret_cast<const wmma_scalar_t*>(b);

    for (int k_step = 0; k_step < k; k_step += kWmmaMKN) {
        __syncwarp();
        for (int i = threadIdx.x; i < kWmmaMKN * kWmmaMKN;
             i += kWarpSize) {
            auto m_global = m_base + (i / kWmmaMKN);
            auto k_global = k_step + (i % kWmmaMKN);

            // a_frag = a[m_base][k_step]
            if (m_global < m && k_global < k) {
                a_shm[i] = a_ptr[m_global * k + k_global];
            } else {
                a_shm[i] =
                    WmmaTypeMapper<scalar_t>::from_float(
                        0.0f
                    );
            }
        }
        for (int i = threadIdx.x; i < kWmmaMKN * kWmmaMKN;
             i += kWarpSize) {
            auto k_global = k_step + (i / kWmmaMKN);
            auto n_global = n_base + (i % kWmmaMKN);

            // b_frag = b[k_step][n_base]
            if (k_global < k && n_global < n) {
                b_shm[i] = b_ptr[k_global * n + n_global];
            } else {
                b_shm[i] =
                    WmmaTypeMapper<scalar_t>::from_float(
                        0.0f
                    );
            }
        }
        __syncwarp();

        wmma::load_matrix_sync(a_frag, a_shm, kWmmaMKN);
        wmma::load_matrix_sync(b_frag, b_shm, kWmmaMKN);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(
        c_shm,
        acc_frag,
        // n,
        kWmmaMKN,
        wmma::mem_row_major
    );
    __syncwarp();

    for (int i = threadIdx.x; i < kWmmaMKN * kWmmaMKN;
         i += kWarpSize) {
        auto m_global = m_base + (i / kWmmaMKN);
        auto n_global = n_base + (i % kWmmaMKN);
        if (m_global < m && n_global < n) {
            c[m_global * n + n_global] =
                static_cast<scalar_t>(c_shm[i]);
        }
    }
}

torch::Tensor
GemmWmmaCuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda());
    TORCH_CHECK(b.is_cuda());
    TORCH_CHECK(a.is_contiguous());
    TORCH_CHECK(b.is_contiguous());

    TORCH_CHECK_EQ(a.dim(), 2);
    TORCH_CHECK_EQ(b.dim(), 2);
    TORCH_CHECK_EQ(a.size(1), b.size(0));
    int m = a.size(0);
    int k = a.size(1);
    int n = b.size(1);

    auto c = torch::empty({m, n}, a.options());
    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        a.scalar_type(),
        "GemmWmmaCuda",
        [&]() {
            GemmWmmaKernel<<<
                dim3(
                    (m + kWmmaMKN - 1) / kWmmaMKN,
                    (n + kWmmaMKN - 1) / kWmmaMKN
                ),
                kWarpSize>>>(
                a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(),
                c.data_ptr<scalar_t>(),
                m,
                k,
                n
            );
        }
    );

    return c;
}
