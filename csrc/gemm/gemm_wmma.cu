#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

constexpr int kMTileSize = 16;
constexpr int kKTileSize = 16;
constexpr int kNTileSize = 16;

template <typename scalar_t>
__global__ void GemmWmmaKernel(

    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    int m,
    int k,
    int n

) {
    ;
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
            GemmWmmaKernel<<<1, 1>>>(
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
