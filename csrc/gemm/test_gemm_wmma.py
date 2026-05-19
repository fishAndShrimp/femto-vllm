import torch

import femtovllm

M = 83
K = 59
N = 67
DTYPE = torch.bfloat16


a = torch.rand((M, K), dtype=DTYPE, device="cuda")
b = torch.rand((K, N), dtype=DTYPE, device="cuda")


# a = a * 0 + 1
# b = b * 0 + 2


out_torch = a @ b
out_cuda = femtovllm._C.GemmWmmaCuda(a, b)


print(a)
print(b)
print(out_torch)
print(out_cuda)
print(
    f"{torch.allclose(out_torch, out_cuda)=}",
)
print(
    f"{torch.allclose(out_torch, out_cuda,rtol=1e-2,atol=1e-2)=}",
)
print(f"{(out_torch - out_cuda).abs().max()=}")
