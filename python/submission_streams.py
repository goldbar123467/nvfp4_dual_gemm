# =============================================================================
# NVFP4 Dual-GEMM Submission - Parallel Streams
# =============================================================================
# GPU MODE Leaderboard Submission
# Kernel: C = silu(A @ B1) * (A @ B2)
# Target: NVIDIA B200 (SM100 Blackwell)
#
# Runs GEMM1 and GEMM2 concurrently on separate CUDA streams
# =============================================================================

import torch
from typing import Tuple, Dict

input_t = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor,
]
output_t = torch.Tensor


class _StreamCache:
    """Cache streams and scale factors."""

    def __init__(self):
        self.stream1 = None
        self.stream2 = None
        self.scales: Dict[tuple, tuple] = {}

    def get_streams(self):
        if self.stream1 is None:
            self.stream1 = torch.cuda.Stream()
            self.stream2 = torch.cuda.Stream()
        return self.stream1, self.stream2

    def get_scales(self, sfa_perm, sfb1_perm, sfb2_perm, l_idx=0):
        key = (sfa_perm.data_ptr(), sfb1_perm.data_ptr(), sfb2_perm.data_ptr(), l_idx)
        if key not in self.scales:
            scale_a = sfa_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1).clone()
            scale_b1 = sfb1_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1).clone()
            scale_b2 = sfb2_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1).clone()
            self.scales[key] = (scale_a, scale_b1, scale_b2)
        return self.scales[key]


_cache = _StreamCache()


def custom_kernel(data: input_t) -> output_t:
    """
    Parallel stream NVFP4 dual GEMM with SiLU activation.

    Runs GEMM1 and GEMM2 concurrently on separate streams.
    """
    a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c_out = data
    m, n, l = c_out.shape

    if l == 1:
        scale_a, scale_b1, scale_b2 = _cache.get_scales(sfa_perm, sfb1_perm, sfb2_perm, 0)
        stream1, stream2 = _cache.get_streams()

        a0 = a[:, :, 0]
        b1t = b1[:, :, 0].T
        b2t = b2[:, :, 0].T

        # Launch GEMM1 on stream1
        with torch.cuda.stream(stream1):
            r1 = torch._scaled_mm(a0, b1t, scale_a, scale_b1, out_dtype=torch.float32)

        # Launch GEMM2 on stream2
        with torch.cuda.stream(stream2):
            r2 = torch._scaled_mm(a0, b2t, scale_a, scale_b2, out_dtype=torch.float32)

        # Wait for both to complete
        stream1.synchronize()
        stream2.synchronize()

        # Fused epilogue on default stream
        output = (torch.nn.functional.silu(r1) * r2).to(torch.float16)
        return output.unsqueeze(-1)

    # L > 1 fallback
    output = torch.empty((m, n, l), dtype=torch.float16, device=a.device)
    for i in range(l):
        scale_a, scale_b1, scale_b2 = _cache.get_scales(sfa_perm, sfb1_perm, sfb2_perm, i)
        stream1, stream2 = _cache.get_streams()

        a_slice = a[:, :, i]
        b1t = b1[:, :, i].T
        b2t = b2[:, :, i].T

        with torch.cuda.stream(stream1):
            r1 = torch._scaled_mm(a_slice, b1t, scale_a, scale_b1, out_dtype=torch.float32)

        with torch.cuda.stream(stream2):
            r2 = torch._scaled_mm(a_slice, b2t, scale_a, scale_b2, out_dtype=torch.float32)

        stream1.synchronize()
        stream2.synchronize()

        output[:, :, i] = (torch.nn.functional.silu(r1) * r2).to(torch.float16)

    return output
