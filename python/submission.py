import torch
from typing import Tuple

input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
output_t = torch.Tensor

def custom_kernel(data: input_t) -> output_t:
    a, b1, b2, _, _, _, sfa_perm, sfb1_perm, sfb2_perm, c_out = data
    m, n, l = c_out.shape

    if l == 1:
        scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)

        a0 = a[:, :, 0]
        b1t = b1[:, :, 0].T
        b2t = b2[:, :, 0].T

        r1 = torch._scaled_mm(a0, b1t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a0, b2t, scale_a, scale_b2, out_dtype=torch.float32)

        return (torch.nn.functional.silu(r1) * r2).to(torch.float16).unsqueeze(-1)

    output = torch.empty((m, n, l), dtype=torch.float16, device="cuda")
    for i in range(l):
        scale_a = sfa_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).reshape(-1)
        r1 = torch._scaled_mm(a[:, :, i], b1[:, :, i].T, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a[:, :, i], b2[:, :, i].T, scale_a, scale_b2, out_dtype=torch.float32)
        output[:, :, i] = (torch.nn.functional.silu(r1) * r2).to(torch.float16)
    return output
