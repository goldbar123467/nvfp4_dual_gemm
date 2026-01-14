import torch
from typing import Tuple

input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
output_t = torch.Tensor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

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

        return (torch.nn.functional.silu(r1) * r2).half().unsqueeze(-1)

    out = torch.empty((m, n, l), dtype=torch.float16, device=a.device)
    for i in range(l):
        scale_a = sfa_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).reshape(-1)
        r1 = torch._scaled_mm(a[:, :, i], b1[:, :, i].T, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a[:, :, i], b2[:, :, i].T, scale_a, scale_b2, out_dtype=torch.float32)
        out[:, :, i] = (torch.nn.functional.silu(r1) * r2).half()
    return out
