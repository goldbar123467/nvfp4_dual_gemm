import torch
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch.set_num_threads(1)

def custom_kernel(data):
    a, b1, b2 = data[0], data[1], data[2]
    sfa_perm, sfb1_perm, sfb2_perm = data[6], data[7], data[8]
    m, n, l = data[9].shape

    if l == 1:
        scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).view(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).view(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).view(-1)

        a0, b1t, b2t = a[:, :, 0], b1[:, :, 0].T, b2[:, :, 0].T

        # Try FP16 output directly from scaled_mm
        r1 = torch._scaled_mm(a0, b1t, scale_a, scale_b1, out_dtype=torch.float16)
        r2 = torch._scaled_mm(a0, b2t, scale_a, scale_b2, out_dtype=torch.float16)

        # Manual silu in FP16
        return (r1 * torch.sigmoid(r1) * r2).unsqueeze(-1)

    output = torch.empty((m, n, l), dtype=torch.float16, device="cuda")
    for i in range(l):
        scale_a = sfa_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).view(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).view(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).view(-1)
        r1 = torch._scaled_mm(a[:, :, i], b1[:, :, i].T, scale_a, scale_b1, out_dtype=torch.float16)
        r2 = torch._scaled_mm(a[:, :, i], b2[:, :, i].T, scale_a, scale_b2, out_dtype=torch.float16)
        output[:, :, i] = r1 * torch.sigmoid(r1) * r2
    return output
