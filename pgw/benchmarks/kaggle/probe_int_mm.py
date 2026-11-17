"""Which torch._int_mm shapes/layouts work on this GPU (cuBLASLt int8 support differs by GPU and CUDA)."""
import json

import torch

res = {}
for (t, n, m) in [(32, 8, 8), (32, 8, 32), (32, 16, 16), (32, 32, 32), (32, 64, 64), (32, 64, 8), (32, 8, 64),
                  (17, 64, 64), (24, 64, 64), (32, 896, 1152), (32, 3584, 4608)]:
    a = torch.randint(-128, 128, (t, n), dtype=torch.int8, device="cuda")
    w = torch.randint(-128, 128, (m, n), dtype=torch.int8, device="cuda")
    ref = (a.cpu().double() @ w.cpu().double().t()).long()
    for layout, b in (("col_major", w.t()), ("row_major", w.t().contiguous())):
        try:
            ok = bool(torch.equal(torch._int_mm(a, b).cpu().long(), ref))
            res[f"{t}x{n}x{m} {layout}"] = "exact" if ok else "WRONG"
        except Exception as exc:  # noqa: BLE001
            res[f"{t}x{n}x{m} {layout}"] = f"{type(exc).__name__}: {str(exc)[:120]}"
print(json.dumps(res, indent=1))
