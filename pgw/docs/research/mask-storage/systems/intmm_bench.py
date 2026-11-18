import torch, time, sys
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv)>1 else 2)
def bench(B,m,n,dtype="int8",layout="nt"):
    if dtype=="int8":
        a=torch.randint(-128,128,(B,n),dtype=torch.int8); w=torch.randint(-128,128,(m,n),dtype=torch.int8)
        wt = w.t() if layout=="nt" else w.t().contiguous()
        f=lambda: torch._int_mm(a,wt)
    else:
        a=torch.randn(B,n,dtype=dtype); w=torch.randn(m,n,dtype=dtype); wt=w.t().contiguous()
        f=lambda: a@wt
    f(); f()
    t=time.perf_counter(); it=0
    while time.perf_counter()-t<2.0: f(); it+=1
    dt=(time.perf_counter()-t)/it
    return 2*B*m*n/dt/1e12
for B in (64,256,1280,4096):
  for (m,n) in [(4864,896),(9728,896),(37888,3584),(3584,18944)]:
    if B*max(m,n)*4>2e9: continue
    print(f"B={B} m={m} n={n} int8 nt {bench(B,m,n):.3f} TOPS  int8 contig {bench(B,m,n,layout='c'):.3f}", flush=True)
for B in (1024,):
    print("bf16", bench(B,4864,896,torch.bfloat16), "f32", bench(B,4864,896,torch.float32))
