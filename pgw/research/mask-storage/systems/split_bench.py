import torch,time,sys
torch.set_num_threads(2)
B=int(sys.argv[1]); K=int(sys.argv[2]); limbs=K//8; mask=(1<<K)-1
for (name,m,n) in [("gate_up7b",37888,3584),("down7b",3584,18944)]:
    W=torch.randint(-128,128,(m,n),dtype=torch.int8)
    d=torch.randint(-128,128,(limbs*B,n),dtype=torch.int8)
    y=torch._int_mm(d,W.t())
    def comb(y):
        acc=y[:B].to(torch.int64)
        for i in range(1,limbs): acc+= y[i*B:(i+1)*B].to(torch.int64)<<(8*i)
        return acc&mask
    comb(y)
    tg=min((lambda t:(torch._int_mm(d,W.t()),time.perf_counter()-t)[1])(time.perf_counter()) for _ in range(3))
    tc=min((lambda t:(comb(y),time.perf_counter()-t)[1])(time.perf_counter()) for _ in range(3))
    tr=min((lambda t:(torch.randint(-128,128,(limbs*B,n),dtype=torch.int8),time.perf_counter()-t)[1])(time.perf_counter()) for _ in range(3))
    print(f"{name} B={B} K={K} gemm {tg:.3f}s ({2*limbs*B*m*n/tg/1e12:.2f} TOPS) combine {tc:.3f}s randint {tr:.4f}s")
