"""(c) Local plaintext compute of 7B linears on the gateway CPU (AMX int8 via torch._int_mm).
A12 activations need 2 int8 limbs (rows doubled); A8 needs 1. Times one layer per group for token batch T.
Also checks the low-bit decode trick numerically."""
import torch,time,json,sys
torch.set_num_threads(2)
G={"qkv":(4608,3584),"o":(3584,3584),"gate_up":(37888,3584),"down":(3584,18944)}
W={g:torch.randint(-128,128,s,dtype=torch.int8) for g,s in G.items()}
def t(f,reps=5):
    f(); best=1e9
    for _ in range(reps):
        s=time.perf_counter(); f(); best=min(best,time.perf_counter()-s)
    return best
out={}
for T in (1,8,32,128,512):
    for limbs in (1,2):
        rows=max(T*limbs,1)
        res={}
        for g,(m,n) in G.items():
            x=torch.randint(-128,128,(rows,n),dtype=torch.int8)
            res[g]=t(lambda: torch._int_mm(x,W[g].t()))
        s_layer=sum(res.values()); s_tok=s_layer*28/T
        out[f"T{T}_limbs{limbs}"]=dict(per_group_s=res,s_per_token=s_tok,tok_per_s_2cores=1/s_tok,
            weight_GBps=sum(m*n for m,n in G.values())/s_layer/1e9)
        print(T,limbs,"s/token %.4f tok/s(2c) %.1f weightBW %.1f GB/s"%(s_tok,1/s_tok,out[f'T{T}_limbs{limbs}']['weight_GBps']),{k:round(v*1e3,2) for k,v in res.items()},flush=True)
# low-bit decode check
K=40;b=29;qmax=2047;m,n=64,3584
w=W["o"][:m].to(torch.int64); x=torch.randint(-qmax,qmax+1,(16,n)); r=torch.randint(0,1<<K,(16,n))
xh=(x+r)&((1<<K)-1); yh=(xh@w.T)&((1<<K)-1); wr_low=(r@w.T)&((1<<b)-1)
d=(yh-wr_low)&((1<<b)-1); d=torch.where(d>=1<<(b-1),d-(1<<b),d)
S=w.abs().sum(1); assert (2*qmax*S+1 <= 1<<b).all()
print("lowbit decode exact:",bool((d==x@w.T).all()))
json.dump(out,open("local_compute_t2.json","w"),indent=1)
