"""Prototype JIT mask factory on CPU (AMX int8 via torch._int_mm).
r is generated directly as K/8 uniform int8 digit planes (r = sum d_i 256^i mod 2^K is uniform),
so no digit decomposition is needed. Verifies exactness on a small slice vs python ints.
Times one transformer layer of a given shape for B slots and extrapolates to L layers."""
import torch, time, sys, json, argparse
p=argparse.ArgumentParser()
p.add_argument("--model",default="qwen2.5-7b"); p.add_argument("--B",type=int,default=256)
p.add_argument("--K",type=int,default=40); p.add_argument("--threads",type=int,default=2)
p.add_argument("--reps",type=int,default=3); p.add_argument("--groups",default="qkv,o,gate_up,down")
a=p.parse_args(); torch.set_num_threads(a.threads)
SH={ # (hidden, inter, q+2kv out, layers)
 "qwen2.5-0.5b":(896,4864,896+2*128,24),
 "qwen2.5-7b":(3584,18944,3584+2*512,28),
 "llama3-70b":(8192,28672,8192+2*1024,80)}
h,I,qkv,L=SH[a.model]
G={"qkv":(qkv,h),"o":(h,h),"gate_up":(2*I,h),"down":(h,I)}
groups=a.groups.split(","); limbs=a.K//8; mask=(1<<a.K)-1
W={g:torch.randint(-128,128,G[g],dtype=torch.int8) for g in groups}
def make(g,B):
    m,n=G[g]
    d=torch.randint(-128,128,(limbs*B,n),dtype=torch.int8)   # stand-in for ChaCha20 output bytes
    y=torch._int_mm(d,W[g].t())                               # int32 (limbs*B, m); n<=65536 so exact
    acc=y[:B].to(torch.int64)
    for i in range(1,limbs):
        acc+= y[i*B:(i+1)*B].to(torch.int64) << (8*i)
    acc&=mask
    return d,acc
# exactness check on small piece
d,acc=make("o",4); m,n=G["o"]
for row in range(2):
    r=sum(int(d[i*4+row].to(torch.int64).dot(torch.ones(1,dtype=torch.int64).expand(0)) if False else 0) for i in range(0))
    rv=[sum(int(d[i*4+row,j])*(256**i) for i in range(limbs))%(1<<a.K) for j in range(n)]
    for o in range(3):
        exp=sum(int(W["o"][o,j])*rv[j] for j in range(n))%(1<<a.K)
        assert exp==int(acc[row,o]),"inexact"
for g in groups: make(g,a.B)
res={}
for g in groups:
    ts=[]
    for _ in range(a.reps):
        t=time.perf_counter(); make(g,a.B); ts.append(time.perf_counter()-t)
    res[g]=min(ts)
per_layer=sum(res.values()); per_tok=per_layer*L/a.B
macs=sum(G[g][0]*G[g][1] for g in groups)*L*limbs
out=dict(model=a.model,K=a.K,B=a.B,threads=a.threads,groups=groups,per_group_s_per_layer=res,
 s_per_token=per_tok,tokens_per_s=1/per_tok,tokens_per_s_per_core=1/per_tok/a.threads,
 effective_TOPS=2*macs/per_tok/1e12, bytes_per_token=sum(G[g][0] for g in groups)*L*a.K//8)
print(json.dumps(out))
