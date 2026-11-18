"""Range-fetch selected bf16 tensors from HF safetensors shards; quantize per-row int8 (absmax) and
measure the bits of W8 r the gateway really needs (A8/A12/A20)."""
import json, struct, sys, math, urllib.request, numpy as np, torch
repo=sys.argv[1]; layers=[int(x) for x in sys.argv[2].split(',')]
base=f"https://huggingface.co/{repo}/resolve/main/"
idx=json.load(urllib.request.urlopen(base+"model.safetensors.index.json"))["weight_map"]
hdr_cache={}
def header(shard):
    if shard not in hdr_cache:
        r=urllib.request.Request(base+shard,headers={"Range":"bytes=0-7"}); n=struct.unpack('<Q',urllib.request.urlopen(r).read())[0]
        r=urllib.request.Request(base+shard,headers={"Range":f"bytes=8-{8+n-1}"}); hdr_cache[shard]=(n,json.loads(urllib.request.urlopen(r).read()))
    return hdr_cache[shard]
def get(name):
    shard=idx[name]; n,h=header(shard); e=h[name]; a,b=e["data_offsets"]
    r=urllib.request.Request(base+shard,headers={"Range":f"bytes={8+n+a}-{8+n+b-1}"})
    raw=urllib.request.urlopen(r).read(); assert e["dtype"]=="BF16"
    t=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(e["shape"]).float(); return t
A={"A8":127,"A12":2047,"A20":(1<<19)-1}
res={}
for L in layers:
    for g,names in {"qkv":["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"],"o":["self_attn.o_proj"],
                    "gate_up":["mlp.gate_proj","mlp.up_proj"],"down":["mlp.down_proj"]}.items():
        w=torch.cat([get(f"model.layers.{L}.{nm}.weight") for nm in names])
        w8=torch.round(w/w.abs().amax(1,keepdim=True)*127).clamp(-127,127)
        S=w8.abs().sum(1).double(); m,n=w8.shape
        for a,q in A.items():
            b=torch.ceil(torch.log2(2*q*S+1))
            res.setdefault(f"{a}.{g}",[]).append(dict(layer=L,m=m,n=n,mean_bits=float(b.mean()),max_bits=int(b.max()),generic=math.ceil(math.log2(2*q*127*n+1))))
        print(L,g,m,n,"A12 mean %.2f max %d"%(res[f'A12.{g}'][-1]['mean_bits'],res[f'A12.{g}'][-1]['max_bits']),flush=True)
        del w,w8
json.dump(res,open(sys.argv[3],"w"),indent=1)
