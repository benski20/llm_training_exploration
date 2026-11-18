"""Bits of W8 r actually needed by the gateway: decoding yq=centered((y_hat-Wr) mod 2^b) is exact iff
2*max|yq|+1 <= 2^b, and |yq_i| <= qmax*sum_j|W8_ij| (guaranteed bound). So per row
b_i = ceil(log2(2*qmax*S_i+1)) bits suffice; W r mod 2^b depends only on r mod 2^b."""
import sys, math, json, torch
from safetensors import safe_open
path=sys.argv[1]
f=safe_open(path,'pt')
A={"A8":127,"A12":2047,"A20":(1<<19)-1}
K={"A8":32,"A12":40,"A20":48}
tot={a:{"rowvar_bits":0,"perlinear_bits":0,"genericbound_bits":0,"K_bits":0} for a in A}
group={}
for k in f.keys():
    if not k.endswith('.weight'): continue
    w=f.get_tensor(k).to(torch.int64); m,n=w.shape; g=k.split('.')[2]
    S=w.abs().sum(1).double()
    for a,q in A.items():
        b=torch.ceil(torch.log2(2*q*S+1)).long()
        bg=math.ceil(math.log2(2*q*127*n+1))
        tot[a]["rowvar_bits"]+=int(b.sum()); tot[a]["perlinear_bits"]+=int(b.max())*m
        tot[a]["genericbound_bits"]+=bg*m; tot[a]["K_bits"]+=K[a]*m
        gg=group.setdefault((a,g),{"m":0,"rowvar":0,"max":0,"generic":bg,"n":n})
        gg["m"]+=m; gg["rowvar"]+=int(b.sum()); gg["max"]=max(gg["max"],int(b.max()))
out={a:{kk:v/8/1e6 for kk,v in t.items()} for a,t in tot.items()}
print(json.dumps({"MB_per_token":out},indent=1))
for (a,g),v in sorted(group.items()):
    print(a,g,"n",v["n"],"mean rowbits %.2f"%(v["rowvar"]/v["m"]),"max",v["max"],"generic",v["generic"],"K",K[a])
