"""Cost model per 1M Qwen2.5-7B tokens/day. Labels: M=measured here, C=cited, E=estimate, A=analytic."""
import json
TOK_DAY=1e6; AVG=TOK_DAY/86400; PEAK=3*AVG                          # E: peak 3x mean (as in sizing.md)
L=28; G={"qkv":(4608,3584),"o":(3584,3584),"gate_up":(37888,3584),"down":(3584,18944)}
MB_K40=L*sum(m for m,n in G.values())*5/1e6                            # A 6.95
MB_LOW=5.066                                                            # M (3 of 28 layers sampled, per-linear max bits)
MB_LOW_A8=4.371                                                         # M
VCPU_H=0.0892/2                                                         # C: c7i.large $0.0892/h, 2 vCPU (DoiT)
T4_H=0.526                                                              # C: g4dn.xlarge on-demand list (AWS)
T4_TOK_H=169e3                                                          # given
JIT_TOK_S_VCPU=6.5                                                      # M: proto 7B K40, 12.95 tok/s on 2 vCPU (B=256)
JIT_TOK_S_VCPU_LOW=JIT_TOK_S_VCPU*5/4                                   # E: 4 limbs instead of 5 (GEMM-bound)
TB_MO={"gp3":80,"st1":45,"sc1":15,"s3_std":23,"onprem_hdd":0.6}         # C (AWS list via aggregators) / E onprem
LOCAL_TOK_S_VCPU=25                                                     # M: all 7B linears, 2 limbs, batched T>=32: 50-72 tok/s per 2 vCPU
def stored(mb,buf_days=2,frac=1.0,tier="gp3"):
    tb=TOK_DAY*frac*mb*buf_days/1e6
    fact=TOK_DAY*frac/T4_TOK_H*T4_H*30
    return dict(storage_TB=round(tb,2),storage_usd_mo=round(tb*TB_MO[tier]),factory=f"{TOK_DAY*frac/T4_TOK_H:.1f} T4-h/day",factory_usd_mo=round(fact),cpu_vcpu=0)
def jit(rate,mb,frac=1.0,buf_min=15,provision="peak"):
    need=(PEAK if provision=="peak" else AVG)*frac/rate
    buf_tb=PEAK*frac*buf_min*60*mb/1e6
    return dict(storage_TB=round(buf_tb,3),storage_usd_mo=round(buf_tb*TB_MO["gp3"]),cpu_vcpu=round(need,2),cpu_usd_mo=round(need*VCPU_H*730))
rows={}
rows["baseline stored, 2-day buffer, gp3"]=stored(MB_K40)
rows["baseline stored, 2-day buffer, S3"]=stored(MB_K40,tier="s3_std")
rows["low-bit Wr storage (K40A12, 29/31 bits)"]=stored(MB_LOW)
rows["low-bit + K32A8"]=stored(MB_LOW_A8)
for wl,f in {"chat":0.45,"rag":0.51,"agent":0.16}.items():   # fraction of tokens still needing masks after prefix caching (E from cited stats)
    rows[f"prefix cache ({wl}) + low-bit, stored 2d"]=stored(MB_LOW,frac=f)
rows["JIT CPU factory (proto), provisioned for 3x peak, 15-min buffer"]=jit(JIT_TOK_S_VCPU,MB_K40)
rows["JIT CPU + low-bit (4 limbs)"]=jit(JIT_TOK_S_VCPU_LOW,MB_LOW)
for wl,f in {"chat":0.45,"rag":0.51,"agent":0.16}.items():
    rows[f"JIT + low-bit + prefix cache ({wl})"]=jit(JIT_TOK_S_VCPU_LOW,MB_LOW,frac=f)
rows["reference: all linears locally on CPU (no PGW)"]=dict(storage_TB=0,cpu_vcpu=round(PEAK/LOCAL_TOK_S_VCPU,2),cpu_usd_mo=round(PEAK/LOCAL_TOK_S_VCPU*VCPU_H*730))
for k,v in rows.items(): print(f"{k:62s} {v}")
# (c) MB saved per local GOP
print("\n(c) per-group, 7B, K40A12, per token")
tot_mb=MB_K40; tot_gop=sum(2*m*n for m,n in G.values())*L/1e9
for g,(m,n) in G.items():
    mb=L*m*5/1e6; gop=2*L*m*n/1e9
    print(f"{g:8s} masks {mb:.3f} MB ({mb/tot_mb:.1%})  local {gop:.2f} GOP ({gop/tot_gop:.1%})  MB saved/GOP {mb/gop:.3f}  low-bit MB/GOP {mb*(29 if g!='down' else 31)/40/gop:.3f}")
G70={"qkv":(10240,8192),"o":(8192,8192),"gate_up":(57344,8192),"down":(8192,28672)}
t70=sum(m for m,n in G70.values())*80*5/1e6; g70=sum(2*m*n for m,n in G70.values())*80/1e9
print(f"70B masks {t70:.1f} MB/token, linears {g70:.0f} GOP/token(1 limb); factory K40 {5*g70:.0f} GOP/token")
for g,(m,n) in G70.items(): print(f"70B {g:8s} masks {80*m*5/1e6:.2f} MB  local {2*80*m*n/1e9:.1f} GOP  MB/GOP {m*5/(2*m*n)*1e3:.3f}")
print(f"7B linears {tot_gop:.1f} GOP/token(1 limb); factory K40 {5*tot_gop:.1f} GOP")
json.dump(rows,open("cost_table.json","w"),indent=1)
