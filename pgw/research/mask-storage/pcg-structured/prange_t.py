"""Minimal noise weight t for dual-LPN masks r = H e (H: n x N, e weight t, odd entries).
Mod-2 reduction gives F2 syndrome decoding [N, k'=N-n, t]. Prange success prob per iteration
= C(N-k', t)/C(N, t); for sublinear weight all ISD variants share this exponent asymptotically
(Canto Torres & Sendrier, PQCrypto 2016). We require log2(1/p) >= TARGET (no credit for poly factors)."""
from math import lgamma, log
def lc(a,b): return (lgamma(a+1)-lgamma(b+1)-lgamma(a-b+1))/log(2)
def bits(N,kp,t): return lc(N,t)-lc(N-kp,t)
import json
out=[]
for TARGET in (128,140):
  for n in (896,3584,8192,18944,28672):
    for ratio in (1.125,1.25,1.5,2.0,3.0,4.0):
        N=int(n*ratio); kp=N-n; t=1
        while bits(N,kp,t)<TARGET: t+=1
        out.append(dict(target=TARGET,n=n,N=N,extra_cols=kp,t=t,bits=round(bits(N,kp,t),1)))
for r in out: print(r)
json.dump(out,open("prange_t.json","w"),indent=1)
