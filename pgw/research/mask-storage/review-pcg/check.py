import os,time,math
os.environ["OMP_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"
import numpy as np
from math import lgamma,log
def lb(n,k): return (lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1))/log(2)
# plaintext local linear cost (float32 GEMM exact: |x|<2^11, |W|<=127, n<=18944 -> |y|<2^11*127*18944 ~2^32 not exact in f32; use f64 single pass: exact <2^53)
m,n,b=8192,3584,128
W=np.random.randint(-127,128,(m,n)).astype(np.float64); X=np.random.randint(-2048,2048,(n,b)).astype(np.float64)
for dt in (np.float64,np.float32):
  Wd=W.astype(dt);Xd=X.astype(dt); Wd@Xd
  t=time.perf_counter(); 
  for _ in range(5): Wd@Xd
  el=(time.perf_counter()-t)/5
  print(dt.__name__, "ps/MAC", el/(m*n*b)*1e12, "7B s/token(2thr)", 6.525e9*el/(m*n*b))
# Prange and Stern (p=2,l opt) bit cost for dual-LPN = SD with code length N, dim k=N-n (syndrome length n), weight w
def prange(N,k,w): return lb(N,w)-lb(N-k,w)
def stern(N,k,w):
  best=1e9
  for p in range(0,9,2):
    for l in range(0,80):
      if N-k-l<w-p or p/2>k/2: continue
      L=lb((k+l)//2,p//2)
      iters=lb(N,w)-lb(N-k-l,w-p)-2*lb((k+l)//2,p//2)
      c=iters+ max(L, 2*L-l)+math.log2(N)
      best=min(best,c)
  return best
for n in (3584,18944):
  for N,w in ((2*n,150),(4*n,75),(2*n,300)):
    print(n,N,w,"prange %.1f stern %.1f"%(prange(N,N-n,w),stern(N,N-n,w)))
