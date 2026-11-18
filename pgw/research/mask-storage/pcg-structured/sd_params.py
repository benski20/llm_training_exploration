# Find minimal noise weight t s.t. F2 syndrome decoding (dual LPN after mod-2 reduction) costs >= 128 bits.
# r = H e in Z_{2^K}^n, H: n x N, e: weight t with odd (unit) entries => r mod 2 is an F2 SD instance [N, N-n, t].
import sys, json, time
from cryptographic_estimators.SDEstimator import SDEstimator
def cost(N, k, w):
    E = SDEstimator(n=N, k=k, w=w, bit_complexities=True, memory_access=0)
    t = E.table  # not used
    return E.fastest_algorithm().time_complexity(), E.fastest_algorithm().__class__.__name__
def cost2(N,k,w):
    E = SDEstimator(n=N, k=k, w=w)
    res = E.estimate()
    best = min(res.items(), key=lambda kv: kv[1]['estimate']['time'])
    return best[1]['estimate']['time'], best[0]
res=[]
for n in [int(a) for a in sys.argv[1].split(',')]:
  for ratio in [1.125,1.25,1.5,2.0,4.0]:
    N=int(n*ratio); k=N-n
    lo,hi=8,min(N-k-1,3000)
    while lo<hi:
        mid=(lo+hi)//2
        c,_=cost2(N,k,mid)
        if c>=128: hi=mid
        else: lo=mid+1
    c,alg=cost2(N,k,lo)
    row=dict(n=n,N=N,k=k,t=lo,bits=round(c,1),alg=alg)
    print(row,flush=True); res.append(row)
json.dump(res,open('sd_params_%s.json'%sys.argv[1].replace(',','_'),'w'),indent=1)
