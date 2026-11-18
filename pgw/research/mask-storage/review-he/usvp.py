# primal uSVP (2016 estimate, ADPS16 style) core-SVP 0.292*beta; secret scaled to match error
import math
def delta(b): return ((math.pi*b)**(1/b)*b/(2*math.pi*math.e))**(1/(2*(b-1)))
def est(n,logq,se,ss):
    q=2.0**logq; nu=se/ss  # scale secret coords by nu -> volume factor
    best=None
    for b in range(100,2000,5):
        for m in range(n//4, 3*n, 32):
            d=n+m+1
            # lattice vol: q^m * nu^n ; target norm ~ se*sqrt(b)
            lhs=se*math.sqrt(b)
            logvol=(m*math.log(q)+n*math.log(nu))/d
            rhs=delta(b)**(2*b-d-1)*math.exp(logvol)
            if lhs<=rhs:
                if best is None or b<best[0]: best=(b,m)
                break
        if best: break
    return best, 0.292*best[0] if best else None
for n,lq in [(2048,54),(2560,56),(2560,64),(3072,64),(4096,109)]:
    print(n,lq,est(n,lq,3.24,1.0))
