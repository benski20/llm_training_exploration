import sys, time
from cryptographic_estimators.SDEstimator import SDEstimator, Prange, Stern, Dumer, BallCollision, BJMM
N,k,w = map(int, sys.argv[1:4])
t0=time.time()
import cryptographic_estimators.SDEstimator as S
keep={"Prange","Stern","Dumer","BallCollision","BJMM","MayOzerov","BothMay"}
exc=[getattr(S,a) for a in dir(S) if a[:1].isupper() and a not in keep and a not in ("SDEstimator","SDProblem","SDAlgorithm") and isinstance(getattr(S,a),type)]
E = SDEstimator(n=N, k=k, w=w, excluded_algorithms=exc)
r = E.estimate()
for a,v in r.items(): print(a, round(v['estimate']['time'],1))
print("secs", round(time.time()-t0,1))
