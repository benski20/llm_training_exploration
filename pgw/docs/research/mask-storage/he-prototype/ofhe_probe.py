import time, os
from openfhe import *
#print([m for m in dir(CryptoContext) if 'Eval' in m and ('Mult' in m or 'Sum' in m or 'Linear' in m)][:40])
p = CCParamsBFVRNS()
p.SetPlaintextModulus(1<<40)
p.SetMultiplicativeDepth(int(os.environ.get("DEPTH","1")))
#p.SetRingDim(4096)
p.SetSecurityLevel(HEStd_128_classic)
try:
    pass
except Exception as e: print('scal',e)
cc = GenCryptoContext(p)
cc.Enable(PKESchemeFeature.PKE); cc.Enable(PKESchemeFeature.LEVELEDSHE)
print('N', cc.GetRingDimension(), 'Q', cc.GetModulus(), 'bits', int(cc.GetModulus()).bit_length())
kp = cc.KeyGen()
import random
N=cc.GetRingDimension()
m=[random.randrange(-(1<<39),1<<39) for _ in range(N)]
pt=cc.MakeCoefPackedPlaintext(m)
t0=time.time(); ct=cc.Encrypt(kp.secretKey, pt); print('enc', time.time()-t0)
w=cc.MakeCoefPackedPlaintext([-77])
t0=time.time(); 
for _ in range(100): c2=cc.EvalMult(ct,w)
print('ptmult per op', (time.time()-t0)/100)
t0=time.time()
for _ in range(100): c3=cc.EvalAdd(c2,ct)
print('add per op', (time.time()-t0)/100)
t0=time.time(); d=cc.Decrypt(kp.secretKey,c3); print('dec',time.time()-t0)
d.SetLength(N)
v=d.GetCoefPackedValue()
T=1<<40
ok=all(((a-(-77*b+b))%T)==0 for a,b in zip(v,m))
print('ok',ok)
