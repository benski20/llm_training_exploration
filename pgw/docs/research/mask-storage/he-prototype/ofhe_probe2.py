import time, os, random
from openfhe import *
def isprime(n):
    if n<2: return False
    for p in [2,3,5,7,11,13,17,19,23,29,31,37]:
        if n%p==0: return n==p
    d=n-1;s=0
    while d%2==0: d//=2;s+=1
    for a in [2,3,5,7,11,13,17,19,23,29,31,37]:
        x=pow(a,d,n)
        if x in(1,n-1): continue
        for _ in range(s-1):
            x=x*x%n
            if x==n-1: break
        else: return False
    return True
M=1<<15
t=((1<<40)//M)*M+1
while not isprime(t): t+=M
print('t',t, t.bit_length())
p = CCParamsBFVRNS(); p.SetPlaintextModulus(t); p.SetMultiplicativeDepth(int(os.environ.get('DEPTH','0')))
p.SetSecurityLevel(HEStd_128_classic)
if os.environ.get('N'): p.SetRingDim(int(os.environ['N']))
cc = GenCryptoContext(p); cc.Enable(PKESchemeFeature.PKE); cc.Enable(PKESchemeFeature.LEVELEDSHE)
N=cc.GetRingDimension(); print('N',N,'logQ',int(cc.GetModulus()).bit_length())
kp=cc.KeyGen()
m=[random.randrange(0,t) for _ in range(N)]
pt=cc.MakePackedPlaintext(m)
t0=time.time(); ct=cc.Encrypt(kp.secretKey,pt); print('enc',time.time()-t0)
w=cc.MakePackedPlaintext([t-77]*N)
t0=time.time()
for _ in range(50): c2=cc.EvalMult(ct,w)
print('ct*pt',(time.time()-t0)/50)
try:
    t0=time.time()
    for _ in range(50): c4=cc.EvalMult(ct,-77)
    print('ct*int',(time.time()-t0)/50)
except Exception as e: print('ct*int fail',str(e)[:100])
t0=time.time()
for _ in range(50): c3=cc.EvalAdd(c2,ct)
print('add',(time.time()-t0)/50)
try:
    t0=time.time(); c5=cc.EvalLinearWSum([ct,ct],[3,4]); print('wsum ok',time.time()-t0)
except Exception as e: print('wsum fail',str(e)[:120])
t0=time.time(); d=cc.Decrypt(kp.secretKey,c3); print('dec',time.time()-t0)
d.SetLength(N); v=d.GetPackedValue()
print('ok', all((a-(-76*b))%t==0 for a,b in zip(v,m)))
s=Serialize(ct,BINARY) if 'Serialize' in globals() else None
