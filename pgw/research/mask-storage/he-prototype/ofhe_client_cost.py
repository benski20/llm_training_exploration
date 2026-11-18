# Production-library client cost per ciphertext at N=4096 (NTT-based C++), to replace numpy-FFT client numbers.
import time, json, os
from openfhe import *
out={}
for t in [65537, 786433]:
    p=CCParamsBFVRNS(); p.SetPlaintextModulus(t); p.SetMultiplicativeDepth(0); p.SetRingDim(4096); p.SetSecurityLevel(HEStd_128_classic)
    try:
        cc=GenCryptoContext(p)
    except Exception as e:
        print('fail',t,str(e)[-120:]); continue
    cc.Enable(PKESchemeFeature.PKE); cc.Enable(PKESchemeFeature.LEVELEDSHE)
    kp=cc.KeyGen(); N=cc.GetRingDimension()
    pts=[cc.MakeCoefPackedPlaintext([i%7]*N) for i in range(200)]
    t0=time.perf_counter(); cts=[cc.Encrypt(kp.secretKey,pt) for pt in pts]; te=(time.perf_counter()-t0)/200
    t0=time.perf_counter()
    for c in cts: d=cc.Decrypt(kp.secretKey,c)
    td=(time.perf_counter()-t0)/200
    w=cc.MakeCoefPackedPlaintext([3])
    t0=time.perf_counter()
    for c in cts: x=cc.EvalMult(c,w)
    tm=(time.perf_counter()-t0)/200
    r=dict(t=t,N=N,logQ=int(cc.GetModulus()).bit_length(),enc_ms=te*1e3,dec_ms=td*1e3,ptmult_ms=tm*1e3,
           enc_ns_per_coeff=te/N*1e9, dec_ns_per_coeff=td/N*1e9)
    print(r); out[str(t)]=r
json.dump(out,open('results_openfhe_client_N4096.json','w'),indent=1)
