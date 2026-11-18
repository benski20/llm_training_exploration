# Library baseline: OpenFHE BFVrns, token-batched column packing.
# ct_i = Enc(r_i[tok_0..tok_{N-1}]) for each input feature i; out_j = sum_i W_ji * ct_i (plaintext-constant mults).
import time, os, json, random, sys
import numpy as np
from openfhe import *
from safetensors.numpy import load_file
sys.path.insert(0, os.path.dirname(__file__))
from nt import isprime  # noqa
DOUT=int(os.environ.get('DOUT','8'))
M=1<<15
t=((1<<40)//M)*M+1
while not isprime(t): t+=M
p=CCParamsBFVRNS(); p.SetPlaintextModulus(t); p.SetMultiplicativeDepth(0); p.SetSecurityLevel(HEStd_128_classic)
p.SetKeySwitchTechnique(BV)
cc=GenCryptoContext(p); cc.Enable(PKESchemeFeature.PKE); cc.Enable(PKESchemeFeature.LEVELEDSHE)
N=cc.GetRingDimension(); logQ=int(cc.GetModulus()).bit_length()
kp=cc.KeyGen()
Wfull=load_file('/root/.pgw-client-demo/packs/qwen2.5-0.5b-instruct/linears-00000-of-00001.safetensors')['linears.0.gate_up.weight']
W=Wfull[:DOUT,:896].astype(np.int64); din=W.shape[1]
rng=np.random.default_rng(1)
R=rng.integers(0,t,size=(din,N),dtype=np.int64)
t0=time.time(); cts=[cc.Encrypt(kp.secretKey, cc.MakePackedPlaintext(R[i].tolist())) for i in range(din)]; tenc=time.time()-t0
# plaintext constants
t0=time.time(); P={}
for w in np.unique(W): P[int(w)]=cc.MakePackedPlaintext([int(w)%t]*N)
tpt=time.time()-t0
t0=time.time(); outs=[]
for j in range(DOUT):
    acc=cc.EvalMult(cts[0],P[int(W[j,0])])
    for i in range(1,din): acc=cc.EvalAdd(acc, cc.EvalMult(cts[i],P[int(W[j,i])]))
    outs.append(acc)
tsrv=time.time()-t0
t0=time.time(); dec=[]
for c in outs:
    d=cc.Decrypt(kp.secretKey,c); d.SetLength(N); dec.append(d.GetPackedValue())
tdec=time.time()-t0
dec=np.array(dec,dtype=object)%t
ref=(W.astype(object)@R.astype(object))%t
ok=bool((dec==ref).all())
ctbytes=len(Serialize(cts[0],BINARY)) if 'Serialize' in globals() else None
try:
    ctbytes=len(SerializeToString(cts[0],BINARY))
except Exception as e:
    ctbytes=str(e)[:80]
res=dict(lib='openfhe-1.6.1 BFVrns',N=N,logQ=logQ,t=t,din=din,dout=DOUT,ok=ok,enc_s_per_ct=tenc/din,
 server_s_per_mac=tsrv/(din*DOUT), dec_s_per_ct=tdec/DOUT, ct_bytes=ctbytes,
 server_s_per_token_mac=tsrv/(din*DOUT)/N)
print(json.dumps(res,indent=1)); json.dump(res,open('results_openfhe.json','w'),indent=1)
