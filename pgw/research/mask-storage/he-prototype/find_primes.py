# Find RNS primes q_d (~2^45, NTT-friendly for N=4096) and q_e with q_d*q_e == 1 mod 2^40 (so Q == 1 mod t, Delta=(Q-1)/t exact).
from nt import isprime
T=1<<40; M=8192
def find(bits_d, max_e_bits=40):
    q=((1<<bits_d)//M)*M+1
    while True:
        q-=M
        if not isprime(q): continue
        inv=pow(q,-1,T)
        if inv.bit_length()>=max_e_bits-1 and isprime(inv): return q,inv
if __name__=='__main__':
    qd,qe=find(45); print(qd,qd.bit_length(),qe,qe.bit_length(), (qd*qe)%T, (qd*qe).bit_length(), qe%M)
