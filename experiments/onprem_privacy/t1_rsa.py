"""T1: check the textbook-RSA homomorphism demo, and show why textbook RSA is not secure as used."""
import json
n, e = 33, 3            # p=3, q=11, phi=20, gcd(3,20)=1
phi = 20; d = pow(e, -1, phi)
enc = lambda m: pow(m, e, n); dec = lambda c: pow(c, d, n)
c2, c5 = enc(2), enc(5)
prod = (c2 * c5) % n
res = {"enc2": c2, "enc5": c5, "product_ct": prod, "decrypts_to": dec(prod),
       "enc10": enc(10), "homomorphic_ok": dec(prod) == 10}
# Weakness 1: deterministic -> attacker with a dictionary of likely plaintexts reads everything
table = {enc(m): m for m in range(n)}
res["dictionary_attack_recovers_all"] = all(table[enc(m)] == m for m in range(n))
# Weakness 2: fixed points: m^3 = m mod 33 means 'ciphertext == plaintext'
res["fixed_points"] = [m for m in range(n) if enc(m) == m]
print(json.dumps(res, indent=1)); json.dump(res, open("results/t1_rsa.json", "w"), indent=1)
