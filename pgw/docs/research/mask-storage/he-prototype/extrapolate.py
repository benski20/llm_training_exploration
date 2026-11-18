import json
N=4096
m7 = json.load(open('results64_slice7b.json')); mc = json.load(open('results64_qwen05_gate_C.json'))
models = {
 'qwen2.5-7b': dict(L=28, outs=[4608,3584,37888,3584], ins=[3584,3584,3584,18944]),
 'llama-3-70b': dict(L=80, outs=[10240,8192,57344,8192], ins=[8192,8192,8192,28672]),
}
# measured per-unit costs (2 BLAS threads; numpy FFT 1 thread)
srv_ns = {v: m7['variant_'+v]['server_ns_per_token_MAC'] for v in 'AB'}; srv_ns['C'] = srv_ns['B']
enc_ns = {v: m7['variant_'+v]['client_ns_per_token_per_elem(enc per in, dec per out)'][0] for v in 'AB'}
enc_ns['C'] = mc['variant_C']['client_ns_per_token_per_elem(enc per in, dec per out)'][0]
dec_ns = {v: m7['variant_'+v]['client_ns_per_token_per_elem(enc per in, dec per out)'][1] for v in 'AB'}; dec_ns['C'] = dec_ns['B']
up_bits = dict(A=64, B=64, C=25); down_bits = dict(A=44+50, B=44, C=44); passes = dict(A=16, B=8, C=8)
T4_FACTORY_K40_TOK_PER_H_7B = 169_000   # given in task (measured by the team), 5 int8 GEMM passes
out = {}
for name, m in models.items():
    L = m['L']; so = L*sum(m['outs']); si = L*sum(m['ins']); macs = L*sum(o*i for o,i in zip(m['outs'], m['ins']))
    for v in 'ABC':
        r = dict(outputs_per_token=so, inputs_per_token=si, MACs_per_token=macs,
          stored_Wr_today_MB_per_token=so*5/1e6,
          upload_MB_per_token=si*up_bits[v]/8/1e6, download_MB_per_token=so*down_bits[v]/8/1e6,
          server_cpu_s_per_token_2thr_float64=macs*srv_ns[v]*1e-9,
          server_cpu_s_per_token_int8_1TOPS_derived=passes[v]*2*macs/1e12,
          client_cpu_s_per_token_numpy=(si*enc_ns[v]+so*dec_ns[v])*1e-9,
          hint_GB_one_time=(so*N*8/1e9 if v in 'BC' else 0.0),
          batch_buffer_GB=N*so*5/1e9,
          int8_gemm_passes=passes[v])
        if name == 'qwen2.5-7b':
            r['T4_tokens_per_h_derived'] = T4_FACTORY_K40_TOK_PER_H_7B*5/passes[v]
        out[f'{name}/{v}'] = r
        print(name, v, {k:(round(x,4) if isinstance(x,float) else x) for k,x in r.items()})
json.dump(out, open('extrapolation.json','w'), indent=1)
