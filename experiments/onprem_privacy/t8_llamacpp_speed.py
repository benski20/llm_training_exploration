"""T8: direct CPU measurement of quantized (Q4_K_M) 1.5B and 7B models with llama.cpp on this 4-core machine."""
import json, sys, time
from llama_cpp import Llama
R = []
for path in sys.argv[1:]:
    llm = Llama(model_path=path, n_ctx=2048, n_threads=4, verbose=False)
    prompt = "Write a short follow-up plan for a student whose attendance has dropped. " * 8
    t = time.perf_counter(); llm(prompt, max_tokens=1); t_pre = time.perf_counter() - t
    t = time.perf_counter(); o = llm(prompt, max_tokens=96, temperature=0); dt = time.perf_counter() - t
    n = o["usage"]["completion_tokens"]; p = o["usage"]["prompt_tokens"]
    R.append({"model": path.split("/")[-1], "prompt_tokens": p, "prefill_s": round(t_pre, 2),
              "decode_tok_per_s": round(n / (dt - t_pre), 2), "gen_tokens": n})
    print(R[-1], flush=True)
json.dump(R, open("results/t8_llamacpp_speed.json", "w"), indent=1)
