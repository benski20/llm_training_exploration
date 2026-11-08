"""Kaggle kernel: an "independent recompute" tier for numeric answers over passages and filings.

The omni verifier's weakest grounded domains are HaluBench DROP and FinanceBench (AUROC 0.5-0.65 at every reader size
and for NLI): the answers are computed metrics, and the wrong ones are plausible near-misses built from real numbers.
This tier re-derives the answer instead of reading the reply:

    1. an open model (Qwen/Qwen3-8B, non-thinking) sees the passage and the question, NOT the stated answer, and writes
       a short Python program (program-of-thought) that copies the needed numbers and computes `answer`;
       k samples per item (temperature 0.7) for self-consistency;
    2. each program runs in a sandbox: an AST whitelist (no imports but math, no dunder names or attributes, no
       eval/exec/open/getattr/...), restricted builtins, in a separate `python -I -S` subprocess with an empty
       environment, a temp cwd, CPU/memory/file-size rlimits, a 10 s timeout and (when available) no network
       namespace (`unshare -n`);
    3. score = disagreement between the stated answer and the recomputed values (tolerant to rounding, percent vs
       fraction, thousands / millions / billions scale), plus the agreement rate across samples.

Evaluated on the corpus's halubench_drop / halubench_finance (200 each, balanced; rows rebuilt with
experiments/omni_corpus.py:halubench, keyed by md5 so they can be joined to corpus ids), a fresh HaluBench sample not
in the corpus (200 each), and ProcessBench pb_gsm8k / pb_math (final-answer recompute).

    python kaggle/kgrun.py --slug omni-numrec-smoke --script kaggle/omni/numeric_recompute.py \
        --bundle vt/omni.py vt/mathcheck.py vt/correctness.py experiments/omni_corpus.py \
        --args "--smoke" --out results/kaggle/omni-numrec-smoke --wait
"""
import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time

SRC = "/kaggle/working/src"
WORK = "/kaggle/working"
TMP = "/tmp/numrec"
SELF = os.path.join(SRC, "kaggle", "omni", "numeric_recompute.py")
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def key(domain, user, response):
    return hashlib.md5(f"{domain}\x00{user}\x00{response}".encode()).hexdigest()


# ---------------------------------------------------------------- data

def split_user(user):
    """corpus halubench user turn 'Context: <passage>\\n\\nQuestion: <q>' -> (passage, question)"""
    i = user.rfind("\n\nQuestion: ")
    return user[len("Context: "):i], user[i + len("\n\nQuestion: "):]


def build_items(n_hb, n_pb, n_fresh):
    import omni_corpus
    from datasets import load_dataset
    items = []
    hb = omni_corpus.halubench(n_hb)
    for r in hb:
        if r["domain"] not in ("halubench_drop", "halubench_finance"):
            continue
        p, q = split_user(r["messages"][1]["content"])
        items.append({"set": "corpus", "domain": r["domain"], "key": key(r["domain"], r["messages"][1]["content"],
                                                                          r["response"]),
                      "kind": "passage", "passage": p, "question": q, "stated": r["response"], "label": r["label"]})
    used = {it["key"] for it in items}
    d = load_dataset("PatronusAI/HaluBench", split="test")  # nosec B615: benchmark data
    for raw, short in (("DROP", "drop"), ("FinanceBench", "finance")):
        dom = f"halubench_{short}"
        rows = []
        for x in d:
            if x["source_ds"] != raw:
                continue
            u = f"Context: {x['passage']}\n\nQuestion: {x['question']}"
            k = key(dom, u, str(x["answer"]))
            if k in used:
                continue
            rows.append({"set": "fresh", "domain": dom, "key": k, "kind": "passage", "passage": x["passage"],
                         "question": x["question"], "stated": str(x["answer"]), "label": int(x["label"] == "FAIL")})
        items += omni_corpus.balanced(rows, n_fresh, seed=11)
    if n_pb:
        for r in omni_corpus.processbench(n_pb):
            if r["domain"] not in ("pb_gsm8k", "pb_math"):
                continue
            u = r["messages"][1]["content"]
            items.append({"set": "corpus", "domain": r["domain"], "key": key(r["domain"], u, r["response"]),
                          "kind": "problem", "passage": "", "question": u.split("\n\n", 1)[1],
                          "stated": final_answer(r["response"]), "label": r["label"]})
    return items


def final_answer(reply):
    """the final answer of a step-by-step solution: the last \\boxed{...}, else the last number"""
    i = reply.rfind("\\boxed{")
    if i >= 0:
        j, depth, k = i + 7, 1, i + 7
        while k < len(reply) and depth:
            depth += {"{": 1, "}": -1}.get(reply[k], 0)
            k += 1
        return reply[j:k - 1]
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", reply)
    return nums[-1] if nums else ""


# ---------------------------------------------------------------- prompts

SYS = "You are a careful financial and data analyst. You answer questions by writing short, correct Python programs."
P_PASSAGE = """Read the passage and answer the question by writing a short Python program.
- Copy every number you need from the passage into a variable, with a comment quoting where it comes from.
- Compute the answer step by step and assign the final result to a variable named `answer`.
- `answer` is a number if the question asks for a quantity (follow the question's units, scale and rounding: e.g.
  12.5 for "12.5%", 330.0 for "$330 million" when asked in USD millions). If the answer is not a quantity, set
  `answer` to a short string (a name, a span of the passage, or "yes"/"no").
- Use only plain Python and the math module: no other imports, no input, no files, no printing needed.

Passage:
{passage}

Question: {question}

Reply with only the code, in one ```python block."""
P_PROBLEM = """Solve the problem by writing a short Python program.
- Compute the answer step by step and assign the final result to a variable named `answer` (a number; use a float
  for fractions, e.g. 2/3).
- Use only plain Python and the math module: no other imports, no input, no files, no printing needed.

Problem:
{question}

Reply with only the code, in one ```python block."""


def prompt_messages(it, max_chars=24000):
    if it["kind"] == "passage":
        p = it["passage"]
        if len(p) > max_chars:
            p = p[:max_chars] + "\n[...]"
        u = P_PASSAGE.format(passage=p, question=it["question"])
    else:
        u = P_PROBLEM.format(question=it["question"])
    return [{"role": "system", "content": SYS}, {"role": "user", "content": u}]


# ---------------------------------------------------------------- generation workers

def chat_ids(tok, it):
    text = tok.apply_chat_template(prompt_messages(it), tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return list(tok(text, add_special_tokens=False)["input_ids"])


def gen_vllm(cfg):
    from vllm import LLM, SamplingParams
    items = [json.loads(l) for l in open(cfg["items"])]
    llm = LLM(cfg["model"], dtype="half", tensor_parallel_size=2, max_model_len=8192, gpu_memory_utilization=0.88,
              enable_prefix_caching=True, seed=0, max_num_seqs=64)
    tok = llm.get_tokenizer()
    prompts = []
    for it in items:
        ids = chat_ids(tok, it)
        if len(ids) > 7400:          # the longest filings: keep the head of the passage
            it = dict(it, passage=it["passage"][:int(len(it["passage"]) * 7000 / len(ids))])
            ids = chat_ids(tok, it)
        prompts.append({"prompt_token_ids": ids})
    sp = SamplingParams(n=cfg["k"], temperature=0.7, top_p=0.8, top_k=20, max_tokens=cfg["max_new"], seed=0)
    t = time.time()
    outs = llm.generate(prompts, sp)
    wall = time.time() - t
    with open(cfg["out"], "w") as f:
        for it, o in zip(items, outs):
            f.write(json.dumps({"key": it["key"], "texts": [c.text for c in o.outputs],
                                "n_prompt": len(o.prompt_token_ids),
                                "n_gen": [len(c.token_ids) for c in o.outputs]}) + "\n")
    json.dump({"backend": "vllm", "wall_s": wall, "gpus": 2, "n": len(items)}, open(cfg["out"] + ".time", "w"))


def gen_hf(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    items = [json.loads(l) for l in open(cfg["items"])]
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float16, device_map="auto",
                                                 attn_implementation="sdpa").eval()
    enc = []
    for it in items:
        ids = chat_ids(tok, it)
        if len(ids) > 6000:
            it = dict(it, passage=it["passage"][:int(len(it["passage"]) * 5600 / len(ids))])
            ids = chat_ids(tok, it)
        enc.append(ids)
    order = sorted(range(len(items)), key=lambda i: -len(enc[i]))
    res, t, i = {}, time.time(), 0
    torch.manual_seed(0)
    while i < len(order):
        L = len(enc[order[i]])
        bs = max(1, min(16, int(24000 // ((L + cfg["max_new"]) * cfg["k"]) or 1)))
        b = order[i:i + bs]
        m = max(len(enc[j]) for j in b)
        x = torch.tensor([[tok.pad_token_id] * (m - len(enc[j])) + enc[j] for j in b]).to(model.device)
        a = torch.tensor([[0] * (m - len(enc[j])) + [1] * len(enc[j]) for j in b]).to(model.device)
        with torch.no_grad():
            g = model.generate(input_ids=x, attention_mask=a, do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                               max_new_tokens=cfg["max_new"], num_return_sequences=cfg["k"],
                               pad_token_id=tok.pad_token_id)
        g = g[:, m:].view(len(b), cfg["k"], -1)
        for r, j in enumerate(b):
            res[j] = [tok.decode(s, skip_special_tokens=True) for s in g[r]]
        i += bs
        log(f"hf gen {i}/{len(order)}")
    wall = time.time() - t
    with open(cfg["out"], "w") as f:
        for j, it in enumerate(items):
            f.write(json.dumps({"key": it["key"], "texts": res[j], "n_prompt": len(enc[j])}) + "\n")
    json.dump({"backend": "hf", "wall_s": wall, "gpus": 2, "n": len(items)}, open(cfg["out"] + ".time", "w"))


# ---------------------------------------------------------------- sandboxed execution

BANNED = {"eval", "exec", "compile", "open", "input", "globals", "locals", "vars", "getattr", "setattr", "delattr",
          "hasattr", "breakpoint", "help", "exit", "quit", "memoryview", "type", "object", "super", "classmethod",
          "staticmethod", "property", "dir", "id", "__import__", "__builtins__"}
RUNNER = r'''
import json, math, sys
SAFE = {n: __builtins__.__dict__[n] if hasattr(__builtins__, "__dict__") else __builtins__[n] for n in
        ("abs", "min", "max", "sum", "round", "len", "range", "int", "float", "str", "list", "dict", "tuple", "set",
         "sorted", "enumerate", "zip", "map", "filter", "any", "all", "bool", "pow", "divmod", "reversed",
         "isinstance", "ZeroDivisionError", "ValueError", "Exception", "True", "False", "None")
        if (n in __builtins__.__dict__ if hasattr(__builtins__, "__dict__") else n in __builtins__)}
SAFE["print"] = lambda *a, **k: None
code = sys.stdin.read()
g = {"__builtins__": SAFE, "math": math}
try:
    exec(compile(code, "<program>", "exec"), g)
    a = g.get("answer", None)
    def conv(v):
        if isinstance(v, bool): return float(v)
        if isinstance(v, (int, float)): return float(v) if math.isfinite(float(v)) else None
        if isinstance(v, str): return v[:500]
        if isinstance(v, (list, tuple, set)): return [conv(u) for u in list(v)[:20]]
        return None
    c = conv(a)
    sys.stdout.write("@@ANSWER@@" + json.dumps({"ok": c is not None and c != [], "answer": c}))
except BaseException as e:
    sys.stdout.write("@@ANSWER@@" + json.dumps({"ok": False, "error": type(e).__name__}))
'''


def extract_code(text):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)(?:```|$)", text, re.S)
    return max(m, key=len) if m else text


def validate(code):
    """the program, with `import math` removed, if it passes the whitelist; else None"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            if all(a.name == "math" and a.asname in (None, "math") for a in node.names):
                continue
            return None
        if isinstance(node, ast.ImportFrom):
            return None
        keep.append(node)
    for node in ast.walk(ast.Module(body=keep, type_ignores=[])):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef,
                             ast.Await, ast.Yield, ast.YieldFrom, ast.With, ast.AsyncWith, ast.AsyncFor)):
            return None
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return None
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in BANNED):
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            return None
    if len(code) > 20000:
        return None
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


def _limits():
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    os.setsid()


UNSHARE = None


def sandbox_prefix():
    global UNSHARE
    if UNSHARE is None:
        try:
            ok = subprocess.run(["unshare", "-n", "true"], capture_output=True, timeout=10).returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
        UNSHARE = ["unshare", "-n"] if ok else []
        log("network namespace isolation:", "unshare -n" if ok else "unavailable (AST whitelist + no imports only)")
    return UNSHARE


def run_program(text, runner, cwd):
    code = validate(extract_code(text))
    if code is None:
        return {"ok": False, "error": "rejected"}
    try:
        p = subprocess.run(sandbox_prefix() + [sys.executable, "-I", "-S", runner], input=code, capture_output=True,
                           text=True, timeout=10, env={}, cwd=cwd, preexec_fn=_limits)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    out = p.stdout
    i = out.rfind("@@ANSWER@@")
    if i < 0:
        return {"ok": False, "error": f"crash rc={p.returncode}"}
    try:
        return json.loads(out[i + 10:])
    except ValueError:
        return {"ok": False, "error": "bad output"}


# ---------------------------------------------------------------- comparison

NUMRE = re.compile(r"\(?-?\$?\s*\d+(?:,\d{3})*(?:\.\d+)?\)?")
SCALES = (1, 100, 0.01, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9)


def numbers(s):
    """(value, decimals) of each number written in s; '(12)' is negative (accounting)"""
    out = []
    for m in NUMRE.finditer(str(s)):
        t = m.group(0)
        neg = t.startswith("(") and t.endswith(")") or "-" in t
        t = t.strip("()").replace("$", "").replace(",", "").replace("-", "").strip()
        try:
            v = float(t)
        except ValueError:
            continue
        out.append((-v if neg else v, len(t.split(".")[1]) if "." in t else 0))
    return out


def tokens(s):
    return re.findall(r"[a-z0-9]+", str(s).lower().replace(",", ""))


def f1(a, b):
    a, b = tokens(a), tokens(b)
    if not a or not b:
        return 0.0
    from collections import Counter
    c = sum((Counter(a) & Counter(b)).values())
    if not c:
        return 0.0
    p, r = c / len(b), c / len(a)
    return 2 * p * r / (p + r)


def stated_value(s, kind):
    """numbers of the stated answer; for a math answer also \\frac / plain arithmetic"""
    if kind == "problem":
        sys.path.insert(0, SRC)
        from vt.mathcheck import evaluate, plain
        t = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
        t = re.sub(r"\\(?:text|mathrm|mbox)\{[^{}]*\}|\\[,!; ]|\\%|%|\^\\circ|\\\$", "", t)
        t = plain(t).replace("§", "")
        if re.search(r",\s|,-|;", t):          # several answers: not compared
            return []
        try:
            return [(evaluate(t.strip().strip(".").replace(" ", "")), 6)]
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            return numbers(t) if re.fullmatch(r"\s*-?\$?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*\w*\s*", t) else []
    return numbers(s)


def compare(stated, kind, ans):
    """(agree, relerr or None) of one recomputed answer against the stated one; None if not comparable"""
    sv = stated_value(stated, kind)
    if kind == "problem" and not sv:
        return None
    vals = []
    if isinstance(ans, list):
        flat = [a for a in ans if a is not None]
        if all(isinstance(a, float) for a in flat):
            vals = flat
        else:
            ans = ", ".join(str(a) for a in flat)
    if isinstance(ans, float):
        vals = [ans]
    if vals and not sv:
        return (f1(stated, " ".join(f"{v:g}" for v in vals)) >= 0.5, None)
    if isinstance(ans, str):
        an = numbers(ans)
        if sv and an and (len(tokens(ans)) <= 4 or len(tokens(stated)) <= 4):
            vals = [v for v, _ in an]
        else:
            ag = f1(stated, ans) >= 0.5 or (len(tokens(ans)) > 0 and set(tokens(ans)) <= set(tokens(stated))) \
                or (len(tokens(stated)) > 0 and set(tokens(stated)) <= set(tokens(ans)))
            return (ag, None)
    if not vals or not sv:
        return None
    agree, rel = False, math.inf
    scales = (1,) if kind == "problem" else SCALES
    for x, dec in sv:
        tol = max(0.5 * 10 ** -dec * 1.0001, 5e-4 * abs(x), 1e-9) if kind == "passage" else max(1e-6, 1e-4 * abs(x))
        for v in vals:
            for s in scales:
                y = v * s
                for z in ((y, round(y, dec), abs(y)) if kind == "passage" else (y,)):
                    d = abs(z - x)
                    rel = min(rel, d / max(abs(x), 1e-9))
                    agree = agree or d <= tol
    return (agree, rel)


def cluster(vals):
    """majority share among recomputed values (numbers close within 0.5%, strings by f1>=0.5)"""
    if not vals:
        return 0.0, None
    best, best_n = None, 0
    for a in vals:
        n = sum(1 for b in vals if same(a, b))
        if n > best_n:
            best, best_n = a, n
    return best_n / len(vals), best


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= max(1e-9, 0.005 * max(abs(a), abs(b)))
    return f1(str(a), str(b)) >= 0.5


def score_item(it, runs, ks=(1, 3, 5)):
    out = {"key": it["key"], "set": it["set"], "domain": it["domain"], "label": it["label"],
           "numeric_stated": bool(stated_value(it["stated"], it["kind"])), "stated": it["stated"][:200]}
    for k in ks:
        rr = runs[:k]
        ok = [r["answer"] for r in rr if r.get("ok")]
        cmp = [compare(it["stated"], it["kind"], a) for a in ok]
        cmp = [c for c in cmp if c is not None]
        share, _ = cluster([a if not isinstance(a, list) else str(a) for a in ok])
        if not cmp:
            out[f"k{k}"] = {"valid": 0, "disagree": 0.5, "maj": 0.5, "relerr": 0.5, "share": 0.0}
            continue
        dis = 1 - sum(c[0] for c in cmp) / len(cmp)
        rels = [c[1] for c in cmp if c[1] is not None]
        relerr = float(sorted(rels)[len(rels) // 2]) if rels else (0.0 if dis < 0.5 else 1.0)
        maj = 0.5 + (0.5 * share if dis > 0.5 else -0.5 * share if dis < 0.5 else 0.0)
        out[f"k{k}"] = {"valid": len(cmp), "disagree": dis, "maj": maj,
                        "relerr": math.log10(1 + min(relerr, 1e6)), "share": share,
                        "flag": int(dis > 0.5 and share >= 0.6)}
    out["answers"] = [r.get("answer") if r.get("ok") else r.get("error") for r in runs]
    return out


# ---------------------------------------------------------------- metrics

def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else None


def metrics(rows, k):
    y = [r["label"] for r in rows]
    m = {"n": len(rows), "wrong": sum(y)}
    for s in ("disagree", "maj", "relerr"):
        m[f"auroc_{s}"] = auroc(y, [r[f"k{k}"][s] for r in rows])
    m["coverage"] = sum(r[f"k{k}"]["valid"] > 0 for r in rows) / len(rows)
    fl = [r[f"k{k}"].get("flag", 0) for r in rows]
    tp = sum(f and l for f, l in zip(fl, y))
    m["flag_rate"] = sum(fl) / len(rows)
    m["precision"] = tp / max(sum(fl), 1)
    m["recall"] = tp / max(sum(y), 1)
    return m


def report(scored, timing):
    res = {"timing": timing, "groups": {}}
    groups = {}
    for r in scored:
        groups.setdefault(f"{r['set']}/{r['domain']}", []).append(r)
        if r["domain"].startswith("halubench") and r["numeric_stated"]:
            groups.setdefault(f"{r['set']}/{r['domain']}/numeric", []).append(r)
    for g, rows in sorted(groups.items()):
        res["groups"][g] = {f"k{k}": metrics(rows, k) for k in (1, 3, 5) if f"k{k}" in rows[0]}
        m = res["groups"][g].get("k5") or res["groups"][g]["k3"]
        log(f"{g:40s} n={m['n']:3d} cov={m['coverage']:.2f} AUROC dis={m['auroc_disagree']} maj={m['auroc_maj']} "
            f"rel={m['auroc_relerr']} flag={m['flag_rate']:.2f} P={m['precision']:.2f} R={m['recall']:.2f}")
    return res


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--vllm", default="vllm==0.11.0 transformers==4.57.1")
    ap.add_argument("--no-pb", action="store_true")
    ap.add_argument("--worker", default="")
    a = ap.parse_args()
    if a.worker:
        cfg = json.load(open(a.worker))
        return {"vllm": gen_vllm, "hf": gen_hf}[cfg["backend"]](cfg)
    os.makedirs(TMP, exist_ok=True)
    subprocess.run("nvidia-smi; python -c 'import torch;print(torch.__version__, torch.version.cuda)'", shell=True)
    n_hb, n_pb, n_fresh = (200, 240, 200)
    items = build_items(n_hb, 0 if a.no_pb else n_pb, n_fresh)
    if a.smoke:
        rng = random.Random(0)  # nosec B311
        by = {}
        for it in items:
            by.setdefault((it["set"], it["domain"]), []).append(it)
        items = [x for v in by.values() for x in rng.sample(v, min(8, len(v)))]
    log("items", len(items), {g: sum(1 for i in items if f"{i['set']}/{i['domain']}" == g)
                              for g in sorted({f"{i['set']}/{i['domain']}" for i in items})})
    ipath = os.path.join(TMP, "items.jsonl")
    with open(ipath, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    gpath = os.path.join(WORK, "generations.jsonl")
    backends = ["vllm", "hf"] if a.backend == "auto" else [a.backend]
    for b in backends:
        if b == "vllm":
            rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + a.vllm.split()).returncode
            log("pip vllm rc", rc)
            if rc:
                continue
        else:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.56", "accelerate"])
        cfg = {"backend": b, "model": a.model, "items": ipath, "out": gpath, "k": a.k, "max_new": a.max_new}
        cp = os.path.join(TMP, f"cfg_{b}.json")
        json.dump(cfg, open(cp, "w"))
        env = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD="spawn", TOKENIZERS_PARALLELISM="false")
        rc = subprocess.run([sys.executable, SELF, "--worker", cp], env=env,
                            stdout=open(os.path.join(WORK, f"gen_{b}.log"), "w"), stderr=subprocess.STDOUT).returncode
        log("gen", b, "rc", rc)
        if rc == 0 and os.path.exists(gpath + ".time"):
            break
        subprocess.run(f"tail -40 {WORK}/gen_{b}.log", shell=True)
    timing = json.load(open(gpath + ".time"))
    gens = {g["key"]: g for g in (json.loads(l) for l in open(gpath))}
    runner = os.path.join(TMP, "runner.py")
    open(runner, "w").write(RUNNER)
    cwd = tempfile.mkdtemp()
    from concurrent.futures import ThreadPoolExecutor
    t = time.time()
    flat = [(it["key"], j, txt) for it in items for j, txt in enumerate(gens[it["key"]]["texts"])]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda x: run_program(x[2], runner, cwd), flat))
    runs = {}
    for (kk, j, _), r in zip(flat, res):
        runs.setdefault(kk, []).append(r)
    timing["exec_cpu_wall_s"] = time.time() - t
    timing["programs"] = len(flat)
    from collections import Counter
    timing["exec_outcomes"] = dict(Counter("ok" if r.get("ok") else r.get("error", "none") for r in res))
    timing["gpu_s_per_check"] = timing["wall_s"] * timing["gpus"] / timing["n"]
    timing["mean_prompt_tokens"] = sum(g["n_prompt"] for g in gens.values()) / len(gens)
    if "n_gen" in next(iter(gens.values())):
        timing["mean_gen_tokens"] = sum(sum(g["n_gen"]) for g in gens.values()) / sum(len(g["n_gen"]) for g in gens.values())
    log("timing", timing)
    ks = tuple(k for k in (1, 3, 5) if k <= a.k)
    scored = [score_item(it, runs[it["key"]], ks) for it in items]
    with open(os.path.join(WORK, "scored.jsonl"), "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    out = report(scored, timing)
    out["config"] = vars(a)
    json.dump(out, open(os.path.join(WORK, "numeric_recompute.json"), "w"), indent=1)
    # a few examples for inspection
    with open(os.path.join(WORK, "examples.txt"), "w") as f:
        for it in items[:40]:
            f.write(f"=== {it['set']} {it['domain']} label={it['label']} stated={it['stated'][:120]!r}\n"
                    f"Q: {it['question'][-300:]}\n--- program 0:\n{gens[it['key']]['texts'][0][:1500]}\n"
                    f"--- runs: {[r.get('answer') if r.get('ok') else r.get('error') for r in runs[it['key']]]}\n\n")
    log("done")


if __name__ == "__main__":
    main()
