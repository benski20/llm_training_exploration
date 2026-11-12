"""Training data for a hallucination-detection probe on document-grounded answers.

Builds one jsonl per source under --out (default data/probe_train/), one record per judged answer:

    id, source, task (qa|summary|data2text|dialogue|other), license, label_origin (human|llm|synthetic),
    question (the request, without documents), documents (list of str), response,
    label (1 = contains unsupported / contradicted content, 0 = faithful),
    spans ([[start, end], ...] char offsets into response; [] if none or unknown), has_spans (bool)

and data/probe_train/MANIFEST.md (licences quoted from the dataset cards, commercial use, counts, exclusions).

Held-out data never enters the output. data/probe_train/exclude_hashes.txt (committed) lists sha1s of
the normalised responses of
    * the RAGTruth test split (wandb/RAGTruth-processed, split=test: our benchmark),
    * data/rag_suite.jsonl records with "split": "test" (the live demo set),
    * /home/user/omni/corpus.jsonl (the 21-domain cross-dataset evaluation corpus).
A training record is dropped when its (normalised response, documents fingerprint) matches, or when its
normalised response alone matches and is at least MIN_KEY_LEN characters long: short responses ("yes",
"1768", "Unable to answer based on given passages.") are not distinctive on their own. The builder only reads
that file, so it runs anywhere with internet (e.g. a fresh Kaggle session); regenerate it with
--make-exclusions on a machine that has the private files.

Within the output, records are deduplicated by normalised response (when >= MIN_KEY_LEN) and by
(question, response); the first source in SOURCES order keeps a duplicate. Records whose documents total
> 12,000 characters or whose response is > 4,000 characters are dropped. Official test splits of the
third-party sets are left out unless --with-test-splits (the RAGTruth test split never enters).

    python experiments/probe_data.py                                  # all sources -> data/probe_train/
    python experiments/probe_data.py --out /kaggle/working/probe --sources ragtruth,faithdial
    python experiments/probe_data.py --make-exclusions --rag-suite data/rag_suite.jsonl \
        --omni /home/user/omni/corpus.jsonl                           # rewrite exclude_hashes.txt

Dependencies: datasets (and what it pulls in: huggingface_hub, pyarrow), numpy, stdlib.
"""
import argparse
import ast
import collections
import csv
import hashlib
import io
import json
import os
import re
import sys
import unicodedata
import urllib.request
import zipfile

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_OUT = os.path.join(REPO, "data", "probe_train")
EXCLUDE_FILE = os.path.join(REPO, "data", "probe_train", "exclude_hashes.txt")
MAX_DOC_CHARS = 12_000
MAX_RESP_CHARS = 4_000
MIN_KEY_LEN = 50          # normalised chars below which a response alone is not a dedup / exclusion key
                          # (50 keeps templates such as "Unable to answer based on given passages." out of it)
TASKS = {"qa", "summary", "data2text", "dialogue", "other"}

csv.field_size_limit(sys.maxsize)


# ---------------------------------------------------------------- normalisation and keys

def norm(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    return " ".join(re.sub(r"[\W_]+", " ", s).split())


def sha1(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()  # nosec B324: content key, not security


def doc_fp(docs):
    """Formatting-insensitive fingerprint of the grounding text (alphanumerics only, first 300)."""
    if isinstance(docs, str):
        docs = [docs]
    return re.sub(r"\s+", "", norm(" ".join(docs or [])))[:300]


def resp_key(resp):
    n = norm(resp)
    return "r:" + sha1(n) if len(n) >= MIN_KEY_LEN else None


def pair_doc_key(resp, docs):
    return "p:" + sha1(norm(resp) + "\x1f" + doc_fp(docs))


def qr_key(question, resp):
    return "q:" + sha1(norm(question) + "\x1f" + norm(resp))


# ---------------------------------------------------------------- downloads

def hf_file(repo, filename, rev):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type="dataset", revision=rev)  # nosec B615: pinned revision


def parquet_rows(repo, filename, rev, columns=None):
    import pyarrow.parquet as pq
    return pq.read_table(hf_file(repo, filename, rev), columns=columns).to_pylist()


def jsonl_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- source loaders
# Each yields dicts with id, task, question, documents, response, label, spans, has_spans.

def rec(id_, task, question, documents, response, label, spans=None, has_spans=False):
    return {"id": id_, "task": task, "question": question or "", "documents": [d for d in documents if d],
            "response": response, "label": int(label), "spans": spans or [], "has_spans": bool(has_spans)}


RAGTRUTH = ("wandb/RAGTruth-processed", "eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b")


def load_ragtruth(args, stats):
    repo, rev = RAGTRUTH
    task_of = {"QA": "qa", "Summary": "summary", "Data2txt": "data2text"}
    for x in parquet_rows(repo, "data/train-00000-of-00001.parquet", rev):  # train ONLY
        resp = x["output"]
        spans = []
        for h in json.loads(x["hallucination_labels"] or "[]"):
            s, e = int(h["start"]), int(h["end"])
            if resp[s:e] != h.get("text", resp[s:e]):
                j = resp.find(h["text"])
                if j < 0:
                    stats["span_unlocated"] += 1
                    continue
                s, e = j, j + len(h["text"])
            spans.append([s, e])
        task = task_of[x["task_type"]]
        docs = [p for p in x["context"].split("\n\n") if p.strip()] if task == "qa" else [x["context"].strip()]
        label = int(bool(json.loads(x["hallucination_labels"] or "[]")))
        yield rec(f"ragtruth:{x['id']}", task, x["query"].strip(), docs, resp, label, sorted(spans), True)


HALUEVAL = ("pminervini/HaluEval", "12a856119f03975a94509091e8cada3e6be6ead7")


def _halueval(sub, task):
    repo, rev = HALUEVAL
    rows = parquet_rows(repo, f"{sub}/data-00000-of-00001.parquet", rev)
    for i, x in enumerate(rows):
        if sub == "qa":
            q, docs, pair = x["question"], [x["knowledge"]], (x["right_answer"], x["hallucinated_answer"])
        elif sub == "dialogue":
            q, docs, pair = x["dialogue_history"], [x["knowledge"]], (x["right_response"], x["hallucinated_response"])
        else:
            q, docs, pair = "Summarize the following document.", [x["document"]], (x["right_summary"], x["hallucinated_summary"])
        yield rec(f"halueval_{task}:{i}:right", task, q, docs, pair[0], 0)
        yield rec(f"halueval_{task}:{i}:hallucinated", task, q, docs, pair[1], 1)


def load_halueval_qa(args, stats):
    return _halueval("qa", "qa")


def load_halueval_dialogue(args, stats):
    return _halueval("dialogue", "dialogue")


def load_halueval_summary(args, stats):
    return _halueval("summarization", "summary")


FAITHDIAL = ("McGill-NLP/FaithDial", "7a414e80725eac766f2602676dc8b39f80b061e4")


def load_faithdial(args, stats):
    """Original Wizard-of-Wikipedia replies carry crowd BEGIN labels (Hallucination -> 1); the edited
    (or unedited, BEGIN=Entailment) FaithDial reply is the faithful one (0)."""
    repo, rev = FAITHDIAL
    splits = ["train", "valid"] + (["test"] if args.with_test_splits else [])
    for split in splits:
        with open(hf_file(repo, f"data/{split}.json", rev), encoding="utf-8") as f:
            data = json.load(f)
        for di, dlg in enumerate(data):
            for ui, u in enumerate(dlg["utterances"]):
                hist = u["history"]
                n = len(hist)
                q = "\n".join(("User: " if (n - 1 - k) % 2 == 0 else "Assistant: ") + t for k, t in enumerate(hist))
                docs = [u["knowledge"]]
                begin = set(u.get("BEGIN") or [])
                base = f"faithdial:{split}:{di}:{ui}"
                yield rec(base + ":response", "dialogue", q, docs, u["response"], 0)
                if u.get("original_response"):
                    yield rec(base + ":original", "dialogue", q, docs, u["original_response"],
                              int("Hallucination" in begin))


ATTRIB = ("osunlp/AttributionBench", "62569e644f4186606f54f742178a4517431b42e1")


def load_attributionbench(args, stats):
    repo, rev = ATTRIB
    files = ["merged_train.jsonl", "merged_dev.jsonl"] + (["merged_test.jsonl", "test_ood.jsonl"] if args.with_test_splits else [])
    for fn in files:
        for i, x in enumerate(jsonl_rows(hf_file(repo, fn, rev))):
            lab = {"attributable": 0, "not attributable": 1}.get(x["attribution_label"])
            if lab is None:
                stats["unknown_label"] += 1
                continue
            yield rec(f"attributionbench:{fn.split('.')[0]}:{i}:{x.get('src_dataset')}", "qa", x["question"],
                      list(x["references"] or []), x["claim"], lab)


SUMMEDITS = ("Salesforce/summedits", "ce0c479aaf59259abb6b67e42248b2f49004b7d5")


def load_summedits(args, stats):
    repo, rev = SUMMEDITS
    with open(hf_file(repo, "summedits.json", rev), encoding="utf-8") as f:
        data = json.load(f)
    for x in data:
        yield rec(f"summedits:{x['id']}", "summary", "Summarize the following document.", [x["doc"]],
                  x["summary"], 1 - int(x["label"]))


ANAH = ("opencompass/anah", "12238ed44444a5ab4df100a1ef2b395cf49b624e")


def _lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def load_anah(args, stats):
    """English only. Sentence annotations: '<Hallucination> None' -> ok, '<No Fact>' -> ok (no checkable
    fact), any other hallucination type (Contradictory, Unverifiable, ...) -> 1 for the whole answer."""
    repo, rev = ANAH
    for fn in ("event.jsonl", "location.jsonl", "person.jsonl", "thing.jsonl"):
        for ti, x in enumerate(jsonl_rows(hf_file(repo, fn, rev))):
            if x.get("language") != "en":
                continue
            try:
                docs = [str(d) for d in _lit(x["documents"])]
                qs = _lit(x["selected_questions"])
            except (ValueError, SyntaxError):
                stats["unparsable"] += 1
                continue
            for model, akey, nkey in (("gpt35", "GPT3.5_answers_D", "human_GPT3.5_answers_D_ann"),
                                      ("internlm", "InternLM_answers", "human_InternLM_answers_ann")):
                try:
                    answers, anns = _lit(x[akey]), _lit(x[nkey])
                except (ValueError, SyntaxError, KeyError):
                    stats["unparsable"] += 1
                    continue
                for qi, (q, a, ann) in enumerate(zip(qs, answers, anns)):
                    types = []
                    for s in ann or []:
                        if "<No Fact>" in s:
                            types.append("NoFact")
                            continue
                        m = re.search(r"<Hallucination>\s*([A-Za-z ]+?)\s*(\n|<|$)", s)
                        types.append(m.group(1).strip() if m else None)
                    if not types:
                        stats["no_annotation"] += 1
                        continue
                    if None in types:
                        stats["unparsable"] += 1
                        continue
                    stats["types:" + ",".join(sorted(set(types)))] += 1
                    label = int(any(t not in ("None", "NoFact") for t in types))
                    yield rec(f"anah:{fn.split('.')[0]}:{ti}:{qi}:{model}", "qa", q, docs, a, label)


CLEANLAB = ("Cleanlab/FinQA-hallucination-detection", "0eeb92f4f19db3defcbc5e42db3d1cf4609b6008")


def load_cleanlab_finqa(args, stats):
    repo, rev = CLEANLAB
    with open(hf_file(repo, "finqa-hallucination-detection.csv", rev), encoding="utf-8", newline="") as f:
        for i, x in enumerate(csv.DictReader(f)):
            yield rec(f"cleanlab_finqa:{i}", "qa", x["query"], [x["context"]], x["llm_response"],
                      1 - int(x["is_correct"]))


WANDB_FINQA = ("wandb/finqa-data-processed-hallucination", "b18fbf67b43a259bf8a4b3a05fc87c7251c58f5a")


def load_wandb_finqa(args, stats):
    repo, rev = WANDB_FINQA
    files = ["data/train-00000-of-00001.parquet"] + (["data/test-00000-of-00001.parquet"] if args.with_test_splits else [])
    for fn in files:
        sp = fn.split("/")[1].split("-")[0]
        for i, x in enumerate(parquet_rows(repo, fn, rev, ["query", "context", "output", "id", "is_hallucination"])):
            yield rec(f"wandb_finqa:{sp}:{i}:{x['id']}", "qa", x["query"], [x["context"]], x["output"],
                      int(bool(x["is_hallucination"])))


RAGBENCH = ("galileo-ai/ragbench", "97808f3e5fd16ede40bbff6c2949af8139b2eb7b")
RAGBENCH_OPEN = ["covidqa", "cuad", "expertqa", "finqa", "hagrid", "hotpotqa", "pubmedqa", "tatqa"]
RAGBENCH_RESTRICTED = ["delucionqa", "emanual", "msmarco", "techqa"]


def _ragbench(args, stats, subsets, source):
    repo, rev = RAGBENCH
    cols = ["id", "question", "documents", "response", "response_sentences",
            "unsupported_response_sentence_keys", "adherence_score"]
    splits = ["train", "validation"] + (["test"] if args.with_test_splits else [])
    for sub in subsets:
        for sp in splits:
            for i, x in enumerate(parquet_rows(repo, f"{sub}/{sp}-00000-of-00001.parquet", rev, cols)):
                if x["adherence_score"] is None:
                    stats["no_label"] += 1
                    continue
                label = 0 if x["adherence_score"] else 1
                resp = x["response"]
                sents = {k: t for k, t in (x["response_sentences"] or [])}
                bad = list(x["unsupported_response_sentence_keys"] or [])
                spans, ok = [], True
                if label == 1:
                    for k in bad:
                        t = (sents.get(k) or "").strip()
                        j = resp.find(t) if t else -1
                        if j < 0:
                            ok = False
                            break
                        spans.append([j, j + len(t)])
                    ok = ok and bool(spans)
                    if not ok:
                        stats["label1_without_locatable_sentences"] += 1
                        spans = []
                elif bad:
                    stats["label0_with_unsupported_keys"] += 1
                yield rec(f"{source}:{sub}:{sp}:{i}:{x['id']}", "qa", x["question"], list(x["documents"] or []), resp,
                          label, sorted(spans), ok)


def load_ragbench(args, stats):
    return _ragbench(args, stats, RAGBENCH_OPEN, "ragbench")


def load_ragbench_restricted(args, stats):
    return _ragbench(args, stats, RAGBENCH_RESTRICTED, "ragbench_restricted")


FAVA = ("fava-uw/fava-data", "f4e40415d525b18bcb49ba241f26fd8e12eeb606")
FAVA_ERR = {"entity", "relation", "contradictory", "unverifiable", "invented", "subjective",
            "contrdictory", "contradiction"}
FAVA_SENT = {"contradictory", "unverifiable", "invented", "subjective", "contrdictory", "contradiction"}


def fava_parse(completion):
    """Tagged FAVA passage -> (erroneous text, error spans in it, corrected text) or None if malformed.
    Erroneous: keep <delete> text, drop <mark>/<insert> text. Corrected: keep <mark>, drop <delete> and the
    bare text of sentence-level error tags (unverifiable, invented, subjective, contradictory)."""
    stack, err, cor, spans = [], [], [], []
    pos = 0
    for m in re.finditer(r"<(/?)([a-z]+)>|[^<]+|<", completion):
        if m.group(2):
            tag, closing = m.group(2), bool(m.group(1))
            if tag == "b":
                continue
            if closing:
                if not stack or stack[-1] != tag:
                    return None
                stack.pop()
            else:
                stack.append(tag)
            continue
        t = m.group(0)
        in_mark = "mark" in stack or "insert" in stack
        in_del = "delete" in stack
        in_err = any(s in FAVA_ERR for s in stack)
        if not in_mark:
            if in_err:
                spans.append([pos, pos + len(t)])
            err.append(t)
            pos += len(t)
        if in_mark or (not in_del and not any(s in FAVA_SENT for s in stack)):
            cor.append(t)
    if stack:
        return None
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    text = "".join(err)
    # trim whitespace inside spans
    out = []
    for s, e in merged:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append([s, e])
    corrected = re.sub(r"[ \t]{2,}", " ", "".join(cor))
    corrected = re.sub(r" +([.,;:!?])", r"\1", corrected)
    corrected = re.sub(r"(?<!\.)\.\s*\.(?!\.)", ".", corrected).strip()   # "century.. At" after a dropped sentence
    return text, out, corrected


def load_fava(args, stats):
    repo, rev = FAVA
    with open(hf_file(repo, "training.json", rev), encoding="utf-8") as f:
        data = json.load(f)
    for i, x in enumerate(data):
        prompt = x["prompt"]
        m = re.search(r"\nPlease identify all the errors in the following passage using the references provided and suggest edits:\nText: (.*)\Z", prompt, re.S)
        if not m:
            stats["no_text_in_prompt"] += 1
            continue
        refs = re.split(r"(?:^|\n)Reference \[\d+\]: ", prompt[:m.start()].replace("Read the following references:", "", 1))
        docs = [r.strip() for r in refs if r.strip()]
        text_in_prompt = m.group(1)
        p = fava_parse(x["completion"])
        if p is None:
            stats["malformed_tags"] += 1
            continue
        err_text, spans, corrected = p
        lead = len(err_text) - len(err_text.lstrip())
        if " ".join(err_text.split()) != " ".join(text_in_prompt.split()):
            stats["tagged_text_differs_from_prompt"] += 1
            continue
        resp = err_text.strip()
        spans = [[s - lead, e - lead] for s, e in spans if e - lead <= len(resp)]
        if spans:
            yield rec(f"fava:{i}:erroneous", "other", "", docs, resp, 1, spans, True)
            if corrected:
                yield rec(f"fava:{i}:corrected", "other", "", docs, corrected, 0, [], True)
        else:
            yield rec(f"fava:{i}:clean", "other", "", docs, resp, 0, [], True)


MINICHECK = ("lytang/C2D-and-D2C-MiniCheck", "1f698000d2f0e39c6830c74d66f793c02a1d9259")


def load_minicheck(args, stats):
    repo, rev = MINICHECK
    for name, fn in (("c2d", "data/c2d-00000-of-00001-5a473df7966a0bbe.parquet"),
                     ("d2c", "data/d2c-00000-of-00001-fb38b2c3dc66f609.parquet")):
        for i, x in enumerate(parquet_rows(repo, fn, rev)):
            yield rec(f"minicheck:{name}:{i}", "other", "", [x["doc"]], x["claim"], 1 - int(x["label"]))


SUMMACOZ = ("nkwbtb/SummaCoz", "0751b1d69660bc1d6dafd132eef88bc02212c6be")


def load_summacoz(args, stats):
    repo, rev = SUMMACOZ
    for sp in ("train", "validation"):
        for i, x in enumerate(parquet_rows(repo, f"data/{sp}-00000-of-00001.parquet", rev)):
            lab = {0: 0, 2: 1}.get(int(x["label"]))
            if lab is None:
                stats["unknown_label"] += 1
                continue
            yield rec(f"summacoz:{sp}:{i}:{x['dataset']}", "summary", "Summarize the following document.",
                      [x["article"]], x["summary"], lab)


HALUBENCH = ("PatronusAI/HaluBench", "5966a87929f51c204ab3cbef986b449495cc97b6")


def load_halubench(args, stats):
    repo, rev = HALUBENCH
    for i, x in enumerate(parquet_rows(repo, "data/test-00000-of-00001.parquet", rev)):
        if x["source_ds"] == "RAGTruth":   # drawn from the RAGTruth test split (our benchmark): drop all of it
            stats["dropped_ragtruth_subset"] += 1
            continue
        lab = {"PASS": 0, "FAIL": 1}.get(x["label"])
        if lab is None:
            stats["unknown_label"] += 1
            continue
        yield rec(f"halubench:{x['source_ds']}:{i}:{x['id']}", "qa", x["question"], [x["passage"]], str(x["answer"]), lab)


HALLUDIAL_URL = "https://drive.usercontent.google.com/download?id=1duxf2fN2A0MMy1gx85LRSk1-EPVE1tG6&export=download&confirm=t"


def load_halludial(args, stats):
    """FlagEval/HalluDial (HF repo holds only a loader script; data on Google Drive, 'all' config)."""
    cache = os.path.join(args.cache, "halludial_all.zip")
    if not os.path.exists(cache):
        os.makedirs(args.cache, exist_ok=True)
        with urllib.request.urlopen(HALLUDIAL_URL, timeout=300) as r:  # nosec B310: fixed https URL
            blob = r.read()
        with open(cache, "wb") as f:
            f.write(blob)
    with open(cache, "rb") as f:
        blob = f.read()
    stats["zip_sha256:" + hashlib.sha256(blob).hexdigest()] += 1
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        data = json.load(z.open("all/all_train.json"))
    for i, x in enumerate(data):
        t = (x.get("target") or "").strip()
        lab = 1 if t.startswith("Yes") else 0 if t.startswith("No") else None
        if lab is None:
            stats["unknown_label"] += 1
            continue
        resp = re.sub(r"^\s*\[Assistant\]:\s*", "", x["response"] or "")
        yield rec(f"halludial:{x['dialogue_id']}:{x['turn']}:{i}", "dialogue", x["dialogue_history"],
                  [x["knowledge"]], resp, lab)


# ---------------------------------------------------------------- source registry (order = dedup priority)

def S(loader, hf, rev, licence, quote, commercial, origin, spans, notes):
    return dict(loader=loader, hf=hf, rev=rev, license=licence, quote=quote, commercial=commercial,
                label_origin=origin, spans=spans, notes=notes)


SOURCES = {
    "ragtruth": S(load_ragtruth, *RAGTRUTH, "MIT", '"This dataset is released under the MIT License." (card prose; no YAML licence field; upstream GitHub ParticleMedia/RAGTruth LICENSE is MIT)',
                  "yes", "human", True, "train split only (15,090); the 2,700-answer test split is never loaded. Responses from GPT-3.5/4, Llama-2, Mistral; expert span annotations. QA contexts split on blank lines into passages."),
    "faithdial": S(load_faithdial, *FAITHDIAL, "MIT", "license:\n- mit", "yes", "human", False,
                   "train+valid. Original Wizard-of-Wikipedia reply labelled by crowd BEGIN tags (label 1 iff 'Hallucination' in BEGIN; Generic/Uncooperative/Entailment -> 0); the FaithDial reply (crowd-edited to be faithful) -> 0. Question = dialogue history."),
    "attributionbench": S(load_attributionbench, *ATTRIB, "Apache-2.0", "license: apache-2.0", "yes", "human", False,
                          "merged_train + merged_dev. Claim-level: response = one cited claim, documents = its references; label 1 = 'not attributable'. Subsets ExpertQA, Stanford-GenSearch, LFQA, AttributedQA (human attribution judgments; upstream subset licences not individually verified). Covers ExpertQA."),
    "anah": S(load_anah, *ANAH, "Apache-2.0", "license: apache-2.0", "yes", "human", False,
              "English topics only (language == 'en'). Human-in-the-loop sentence annotations aggregated to answer level (any Contradictory/Unverifiable/... sentence -> 1; None/No Fact -> 0). English annotations carry no sentence text, so no spans. Documents are whole reference articles: many exceed 12k chars and are dropped."),
    "summedits": S(load_summedits, *SUMMEDITS, "CC-BY-4.0 (card also says research-only)", 'license: cc-by-4.0  /  "This release is for research purposes only in support of an academic paper."',
                   "unclear", "human", False, "All 6,348 (benchmark, no train split). label = 1 - card label (card: 1 = consistent). Edited summaries are LLM-written, consistency labels human-annotated."),
    "cleanlab_finqa": S(load_cleanlab_finqa, *CLEANLAB, "MIT", "license: mit", "yes", "human", False,
                        "GPT-4o answers to FinQA (test subset) questions; label = 1 - is_correct, from comparison with the human FinQA gold answer (some golds manually corrected). Short numeric answers."),
    "halueval_qa": S(load_halueval_qa, *HALUEVAL, "Apache-2.0", "license: apache-2.0", "yes", "synthetic", False,
                     "10k items -> right + hallucinated answer = 2 records each. Hallucinated answers ChatGPT-generated and filtered; knowledge/questions from HotpotQA (CC BY-SA 4.0). Upstream GitHub RUCAIBox/HaluEval LICENSE is MIT."),
    "halueval_dialogue": S(load_halueval_dialogue, *HALUEVAL, "Apache-2.0 (card); dialogues from OpenDialKG (CC-BY-NC-4.0)", "license: apache-2.0",
                           "unclear", "synthetic", False, "10k items x 2. Dialogue histories and knowledge come from OpenDialKG, whose README states CC-BY-NC-4.0, so commercial use is doubtful despite the card."),
    "halueval_summary": S(load_halueval_summary, *HALUEVAL, "Apache-2.0", "license: apache-2.0", "yes", "synthetic", False,
                          "10k items x 2. Documents are CNN/DailyMail news articles (publisher copyright; same caveat as any CNN/DM use). Question text is a fixed 'Summarize the following document.'"),
    "ragbench": S(load_ragbench, *RAGBENCH, "CC-BY-4.0", "license: cc-by-4.0", "yes", "llm", True,
                  "Subsets " + ", ".join(RAGBENCH_OPEN) + "; train+validation. Responses from GPT-3.5 / Claude-3-Haiku; label = not adherence_score (GPT-4-family annotator); spans = located unsupported response sentences. Upstream corpora have their own licences (not individually verified here)."),
    "ragbench_restricted": S(load_ragbench_restricted, *RAGBENCH, "CC-BY-4.0 (card); upstream corpora restricted/unverified", "license: cc-by-4.0",
                             "unclear", "llm", True, "Subsets " + ", ".join(RAGBENCH_RESTRICTED) + " kept apart: MS MARCO passages are under Microsoft's non-commercial terms; DelucionQA's README says the car-manual context is withheld 'due to licensing issues' (its data files are CC-BY-4.0, code AGPL-3.0); TechQA and EManual upstream terms not verified."),
    "fava": S(load_fava, *FAVA, "CC-BY-4.0", "license: cc-by-4.0", "yes", "synthetic", True,
              "training.json (synthetic errors inserted into reference-grounded passages). Each tagged item -> erroneous passage (label 1, spans = error-tag text) + corrected passage (label 0); untagged -> one label-0 record. The prompt has no user request, so question = ''. annotations.json (460 human-annotated passages) not used: no grounding documents."),
    "wandb_finqa": S(load_wandb_finqa, *WANDB_FINQA, "MIT (card prose)", '"This dataset follows the MIT License of the original FinQA dataset." (no YAML licence field)',
                     "yes", "synthetic", False, "train split only. Original FinQA gold output (0) paired with an o1-mini-perturbed output (1)."),
    "minicheck": S(load_minicheck, *MINICHECK, "MIT", "license: mit", "yes", "synthetic", False,
                   "C2D + D2C synthetic claim/document pairs (MiniCheck training data); response = claim, question = ''. label = 1 - card label (1 = supported). C2D/D2C pair each claim with several documents (supporting and not), so response-text dedup keeps only the first document per claim and drops most rows."),
    "summacoz": S(load_summacoz, *SUMMACOZ, "CC-BY-ND-4.0", "license: cc-by-nd-4.0", "unclear", "human", False,
                  "SummaC benchmark (CoGenSumm, XSumFaith, Polytope, FactCC, SummEval, FRANK) with explanations; label 2 -> 1, 0 -> 0. NoDerivatives licence: whether training a commercial model is permitted is unclear. Covers FactCC / XSumFaith / FRANK."),
    "halubench": S(load_halubench, *HALUBENCH, "CC-BY-NC-2.0", "license: cc-by-nc-2.0", "no", "synthetic", False,
                   "Single test split (benchmark). Mix: perturbed answers for DROP/CovidQA/PubMedQA/FinanceBench plus HaluEval items (deduplicated against halueval_qa) and 900 RAGTruth items, all dropped: every one has the normalised response text of a RAGTruth test-split answer (some are the generic refusal, reformatted with 'passage N:' prefixes). Marked synthetic. Research evaluation only."),
    "halludial": S(load_halludial, "FlagEval/HalluDial", "52dcc4e903fc0322c24d408e3c617846f34ae8b1", "CC-BY-NC-SA-4.0", 'license: cc-by-nc-sa-4.0 (loader script: _LICENSE = "CC BY-NC-SA")',
                   "no", "llm", False, "HF repo holds only a loader script; data ('all' config, 20,000 public train responses) fetched from the Google Drive URL in that script. Label from the GPT-4 'target' (Yes./No.). Research evaluation only."),
}


# ---------------------------------------------------------------- exclusion set

def make_exclusions(args):
    rows = []  # (origin, response, doc)
    repo, rev = RAGTRUTH
    for x in parquet_rows(repo, "data/test-00000-of-00001.parquet", rev, ["output", "context"]):
        rows.append(("ragtruth_test", x["output"], x["context"]))
    for x in jsonl_rows(args.rag_suite):
        if x.get("split") == "test":
            rows.append(("rag_suite_test", x["response"], x.get("doc") or ""))
    for x in jsonl_rows(args.omni):
        rows.append(("omni_corpus", x["response"], x.get("doc") or ""))
    keys = collections.defaultdict(set)
    for origin, resp, doc in rows:
        keys["r:" + sha1(norm(resp))].add(origin)
        keys[pair_doc_key(resp, doc)].add(origin)
    os.makedirs(os.path.dirname(args.exclude), exist_ok=True)
    with open(args.exclude, "w") as f:
        f.write("# Held-out responses that must never enter probe training data (experiments/probe_data.py).\n")
        f.write("# r:<sha1(norm(response))>  p:<sha1(norm(response) + '\\x1f' + doc_fingerprint)>  then origins.\n")
        f.write("# Sources: " + ", ".join(f"{k}={v}" for k, v in sorted(collections.Counter(o for o, _, _ in rows).items())) + "\n")
        for k in sorted(keys):
            f.write(f"{k}\t{','.join(sorted(keys[k]))}\n")
    print(f"wrote {len(keys)} keys from {len(rows)} held-out responses -> {args.exclude}")


def load_exclusions(path):
    ex = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            k, origins = line.rstrip("\n").split("\t")
            ex[k] = origins
    return ex


# ---------------------------------------------------------------- build

def build(args):
    ex = load_exclusions(args.exclude)
    names = list(SOURCES) if not args.sources else [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        sys.exit(f"unknown sources: {unknown}; known: {list(SOURCES)}")
    names = [n for n in SOURCES if n in names]   # fixed priority order
    os.makedirs(args.out, exist_ok=True)
    seen = {}
    report = {}
    for name in names:
        meta = SOURCES[name]
        st = collections.Counter()
        print(f"[{name}] loading {meta['hf']}@{meta['rev'][:8]}", flush=True)
        recs = sorted(meta["loader"](args, st), key=lambda r: r["id"])
        kept = []
        for r in recs:
            st["raw"] += 1
            assert r["task"] in TASKS, r["task"]
            if not r["response"] or not r["response"].strip():
                st["drop_empty_response"] += 1
                continue
            if sum(len(d) for d in r["documents"]) > MAX_DOC_CHARS:
                st["drop_docs_gt_12k"] += 1
                continue
            if len(r["response"]) > MAX_RESP_CHARS:
                st["drop_response_gt_4k"] += 1
                continue
            rk = resp_key(r["response"])
            hit = ex.get(pair_doc_key(r["response"], r["documents"])) or (ex.get(rk) if rk else None)
            if hit:
                st["excluded_heldout"] += 1
                for o in hit.split(","):
                    st["excluded:" + o] += 1
                continue
            keys = [qr_key(r["question"], r["response"])] + ([rk] if rk else [])
            dup = next((seen[k] for k in keys if k in seen), None)
            if dup is not None:
                st["dedup_within" if dup == name else "dedup_vs_" + dup] += 1
                continue
            for k in keys:
                seen[k] = name
            r = {"id": r["id"], "source": name, "task": r["task"], "license": meta["license"],
                 "label_origin": meta["label_origin"], "question": r["question"], "documents": r["documents"],
                 "response": r["response"], "label": r["label"],
                 "spans": r["spans"] if r["label"] == 1 else [], "has_spans": r["has_spans"]}
            kept.append(r)
        ids = [r["id"] for r in kept]
        assert len(ids) == len(set(ids)), f"duplicate ids in {name}"
        path = os.path.join(args.out, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        st["total"] = len(kept)
        st["hallucinated"] = sum(r["label"] for r in kept)
        st["with_spans"] = sum(r["has_spans"] for r in kept)
        st["bytes"] = os.path.getsize(path)
        report[name] = st
        print(f"[{name}] {dict(st)}", flush=True)
    write_manifest(args, names, report, ex)


def write_manifest(args, names, report, ex):
    origins = collections.Counter()
    for v in ex.values():
        for o in v.split(","):
            origins[o] += 1
    L = ["# Probe training data (document-grounded hallucination detection)", "",
         "Built by `experiments/probe_data.py` (" + ("all sources" if not args.sources else "sources: " + args.sources)
         + ("; official test splits INCLUDED" if args.with_test_splits else "; official test splits of third-party sets left out") + ").",
         "One jsonl per source; fields `id, source, task, license, label_origin, question, documents, response, label, spans, has_spans`. "
         "label 1 = contains unsupported or contradicted content. `spans` are [start, end) character offsets into `response`; "
         "`has_spans` = the record carries span-level annotation (for label 0 that means 'no span').", "",
         "Commercial OK is our reading of the licence quoted from the dataset card at the pinned revision plus the noted upstream terms; it is not legal advice. "
         "Sources with commercial OK = no/unclear are kept in their own files for research evaluation only.", "",
         "| source (file) | HF id @ revision | licence as stated on the card | commercial OK | label origin | total | hallucinated | spans | notes / exclusions |",
         "|---|---|---|---|---|---:|---:|---|---|"]
    tot = hal = 0
    for n in names:
        m, st = SOURCES[n], report[n]
        tot += st["total"]
        hal += st["hallucinated"]
        drops = []
        for k in ("drop_docs_gt_12k", "drop_response_gt_4k", "drop_empty_response", "excluded_heldout"):
            if st[k]:
                drops.append(f"{k}={st[k]}")
        drops += [f"{k}={v}" for k, v in sorted(st.items()) if k.startswith(("excluded:", "dedup_"))]
        drops += [f"{k}={v}" for k, v in sorted(st.items()) if k.startswith(("unparsable", "no_annotation", "dropped_ragtruth_subset", "malformed", "tagged_text", "no_text", "unknown_label", "no_label", "label0_with", "label1_without", "span_unlocated", "zip_sha256"))]
        quote = m["quote"].replace("\n", " ").replace("|", "\\|")
        L.append(f"| {n} (`{n}.jsonl`) | `{m['hf']}` @ `{m['rev'][:12]}` | {quote} | {m['commercial']} | {m['label_origin']} | "
                 f"{st['total']:,} | {st['hallucinated']:,} ({100 * st['hallucinated'] / max(st['total'], 1):.0f}%) | "
                 f"{'yes' if m['spans'] else 'no'} ({st['with_spans']:,}) | {m['notes']} Raw {st['raw']:,}; {', '.join(drops) or 'nothing dropped'}. |")
    com = [n for n in names if SOURCES[n]["commercial"] == "yes"]
    ctot = sum(report[n]["total"] for n in com)
    chal = sum(report[n]["hallucinated"] for n in com)
    L += ["", f"**Total: {tot:,} records, {hal:,} hallucinated ({100 * hal / max(tot, 1):.1f}%).** "
          f"Commercial OK = yes only: {ctot:,} records, {chal:,} hallucinated ({100 * chal / max(ctot, 1):.1f}%).", "",
          "## Held-out exclusions", "",
          f"`exclude_hashes.txt` holds {len(ex):,} keys (" + ", ".join(f"{o}: {c:,}" for o, c in sorted(origins.items())) + " keys; each held-out response contributes an `r:` and a `p:` key). "
          "Sources: the RAGTruth test split (2,700 answers), `data/rag_suite.jsonl` records with split=test (360), `/home/user/omni/corpus.jsonl` (5,107). "
          f"A record is excluded when sha1 of (normalised response, documents fingerprint) matches, or sha1 of its normalised response (NFKC, lower-case, punctuation -> space, whitespace collapsed) matches and that response has at least {MIN_KEY_LEN} normalised characters, so a held-out 'Yes' does not remove every 'Yes'. "
          "A held-out response that is matched always counts once in `excluded_heldout`; the `excluded:<origin>` entries say which held-out sets it occurs in (one response can be in several, e.g. a RAGTruth test answer that is also in the omni corpus), so they can sum to more. "
          "The omni corpus was built from RAGTruth train+test, HaluEval and HaluBench, which is why it removes training items from those sources.", "",
          "## Other filters", "",
          f"* Documents totalling > {MAX_DOC_CHARS:,} characters or responses > {MAX_RESP_CHARS:,} characters are dropped (`drop_*`).",
          f"* Deduplication across and within sources by normalised response (when >= {MIN_KEY_LEN} chars) and by normalised (question, response); the earlier source in the table keeps the record (`dedup_vs_<source>`, `dedup_within`).",
          "* Records are sorted by `id` within each file; output is deterministic for fixed revisions.", "",
          "## Candidates not included", "",
          "| candidate | why |", "|---|---|"]
    L += [f"| {a} | {b} |" for a, b in NOT_INCLUDED]
    with open(os.path.join(args.out, "MANIFEST.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"total {tot} hallucinated {hal}")


NOT_INCLUDED = [
    ("RAGTruth test split (`wandb/RAGTruth-processed`, test)", "Our benchmark: never loaded for training; its responses are in the exclusion set."),
    ("RAGTruth++ (`blue-guardrails/ragtruth-plus-plus`)", "All 408 responses are re-annotations of RAGTruth *test* items (every original_id is in the test split). Also no licence on the card."),
    ("LLM-AggreFact (`lytang/LLM-AggreFact`, cc-by-nd-4.0, gated)", "Gate text: \"Data in the benchmark should not be used in pretraining or fine-tuning any NLP models.\" Also contains RAGTruth data. TofuEval, ClaimVerify, Reveal, Wice, AggreFact-CNN/XSum are only reachable through it on HF."),
    ("TofuEval", "Not on the HF Hub as a standalone dataset (`Dornavineeth/TOFUEval` is the unrelated TOFU unlearning benchmark); only via LLM-AggreFact, excluded above."),
    ("HaluEval `general`", "ChatGPT replies to user queries without grounding documents."),
    ("ExpertQA (`cmalaviya/expertqa`, MIT) standalone", "Its human claim-level attribution judgments are already in AttributionBench (ExpertQA subset); answer-level aggregation not attempted. RAGBench also has an expertqa subset."),
    ("DelucionQA (`corvicai/delucionqa`, MIT)", "184 questions with gold answers only, no hallucination labels; DelucionQA appears via RAGBench (`ragbench_restricted`)."),
    ("XSumFaith (`google-research-datasets/xsum_factuality`), FactCC / FRANK (`mtc/*`)", "XSumFaith repo is a loader script without the source articles; the `mtc/*` copies have no licence on the card. Their annotations are covered through SummaCoz (SummaC benchmark)."),
    ("ANAH-v2", "No ANAH-v2 dataset found on the HF Hub; ANAH (v1, `opencompass/anah`) is used, English only."),
    ("FAVA annotations.json", "460 human-annotated passages but no grounding documents (web-checked)."),
    ("`oumi-ai/oumi-groundedness-benchmark`", "Llama 3.1 Community License (output-use conditions); card does not say how the gold labels were produced; test-only benchmark."),
    ("`aporia-ai/rag_hallucinations`", "No licence and no description of how responses/labels were produced."),
    ("`neural-bridge/rag-hallucination-dataset-1000`", "No hallucination labels (context/question/answer only)."),
    ("`potsawee/wiki_bio_gpt3_hallucination` (SelfCheckGPT)", "Closed-book generation judged for world factuality, not grounded in a provided document."),
    ("`reglab/legal_rag_hallucinations`, `yushihu/halueval-wild`", "No grounding documents in the data."),
    ("`vectara/open_ragbench`", "No hallucination labels (cc-by-nc-4.0)."),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--sources", default="", help="comma-separated subset of: " + ",".join(SOURCES))
    ap.add_argument("--exclude", default=EXCLUDE_FILE, help="exclusion hash file")
    ap.add_argument("--with-test-splits", action="store_true",
                    help="also use official test splits of third-party sets (never RAGTruth test)")
    ap.add_argument("--cache", default=os.path.join(os.path.expanduser("~"), ".cache", "probe_data"),
                    help="cache for non-HF downloads (HalluDial)")
    ap.add_argument("--make-exclusions", action="store_true", help="(re)write --exclude from the private files")
    ap.add_argument("--rag-suite", default=os.path.join(REPO, "data", "rag_suite.jsonl"))
    ap.add_argument("--omni", default="/home/user/omni/corpus.jsonl")
    args = ap.parse_args()
    if args.make_exclusions:
        make_exclusions(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
