import random

from transformers import AutoTokenizer

from vt.claims import FORMATS, build, claim_tokens, label_claims

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
QA = [{"q": "Who wrote Hamlet?", "aliases": ["Shakespeare", "William Shakespeare"]},
      {"q": "What is the capital of France?", "aliases": ["Paris"]},
      {"q": "Which planet is the largest?", "aliases": ["Jupiter"]},
      {"q": "Who painted the Mona Lisa?", "aliases": ["Leonardo da Vinci", "Leonardo"]}]


def ids(s):
    return TOK(s).input_ids


def tokstr(r, j):
    return TOK.decode(r[j:j + 1]).strip()


def test_claim_tokens_skip_words_from_the_question():
    r = ids("The Chipmunks were created by Jim Henson in 1958.")
    got = [tokstr(r, j) for j in claim_tokens(TOK, r, "Who was the man behind The Chipmunks?")]
    assert got[:2] == ["Jim", "H"] or got[:1] == ["Jim"]
    assert got[-1] == "1"                                   # a number's claim token is its first digit


def test_label_short_and_multiturn():
    rng = random.Random(0)
    for fmt in ("short", "multiturn"):
        it = build(fmt, QA, rng)
        gold = it["golds"][0][0]
        right = label_claims(TOK, it, ids(f"It was {gold}."))
        wrong = label_claims(TOK, it, ids("It was Bartholomew Quint."))
        assert right and right[0][1] == 1 and wrong and wrong[0][1] == 0


def test_label_multi_lines():
    it = build("multi", QA, random.Random(1), k=3)
    lines = [f"{i}. {g[0] if i != 2 else 'Zorblax'}" for i, g in enumerate(it["golds"], 1)]
    lab = label_claims(TOK, it, ids("\n".join(lines)))
    assert [y for _, y in lab] == [1, 0, 1]


def test_label_reasoning_and_math():
    it = build("reasoning", QA, random.Random(2))
    g = it["golds"][0][0]
    lab = label_claims(TOK, it, ids(f"Let me think. Many people know this.\nAnswer: {g}"))
    assert lab and lab[0][1] == 1
    assert label_claims(TOK, it, ids("I am not certain.")) == []        # no answer line, no claim
    m = build("math", [{"q": "Tom has 3 apples and buys 4. How many?", "answer": "7"}], random.Random(0))
    assert label_claims(TOK, m, ids("3 + 4 = 7.\nAnswer: 7"))[0][1] == 1
    assert label_claims(TOK, m, ids("3 + 4 = 8.\n**Answer:** 8"))[0][1] == 0


def test_every_format_builds_chat_messages():
    for fmt in FORMATS:
        pool = QA if fmt != "math" else [{"q": "1+1?", "answer": "2"}] * 4
        it = build(fmt, pool, random.Random(0))
        assert it["messages"][0]["role"] == "system" and it["messages"][-1]["role"] == "user"
        TOK.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True)
