"""vt/ledger.py: every statement of every reply put to one judge, with all the evidence there is."""
from vt.ledger import Ledger, exact_checks, overall, units

A, B, C, D = (0.9, 0.04, 0.03, 0.03), (0.04, 0.9, 0.03, 0.03), (0.04, 0.03, 0.9, 0.03), (0.03, 0.03, 0.04, 0.9)


class Judged(Ledger):
    """The ledger with a stand-in judge that reads the prompt: what a model would conclude from it."""

    def __init__(self, retriever=None):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        self.retriever, self.max_lookups, self.max_units, self.max_passages = retriever, 3, 24, 6
        self.prompts = []

    def _judge(self, prompts):
        self.prompts += prompts
        out = []
        for p in prompts:
            stmt = p.split("from the reply")[1].lower()
            ev = p.split("Statement from the reply")[0].split("Code from the reply")[0]
            if "hope" in stmt or "waves" in stmt:
                out.append(D)
            elif "does not parse" in ev or "401 is wrong" in ev or "rome" in stmt and "Paris" in ev:
                out.append(B)
            elif "parses." in ev or "= 391 exactly" in ev or "1889" in stmt and "1889" in ev:
                out.append(A)
            else:
                out.append(C)
        return out


def test_units_and_exact_checks():
    us = units("So 12 * 4 = 50.\n```python\ndef f(x)\n    return x\n```\nHope this helps!")
    assert [("code" in u) for u in us] == [True, False, False]
    assert "does not parse" in exact_checks(us[0])[0]
    assert exact_checks(us[1]) == ["12 × 4 = 48 exactly, so the statement's 50 is wrong."]
    assert exact_checks(us[2]) == []


def test_one_judge_for_every_kind_of_reply():
    looked = []
    led = Judged(retriever=lambda q: looked.append(q) or ["Eiffel Tower (Wikipedia): Opened in Paris in 1889."])
    poem, calc, summary, code = led.build_batch([
        {"text": "The waves sing. The tower rose in Rome.", "messages": [{"role": "user", "content": "A poem?"}]},
        {"text": "17 × 23 = 401.", "messages": [{"role": "user", "content": "17 times 23?"}]},
        {"text": "It opened in 1889.", "messages": [{"role": "user", "content": "Summarize:\n\nIt opened in 1889."}]},
        {"text": "```python\ndef f(x):\n    return x\n```", "messages": [{"role": "user", "content": "Code?"}]}])
    assert [u["verdict"] for u in poem] == ["not a factual claim", "contradicted"]   # looked up, then contradicted
    assert calc[0]["verdict"] == "contradicted" and "391" in calc[0]["evidence"][0]
    assert summary[0]["verdict"] == "supported" and "sources" not in summary[0]   # settled by the user's own text
    assert code[0]["verdict"] == "supported"
    assert len(looked) == 1 and "Rome" in looked[0]                               # only the unsettled statement
    assert overall(poem)["verdict"] == "likely wrong" and overall(summary)["verdict"] == "likely right"


def test_without_a_retriever_unsettled_stays_unverified():
    led = Judged()
    out = led.build_batch([{"text": "Napoleon was short. Hope this helps!", "question": "Was he?"}])[0]
    assert [u["verdict"] for u in out] == ["unverified", "not a factual claim"]
    assert overall(out)["verdict"] == "unverified"
    assert overall([{"claim": "Hi", "verdict": "not a factual claim"}])["verdict"] == "no factual claims"


def test_documents_and_earlier_lookups_are_evidence():
    led = Judged()
    led.build_batch([{"text": "It opened in 1889.", "messages": [{"role": "user", "content": "When?"}],
                      "documents": ["Doc text."], "found": ["Page (Wikipedia): passage."]}])
    assert "Documents the user gave:\nDoc text." in led.prompts[0] and "Found by search:\nPage" in led.prompts[0]


def test_lookups_are_pooled_across_a_reply():
    looked = []
    led = Judged(retriever=lambda q: looked.append(q) or [f"Page {len(looked)} (Wikipedia): Paris 1889."])
    led.max_lookups = 1
    out = led.build_batch([{"text": "It was in Rome. It opened in 1889.", "question": "Where and when?"}])[0]
    assert len(looked) == 1                                                    # one lookup for the reply
    assert [u["verdict"] for u in out] == ["contradicted", "supported"]        # both judged against what it found
