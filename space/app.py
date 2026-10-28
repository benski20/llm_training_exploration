"""VeriChat as a Hugging Face Space (ZeroGPU). Not published: prepared for when hosting is confirmed.

Every name, number and date in a reply is scored by the model's own internals (P(correct)); with
documents, each claim says whether the documents or the model's memory produced it, with the sentence.
A second tab checks an answer written by any AI with the general hallucination detector.
"""
import html as H
import os
import sys

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import spaces                                                    # ZeroGPU
    gpu = spaces.GPU(duration=120)
except ImportError:
    def gpu(f):
        return f

from vt.verichat import VeriChat

MODEL = os.environ.get("VERICHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HEAD = os.environ.get("VERICHAT_HEAD", "heads/Qwen2.5-7B-Instruct_claims_head.pt")
bot = VeriChat.load(MODEL, HEAD)
DET_HEAD = os.environ.get("VERICHAT_DETECTOR", "heads/Qwen2.5-3B-Instruct_detector.pt")
det = None
if os.path.exists(DET_HEAD):
    import torch
    from vt.detector import Detector
    h = torch.load(DET_HEAD, map_location="cpu", weights_only=True)
    det = Detector(bot.model, bot.tok, h) if h["model"] == MODEL else Detector.load(h["model"], DET_HEAD)
if os.environ.get("VERICHAT_MODERATION", "1") == "1":             # on by default for a public Space
    from vt.moderation import Moderator
    bot.moderator = Moderator.load()
if os.environ.get("VERICHAT_CORPUS"):                                 # a folder of .txt/.md: look up doubted answers
    from vt.retrieval import Corpus
    bot.retriever = Corpus.from_dir(os.environ["VERICHAT_CORPUS"])

COLORS = {"likely right": "#1f7a4d", "uncertain": "#8a5a00", "doubtful": "#a8231c"}


def render(r):
    t, html, pos = r["text"], "", 0
    for c in r["claims"]:
        k = c.get("start", t.find(c["text"], pos))
        if k is None or k < pos or not c.get("text"):
            continue
        html += H.escape(t[pos:k])
        html += f'<b style="color:{COLORS[c["verdict"]]}" title="P(correct) {c["score"]:.2f}">{H.escape(t[k:k + len(c["text"])])}</b>'
        pos = k + len(c["text"])
    html += H.escape(t[pos:])
    html = html.replace("[unsure:] ", "").replace("[unsure:]", "")
    rows = []
    for c in r["claims"]:
        src = f' · from {H.escape(c["source"])}' + (f': “{H.escape(c["citation"]["sentence"])}”' if c.get("citation") else "") \
            if c.get("source") else ""
        rows.append(f'<li><b>{H.escape(c["text"])}</b> — {c["verdict"]} ({c["score"]:.2f}){src}</li>')
    note = ""
    if r.get("retrieval", {}).get("retrieved"):
        note = f'<p><i>The model doubted its first answer ({H.escape(r["retrieval"]["closed_book_text"])}), so it looked it up.</i></p>'
    return html + ("<ul>" + "".join(rows) + "</ul>" if rows else "") + note + \
        ("<p><i>Answer withheld: the model doubted it.</i></p>" if r["withheld"] else "")


@gpu
def chat(message, history, documents, mode):
    msgs = []
    for h in history or []:                                          # "messages" or (user, assistant) tuples
        if isinstance(h, dict):
            if h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str):
                msgs.append({"role": h["role"], "content": h["content"]})
        else:
            msgs += [{"role": "user", "content": h[0]}, {"role": "assistant", "content": h[1]}]
    msgs.append({"role": "user", "content": message})
    docs = [d.strip() for d in (documents or "").split("\n---\n") if d.strip()]
    return render(bot.reply(msgs, documents=docs, mode=mode))


@gpu
def judge(question, answer, documents):
    if det is None or not (question or "").strip() or not (answer or "").strip():
        return "<p>Enter a question and an answer.</p>"
    docs = [d.strip() for d in (documents or "").split("\n---\n") if d.strip()]
    r = det.judge([{"role": "user", "content": question}], answer, documents=docs)
    out, pos = "", 0
    for s in r["spans"]:
        out += H.escape(answer[pos:s["start"]]) + f'<b style="color:#a8231c" title="P {s["score"]:.2f}">' \
            f'{H.escape(answer[s["start"]:s["end"]])}</b>'
        pos = s["end"]
    out += H.escape(answer[pos:])
    verdict = "Likely contains a hallucination" if r["flagged"] else "No hallucination detected"
    return f"<p>{out}</p><p><b>{verdict}</b> (P = {r['p_hallucinated']:.2f}). Red: the spans the reader doubts most.</p>"


with gr.Blocks(title="VeriChat") as demo:
    gr.Markdown("# VeriChat\nEvery name, number and date is scored by the model's own internals. "
                "Green: likely right · amber: uncertain · red: doubtful. With documents, claims say where they came from.")
    with gr.Tab("Chat"):
        docs = gr.Textbox(label="Documents (optional; separate with a line containing ---)", lines=4)
        mode = gr.Radio(["hedge", "stop", "flag"], value="hedge", label="Doubtful claims: mark / withhold / only score")
        gr.ChatInterface(chat, additional_inputs=[docs, mode], type="messages")
    if det is not None:
        with gr.Tab("Check any AI's answer"):
            q = gr.Textbox(label="The question or prompt", lines=2)
            a = gr.Textbox(label="The answer (from any model)", lines=6)
            jd = gr.Textbox(label="Source documents (optional; separate with ---)", lines=3)
            out = gr.HTML()
            gr.Button("Check").click(judge, [q, a, jd], out)

if __name__ == "__main__":
    demo.launch()
