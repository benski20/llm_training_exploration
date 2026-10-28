"""VeriChat as a Hugging Face Space (ZeroGPU). Not published: prepared for when hosting is confirmed.

Every name, number and date in a reply is scored by the model's own internals (P(correct)); with
documents, each claim says whether the documents or the model's memory produced it, with the sentence.
"""
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

COLORS = {"likely right": "#1f7a4d", "uncertain": "#8a5a00", "doubtful": "#a8231c"}


def render(r):
    t, html, pos = r["text"], "", 0
    for c in r["claims"]:
        k = c.get("start", t.find(c["text"], pos))
        if k is None or k < pos or not c.get("text"):
            continue
        html += gr.utils.sanitize_html(t[pos:k]) if hasattr(gr.utils, "sanitize_html") else t[pos:k]
        html += f'<b style="color:{COLORS[c["verdict"]]}" title="P(correct) {c["score"]:.2f}">{t[k:k + len(c["text"])]}</b>'
        pos = k + len(c["text"])
    html += t[pos:]
    html = html.replace("[unsure:] ", "").replace("[unsure:]", "")
    rows = []
    for c in r["claims"]:
        src = f' · from {c["source"]}' + (f': “{c["citation"]["sentence"]}”' if c.get("citation") else "") if c.get("source") else ""
        rows.append(f'<li><b>{c["text"]}</b> — {c["verdict"]} ({c["score"]:.2f}){src}</li>')
    return html + ("<ul>" + "".join(rows) + "</ul>" if rows else "") + ("<p><i>Answer withheld: the model doubted it.</i></p>" if r["withheld"] else "")


@gpu
def chat(message, history, documents, mode):
    msgs = []
    for u, a in history or []:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": message})
    docs = [d.strip() for d in (documents or "").split("\n---\n") if d.strip()]
    return render(bot.reply(msgs, documents=docs, mode=mode))


with gr.Blocks(title="VeriChat") as demo:
    gr.Markdown("# VeriChat\nEvery name, number and date is scored by the model's own internals. "
                "Green: likely right · amber: uncertain · red: doubtful. With documents, claims say where they came from.")
    docs = gr.Textbox(label="Documents (optional; separate with a line containing ---)", lines=4)
    mode = gr.Radio(["hedge", "stop", "flag"], value="hedge", label="Doubtful claims: mark / withhold / only score")
    gr.ChatInterface(chat, additional_inputs=[docs, mode])

if __name__ == "__main__":
    demo.launch()
