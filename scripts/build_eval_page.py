#!/usr/bin/env python3
"""Blind-eval stage 3 (primary judge: Stefan): build the local blind A/B page, SFT vs DPO.

Same protocol as the judge spot-check: sides randomized per pair, pick first, reveal after,
running tally, undo. The final percentage is THE pre-registered shipping verdict:
DPO ships only if it wins the decided pairs.

  python3 scripts/build_eval_page.py [--n 60] [--answers ...] [--out ~/Desktop/serblo_blindeval.html]
"""
import argparse
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[1]

PAGE = """<!DOCTYPE html>
<html lang="sr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Serblo — SLEPI DUEL: SFT vs DPO</title>
<style>
:root{--bg:#101418;--panel:#1a2026;--line:#2c343c;--txt:#e8ecef;--mut:#8b98a5;--acc:#c9a35f;--ok:#7fc98a;--bad:#e08484;}
@media (prefers-color-scheme: light){:root{--bg:#f5f4f0;--panel:#fff;--line:#ddd;--txt:#1c2226;--mut:#68737d;--acc:#9a7328;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font:16px/1.55 system-ui,sans-serif;padding:24px}
main{max-width:900px;margin:0 auto}h1{font-size:1.1rem;letter-spacing:.04em}
.meta{color:var(--mut);font-size:.85rem;margin-bottom:18px}
.q{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px;font-weight:600}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.ans{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;cursor:pointer;white-space:pre-wrap}
.ans:hover{border-color:var(--acc)}
.ans.pick{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc)}
.tag{display:inline-block;margin-top:10px;font-size:.75rem;letter-spacing:.08em;color:var(--mut)}
.bar{display:flex;gap:10px;align-items:center;margin:16px 0}
button{background:var(--panel);color:var(--txt);border:1px solid var(--line);border-radius:8px;padding:8px 14px;cursor:pointer;font-size:.9rem}
button:hover{border-color:var(--acc)}button:disabled{opacity:.4;cursor:default}
.tally{margin-left:auto;color:var(--mut);font-variant-numeric:tabular-nums}
.verdict{min-height:1.4em;margin:10px 0;font-weight:600}
.done{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:24px;text-align:center;font-size:1.05rem}
.win{color:var(--ok)}.lose{color:var(--bad)}
@media(max-width:700px){.pair{grid-template-columns:1fr}}
</style></head><body><main>
<h1>SERBLO · slepi duel — SFT protiv DPO</h1>
<div class="meta">Isti promptovi, po jedan odgovor iz svakog modela, strane nasumične. Klikni BOLJI (ili „nerešeno"). Posle izbora piše koji je model koji.</div>
<div id="app"></div>
<script>
const DATA = __DATA__;
let i = 0, dpo = 0, sft = 0, ties = 0;
let hist = [];
const app = document.getElementById('app');
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
function render(){
  if(i >= DATA.length){
    const dec = dpo + sft;
    const pct = dec ? Math.round(dpo/dec*100) : 0;
    const cls = dpo > sft ? 'win' : 'lose';
    app.innerHTML = `<div class="done">Kraj. <b class="${cls}">DPO ${dpo} : ${sft} SFT</b>`+
      `${ties?` · nerešeno ${ties}`:''}<br>DPO uzima ${pct}% odlučenih.<br><br>Javi Klodu ove brojke.</div>`;
    return;
  }
  const it = DATA[i];
  app.innerHTML = `
    <div class="meta">duel ${i+1} / ${DATA.length}</div>
    <div class="q">${esc(it.q)}</div>
    <div class="pair">
      <div class="ans" id="L" onclick="pick('left')">${esc(it.left)}<div class="tag" id="tL"></div></div>
      <div class="ans" id="R" onclick="pick('right')">${esc(it.right)}<div class="tag" id="tR"></div></div>
    </div>
    <div class="bar">
      <button onclick="pick('tie')">nerešeno / oba loša</button>
      <button onclick="undo()" ${hist.length?'':'disabled'}>↩ vrati poslednji</button>
      <span class="tally">DPO ${dpo} : ${sft} SFT${ties?` · X ${ties}`:''}</span>
    </div>
    <div class="verdict" id="v"></div>`;
}
function pick(side){
  hist.push({i, dpo, sft, ties});
  const it = DATA[i];
  document.getElementById('tL').textContent = it.leftModel.toUpperCase();
  document.getElementById('tR').textContent = it.rightModel.toUpperCase();
  const v = document.getElementById('v');
  if(side === 'tie'){ ties++; v.innerHTML = '<span class="meta">nerešeno</span>'; }
  else{
    const winner = side === 'left' ? it.leftModel : it.rightModel;
    if(winner === 'dpo'){ dpo++; v.innerHTML = '<span class="win">→ izabrao si DPO</span>'; }
    else{ sft++; v.innerHTML = '<span class="lose">→ izabrao si SFT</span>'; }
    document.getElementById(side === 'left' ? 'L' : 'R').classList.add('pick');
  }
  document.querySelectorAll('.ans').forEach(a => a.onclick = null);
  setTimeout(()=>{ i++; render(); }, 1000);
}
function undo(){
  if(!hist.length) return;
  const h = hist.pop();
  i = h.i; dpo = h.dpo; sft = h.sft; ties = h.ties;
  render();
}
render();
</script></main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=str(ROOT / "data/dpo/eval_answers.jsonl"))
    ap.add_argument("--out", default=str(pathlib.Path.home() / "Desktop/serblo_blindeval.html"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.answers) if l.strip()]
    rng = random.Random(a.seed)
    rows = rng.sample(rows, min(a.n, len(rows)))
    items = []
    for r in rows:
        q = next((m["content"] for m in r["prompt_messages"] if m["role"] == "user"), "")
        flip = rng.random() < 0.5
        left, right = ("dpo", "sft") if flip else ("sft", "dpo")
        items.append({"q": q, "left": r[left], "right": r[right],
                      "leftModel": left, "rightModel": right})
    out = pathlib.Path(a.out)
    out.write_text(PAGE.replace("__DATA__", json.dumps(items, ensure_ascii=False)))
    print(f"written: {out} ({len(items)} duels)")


if __name__ == "__main__":
    main()
