"""Self-contained read-only HTML; all imported content is escaped."""

from html import escape

from .analysis import analyze
from .io import dumps, write_new


def render(store, output, comparison=None):
    rows = store.history()
    cards = []
    for row in rows:
        if row["integrity"] != "verified":
            cards.append(f'<article><h2>{escape(row["id"])}</h2><p>证据校验失败</p></article>')
            continue
        item = analyze(store, row["id"])
        facts = item["facts"]
        cards.append(f'''<article><span class="tag">{escape(row['evidence_kind'])}</span>
<h2>{escape(row['id'])}</h2><p>{escape(str(row['status']))} · {facts['recorded_events']} 条观测</p>
<dl><dt>执行步数</dt><dd>{facts['completed_execution_steps']}</dd>
<dt>最高 reward</dt><dd>{escape(str(facts['maximum_reward']))}</dd>
<dt>episode success</dt><dd>{escape(str(facts['episode_success']))}</dd></dl>
<p>Gate：{escape(dumps(facts['gate']))}</p>
<details><summary>事实、统计与证据边界</summary><pre>{escape(dumps(item))}</pre></details></article>''')
    comparison_html = ""
    if comparison is not None:
        comparison_html = '<section><h2>运行比较</h2><p>最早观测差异不等于根因。</p><pre>' + escape(dumps(comparison)) + '</pre></section>'
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Basin · 运行证据</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#101820;color:#e1e9ed;font:16px/1.7 system-ui,sans-serif}
main{max-width:1180px;margin:64px auto;padding:0 24px}header{border-bottom:1px solid #38505e;padding-bottom:28px;margin-bottom:28px}
h1{font-size:48px;letter-spacing:-2px;margin:4px 0}h2{font-size:18px;overflow-wrap:anywhere}.eyebrow,.tag{color:#80ddbe;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}article,section{padding:24px;background:#192731;border:1px solid #344c5a;border-radius:12px}
section{margin-top:24px}p{color:#b4c5ce}dl{display:grid;grid-template-columns:1fr 1fr}dd{margin:0;text-align:right}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 ui-monospace,monospace;max-height:540px;overflow:auto}summary{cursor:pointer;color:#80ddbe}footer{margin-top:32px;color:#99aeb9}
</style><main><header><div class="eyebrow">BASIN / EVIDENCE FIRST</div><h1>先查证，再解释。</h1>
<p>采集适配 → 不可覆盖的 history → 确定性分析 → 人与 agent 调查</p>
<p>独立读取 Rosetta 已有记录；报告不执行训练，也不授予 Gate 或 M2 通过。</p></header>
<div class="grid">''' + "".join(cards) + "</div>" + comparison_html + '''
<footer>SHA-256 检查用于发现文件变化；不证明来源可信。未采集的张量、干预和因果证据仍然缺失。</footer></main></html>'''
    write_new(output, html)
    return {"report": str(output), "runs": len(rows)}
