"""Small, portable listening report; external FLACs load only when played."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def quantiles(values):
    values = [v for v in values if v is not None]
    return (
        {
            "n": len(values),
            "p50": round(float(np.percentile(values, 50)), 1),
            "p95": round(float(np.percentile(values, 95)), 1),
            "max": round(max(values), 1),
        }
        if values
        else {"n": 0}
    )


def delta(marks, end, start):
    return (
        round((marks[end] - marks[start]) / 1e6, 2)
        if end in marks and start in marks
        else None
    )


def heard_utterances(events, audit, origin):
    """Show only locally observed text, including explicitly unfinished finals."""
    starts, complete, partial = {}, {}, {}
    for event in events:
        ident = event.get("item_id")
        if not ident:
            continue
        if event["kind"] == "target_first_play":
            starts[ident] = event["mono_ns"]
        elif event["kind"] == "heard_agent_complete":
            complete[ident] = event
        elif event["kind"] == "heard_partial":
            partial[ident] = event
    rows = []
    for ident in dict.fromkeys([*starts, *complete, *partial]):
        event = complete.get(ident) or partial.get(ident) or {}
        stamp = event.get("mono_ns", starts.get(ident, origin))
        rows.append(
            {
                "at": round((starts.get(ident, stamp) - origin) / 1e9, 2),
                "text_available_at": round((stamp - origin) / 1e9, 2),
                "speaker": "助手·本地ASR"
                if ident in complete
                else "助手·本地ASR（未完成尾部刷新）",
                "text": event.get("text", "[已有播放音频，暂无本地转写]"),
                "audit_text": audit.get(ident, ""),
                "interrupted": event.get("interrupted"),
                "item_id": ident,
                "asr_finalized": ident in complete,
            }
        )
    return rows


def scene_record(path):
    """Derive evidence without treating agenda advancement as target success."""
    summary = json.loads((path / "summary.json").read_text())
    case = json.loads((path / "case.json").read_text())
    provenance = json.loads((path / "provenance.json").read_text())
    events = [
        json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()
    ]
    origin = summary.get("origin_ns") or 0
    requests = {
        e.get("request_id"): e
        for e in events
        if e["kind"] == "local_request" and e.get("request_id")
    }
    decisions = []
    for e in events:
        if e["kind"] != "local_result":
            continue
        request = requests.get(e.get("request_id"), {})
        decisions.append(
            {
                "at": round((e["mono_ns"] - origin) / 1e9, 2),
                "phase": e["phase"],
                "ms": e.get("duration_ms"),
                "input_tokens": e.get("usage", {}).get("prompt_tokens"),
                "output_tokens": e.get("usage", {}).get("completion_tokens"),
                "raw": e.get("raw", ""),
                "request_id": e.get("request_id"),
                "input": request.get("body", {}).get("messages", []),
            }
        )
    utterances = []
    for kind, turns in (
        ("用户", summary["turns"]),
        ("附和", summary["backchannels"]),
        ("环境/旁人", summary.get("scenario_audio", [])),
    ):
        for turn in turns:
            m = turn["marks"]
            start = m.get("user_first_voice_tx")
            if not start:
                continue
            utterances.append(
                {
                    "at": round((start - origin) / 1e9, 2),
                    "speaker": kind,
                    "text": turn.get("text") or turn.get("planned_text", ""),
                    "action": turn["action"],
                    "stage": turn.get("pending_intent", {}).get("id", ""),
                    "decision_to_audio_ms": delta(m, "user_first_voice_tx", "decision"),
                    "llm_ttft_ms": delta(m, "llm_first_token", "local_request"),
                    "tts_first_voice_ms": delta(m, "tts_first_voice", "tts_submit"),
                    "audio_queue_ms": delta(
                        m, "user_first_voice_tx", "tts_first_voice"
                    ),
                    "model_generated": turn.get("source") != "frozen_scenario_audio"
                    and turn.get("generation_source")
                    not in {
                        "asked_fields_template",
                        "scene_controlled_text",
                        "scenario_guard_fallback",
                    },
                }
            )
    audit = summary.get("target_transcripts_audit_only", {})
    utterances.extend(heard_utterances(events, audit, origin))
    utterances.sort(key=lambda row: row["at"])
    actions = [e for e in events if e["kind"] == "decision_result"]
    markers = [
        {
            "at": round((e["mono_ns"] - origin) / 1e9, 2),
            "kind": e["kind"],
            "detail": {k: v for k, v in e.items() if k not in {"mono_ns", "kind"}},
        }
        for e in events
        if e["kind"]
        in {
            "scenario_not_triggered",
            "live_error",
            "target_error",
            "scenario_generation_rejected",
            "scenario_private_fields_selected",
            "scenario_reference_computed",
            "scenario_recovery",
            "speech_stale_at_transport",
            "scenario_trigger_selected",
            "target_playback_stop",
            "scenario_observation_end",
        }
    ]
    counts = {
        a: sum(e.get("choice") == a for e in actions)
        for a in ("WAIT", "RESPOND", "INTERRUPT", "BACKCHANNEL")
    }
    clock = [e.get("lateness_ms") for e in events if e["kind"] == "audio_clock_late"]
    metrics = {
        "decision_ms": quantiles(
            [d["ms"] for d in decisions if d["phase"] == "decision"]
        ),
        "decision_to_audio_ms": quantiles(
            [
                u.get("decision_to_audio_ms")
                for u in utterances
                if u["speaker"] == "用户"
            ]
        ),
        "llm_ttft_ms": quantiles([u.get("llm_ttft_ms") for u in utterances]),
        "tts_first_voice_ms": quantiles(
            [u.get("tts_first_voice_ms") for u in utterances]
        ),
        "clock_late_ms_over_5": quantiles(clock),
    }
    return {
        "id": summary["id"],
        "title": summary["title"],
        "status": summary["status"],
        "error": summary["error_class"],
        "seconds": round(summary["duration_s"], 1),
        "event": provenance["event_type"],
        "commit": provenance["commit"],
        "dirty": bool(provenance["dirty"]),
        "goal": case.get("goal"),
        "counts": counts,
        "metrics": metrics,
        "utterances": utterances,
        "decisions": decisions,
        "markers": markers,
        "progress": summary["script_progress"],
        "semantic_verdict": "待人工听测；运行结束不等于目标行为正确",
        "audio": str(path.name) + "/mix.flac",
        "target_incomplete_count": sum(
            r.get("status") == "incomplete" for r in summary["target_responses"]
        ),
        "target_status": [
            {k: r.get(k) for k in ("for_turn", "status", "status_details", "usage")}
            for r in summary["target_responses"]
        ],
    }


PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>τ² · 本地用户模拟器听测</title>
<style>:root{color-scheme:light;--ink:#183329;--muted:#65766f;--line:#dce6df;--paper:#f5f8f3;--accent:#166950}*{box-sizing:border-box}body{overflow-wrap:anywhere;margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 system-ui,sans-serif}header{padding:32px 5vw 24px;background:#15382f;color:#edf8f0}header h1{margin:4px 0;font-size:28px}header p{max-width:1000px;color:#c8ded0}main{display:grid;grid-template-columns:275px minmax(0,1fr);gap:24px;padding:24px 5vw}article{min-width:0}nav{position:sticky;top:16px;align-self:start;max-height:92vh;overflow:auto}button{font:inherit;border:1px solid var(--line);border-radius:8px;background:white;color:var(--ink);padding:10px;cursor:pointer}nav button{display:block;width:100%;text-align:left;margin:0 0 7px}nav button.active{border-color:var(--accent);background:#dfede2}.tag{font-size:12px;color:var(--muted)}section{background:white;border:1px solid var(--line);border-radius:12px;padding:22px;margin-bottom:16px}h2{margin:0 0 8px;font-size:22px}h3{font-size:17px;margin:18px 0 6px}a{color:var(--accent)}audio{width:100%;margin:12px 0}.note{color:var(--muted);font-size:13px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.metric{background:var(--paper);padding:12px;border-radius:8px}.metric strong{font-size:24px;display:block}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid var(--line)}td:last-child{max-width:500px;white-space:normal}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f5f7f4;padding:12px}.utterance{border-left:3px solid #adc8b7;padding:8px 14px;margin:8px 0}.utterance.user{border-color:#d9b65b}summary{cursor:pointer}input{font:inherit;padding:8px;width:100%;border:1px solid var(--line);border-radius:8px;margin-bottom:10px}.time{border:0;background:#edf3ed;padding:2px 7px;font-size:12px}.warn{color:#965119}.ok{color:#166950}@media(max-width:800px){main{display:block;padding:16px}nav{position:static;max-height:240px;margin-bottom:16px}.metrics{grid-template-columns:1fr 1fr}header{padding:24px 16px}}</style>
<header><div>本地审阅 · 集成分支 · 连续真实时钟</div><h1>19 个受控双工场景</h1><p>本地 X-ASR → 单 Qwen3.5-9B 判断与表达 → Qwen3-TTS；目标 gpt-realtime-1.5。ASR、判断、TTS 不暂停目标服务。左声道为用户，右声道为助手。页面和外置音频可以整目录下载后直接打开。</p><span id="suite"></span></header>
<main><nav><input id="filter" placeholder="搜索场景 / ID"><div id="menu"></div></nav><article id="detail"></article></main>
<script>const DATA=__DATA__;const $=id=>document.getElementById(id);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const labels={decision_ms:'B 判断完整耗时',decision_to_audio_ms:'决定开口 → 首音提交',llm_ttft_ms:'B 表达首 token',tts_first_voice_ms:'TTS 请求 → 首个有效音频'};let selected=DATA[0]?.id;
function menu(){const q=$('filter').value.toLowerCase();$('menu').innerHTML=DATA.filter(s=>(s.id+s.title).toLowerCase().includes(q)).map(s=>`<button class="${s.id===selected?'active':''}" onclick="show('${s.id}')"><b>${esc(s.id)}</b> ${esc(s.title)}<br><span class="tag">${esc(s.status)} · ${s.seconds}s · 插入动作 ${s.counts.INTERRUPT}</span></button>`).join('')}
function show(id){document.querySelectorAll('audio').forEach(a=>a.pause());selected=id;const s=DATA.find(x=>x.id===id);location.hash=id;menu();$('detail').innerHTML=`<section><div class="tag">${esc(s.event)} · ${esc(s.commit.slice(0,12))} ${s.dirty?'（有未提交修改）':'（干净提交）'}</div><h2>${esc(s.id+' '+s.title)}</h2><p>${esc(s.goal)}</p><span class="${s.status==='ok'?'ok':'warn'}">运行状态：${esc(s.status)} ${esc(s.error)}</span><p class="note">${esc(s.semantic_verdict)}</p>${s.review_note?`<details open><summary>日志审阅（听感待确认）</summary><p>${esc(s.review_note)}</p></details>`:""}${s.target_incomplete_count?`<p class="warn">目标有 ${s.target_incomplete_count} 次输出未完整结束；查看下方响应状态与原因。</p>`:""}<audio id="player" controls preload="none" src="${esc(s.audio)}"></audio><div><a href="${s.id}/stereo.flac">双声道听测录音</a> · <a href="${s.id}/stereo-timing.flac">时间戳对齐录音</a> · <a href="${s.id}/summary.json">原始摘要</a> · <a href="${s.id}/events.jsonl">完整事件日志</a> · <a href="${s.id}/provenance.json">版本和配置</a></div><p class="note">跳转按钮使用事件时间，选择“时间戳对齐录音”核对精确间隔；普通听测录音保留每段连续 PCM，可能有少量调度差异。client submit 是本地发送完成，不代表远端已听到。</p><button onclick="$('player').src='${s.id}/stereo-timing.flac'">切到时间戳双声道</button> <button onclick="$('player').src='${s.audio}'">切回混音</button></section><section><h3>组件时延（ms）</h3><div class="metrics">${Object.entries(labels).map(([k,l])=>{const m=s.metrics[k];return `<div class="metric"><span>${l}</span><strong>${m.p50??'—'}</strong><span class="note">p50 · p95 ${m.p95??'—'} · n=${m.n}</span></div>`}).join('')}</div><p class="note">p50/p95 由每项实际样本的线性分位数计算；无样本显示 —。预录音频、模板表达没有对应模型生成耗时，不补零。首音统计仅包含实际发出的语音，过期取消单独显示。判断耗时不包含随后表达和 TTS；“决定开口”指判断返回完成。首音用绝对采样幅度 0.005 阈值与 20ms 音频包定位，不是 RMS、音素检测或远端声卡播放时间。</p><p class="note">决策计数是本地动作；INTERRUPT 不等于目标已被打断，旁人/环境声音也可能采用该插入动作。</p><p>决策：${esc(JSON.stringify(s.counts))}。&gt;5ms 时钟迟到：${esc(JSON.stringify(s.metrics.clock_late_ms_over_5))}</p></section><section><h3>对话与实际观察</h3><p class="note">助手条目按首播时刻定位；完整文本在之后逐步识别，未完成尾部刷新会显式标注。服务端完整转写仅作审计。</p>${s.utterances.map(u=>`<div class="utterance ${u.speaker==='用户'?'user':''}"><button class="time" onclick="seek(${u.at})">${u.at.toFixed(2)}s ▶</button> <b>${esc(u.speaker)}</b> <span class="tag">${esc(u.action)} ${esc(u.stage)}</span><div>${esc(u.text)}</div>${u.speaker==='用户'?`<div class="note">决定→首音 ${u.decision_to_audio_ms??'—'}ms；表达 TTFT ${u.llm_ttft_ms??'—'}ms；TTS ${u.tts_first_voice_ms??'—'}ms；${u.model_generated?'模型表达':'预录/模板刺激'}</div>`:''}${u.audit_text?`<details><summary class="note">服务端完整转写（仅审计，B 不可见；可能含未播放内容）</summary>${esc(u.audit_text)}</details>`:''}</div>`).join('')}</section><section><h3>B 输入 token 与耗时</h3><p class="note">每一行对应实际请求；表达请求总时长可能含下游音频队列等待，不等于纯模型解码耗时。展开可核对送入 B 的上下文以及结构化输出。</p><div class="scroll"><table><tr><th>时间</th><th>阶段</th><th>输入/输出 token</th><th>总耗时 ms</th><th>输入与输出</th></tr>${s.decisions.map(d=>`<tr><td>${d.at}s</td><td>${esc(d.phase)}</td><td>${d.input_tokens??'—'} / ${d.output_tokens??'—'}</td><td>${d.ms?.toFixed(1)??'—'}</td><td><details><summary>${esc(d.raw.slice(0,90))}</summary><pre>${esc(JSON.stringify(d,null,2))}</pre></details></td></tr>`).join('')}</table></div></section><section><h3>刺激有效性、异常与推进证据</h3><p class="note">not_triggered 表示未形成预期刺激窗口；不将它算成目标失败。advance 表示完成预定输入，met 只是 B 的判断，都不是独立成功分数。</p>${s.markers.map(m=>`<details><summary>${m.at}s · ${esc(m.kind)}</summary><pre>${esc(JSON.stringify(m.detail,null,2))}</pre></details>`).join('')}<details><summary>完整 agenda 状态与私有信息披露记录</summary><pre>${esc(JSON.stringify(s.progress,null,2))}</pre></details><details><summary>目标响应状态与上下文 token</summary><pre>${esc(JSON.stringify(s.target_status,null,2))}</pre></details></section>`}
function seek(t){const a=$('player');a.currentTime=t;a.play().catch(()=>{})}$('filter').addEventListener('input',menu);$('suite').textContent=`已记录 ${DATA.length}/19 条 · ${DATA.filter(s=>s.status==='ok').length} 条正常结束 · ${DATA.filter(s=>s.status!=='ok').length} 条需检查`;if(DATA.length)show(DATA.some(s=>s.id===location.hash.slice(1))?location.hash.slice(1):DATA[0].id);
</script></html>"""


def build(root):
    """Build from finished trials only; no service or legacy imports needed."""
    rows = [scene_record(p.parent) for p in sorted(root.glob("*/summary.json"))]
    notes_path = root / "review-notes.json"
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    for row in rows:
        note = notes.get(row["id"], {})
        summary_hash = hashlib.sha256(
            (root / row["id"] / "summary.json").read_bytes()
        ).hexdigest()
        if note.get("summary_sha256") == summary_hash:
            row["review_note"] = note["note"]
    data = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    links = [
        f'<a style="color:inherit" href="{name}">{label}</a>'
        for name, label in (
            ("initial/index.html", "首轮19条录音"),
            ("attempts.json", "全部尝试与版本"),
            ("review.md", "完整审阅报告（MD）"),
        )
        if (root / name).exists()
    ]
    page = PAGE.replace("__DATA__", data)
    if links:
        page = page.replace("</header>", "<p>" + " · ".join(links) + "</p></header>")
    (root / "index.html").write_text(page, encoding="utf-8")
    (root / "metrics.json").write_text(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in ("id", "status", "metrics", "counts", "commit", "dirty")
                }
                for r in rows
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--serve", action="store_true", help="Serve locally with audio Range support"
    )
    parser.add_argument("--port", type=int, default=8796)
    args = parser.parse_args()
    print(
        f"Built {len(build(args.directory))} scenes in {args.directory / 'index.html'}"
    )

    if args.serve:
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount
        from starlette.staticfiles import StaticFiles

        app = Starlette(
            routes=[
                Mount(
                    "/",
                    app=StaticFiles(
                        directory=args.directory, html=True, follow_symlink=True
                    ),
                )
            ]
        )
        uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
