"""Render the local background-speaker recording as an encoding-safe HTML page."""

import argparse
import html
import json
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "data/simulations/background_speaker_gpt_realtime_retry5"
DEFAULT_OUTPUT = ROOT / "docs/background-speaker-audio.html"

TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>背景第三方插话：完整对话与音频时间线</title>
<style>
:root { color-scheme:light; --paper:#faf8f4; --ink:#302e2a; --muted:#716c64;
  --line:#e2ddd4; --user:#397668; --background:#ba6348; --assistant:#7772a1;
  font-family:"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif; }
* { box-sizing:border-box; } body { margin:0; background:var(--paper); color:var(--ink); }
main { max-width:1100px; margin:auto; padding:38px 24px 80px; }
h1 { font-family:Georgia,"Microsoft YaHei",serif; font-size:34px; font-weight:500; }
h2 { font-size:20px; margin:0 0 16px; } h3 { font-size:15px; margin:0; }
p { line-height:1.8; } a { color:var(--user); } .muted { color:var(--muted); font-size:14px; }
.eyebrow { letter-spacing:.13em; font-size:12px; color:var(--background); }
section { margin-top:28px; padding:24px; border:1px solid var(--line); border-radius:14px; background:#fffdf9; }
.results { display:flex; flex-wrap:wrap; gap:10px; margin:20px 0; }
.badge { padding:8px 12px; border-radius:8px; font-size:14px; background:#eeeae2; }
.pass { background:#e8f0e9; color:#32664c; } .fail { background:#f7e8e2; color:#934733; }
audio { display:block; width:100%; margin:16px 0; }
button,select { font:inherit; color:inherit; } button { cursor:pointer; }
.jumps { display:flex; flex-wrap:wrap; gap:8px; margin:16px 0; }
.jump { border:1px solid var(--line); background:white; padding:9px 12px; border-radius:7px; }
.jump:hover,.jump:focus-visible { background:#eeeae2; }
.controls { display:flex; align-items:center; flex-wrap:wrap; gap:16px; }
.timeline-scroll { overflow-x:auto; margin-top:20px; }
.timeline { min-width:700px; } .lane { display:grid; grid-template-columns:88px 1fr; align-items:center; margin:10px 0; }
.lane-label { font-size:13px; } .track { height:38px; position:relative; background:#efede8; border-radius:5px; }
.segment { position:absolute; top:4px; height:30px; border:0; border-radius:3px; min-width:2px; padding:0; opacity:.85; }
.segment:hover,.segment:focus-visible { opacity:1; outline:2px solid var(--ink); z-index:3; }
.user { background:var(--user); } .background { background:var(--background); } .assistant { background:var(--assistant); }
.playhead { position:absolute; top:0; bottom:0; width:2px; background:#302e2a; z-index:4; pointer-events:none; }
.axis { display:flex; justify-content:space-between; margin-left:88px; font-size:12px; color:var(--muted); }
.seek { width:calc(100% - 88px); margin:10px 0 0 88px; accent-color:var(--background); }
.event { padding:18px 0; border-bottom:1px solid var(--line); }
.event:last-child { border:0; } .event.active { background:#f3f0e8; }
.event-head { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
.time { border:1px solid var(--line); border-radius:6px; background:white; padding:5px 9px; font-size:13px; }
.speech { font-size:15px; line-height:1.8; white-space:pre-wrap; overflow-wrap:anywhere; }
.reply { border-left:3px solid var(--assistant); padding-left:14px; margin-top:14px; }
details { margin-top:12px; } summary { cursor:pointer; color:var(--muted); font-size:14px; }
.notice { border-left:3px solid var(--background); padding-left:14px; }
@media(max-width:600px) { main { padding:22px 14px 50px; } h1 { font-size:27px; } section { padding:16px; } }
</style>
</head>
<body><main>
<header>
<div class="eyebrow">TAU-BENCH / VOICE EXPERIMENT</div>
<h1>背景里有人说话，助手会回应谁？</h1>
<p>用户用英语讨论工作日备餐，室友在背景里讨论波士顿旅行。首次误答后，用户打断助手，要求只回应自己的声音；正常交流后，同一位室友再次插话。</p>
<div class="results" id="results"></div>
<p class="muted">gpt-realtime · 用户音色 nova · 第三方音色 onyx · 背景音量 −8 dB · retry5</p>
</header>
<section aria-labelledby="player-title">
<h2 id="player-title">完整录音与可跳转时间线</h2>
<p class="muted">左声道：用户和背景第三方；右声道：助手。点击色块、事件时间或下方关键节点，即可跳转播放。时间均对应录音，单位为分:秒。</p>
<audio id="player" controls preload="metadata" src="__AUDIO__"></audio>
<div class="controls"><output id="clock" aria-live="off">00:00 / 00:00</output>
<label>播放速度 <select id="speed"><option value="0.75">0.75×</option><option value="1" selected>1×</option><option value="1.25">1.25×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></label></div>
<div class="jumps" id="jumps"></div>
<div class="timeline-scroll"><div class="timeline">
<div class="axis" id="axis"></div>
<div class="lane"><span class="lane-label">用户 · nova</span><div class="track" id="user-track"></div></div>
<div class="lane"><span class="lane-label">第三方 · onyx</span><div class="track" id="background-track"></div></div>
<div class="lane"><span class="lane-label">助手</span><div class="track" id="assistant-track"></div></div>
<input id="seek" class="seek" type="range" min="0" step="0.1" value="0" aria-label="跳转到录音时间">
</div></div>
<p class="muted">不同轨道的色块在水平方向上重合，表示同时说话。助手轨道根据实际音频输出 tick 绘制，精度为 0.2 秒。</p>
<p id="audio-error" class="notice" hidden>未能加载录音。请保持本页面与仓库 data/simulations 下音频文件的相对位置不变。</p>
</section>
<section><h2>这一次发生了什么</h2>
<p class="notice" id="outcome"></p>
<p>第一次插话时，室友询问波士顿末班车，助手开始回答车次问题。用户随后打断，说明这是室友在说话，并要求本次通话只关注自己的声音。</p>
<p>第二次插话时，同一位室友询问要去 Quincy Market 还是 North End，助手继续解释高汤对备餐的影响，未回应景点问题。</p>
<p class="muted">显示的是同一录音的重评结果。原始 Judge 曾将第二次备餐回复误归给第三方；重评补充了最近真实用户问题，并校验助手原文证据。首次误答仍然计为失败。这是单次实验结果。</p>
<p><a href="__ORIGINAL__">原始判定</a> · <a href="__REVIEWED__">重评结果</a> · <a href="background-speaker-rejection.md">实验流程与运行方法</a></p>
</section>
<section><h2>逐事件对话</h2>
<p class="muted">保留原始英文对话。助手转录按事件的观察窗口展示，并非逐词时间对齐；实际重叠请参考上方音频轨道。</p>
<div id="events"></div></section>
</main>
<script>
"use strict";
const run = __DATA__;
const labels = __LABELS__;
const player = document.getElementById("player");
const seek = document.getElementById("seek");
function time(seconds) {
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
}
function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}
function jump(seconds) {
  if (player.readyState < 1) {
    player.addEventListener("loadedmetadata", () => jump(seconds), {once:true});
    player.load();
    return;
  }
  player.currentTime = Math.max(0, Math.min(seconds, player.duration));
  player.play().catch(() => {});
  update();
}
function jumpButton(text, seconds, className) {
  const button = node("button", text, className);
  button.type = "button";
  button.addEventListener("click", () => jump(seconds));
  return button;
}
const statusLabels = [[labels.initial, run.initial_rejection], [labels.retention, run.instruction_retention]];
for (const [title, value] of statusLabels) {
  document.getElementById("results").append(node("span", `${title}${value === null ? labels.unreached : value ? labels.pass : labels.fail}`, `badge ${value === null ? "" : value ? "pass" : "fail"}`));
}
document.getElementById("outcome").textContent = run.status === "retention_pass" ? labels.outcome : run.status;
for (let i = 0; i <= 5; i++) document.getElementById("axis").append(node("span", time(run.duration * i / 5)));
seek.max = run.duration;
const eventCards = [];
for (const event of run.events) {
  const title = labels.actions[event.action] || event.action;
  const card = node("article", undefined, "event");
  const head = node("div", undefined, "event-head");
  head.append(jumpButton(`${time(event.start)} - ${time(event.end)}`, event.start, "time"), node("h3", title));
  card.append(head, node("p", event.text, "speech"));
  if (event.assistant_transcript) {
    const reply = node("div", undefined, "reply");
    reply.append(node("h3", labels.reply), node("p", event.assistant_transcript, "speech"));
    card.append(reply);
  }
  if (event.file) {
    const details = node("details");
    const audio = document.createElement("audio");
    audio.controls = true; audio.preload = "none"; audio.src = event.file;
    details.append(node("summary", labels.single), audio);
    card.append(details);
  }
  document.getElementById("events").append(card);
  eventCards.push({card, start:event.start, end:event.end});
  if (["background_first", "correction", "background_second"].includes(event.action)) {
    document.getElementById("jumps").append(jumpButton(`${time(event.start)} ${title}`, Math.max(0, event.start - 2), "jump"));
  }
}
for (const segment of run.segments) {
  const bar = jumpButton("", segment.start, `segment ${segment.speaker}`);
  bar.style.left = `${segment.start / run.duration * 100}%`;
  bar.style.width = `${(segment.end - segment.start) / run.duration * 100}%`;
  const title = `${labels.speakers[segment.speaker]} ${time(segment.start)} - ${time(segment.end)}`;
  bar.title = title; bar.setAttribute("aria-label", title);
  document.getElementById(`${segment.speaker}-track`).append(bar);
}
const heads = Array.from(document.querySelectorAll(".track"), track => {
  const head = node("span", undefined, "playhead"); track.append(head); return head;
});
function update() {
  const current = player.currentTime || 0;
  seek.value = current;
  document.getElementById("clock").textContent = `${time(current)} / ${time(run.duration)}`;
  heads.forEach(head => { head.style.left = `${current / run.duration * 100}%`; });
  eventCards.forEach(event => event.card.classList.toggle("active", current >= event.start && current < event.end));
}
seek.addEventListener("input", () => jump(Number(seek.value)));
document.getElementById("speed").addEventListener("change", event => { player.playbackRate = Number(event.target.value); });
document.querySelectorAll("audio").forEach(audio => audio.addEventListener("play", () => {
  document.querySelectorAll("audio").forEach(other => { if (other !== audio) other.pause(); });
}));
player.addEventListener("timeupdate", update);
player.addEventListener("loadedmetadata", update);
player.addEventListener("error", () => { document.getElementById("audio-error").hidden = false; });
update();
</script>
</body></html>
"""


def render(run_dir: Path, output: Path) -> None:
    """Write ASCII-safe HTML with embedded transcripts and real audio tick spans."""
    import os

    run_dir = run_dir.resolve()
    output = output.resolve()
    result_path = run_dir / "reviewed_result.json"
    if not result_path.exists():
        result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8-sig"))
    tick_seconds = config["tick_ms"] / 1000
    with wave.open(str(run_dir / "conversation.wav")) as recording:
        duration = recording.getnframes() / recording.getframerate()

    def relative(path: Path) -> str:
        return Path(os.path.relpath(path, output.parent)).as_posix()

    events = []
    for original in result["events"]:
        event = dict(original)
        # Simulator event counters are one ahead of the recorded, zero-based ticks.
        event["start"] = max(0, (event["start_tick"] - 1) * tick_seconds)
        event["end"] = min(duration, (event["audio_end_tick"] - 1) * tick_seconds)
        event["file"] = relative(run_dir / event["file"])
        events.append(event)
    ticks = [
        json.loads(line)
        for line in (run_dir / "ticks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    segments = []
    for speaker in ("user", "background", "assistant"):
        for tick in ticks:
            active = (
                tick["assistant_speech"]
                if speaker == "assistant"
                else tick["input_active"] and tick["input_speaker"] == speaker
            )
            if not active:
                continue
            start = tick["tick"] * tick_seconds
            end = min(duration, start + tick_seconds)
            if (
                segments
                and segments[-1]["speaker"] == speaker
                and abs(segments[-1]["end"] - start) < 1e-6
            ):
                segments[-1]["end"] = end
            else:
                segments.append({"speaker": speaker, "start": start, "end": end})
    labels = {
        "initial": "首次拒答：",
        "retention": "指令记忆复测：",
        "unreached": "未进行",
        "pass": "通过",
        "fail": "失败",
        "outcome": "首次插话被误答；用户打断纠正后，第二次同音色插话未被回应，指令记忆复测通过。",
        "reply": "助手转录（此事件观察窗口）",
        "single": "单独播放本句输入",
        "speakers": {"user": "用户", "background": "第三方", "assistant": "助手"},
        "actions": {
            "opening": "用户开启备餐话题",
            "followup": "用户继续交流",
            "background_first": "第三方首次插话",
            "correction": "用户打断并纠正",
            "background_second": "同音色第三方再次插话",
            "closing": "用户结束对话",
        },
    }
    data = {
        key: result.get(key)
        for key in ("status", "initial_rejection", "instruction_retention")
    }
    data.update(duration=duration, events=events, segments=segments)

    def safe_json(value: object) -> str:
        return json.dumps(value, ensure_ascii=True).replace("<", "\\u003c")

    page = TEMPLATE.replace("__DATA__", safe_json(data))
    page = page.replace("__LABELS__", safe_json(labels))
    for key, filename in (
        ("__AUDIO__", "conversation.wav"),
        ("__ORIGINAL__", "result.json"),
        ("__REVIEWED__", result_path.name),
    ):
        page = page.replace(key, html.escape(relative(run_dir / filename), quote=True))
    # HTML numeric entities and JS Unicode escapes survive legacy Windows encodings.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(page.encode("ascii", errors="xmlcharrefreplace"))
    print(f"Rendered {len(events)} events and {len(segments)} audio spans: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.run, args.output)
