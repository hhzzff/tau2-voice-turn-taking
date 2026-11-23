"""Combine comparison and story consistency into one portable HTML with WAVs."""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path


def embedded_data(html: str) -> dict:
    """Read the explicitly embedded display payload from a generated report."""
    match = re.search(
        r'<script id="data" type="application/json">(.*?)</script>', html, re.S
    )
    if match is None:
        raise ValueError("The source report has no embedded display data")
    return json.loads(match.group(1))


def build(batch: Path) -> Path:
    """Embed original audio and link story nodes to the existing transcript player."""
    comparison = (batch / "comparison.html").read_text(encoding="utf-8")
    consistency = (batch / "script_consistency.html").read_text(encoding="utf-8")
    data = embedded_data(comparison)
    story = embedded_data(consistency)
    audio_blocks = []
    for index, sample in enumerate(data["samples"]):
        folder = batch / sample["directory"]
        audio = (folder / "audio/both.wav").read_bytes()
        encoded = base64.b64encode(audio).decode("ascii")
        audio_blocks.append(
            f'<script id="embedded-audio-{index}" type="application/octet-stream">'
            + encoded
            + "</script>"
        )
        sample["assessment"] = json.loads(
            (folder / "assessment.json").read_text(encoding="utf-8")
        )
    data["story"] = story
    payload = json.dumps(data, ensure_ascii=False).replace("<", r"\u003c")
    comparison = re.sub(
        r'(<script id="data" type="application/json">).*?(</script>)',
        lambda m: m.group(1) + payload + m.group(2),
        comparison,
        count=1,
        flags=re.S,
    )
    start = consistency.index('<div class="conclusion">')
    end = consistency.index('<footer class="footer">', start)
    analysis_html = consistency[start:end]
    analysis_html = analysis_html.replace('class="axis"', 'class="chart-axis"')
    analysis_html = analysis_html.replace("choose(", "chooseStage(")
    script_start = consistency.index("'use strict';")
    script_end = consistency.index("</script>", script_start)
    analysis_js = consistency[script_start:script_end]
    analysis_js = analysis_js.replace(
        "JSON.parse(document.getElementById('data').textContent)",
        "JSON.parse(document.getElementById('data').textContent).story",
    )
    analysis_js = analysis_js.replace("choose(", "chooseStage(")
    analysis_js = analysis_js.replace(
        "D.samples.map(s=>{const e=s.events[k]",
        "D.samples.map((s,i)=>{const e=s.events[k]",
    )
    analysis_js = analysis_js.replace(
        "</blockquote><details><summary>查看触发依据与最近 ASR</summary>",
        '</blockquote><button class="listen-stage" onclick="listenStage(${i},${k})">'
        "听这段对话 · 查看 transcript</button>"
        "<details><summary>查看触发依据与最近 ASR</summary>",
    )
    analysis_js = analysis_js.replace(
        'onclick="chooseStage(${k})" onkeydown=',
        'onclick="chooseStage(${k});listenStage(${i},${k})" onkeydown=',
    )
    analysis_js = analysis_js.replace(
        "event.preventDefault();chooseStage(${k})}",
        "event.preventDefault();chooseStage(${k});listenStage(${i},${k})}",
    )
    analysis_js += "\nwindow.chooseStage=chooseStage;\n"
    extra_css = (Path(__file__).parent / "templates/voice_standalone.css").read_text(
        encoding="utf-8"
    )
    comparison = comparison.replace("</style>", extra_css + "\n</style>", 1)
    comparison = comparison.replace(
        "<title>全双工语音对比 · 对话与评审</title>",
        "<title>三模型语音实验 · 音频、对话与剧本一致性</title>",
    )
    comparison = comparison.replace(
        "<h1>听见什么，何时回应</h1>",
        "<h1>三模型语音实验：对话与路径</h1>",
    )
    comparison = comparison.replace(
        "三种语音模型，同一场野餐计划。沿着对话查看追问、否定、背景插话与停顿，结合原始录音判断每一次回应。",
        "Grok、Gemini Live、豆包的同主题新录音。先听对话，再比较剧情分支与插入时机。三组原始音频、transcript 和分析已内嵌，单个文件可离线使用。",
    )
    nav = (
        '<nav class="report-nav" aria-label="报告内容">'
        '<a href="#overview-section">综合评审</a>'
        '<a href="#recordings">音频与 transcript</a>'
        '<a href="#analysis">剧本一致性分析</a></nav>'
    )
    comparison = comparison.replace(
        '<section class="summary">',
        nav + '<section class="summary" id="overview-section">',
        1,
    )
    comparison = comparison.replace(
        '<nav id="tabs"', '<div id="recordings"></div><nav id="tabs"', 1
    )
    comparison = comparison.replace(
        '<details class="conditions">',
        '<section id="analysis" aria-label="剧本进程一致性分析">'
        '<header class="analysis-head"><h2>三次测试的剧本进程一致性</h2>'
        '<p class="hint">点击时间线节点，直接播放对应模型的该阶段对话。</p>'
        "</header>" + analysis_html + "</section>"
        '<details class="conditions">',
        1,
    )
    comparison = comparison.replace(
        "let currentIndex=0,transcriptRows=[],lastFollow=-1;",
        "let currentIndex=0,transcriptRows=[],lastFollow=-1;\n"
        + EMBEDDED_PLAYER_HELPERS,
    )
    comparison = comparison.replace(
        "${esc(s.directory)}/audio/both.wav", "${esc(audioUrl(currentIndex))}"
    )
    comparison = comparison.replace(
        "${esc(s.directory)}/assessment.json", "${esc(assessmentUrl(currentIndex))}"
    )
    comparison = comparison.replace(
        "音频加载失败，请确认 HTML 与模型音频目录一起保留。",
        "内嵌音频加载失败，请重新打开完整 HTML 文件。",
    )
    # Wait for embedded media metadata before seeking when switching models.
    old = (
        "p.currentTime=Math.max(0,Number(t)||0);p.play().catch(()=>{});"
        "updatePlayback();if(scroll)"
    )
    new = (
        "const seek=()=>{if(document.getElementById('player')!==p)return;"
        "p.currentTime=Math.max(0,Number(t)||0);p.play().catch(()=>{});updatePlayback()};"
        "if(p.readyState>=1)seek();else p.addEventListener('loadedmetadata',seek,{once:true});"
        "if(scroll)"
    )
    if old not in comparison:
        raise ValueError(
            "Comparison player changed; update standalone seek integration"
        )
    comparison = comparison.replace(old, new, 1)
    comparison = comparison.replace(
        '<a href="${esc(audioUrl(currentIndex))}" download>',
        '<a href="${esc(audioUrl(currentIndex))}" download="${esc(s.directory)}.wav">',
    )
    comparison = comparison.replace(
        '<a href="${esc(assessmentUrl(currentIndex))}">',
        '<a href="${esc(assessmentUrl(currentIndex))}" download="${esc(s.directory)}_assessment.json">',
    )
    comparison = comparison.replace(
        '<script id="data"', "\n".join(audio_blocks) + '\n<script id="data"', 1
    )
    comparison = comparison.replace(
        "</body>", "<script>\n(()=>{\n" + analysis_js + "\n})();\n</script>\n</body>", 1
    )
    output = batch / "voice_report_standalone.html"
    output.write_text(comparison, encoding="utf-8")
    print(f"{output.resolve()} ({output.stat().st_size / 1024**2:.1f} MiB)")
    return output


EMBEDDED_PLAYER_HELPERS = r"""
const audioCache=new Map(),assessmentCache=new Map();
function audioUrl(i){if(!audioCache.has(i)){const raw=atob(document.getElementById('embedded-audio-'+i).textContent.trim()),bytes=new Uint8Array(raw.length);for(let k=0;k<raw.length;k++)bytes[k]=raw.charCodeAt(k);audioCache.set(i,URL.createObjectURL(new Blob([bytes],{type:'audio/wav'})))}return audioCache.get(i)}
function assessmentUrl(i){if(!assessmentCache.has(i))assessmentCache.set(i,URL.createObjectURL(new Blob([JSON.stringify(D.samples[i].assessment,null,2)],{type:'application/json'})));return assessmentCache.get(i)}
function listenStage(i,k){show(i);const e=D.samples[i].events[k];if(!e)return;jump(Math.max(0,e.start_seconds-3));document.getElementById('recordings').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'auto':'smooth',block:'start'})}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    build(parser.parse_args().batch)
