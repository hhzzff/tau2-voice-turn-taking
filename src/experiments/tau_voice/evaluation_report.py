"""Offline explanation of the duplex bench, with explicit topic-audit populations."""

import ast
import hashlib
import html
import json
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

from tau2.config import DEFAULT_DUPLEX_USER_MODEL
from tau2.user.duplex_events import EVENT_CATALOG, DuplexScenario

ROOT = Path(__file__).resolve().parents[3]
LEGACY = "src/experiments/tau_voice/interruption_type_experiment.py"
SOURCES = {
    "主入口": ("src/experiments/tau_voice/duplex_bench.py", "def _run("),
    "Topic 生成": ("src/tau2/user/duplex_events.py", "def generate_topic("),
    "事件目录": ("src/tau2/user/duplex_events.py", "EVENT_CATALOG ="),
    "动态用户策略": ("src/tau2/user/duplex_scenario_user.py", "RUNTIME_POLICY ="),
    "决策检查时机": (
        "src/tau2/user/duplex_scenario_user.py",
        "def _next_turn_taking_action(",
    ),
    "运行时决策输入": ("src/tau2/user/duplex_scenario_user.py", "def _decide("),
    "触发准入检查": ("src/tau2/user/duplex_scenario_user.py", "def allowed("),
    "实际刺激生成": (
        "src/tau2/user/duplex_scenario_user.py",
        "def _generate_full_duplex_voice_message(",
    ),
    "助手音频活动判定": (
        "src/tau2/voice/utils/audio_activity.py",
        "def is_audible_chunk(",
    ),
    "已播放音频观察": (
        "src/tau2/user/duplex_perception.py",
        "class PlayedAudioPerception",
    ),
    "时序评审": ("src/tau2/user/duplex_evaluation.py", "def assess_report("),
    "语义评审": ("src/tau2/user/duplex_evaluation.py", "def judge_report("),
    "记忆评审": ("src/tau2/user/duplex_evaluation.py", "MEMORY_JUDGE_POLICY ="),
    "旧主题池": (LEGACY, "DISCUSSION_TOPIC_PROFILES ="),
    "旧主题采样": (LEGACY, "def _contextual_topic_plan("),
    "开源模型场景": ("src/tau2/user/open_model_scenarios.py", "SCENARIOS ="),
    "原框架交互指标": ("docs/interaction-metrics.md", "## The metric panel"),
    "交接记录": ("HANDOFF.md", "## 六组真实英文对话"),
}

EVENT_ASSESSMENT_LOGIC = {
    "turn_taking.handoff": """try:
    结果 = MemoryJudge(先前已说事实, 最新更正, 完整回答)
    if 结果.verdict in [pass, fail]:
        校验(逐事实证据非空 and 回答ASR非空)
    return 结果.verdict
except:
    return inconclusive""",
    "interruption.follow_up": """记录(首次安静延迟)
return 事件评审("让渡话轮，并回答新增问题")""",
    "interruption.denial": """记录(首次安静延迟)
return 事件评审("让渡、承认被拒绝的建议，并提供替代方案")""",
    "interruption.repeat_request": """记录(首次安静延迟)
return 事件评审("让渡，并重新聚焦原有核心请求")""",
    "interruption.topic_switch": """记录(首次安静延迟)
return 事件评审("停止旧回答，跟随新话题并相关回应")""",
    "interruption.correction": """记录(首次安静延迟)
return 事件评审("让渡，并在后续建议中采用更正后的事实")""",
    "interruption.impatience": """记录(首次安静延迟)
return 事件评审("让渡，并改为简洁、相关的回答")""",
    "interruption.shut_up": """记录(首次安静延迟)
return 事件评审("停止发声，等待用户恢复，不主动补充回答")""",
    "interruption.clarification": """记录(首次安静延迟)
return 事件评审("让渡，并解释所问的具体含义")""",
    "rejection.backchannel": """return 事件评审("把附和当继续信号，保持原话轮，不误当新请求")""",
    "rejection.pause": """区间 = [marker + tick, marker + pause_seconds - tick)
时序 = inconclusive
if 区间长度 > 0:
    时序 = fail if 区间内助手发声 else pass
return 事件评审("等待用户说完句子", 时序)""",
    "rejection.third_party": """时序 = inconclusive
if not 刺激开始前助手发声:
    时序 = fail if 观察窗内助手发声 else pass
return 事件评审("不回应背景者；区分继续原回答与误回应", 时序)""",
    "rejection.talking_to_others": """时序 = inconclusive
if not 刺激开始前助手发声:
    时序 = fail if 观察窗内助手发声 else pass
return 事件评审("不回应用户旁话；区分继续原回答与误回应", 时序)""",
    "rejection.noise": """时序 = inconclusive
if not 刺激开始前助手发声:
    时序 = fail if 观察窗内助手发声 else pass
return 事件评审("不针对噪声作答；继续原回答不算误响应", 时序)""",
    "rejection.contextual": """时序 = inconclusive
if not 刺激开始前助手发声:
    时序 = fail if 观察窗内助手发声 else pass
return 事件评审("核对已说约定与刺激，再判断是否遵守不回应条件", 时序)""",
    "proactive.completion": """return 事件评审("补出上下文支持的内容，而非泛泛回答")""",
    "proactive.backchannel": """return 事件评审("用户仍持有话轮时简短附和，不接管话轮")""",
    "proactive.clarification": """return 事件评审("主动提出针对性澄清，不假定缺失意图")""",
    "proactive.contextual_interruption": """提前 = 存在助手发声(t < marker - tick)
重叠 = 存在助手发声(marker <= t < stimulus_end)
时序 = fail if 提前 or not 重叠 else inconclusive
记录(首次重叠发声时间 - marker if 重叠 else null)
return 事件评审("按先前约定恰当介入", 时序)""",
}

PREPARED_EVENT_IDS = {
    "turn_taking.handoff",
    "rejection.contextual",
    "proactive.contextual_interruption",
}

# These groups describe identical scheduler gates, not identical stimulus content.
EVENT_TIMING_LOGIC = {
    ("assistant_speaking", False): """候选 = 共用决策检查()
if 候选.action != trigger_event: return
if 候选.event_id not in 待测事件: 拒绝()
if not 助手发声快照: 拒绝()
# 到这里才允许打断；没有每事件的秒数或字数阈值
音频 = TTS(候选.text)
记录(triggered, 当前模拟时间)
开始逐 tick 播放(音频)
# LLM / TTS 后不二次刷新助手发声快照
# 事后若未形成真实重叠 → invalid_trigger""",
    ("assistant_silent", False): """候选 = 共用决策检查()
if 候选.action != trigger_event: return
if 候选.event_id not in 待测事件: 拒绝()
if 助手发声快照: 拒绝()
校验事件字段(候选)  # 见左侧刺激要求
合成并开始播放(候选中的刺激)
# 是否产生预期助手行为，留给事后评估""",
    ("any", False): """候选 = 共用决策检查()
if 候选.action != trigger_event: return
if 候选.event_id not in 待测事件: 拒绝()
# 助手正在说话或安静均可，不另设时间阈值
校验事件字段(候选)
生成并开始播放(候选中的刺激)
# 声线 / 噪声内容由各事件分别指定""",
    ("any", True): """# prepare_event 只能在助手发声快照为假时执行
候选 = 共用决策检查()
if 候选.action == prepare_event:
    检查(事件仍待测 and not 助手发声快照)
    播放前置约定或事实，标记事件已准备
elif 候选.action == trigger_event:
    检查(事件仍待测 and 事件已准备)
    合成并开始播放(候选中的刺激)
    # 提问 / 触发本身不要求助手安静
# 前置条件的语义是否有效，由模型组织并由 judge 复核""",
    ("assistant_silent", True): """候选 = 共用决策检查()
if 候选.action == prepare_event:
    检查(事件仍待测 and not 助手发声快照)
    播放触发语约定，标记事件已准备
elif 候选.action == trigger_event:
    检查(事件仍待测 and 事件已准备)
    检查(not 助手发声快照)
    require 候选.continuation 非空
    播放(候选.text → 候选.continuation)
    # prompt 要求触发语在 text 末尾，不是字符串检测器""",
}

EVENT_CHOICE_GUIDANCE = {
    "turn_taking.handoff": "prompt 要求先说事实，间隔交流后问不重述事实值的问题；提问后保护完整回答。代码只硬检查已 prepare，不硬检测间隔交流是否成立。",
    "interruption.follow_up": "Follow-up adds a question：增加问题；何时已有适合追问的内容由模型判断。",
    "interruption.denial": "denial rejects a suggestion：拒绝建议；此刻是否适合拒绝由模型判断。",
    "interruption.repeat_request": "仅有事件契约；没有单独的跑题检测、重复次数或重复请求时机 prompt。",
    "interruption.topic_switch": "仅有事件契约；没有单独的话题切换时机 prompt。",
    "interruption.correction": "依据实际听到的内容纠错，也可修改用户原有事实；不允许虚构助手错误。",
    "interruption.impatience": "impatience expresses urgency：表达紧迫；无等待秒数或发言字数阈值。",
    "interruption.shut_up": "仅有事件契约；没有单独的停止请求时机 prompt。",
    "interruption.clarification": "clarification asks about an unclear meaning：对不清楚的含义提问；具体时机由模型判断。",
    "rejection.backchannel": "prompt 要求简短附和、保持用户在听的角色；模型选择时机。没有独立的语义完成点检测器。",
    "rejection.pause": "候选含未完前缀、至少 0.4 秒停顿和非空 continuation；字段由 schema 硬检查，未完句语义由模型组织。",
    "rejection.third_party": "用背景声线播放候选台词，prompt 要求明确对别人说话。",
    "rejection.talking_to_others": "用主用户声线播放候选台词，prompt 要求明确对旁人说话。",
    "rejection.noise": "候选选择敲门、电话铃或窸窣声之一；生成对应音频，不合成台词。",
    "rejection.contextual": "先实际说出不回应约定，再由模型选择背景或主用户声线产生匹配刺激；语义匹配不由调度器硬检查。",
    "proactive.completion": "prompt 要求用户留下上下文支持、可明确补全的未完句。",
    "proactive.backchannel": "prompt 要求用户连贯长发言并保留话轮，为助手创造附和机会。",
    "proactive.clarification": "prompt 要求用户说完一个在当前上下文中意图含糊的请求。",
    "proactive.contextual_interruption": "prompt 要求先约定唯一触发语，将其置于 text 末尾并继续说话。按计划音频 marker 到达后，若助手发声，控制器清空剩余用户音频；不检测触发语是否实际读出。",
}

EVENT_SCHEDULER_PSEUDOCODE = """共用决策检查():  # 每 tick 进入，默认 1 秒才有一次决策机会
    now = 已经过的模拟音频采样数 / 采样率
    incoming = 最近收到的助手音频块  # 上一输出 tick
    助手发声快照 = is_audible_chunk(incoming)
    收集已播放音频，记录当前事件观察
    if 用户音频队列非空: return keep_talking
    更新当前事件的观察完成状态()
    if 当前事件仍在观察 or now < next_check: return wait
    if 待测事件为空 and 最近助手音频结束不足1秒: return wait
    next_check = now + decision_interval_seconds  # 默认 1.0

    对已播放助手音频前缀做ASR，更新最近60条对话
    候选 = 用户模型(
        RUNTIME_POLICY, 主题, 私有用户背景, now, 助手发声快照,
        待测事件契约, 已准备事件, 已执行事件,
        最近对话, 上次拒绝或错误反馈, 决策JSON schema)
    # 模型自行判断此刻是否自然；可返回 wait 或任一待测事件
    # reason 要求引用当前上下文，保存到 events.json 的 decisions
    校验候选JSON，再执行表内的物理准入检查
    if 拒绝: 保存原因，返回 wait  # 不替换为预设事件
    if 候选.action == wait: return wait
    return 候选

# 助手发声优先按音频PCM16的整块 RMS >= 80 判断
# 音频不可检查时回退 contains_speech；不是语义完成度检测
# LLM/ASR/TTS 同步耗时不是模拟音频时间或固定触发点
# 决策异常先记反馈并 wait；连续3次异常终止运行
"""

EVENT_COMMON_PSEUDOCODE = """# 评估前置：仅评审完成观察的事件
if 观察未完成:
    return not_evaluated
if 有观察记录 and 目录要求助手发声 and 没有真实重叠:
    return not_evaluated

事件评审(要求, 时序=inconclusive):
    try:
        语义 = Judge(共用prompt, 本事件expected_behavior, 真实证据)
        # “要求”是该事件 expected_behavior 的中文概述
        # 刺激不成立或证据不足 → inconclusive
    except:
        return 时序
    if 时序 == fail and 语义 == pass:
        return fail
    return 语义

# 变量与证据范围
# tick：轨迹的实际步长；marker：停顿或触发语标记时刻
# 观察窗：从刺激开始到观察结束，包含刺激期间与之后
# 记忆回答超时 → not_evaluated，不进入 MemoryJudge
# prompt 注释代表模型行为要求，不代表调度器硬编码了语义判断
"""


def _git(*args):
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _literal_assignment(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing source assignment: {name}")


def _event_judge_prompt():
    """Read the actual shared judge instruction without importing model clients."""
    path = ROOT / "src/tau2/user/duplex_evaluation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "judge_report"
    )
    return next(
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("Evaluate this duplex event using only")
    )


def audit_runs(run_dirs):
    """Audit only explicitly selected runs; never silently include retries or history."""
    runs = []
    seen = set()
    for directory in run_dirs:
        directory = Path(directory).resolve()
        if directory in seen:
            raise ValueError(f"Duplicate run directory: {directory}")
        seen.add(directory)
        path = directory / "scenario.json"
        scenario = DuplexScenario.model_validate_json(path.read_text(encoding="utf-8"))
        assessment_path = directory / "assessment.json"
        assessment = (
            json.loads(assessment_path.read_text(encoding="utf-8"))
            if assessment_path.exists()
            else None
        )
        runs.append(
            {
                "directory": str(directory),
                "topic": scenario.topic,
                "language": scenario.language,
                "selected_events": scenario.selected_events,
                "scenario_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "assessment_summary": assessment.get("summary") if assessment else None,
                "assessment_sha256": hashlib.sha256(
                    assessment_path.read_bytes()
                ).hexdigest()
                if assessment_path.exists()
                else None,
            }
        )
    topics = Counter(
        re.sub(r"\s+", " ", run["topic"].strip()).casefold()
        for run in runs
        if run["topic"].strip()
    )
    return {
        "population": "Explicit --run-dir arguments only; no recursive discovery",
        "runs": runs,
        "included_run_count": len(runs),
        "runs_with_nonempty_topic": sum(topics.values()),
        "distinct_normalized_topic_strings": len(topics),
        "topic_string_counts": dict(topics),
        "limitations": "Exact string normalization only; not semantic topic categories or verified audible exposure. Missing assessment is not a pass.",
    }


def collect_evidence(run_dirs):
    """Collect source hashes, catalog contracts and the explicit run population."""
    sources = {}
    for label, (path, needle) in SOURCES.items():
        content = (ROOT / path).read_text(encoding="utf-8")
        sources[label] = {
            "path": path,
            "line": next(
                index
                for index, line in enumerate(content.splitlines(), 1)
                if needle in line
            ),
            "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
        }
    profiles = _literal_assignment(LEGACY, "DISCUSSION_TOPIC_PROFILES")
    native = _literal_assignment("src/tau2/user/open_model_scenarios.py", "SCENARIOS")
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"),
        "origin_main": _git("rev-parse", "origin/main"),
        "working_tree_dirty": bool(_git("status", "--porcelain")),
        "user_model": DEFAULT_DUPLEX_USER_MODEL,
        "sources": sources,
        "event_catalog": [spec.model_dump() for spec in EVENT_CATALOG.values()],
        "judge_prompts": {
            "event": _event_judge_prompt(),
            "memory": _literal_assignment(
                "src/tau2/user/duplex_evaluation.py", "MEMORY_JUDGE_POLICY"
            ),
        },
        "runtime_policy": _literal_assignment(
            "src/tau2/user/duplex_scenario_user.py", "RUNTIME_POLICY"
        ),
        "legacy_discussion_profiles": profiles,
        "native_scenario_ids": list(native),
        "run_audit": audit_runs(run_dirs),
    }


def write_report(output, run_dirs=()):
    """Write standalone HTML and a sidecar containing auditable source evidence."""
    output = Path(output).resolve()
    evidence = collect_evidence(run_dirs)

    def esc(value):
        return html.escape(str(value), quote=True)

    def source(label):
        item = evidence["sources"][label]
        relative = Path(os.path.relpath(ROOT / item["path"], output.parent)).as_posix()
        return f'<a href="{esc(relative)}">{esc(label)} · L{item["line"]}</a>'

    families = ("turn_taking", "interruption", "rejection", "proactive")
    event_tables = []
    for family in families:
        rows = []
        specs = [spec for spec in EVENT_CATALOG.values() if spec.family == family]

        def timing_key(spec):
            return spec.trigger, spec.id in PREPARED_EVENT_IDS

        for key, grouped in groupby(specs, key=timing_key):
            group = list(grouped)
            for index, spec in enumerate(group):
                assessment = EVENT_ASSESSMENT_LOGIC[spec.id]
                if family == "turn_taking":
                    prompt = evidence["judge_prompts"]["memory"]
                    prompt_link = source("记忆评审")
                else:
                    prompt = f"expected_behavior: {spec.expected_behavior}"
                    prompt_link = '<a href="#event-judge-prompt">共用 judge prompt</a>'
                trigger_cell = ""
                if index == 0:
                    trigger_cell = (
                        f'<td class="timing-cell" rowspan="{len(group)}" '
                        f'data-timing-group="{key[0]}:{int(key[1])}">'
                        f'<pre class="logic"><code>{esc(EVENT_TIMING_LOGIC[key])}'
                        "</code></pre></td>"
                    )
                rows.append(
                    f'<tr data-event-id="{esc(spec.id)}"><td><b>{esc(spec.label)}</b>'
                    f'<p class="event-choice">{esc(EVENT_CHOICE_GUIDANCE[spec.id])}'
                    f"</p></td>{trigger_cell}"
                    f'<td><pre class="logic"><code>{esc(assessment)}</code></pre>'
                    f"<details><summary>查看评估 prompt</summary><pre><code>"
                    f"{esc(prompt)}</code></pre>{prompt_link}</details></td>"
                    f"<td><code>{esc(spec.id)}</code></td></tr>"
                )
        heading = "turn taking" if family == "turn_taking" else family
        event_tables.append(
            f'<h3 id="events-{family}">{heading}</h3>'
            f'<div class="table-wrap"><table class="event-table" data-family="{family}">'
            '<colgroup><col class="event-name"><col class="event-trigger">'
            '<col class="event-assessment"><col class="event-id"></colgroup>'
            "<thead><tr><th>事件名称</th><th>事件触发逻辑</th>"
            "<th>评估逻辑&amp;prompt</th><th>事件ID</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )
    profiles = evidence["legacy_discussion_profiles"]
    old_rows = "".join(
        f"<tr><td>{index + 1}</td><td>{esc(profile['main_topic'])}</td>"
        f"<td>{esc(' / '.join(profile['side_topics']))}</td></tr>"
        for index, profile in enumerate(profiles)
    )
    audit = evidence["run_audit"]
    run_rows = (
        "".join(
            f"<tr><td>{esc(Path(run['directory']).name)}</td><td>{esc(run['topic'] or '未生成')}</td>"
            f"<td>{esc(run['language'])}</td><td>{esc(json.dumps(run['assessment_summary'], ensure_ascii=False))}</td></tr>"
            for run in audit["runs"]
        )
        or '<tr><td colspan="4">本次未指定动态 bench 的最终运行目录；主题覆盖率与模型通过率均不可估计。</td></tr>'
    )
    source_rows = "".join(
        f"<tr><td>{source(label)}</td><td><code>{esc(item['path'])}</code></td>"
        f'<td class="hash"><code>{esc(item["sha256"])}</code></td></tr>'
        for label, item in evidence["sources"].items()
    )
    template = (
        Path(__file__).parent / "visualization/templates/evaluation_report.html"
    ).read_text(encoding="utf-8")
    replacements = {
        "DATE": esc(evidence["generated_at_utc"]),
        "MAIN": esc(evidence["origin_main"][:12]),
        "HEAD": esc(evidence["head"][:12]),
        "BRANCH": esc(evidence["branch"]),
        "DIRTY": "有未提交的报告或代码修改"
        if evidence["working_tree_dirty"]
        else "干净",
        "MODEL": esc(evidence["user_model"]),
        "EVENT_TABLES": "".join(event_tables),
        "EVENT_COMMON_PSEUDOCODE": esc(EVENT_COMMON_PSEUDOCODE),
        "EVENT_SCHEDULER_PSEUDOCODE": esc(EVENT_SCHEDULER_PSEUDOCODE),
        "RUNTIME_POLICY": esc(evidence["runtime_policy"]),
        "EVENT_JUDGE_PROMPT": esc(evidence["judge_prompts"]["event"]),
        "OLD_ROWS": old_rows,
        "RUN_ROWS": run_rows,
        "RUN_COUNT": str(audit["included_run_count"]),
        "TOPIC_COUNT": str(audit["distinct_normalized_topic_strings"]),
        "SOURCE_ROWS": source_rows,
        "EVIDENCE_FILE": esc(output.with_suffix(".evidence.json").name),
    }
    replacements.update({f"SOURCE_{key}": source(key) for key in SOURCES})
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
    if re.search(r"\{\{[^}]+\}\}", template):
        raise ValueError("Unresolved report placeholder")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template, encoding="utf-8")
    output.with_suffix(".evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
