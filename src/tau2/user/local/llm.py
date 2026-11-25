"""Single B semantic review and expression, migrated without prompt changes."""

from __future__ import annotations

import copy
import json
import re
import uuid

from .backends import checked
from .policy import evidence_still_valid
from .recording import TurnRecorder, now


def schema_format(name, schema):
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def clause_end(text):
    for match in re.finditer(r"[，,。！？!?；;：:]", text):
        if match.end() >= 6:
            return match.end()
    return None


RECALL_ANSWER = re.compile(
    r"\d|青竹|夜间自驾|夜里.{0,3}开车|[零〇一二两三四五六七八九十百千万亿]{2,}"
    r"|[一二两三四五六七八九]个?[人位年月日号]"
)

DECISION_PROMPT = (
    "你是普通用户模拟器B，不是被测助手。只看实际已播放音频的本地转写。"
    "分别判断agenda.review_task的各met条件、以及next_intent的插入机会。"
    "met只记录任务语义，不决定接话权；私有计划和未发出的台词不代表助手听过。"
    "opening只需用户表达且助手相关回应，不要求逐字重复。recall_probe只需采到回答或明确不记得，"
    "不把正确答案说给助手，不替它纠错。对照当前已说已听证据逐项判断。"
    "普通交接的allowed_actions只有RESPOND时必须RESPOND，即便met还有false。"
    "next_intent是用户已经决定要表达的下一件事，required包含尚未告知的新信息，不是助手已经说过的话。"
    "在目标仍说话时，检查本场景的trigger_condition；条件已经出现就ready=true，并选对应INTERRUPT或BACKCHANNEL。"
    "不要要求助手先说错才能提出用户自己的新决定，也不要因为当前回答正确就一直压住预定的新意图。"
    "evidence必须摘录latest_agent_text里一段完整、连续、原样的短语，不编造引文。"
    "自我更正日期是用户新想法，无需等助手先说错；附和只表示在听。"
    "不把尚未说完的数值、日期或否定句当成错误。"
    "action属于allowed_actions；ready仅表示插入机会是否出现；不要输出台词或思考过程。"
)

EXPRESSION_PROMPT = (
    "你扮演正在与语音助手交谈的普通用户，只输出实际要说的自然中文，不是助手。"
    "何时开口已由控制器确定，不再决定WAIT。speech_intent是本次唯一主意图，"
    "required里的事实和作用范围须完整准确，不照念字段名或内部说明。"
    "先简短承接最近实际交互，必要时回答助手反问，再表达当前意图；不能复述整段背景。"
    "不擅自改变日期、人数、金额、程度或添加没给出的限制。不假装已操作硬件、订票或购买。"
    "不要把建议说成自己已经执行，也不要把未听到的话说成‘你刚才说’。"
    "例句只帮助理解意图，可以自然改写，但不能把自己的要求反问给助手。"
    "recall_probe只询问指定字段，绝不说出或猜测答案，不确认前一题的错答案。"
    "closing只自然结束，不宣称所有问题解决。通常一到三短句；开场要说全必要事实。"
)


def safe_recent(history, recall=False):
    recent = copy.deepcopy(history[-6:])
    if recall:
        # Semantic adjacency without disclosing any answer-bearing utterance.
        return [
            {
                "role": row["role"],
                "content": "[回忆提问阶段隐藏前文事实原文，避免答案泄漏]",
            }
            for row in recent
        ]
    return [{"role": row["role"], "content": row["content"]} for row in recent]


def expression_violations(text, guard):
    return (
        ["forbidden_expression"]
        if any(re.search(p, text) for p in guard.get("forbidden_patterns", []))
        else []
    ) + [
        "missing_required_expression"
        for p in guard.get("required_patterns", [])
        if not re.search(p, text)
    ]


class ScenarioSpeaker:
    def __init__(self, session, turn, state):
        self.session, self.turn, self.state = session, turn, state

    async def segments(self, *args, **kwargs):
        s, turn = self.session, self.turn
        stage = self.state["pending_intent"]
        if stage.get("answer_mode") in {
            "asked_fields_template",
            "scene_controlled_text",
        }:
            text = stage["prepared_answer"]
            turn["generation_source"] = stage["answer_mode"]
            turn["speech_mode"] = "scenario_controlled"
            TurnRecorder(s, turn).emit(
                "scenario_template_expression",
                text=text,
                source=stage["answer_mode"],
                model_generation_used=False,
            )
            yield text
            return
        recall = stage["kind"] == "recall_probe"
        guard = stage.get("expression_guard")
        payload = {
            "persona": s.case["scenario_runtime"]["persona"],
            "speech_mode": turn["action"],
            "kind": stage["kind"],
            "speech_intent": stage["need"],
            "required": stage["required"],
            "reference_expression": stage["example"],
            "recent_exchanges": safe_recent(self.state["heard_history"], recall),
            "latest_agent_utterance": "[回忆阶段不提供答案原文]"
            if recall
            else self.state["currently_heard_agent"],
            "relevant_private_facts": {} if recall else stage.get("facts", {}),
            "agenda_revision": self.state["agenda"]["version"],
        }
        tagged = TurnRecorder(s, turn)
        for attempt in range(2):
            request_id = "scenario-b-" + uuid.uuid4().hex[:12]
            turn["generation_request_id"] = request_id
            turn["generation_source"] = "B"
            turn["speech_mode"] = "recall_protected" if recall else "natural"
            body = {
                "model": s.cfg.llm.model,
                "temperature": 0,
                "max_tokens": 384,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [
                    {"role": "system", "content": EXPRESSION_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
            }
            start = tagged.emit(
                "local_request",
                phase="generation",
                model_role="B",
                request_id=request_id,
                body=body,
                history_chars=len(
                    json.dumps(payload["recent_exchanges"], ensure_ascii=False)
                ),
                attempt=attempt + 1,
            )
            text, pending, usage, finish = "", "", {}, None
            first = True
            async with s.client.stream(
                "POST", s.cfg.llm.base_url + "/chat/completions", json=body
            ) as response:
                checked(response, "scenario-B-expression")
                async for line in response.aiter_lines():
                    if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                        continue
                    packet = json.loads(line[5:].strip())
                    usage = packet.get("usage") or usage
                    for choice in packet.get("choices", []):
                        finish = choice.get("finish_reason") or finish
                        delta = choice.get("delta", {}).get("content") or ""
                        if not delta:
                            continue
                        if first:
                            tagged.emit("llm_first_token")
                            first = False
                        text += delta
                        pending += delta
                        if not recall and not guard:
                            while (end := clause_end(pending)) is not None:
                                part, pending = pending[:end], pending[end:]
                                yield part
            tagged.emit(
                "local_result",
                phase="generation",
                model_role="B",
                request_id=request_id,
                raw=text,
                usage=usage,
                finish_reason=finish,
                duration_ms=(now() - start) / 1e6,
            )
            turn["generation_usage"] = usage
            if not text.strip() or finish == "length":
                raise ValueError("empty or truncated scenario speech")
            if recall and RECALL_ANSWER.search(text):
                tagged.emit(
                    "recall_generation_rejected",
                    request_id=request_id,
                    reason="possible_answer_leak",
                )
                payload["retry_instruction"] = (
                    "上一稿含数字或答案线索。只用不带答案的问句询问指定字段。"
                )
                continue
            if guard and (violations := expression_violations(text, guard)):
                tagged.emit(
                    "scenario_generation_rejected",
                    request_id=request_id,
                    reasons=violations,
                    reason="scenario_stimulus_contract",
                    before_tts=True,
                )
                payload["retry_instruction"] = guard["retry_instruction"]
                continue
            if pending.strip():
                yield pending.strip()
            return
        if guard:
            fallback = guard["fallback"]
            if expression_violations(fallback, guard):
                raise ValueError("scenario fallback violates stimulus contract")
            turn["generation_source"] = "scenario_guard_fallback"
            tagged.emit(
                "scenario_template_expression",
                text=fallback,
                source="scenario_guard_fallback",
                model_generation_used=False,
                attempts=2,
            )
            yield fallback
            return
        raise ValueError("recall generation failed leakage guard twice")


async def review(session, state, allowed, trigger):
    progress = session.progress
    previous = progress.last_sent
    criteria = previous["stage"].get("criteria", {}) if previous else {}
    properties = {
        key: {
            "type": "boolean",
            **(
                {}
                if previous["complete"] and state["current_reply_text"]
                else {"const": False}
            ),
        }
        for key in criteria
    }
    schema = {
        "type": "object",
        "properties": {
            "met": {
                "type": "object",
                "properties": properties,
                "required": list(criteria),
                "additionalProperties": False,
            },
            "action": {"type": "string", "enum": allowed},
            "ready": {"type": "boolean"},
            "evidence": {"type": "string", "maxLength": 48},
        },
        "required": ["met", "action", "ready", "evidence"],
        "additionalProperties": False,
    }
    payload = {
        "agenda": {
            "review_task": {
                "criteria": criteria,
                "actually_said": previous["text"] if previous else "",
                "heard_reply": state["current_reply_text"],
            }
        },
        "scenario_id": session.case["scenario_runtime"]["source_id"],
        "next_intent": {
            "id": progress.active["id"],
            "kind": progress.active["kind"],
            "need": progress.active["need"],
            "required": progress.active.get("required", []),
            "trigger_condition": trigger,
        },
        "latest_agent_text": state["currently_heard_agent"],
        "recent_exchanges": safe_recent(state["heard_history"]),
        "floor": state["floor_permission"],
        "allowed_actions": allowed,
    }
    variants = progress.active.get("variants")
    if variants:
        payload["next_intent"]["variants"] = variants
        payload["variant_rule"] = (
            "选择尚未回答且满足触发条件的变体；已回答的问题不能装作没听见。没有合适问题时variant=none、ready=false、WAIT。"
        )
        schema["properties"]["variant"] = {
            "type": "string",
            "enum": ["none", *variants],
        }
        schema["required"].append("variant")
    body = {
        "model": session.cfg.llm.model,
        "temperature": 0,
        "max_tokens": 320,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": DECISION_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "response_format": schema_format("scenario_review", schema),
    }
    rid = "scenario-decision-" + uuid.uuid4().hex[:12]
    at = session.rec.emit(
        "local_request",
        phase="decision",
        model_role="B",
        request_id=rid,
        turn=len(session.turns) + 1,
        body=body,
    )
    response = await session.client.post(
        session.cfg.llm.base_url + "/chat/completions", json=body
    )
    checked(response, "scenario-B-review")
    data = response.json()
    choice = data["choices"][0]
    raw = choice["message"]["content"]
    wire = json.loads(raw)
    done = session.rec.emit(
        "local_result",
        phase="decision",
        model_role="B",
        request_id=rid,
        raw=raw,
        usage=data.get("usage", {}),
        duration_ms=(now() - at) / 1e6,
        finish_reason=choice.get("finish_reason"),
    )
    if (
        choice.get("finish_reason") == "length"
        or set(wire) != set(schema["required"])
        or set(wire["met"]) != set(criteria)
        or any(type(x) is not bool for x in wire["met"].values())
        or wire["action"] not in {"WAIT", "RESPOND", "INTERRUPT", "BACKCHANNEL"}
        or type(wire["ready"]) is not bool
        or not isinstance(wire["evidence"], str)
    ):
        raise ValueError("invalid scenario decision")
    if variants and wire.get("variant") not in {"none", *variants}:
        raise ValueError("unknown scenario variant")
    if any(wire["met"].values()) and not (
        previous and previous["complete"] and state["current_reply_text"]
    ):
        raise ValueError("semantic success without delivered user and actual reply")
    return wire, rid, done, data.get("usage", {})


async def requested_private_fields(session, state, stage):
    """Select only explicitly asked fields; private answers never enter this request."""
    fields = stage["private_fields"]
    signals = stage.get("request_signal_patterns", [])
    if signals and not any(
        re.search(pattern, state["current_reply_text"]) for pattern in signals
    ):
        stage["other_question"] = ""
        session.rec.emit(
            "scenario_private_fields_selected",
            stage_id=stage["id"],
            requested=[],
            evidence={},
            recovered_by_script=[],
            private_values_visible_to_selector=False,
            source="no_request_signal_guard",
        )
        return []
    schema = {
        "type": "object",
        "properties": {
            "asked": {
                "type": "array",
                "maxItems": len(fields),
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string", "enum": list(fields)},
                        "evidence": {"type": "string", "maxLength": 100},
                    },
                    "required": ["field", "evidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["asked"],
        "additionalProperties": False,
    }
    schema["properties"]["other_question"] = {"type": "string", "maxLength": 100}
    schema["required"].append("other_question")
    payload = {
        "actually_heard": state["current_reply_text"],
        "field_meanings": {k: v["label"] for k, v in fields.items()},
        "already_delivered": session.progress.profile,
    }
    body = {
        "model": session.cfg.llm.model,
        "temperature": 0,
        "max_tokens": 384,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "system",
                "content": "你是用户模拟器的信息选择器。只提取助手本轮实际询问用户的字段，并原样引用对应问题；陈述、猜测、推荐和内部需要不算询问。已经提供过的字段再次被问也必须列出，由Python记录重复，不能改为未知问题。若先列出字段再问‘这些都有了吗’、‘告诉我这些信息’，这是对刚列出字段的成组请求，列全这些字段，引文包含字段与指代问题。‘从哪儿到哪儿’同时问出发地和目的地；‘从哪个城市上车’问出发地。问酒店星级不等于问预算，问待几天不等于问具体日期。问了多项就全部列出，没有对应询问则asked为空。other_question只放这些字段以外的真正未知问题，不重复放已提取问题的概括或‘你能确定哪部分’，‘确认好再告诉我’不是新问题。没有则为空字符串。不要提供答案。"
                + stage.get("field_selection_guidance", ""),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "response_format": schema_format("asked_fields", schema),
    }
    rid = "scenario-fields-" + uuid.uuid4().hex[:12]
    start = session.rec.emit(
        "local_request",
        phase="field_selection",
        model_role="B",
        request_id=rid,
        body=body,
    )
    response = await session.client.post(
        session.cfg.llm.base_url + "/chat/completions", json=body
    )
    checked(response, "scenario-field-selection")
    data = response.json()
    raw = data["choices"][0]["message"]["content"]
    session.rec.emit(
        "local_result",
        phase="field_selection",
        model_role="B",
        request_id=rid,
        raw=raw,
        duration_ms=(now() - start) / 1e6,
        usage=data.get("usage", {}),
    )
    wire = json.loads(raw)
    asked = []
    other = wire.get("other_question", "")
    if other and not evidence_still_valid(
        other, state["current_reply_text"], state["current_reply_text"]
    ):
        raise ValueError("unknown field question lacks heard evidence")
    stage["other_question"] = other
    for row in wire.get("asked", []):
        key = row.get("field")
        if key not in fields or not evidence_still_valid(
            row.get("evidence", ""),
            state["current_reply_text"],
            state["current_reply_text"],
        ):
            raise ValueError(
                "private field disclosure without actual question evidence"
            )
        anchors = fields[key].get("question_terms", [])
        known_choice = key in session.progress.profile and re.search(
            r"你更想|更想哪|倾向哪|选哪|哪一种|哪个", row["evidence"]
        )
        if (
            anchors
            and not known_choice
            and not any(
                re.search(pattern, row["evidence"], re.I) for pattern in anchors
            )
        ):
            session.rec.emit(
                "scenario_private_field_rejected",
                field=key,
                evidence=row["evidence"],
                reason="question_has_no_field_anchor",
            )
            stage["other_question"] = stage["other_question"] or row["evidence"]
            continue
        if key not in asked:
            asked.append(key)
    recovered = []
    for key, definition in fields.items():
        for pattern in definition.get("explicit_question_patterns", []):
            match = re.search(pattern, state["current_reply_text"])
            if key not in asked and match:
                asked.append(key)
                recovered.append(key)
                session.rec.emit(
                    "scenario_private_field_recovered",
                    field=key,
                    evidence=match.group(),
                    source="explicit_question_guard",
                    model_selected=False,
                )
                if compact(stage.get("other_question", "")) == compact(match.group()):
                    stage["other_question"] = ""
    # Resolve explicitly enumerated fields followed by an availability question.
    # Only the field names in this actual utterance are eligible, never all slots.
    for pattern in stage.get("group_request_patterns", []):
        match = re.search(pattern, state["current_reply_text"])
        if not match:
            continue
        for key, terms in stage.get("group_reference_fields", {}).items():
            if key not in asked and any(
                re.search(term, match.group("fields")) for term in terms
            ):
                asked.append(key)
                recovered.append(key)
                session.rec.emit(
                    "scenario_private_field_recovered",
                    field=key,
                    evidence=match.group(),
                    source="enumerated_group_request",
                    model_selected=False,
                )
        heard, other = (
            compact(state["current_reply_text"]),
            compact(stage.get("other_question", "")),
        )
        left = len(compact(state["current_reply_text"][: match.start()]))
        right = len(compact(state["current_reply_text"][: match.end()]))
        start = heard.find(other) if other else -1
        if start >= 0 and start < right and start + len(other) > left:
            stage["other_question"] = ""
    if not asked and not stage.get("other_question"):
        question = re.search(
            r"[^。！？?\n]*吗[？?。]?\s*$", state["current_reply_text"]
        )
        if question:
            stage["other_question"] = question.group().strip()
            session.rec.emit(
                "scenario_unknown_question_recovered",
                evidence=question.group(),
                source="explicit_question_guard",
                private_values_disclosed=False,
            )
    session.rec.emit(
        "scenario_private_fields_selected",
        stage_id=stage["id"],
        requested=asked,
        evidence=wire,
        recovered_by_script=recovered,
        private_values_visible_to_selector=False,
    )
    return asked


def compact(text):
    """Normalize question quotations for deterministic field matching."""
    return re.sub(r"[\W_]+", "", text)
