"""Deterministic agenda and disclosure accounting, independent of transport."""

from __future__ import annotations

import copy
import json
from decimal import Decimal

PROTOCOL = "single-b-floor-scenarios-v17"


def private_answer_text(stage, asked):
    """Minimal P03 answers cannot invent or contradict undisclosed user facts."""
    parts = [stage["private_fields"][key]["answer"].rstrip("。") for key in asked]
    if stage.get("other_question"):
        parts.append("你另外问的细节，我还没确认")
    if not parts:
        raise ValueError("private answer has no actual question")
    return "。".join(parts) + "。"


def settlement_reference(problem, delivered):
    """Exact-cent ledger for this declared equal-share settlement fixture."""
    missing = [
        e["amount_field"]
        for e in problem["expenses"]
        if e["amount_field"] not in delivered
    ]
    if missing:
        return {"kind": "settlement", "ready": False, "missing": missing}
    paid = {name: 0 for name in problem["participants"]}
    owed = dict(paid)
    shares = []
    for expense in problem["expenses"]:
        exact = Decimal(str(delivered[expense["amount_field"]])) * 100
        cents = int(exact)
        if cents != exact or cents < 0 or cents % len(expense["shared_by"]):
            raise ValueError(
                "settlement fixture must have nonnegative exactly divisible cent amounts"
            )
        per_person = cents // len(expense["shared_by"])
        paid[expense["payer"]] += cents
        for name in expense["shared_by"]:
            owed[name] += per_person
        shares.append(
            {
                "expense": expense["id"],
                "total_cents": cents,
                "shared_by": expense["shared_by"],
                "per_person_cents": per_person,
            }
        )
    balances = {name: paid[name] - owed[name] for name in paid}
    creditors = [[name, value] for name, value in balances.items() if value > 0]
    transfers = []
    for debtor, balance in balances.items():
        remaining = max(0, -balance)
        for creditor in creditors:
            amount = min(remaining, creditor[1])
            if amount:
                transfers.append(
                    {"from": debtor, "to": creditor[0], "amount_cents": amount}
                )
                remaining -= amount
                creditor[1] -= amount
    return {
        "kind": "settlement",
        "ready": True,
        "missing": [],
        "shares": shares,
        "balances_cents": balances,
        "paid_cents": paid,
        "owed_cents": owed,
        "transfers": transfers,
        "note": "Reference from fully delivered amounts only; compare the actual agent answer separately.",
    }


class ScenarioAgenda:
    def __init__(self, case, cfg, client, rec, on_settled=None):
        self.case, self.cfg, self.client, self.rec = case, cfg, client, rec
        self.frames = copy.deepcopy(case["scenario_runtime"]["stages"])
        self.cursor = self.version = 0
        self.exchanges, self.reviews = set(), []
        self.task_states, self.profile = {}, {}
        self.last_sent = None
        self.error = None
        self.samples = []
        self.protocol_version = PROTOCOL
        self.protocol_version = case["scenario_runtime"].get("protocol", PROTOCOL)
        self.topics, self.active_topic = {}, None
        self.required_fields = case["scenario_runtime"].get(
            "required_private_fields", []
        )
        self.private_questions, self.private_answers = [], []
        rec.emit(
            "control_protocol",
            version=self.protocol_version,
            scenario_id=case["scenario_runtime"]["source_id"],
            architecture="audio floor -> one B review -> Python agenda -> same B expression",
            met_controls_floor=False,
            speculative=False,
        )
        (rec.path / "scenario-spec.json").write_text(
            json.dumps(case["scenario_runtime"], ensure_ascii=False, indent=2)
        )

    @property
    def active(self):
        return self.frames[self.cursor] if self.cursor < len(self.frames) else None

    def state(self):
        return copy.deepcopy(self.profile), copy.deepcopy(self.active)

    def view(self):
        return {
            "version": self.version,
            "top": copy.deepcopy(self.active),
            "pending": False,
            "expressed_turn": None,
            "remaining_count": len(self.frames) - self.cursor,
            "completed_count": sum(
                bool(t["met"]) and all(t["met"].values())
                for t in self.task_states.values()
            ),
            "sampling_cursor": self.cursor,
        }

    def activate_next(self, reason):
        old = self.active["id"]
        self.cursor += 1
        self.version += 1
        self.rec.emit(
            "scenario_stage_advance",
            previous=old,
            next=self.active["id"] if self.active else None,
            reason=reason,
            semantic_success_implied=False,
        )

    def user_finished(self, turn):
        if not turn["marks"].get("user_first_voice_tx"):
            return
        stage = turn["pending_intent"]
        complete = bool(
            turn["marks"].get("user_audio_complete") and not turn.get("interrupted")
        )
        self.last_sent = {
            "stage": copy.deepcopy(stage),
            "turn": turn["index"],
            "text": turn.get("spoken_text", turn["text"]),
            "complete": complete,
            "end_ns": turn["marks"].get("user_audio_complete", 0),
        }
        prior = self.task_states.get(stage["id"], {})
        self.task_states[stage["id"]] = {
            **prior,
            "delivered": complete,
            "spoken_text": self.last_sent["text"],
            "turn": turn["index"],
            "met": prior.get("met", {}),
            "criteria": stage.get("criteria", {}),
        }
        if complete:
            if stage["kind"] == "private_answer":
                delivered = list(stage.get("update", {}))
                self.private_answers.append(
                    {
                        "turn": turn["index"],
                        "stage_id": stage["id"],
                        "fields": delivered,
                        "new_required": [
                            k
                            for k in delivered
                            if k in self.required_fields and k not in self.profile
                        ],
                        "repeated": [k for k in delivered if k in self.profile],
                    }
                )
            self.profile.update(stage.get("update", {}))
            for topic in stage.get("suspend_topics", []):
                self.topics.setdefault(topic, {})["state"] = "suspended"
            if stage.get("topic"):
                topic = stage["topic"]
                self.topics.setdefault(topic, {}).update(
                    state="active", last_spoken=self.last_sent["text"]
                )
                self.active_topic = topic
        self.rec.emit("scenario_user_evidence", **self.last_sent)

    def record_private_question(self, stage, asked, heard):
        if not asked and not stage.get("other_question"):
            return
        row = {
            "stage_id": stage["id"],
            "heard": heard,
            "asked": list(asked),
            "new_required": [
                k for k in asked if k in self.required_fields and k not in self.profile
            ],
            "repeated": [k for k in asked if k in self.profile],
            "other_question": stage.get("other_question", ""),
            "after_required_complete": bool(self.required_fields)
            and all(k in self.profile for k in self.required_fields),
        }
        self.private_questions.append(row)
        self.rec.emit("scenario_private_question", **row)

    def finish_private_answers_if_ready(self, stage, asked):
        reason = None
        if self.required_fields and all(
            k in self.profile for k in self.required_fields
        ):
            reason = "required_information_delivered_waited_for_reply"
        elif stage.get("observe_only"):
            reason = "clarification_cap_reached_missing_retained"
        elif not asked and not stage.get("other_question"):
            reason = "no_actual_private_question_skip_answer"
        if reason:
            while self.active and self.active["kind"] == "private_answer":
                self.activate_next(reason)
        return bool(reason)

    def information_efficiency(self):
        covered, first_complete = set(), None
        for i, row in enumerate(self.private_answers, 1):
            covered.update(row["fields"])
            if (
                self.required_fields
                and set(self.required_fields) <= covered
                and first_complete is None
            ):
                first_complete = i
        return {
            "required": self.required_fields,
            "questions": copy.deepcopy(self.private_questions),
            "answers": copy.deepcopy(self.private_answers),
            "answer_rounds": len(self.private_answers),
            "first_complete_answer_round": first_complete,
            "missing": [k for k in self.required_fields if k not in self.profile],
            "note": "Coverage counts fully transmitted answers only; task correctness needs separate review.",
        }

    def heard_reply(self, turn):
        self.exchanges.add(turn)

    def observe_reply(self, item_id, for_turn, text, heard_ns, complete=True):
        self.rec.emit(
            "scenario_reply_evidence",
            item_id=item_id,
            turn=for_turn,
            text=text,
            heard_ns=heard_ns,
            playback_complete=complete,
        )

    def update_review(self, met, request_id):
        if not self.last_sent:
            return
        key = self.last_sent["stage"]["id"]
        # The model may retract a previous judgment when newer evidence contradicts it.
        self.task_states[key]["met"] = copy.deepcopy(met)
        self.task_states[key]["review_request_id"] = request_id
        parent = self.last_sent["stage"].get("resume_parent")
        if parent and parent in self.task_states and "resume" in met:
            self.task_states[parent]["met"] = {"complete": met["resume"]}
            self.task_states[parent]["review_request_id"] = request_id
        self.reviews.append(
            {"task_id": key, "met": copy.deepcopy(met), "request_id": request_id}
        )
        self.rec.emit("scenario_semantic_review", **self.reviews[-1])

    def manifest(self):
        return {
            "protocol": self.protocol_version,
            "sampling_cursor": self.cursor,
            "total_stages": len(self.frames),
            "tasks": copy.deepcopy(self.task_states),
            "samples": copy.deepcopy(self.samples),
            "topics": copy.deepcopy(self.topics),
            "active_topic": self.active_topic,
            "profile": copy.deepcopy(self.profile),
            "error": self.error,
            "information_efficiency": self.information_efficiency(),
            "task_reference": (
                settlement_reference(
                    self.case["scenario_runtime"]["settlement_problem"], self.profile
                )
                if self.case["scenario_runtime"].get("settlement_problem")
                else None
            ),
            "criterion": "B semantic judgments; scenario sampling is separate and is not a success score",
        }

    async def close(self):
        return
