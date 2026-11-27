# 在线验证复查（2026-11-26）

复查基于合并提交 `a51bf7b`，使用原开发目录 `.env` 中更新后的 OpenAI key。凭证不复制进代码或测试产物。

## Gemini：本地发包等待累积，不是模型回答超时

Provider 套件使用 200ms tick，并要求每次 `run_tick()` 的实际耗时不超过 300ms。这个配置与全局默认 100ms tick 独立。原失败发生在耗时断言，不能理解为 Gemini 没有生成回答。

实际复现：把网络发送替换为空函数，仍通过真实 adapter 的发包方法发送一块 200ms 音频。音频拆成 10 个 20ms 包，原实现每包发送后都执行 `sleep(20ms)`。当前 Windows 定时条件下，这次等待通常约 31ms，因此一块音频总耗时约 310～315ms。无网络也能复现，排除了模型推理和网络发送是这一现象的必要条件。

仅在诊断进程中临时请求 1ms Windows 定时精度后，同一方法降到约 203～207ms；诊断完成即恢复。修复没有修改系统定时设置，而是把相对等待改成按最初时间表计算剩余等待，避免累积定时误差及网络发送耗时。原 300ms 断言保持不变。

修复后，在默认 Windows 定时精度下，无网络实测为约 197～216ms；真实 Gemini 套件 **11/11 通过**，覆盖连接、重连、两种长度单轮、多轮、工具调用、时延及插话。

之前能完成对话并生成 Web demo，与当前严格时延断言失败不矛盾：对话和报告生成不强制每个 tick 都小于 300ms。当前证据没有追溯确定之前每次运行的系统定时精度或机器负载；不能仅凭生成了 demo 就推定当时满足这一性能条件。

## OpenAI：新 key 生效，连接问题消失

重新加载 `C:/code/tau2-bench/.env`，使用更新后的 `OPENAI_API_KEY`，并覆盖诊断进程内的旧配置。独立连接测试通过。

后续完整 Provider 套件中 **10/11 通过**，包括连接、重连、单轮、多轮、工具调用及实际插话检测。唯一失败为 `test_barge_in_baseline`：测试要求连续回答至少 5 秒，模型本次只生成约 4.2 秒。这一失败与认证、连接或 tick 超时无关；单独复测该项随后通过。没有放宽时长断言，也不能把首次完整运行记成 11/11。

## Grok：账户额度限制，代理与直连一致

读取 WebSocket 403 的响应正文，服务端明确返回：

> Your team ... has either used all available credits or reached its monthly spending limit.

分别使用系统默认代理和显式 `proxy=None` 连接，两者均返回相同的额度错误。另用同一个 key 请求 `https://api.x.ai/v1/models`，默认代理和直连也均返回相同额度错误。

因此，本次 403 已定位为团队余额或月度消费上限；不能凭当前响应进一步区分这两个额度条件。需要补充余额或提高消费上限后才能复验。没有更改系统代理、账户限额或购买服务。

## 测试汇总与证据

修复发包等待后的完整 Provider 运行：**21 passed、8 failed、44 skipped、4 errors**。其中 Gemini 11 项通过，OpenAI 10 项通过、1 项回答长度失败；xAI 7 项失败、4 项连接准备阶段错误。之后 OpenAI 唯一失败项独立复测通过，xAI 仍受额度限制。豆包未在本次复查中重测。

另外修复 `run_provider_suite.py`：JUnit 中的 `<error>` 原来会被明细误显示为 PASS，现在正确显示 ERROR 并保留错误详情。第一次修复后在线套件运行在这项汇总修复之前已启动，因此其旧日志中 xAI 的四个绿色标记是错误标记；后续独立运行的 XML 确认为连接准备失败。

新增 9 项离线回归：100ms/200ms 发包在粗粒度定时器与网络发送延迟下不累积额外等待、慢发送不增加无谓休眠、即时发送与空音频不等待，以及套件 ERROR 明细正确展示。

完整语音与流式离线回归：**608 passed、5 skipped、83 deselected**。Ruff lint、格式及 Git 空白检查通过。

原始记录保留在被忽略的 `data/simulations/merge_validation/`，不纳入代码提交：

- `online-recheck.xml` / `.log`：使用新 key 的首次连接和 Gemini 原始超时复现。
- `network-recheck.jsonl`：Grok 代理/直连及模型列表请求的脱敏响应。
- `pacing-after-fix.json`：默认 Windows 定时条件下修复后的发包耗时。
- `provider-suite-after-pacing.log`：修复后的完整在线套件。
- `remaining-online.xml` / `.log`：OpenAI 回答长度项复测与 xAI 连接错误明细。
- `voice-after-pacing.xml` / `.log`：完整离线回归。
