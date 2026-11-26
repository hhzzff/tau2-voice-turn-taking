# 本地用户模拟器接入与阶段验收

实现分支：`experiment/local-user-simulator`。原克隆目录保留 `main` 基线，
开发位于独立 worktree。按用户授权整理草稿 PR，完成后交由用户最终审阅。

## 实现边界

正式测试仅运行本分支代码，不导入旧 `dynamic_pi` 源码。原库只作为
迁移来源、阶段 debug 与历史对齐证据。模型服务可以复用，但必须保存
模型、配置、声音、样本和代码版本。当前 Python 3.12 已验证可以导入
X-ASR 与重采样依赖，因此先采用同一环境，避免不必要的 IPC。

τ² 的 provider、事件目录与消息协议继续复用；用户内核接入
`FullDuplexUser`。音频收发/播放按独立真实时钟推进，ASR、判断、表达和
TTS 不得冻结目标会话。模型太慢造成的迟到和错过窗口就是实验结果。

## 计划提交与关卡

| 阶段 | 提交范围 | 进入下一阶段前检查 |
|---|---|---|
| 1 | 受控场景、不可变音频、floor 与确定性 agenda | 19 类映射、素材哈希、P01/R06保持、P03只按实发披露与独立计算参考 |
| 2 | 流式用户内核与本地后端 | 无未来文本、ASR修订、过期动作取消、首音和实际发送分离 |
| 3 | τ² 真实时间驱动、目标连接与 CLI | 注入慢 B/TTS 时音频继续、网络/音频格式、回执、会话隔离、P03真实闭环 |
| 4 | 全量听测入口、分阶段指标与审阅报告 | 19条均从本分支运行、错误/无效样本保留、音轨与事件一致 |
| 5（必要时） | 真实测试发现的修复与针对性复测 | 每项修复有对应证据；最终验收绑定明确提交 |

每个阶段先完成检查再提交；提交后审阅该 diff、依赖边界和阶段目标，
审阅结论记录到 Git notes 和本地审计产物。对行为的实质修正单列提交，
不把失败记录覆盖成成功，也不反复重跑无关的旧库测试。

## 正式运行的证据

每个 run 保存提交 SHA、工作树状态、实际导入位置、场景/提示词/配置
哈希、目标配置、模型与声音信息。保存真实传输时间的双声道录音、独立
音轨、ASR观察、B输入输出和 usage、发送回执、阶段推进与控制来源。

结果分开显示刺激有效性、目标语义表现和时延；B的 `met`、场景推进、
通用 mock task reward 都不是独立成功分数。先完成 P03 冒烟，再全量
运行19条。网页采用外置压缩无损音频和按需加载，避免内嵌大文件。

## 当前记录

- 原基线：`0f634c66f9a90f573b6ca380229895fa992c391c`。
- 阶段 1：19类映射和16条素材已迁入（15个独立音频文件）；7项定向
  测试与 `make check-all` 通过。这一阶段只验策略与素材，真实结果见后续阶段。
- 原仓库的15项离线审阅测试不替代本次新增实现的验收。
- 阶段 2：用户内核与目标连接分离；异步 X-ASR、单 B 判断/表达、流式
  Qwen TTS、播放观察、发送回执和后台落盘已接入。14项定向测试和
  `make check-all`通过。首音前重新校验附和时只检查当前发言权/语音，
  不重复检查已预占的次数配额。
- 分支内真实 TTS→ASR验证：2.48秒音频，首个有效PCM约146.156ms；
  “饭钱实际付了二百八十八块”识别成“饭前实际付了二百八十八”，保留
  该识别误差。证据位于忽略目录
  `data/simulations/local-user-integration/stage2-audio/`。
- 阶段 3：新增 `LocalStreamingUser` 注册项和 `duplex_bench local` 显式入口，
  使用 τ² OpenAI provider 的连接/配置/发送/截断接口。接收、播放和发送
  独立运行；服务端 VAD 才触发播放截断，B 的 INTERRUPT 不直接取消目标。
  provider 保留原有参数默认行为，新增可选音色、输入转写语言和输出上限。
- 原共享转换器只支持 8k μ-law；在同一共享模块新增原生 PCM 转换器，
  避免损失带宽。16→24k 使用有状态线性插值（小于一个源样本的延迟），
  24→16k 使用 soxr 抗混叠滤波并在句末排空；不在 provider 内复制转换逻辑。
- P03真实冒烟 `stage3-smoke-01`：117.65秒正常结束，目标一次询问两笔
  金额，实际发送金额后得到小林→我57元、阿哲→我96元，与独立账本一致。
  金额答案只在目标明确询问后披露。普通收尾后保留目标回答窗口。
- 阶段3定向测试18项通过；其中慢B和慢TTS各1.1秒时收发、播放和ASR持续。
  通用 provider suite 已执行：77项因未配置各供应商测试凭据而跳过，
  不能视为77项通过；本阶段的实际OpenAI证据来自上述真实P03。

## 运行入口

安装：`uv sync --extra local-user --extra dev`。把目标密钥放入忽略的
`.env`，变量名 `OPENAI_REALTIME_API_KEY`。本地后端地址和音色默认值在
`tau2/config.py`；ASR权重通过参数指定，不绑定旧代码库。

```bash
python -m experiments.tau_voice.duplex_bench local \
  --asr-model-path /path/to/x-asr-160ms \
  --output data/simulations/local-user-review
```

可加 `--scenes P03` 做单例验证；省略则逐条运行19例。拒绝覆盖已有目录。
这是受控场景接入，不会悄悄替换原有动态用户构建器或领域任务评分。

本地剧本和执行器集中在 `src/experiments/tau_voice/local_user/`，其中用户
适配器继承 τ 的 `FullDuplexUser`，由实验入口显式注册。核心 τ 不反向导入
实验包；共享 provider 与 PCM 转换保留在核心。接口与生命周期约定见
[模块 README](../src/experiments/tau_voice/local_user/README.md)。

其他机器可通过 `--config` 指定 ASR、B/TTS 服务配置，格式见
[JSON 示例](../src/experiments/tau_voice/examples/local_user_config.json)。
配置内提供 `asr.model_path` 时可省略 `--asr-model-path`；显式参数优先。
校验拒绝未知字段，实际配置仍完整写入每次实验的 provenance。

生成便携听测页（可在测试过程中对已结束样例反复更新）：

```bash
python -m experiments.tau_voice.local_user_report data/simulations/local-user-review
python -m experiments.tau_voice.local_user_report data/simulations/local-user-review --serve --port 8796
```

页面默认播放外置 `mix.flac`，同时提供用户/助手分声道、时间戳对齐录音、
原始事件、请求输入输出、token数及各阶段p50/p95。页面数据直接嵌入HTML，
音频使用相对链接和 `preload=none`，整目录下载后也可直接打开。
语义结论默认待人工审阅；不会把正常结束、B的met或agenda推进算成成功。

内置本地服务支持HTTP Range，长录音可以按需跳转；不需要公网部署。

## 首轮全量与修复

首轮19例均从干净提交 `d598655` 运行，位于
`data/simulations/local-user-integration/20261125-all19-r1/`。
16例正常结束，I04/M01触及原时间上限，R05因目标先询问操作系统而未形成
噪声刺激。P03虽已计算出正确净转账，长解释仍触及目标2048输出token上限；
它是目标输出截断，不是本地9B上下文越界，也不应计作完整目标回答。

首轮暴露出收尾ASR尚未排空就被取消的问题。修复在实时音频时钟结束后，
只刷新已经播放的音频，并等待有界观察队列排空；未播缓冲仍不可见。
新增回归测试覆盖延迟最终识别，报告对历史缺少收尾确认的文本显式标注，
不拿服务端全文替代本地观察。清除未使用的WAV/路径包装，保留唯一落盘通道。

可用 `--max-seconds` 和 `--target-max-output-tokens` 为选定场景调整预算，
实际值保存到场景快照、哈希和服务端配置确认。超时/未触发使批处理返回非零，
原始失败记录不覆盖。补测限定为上述有证据的问题。

首音以20ms包和绝对采样幅度0.005阈值定位，不是RMS或音素检测，也不是
远端设备播放确认。流式表达请求总时长可能包含下游队列背压；评估纯响应速度
应结合首token/首可用文本/TTS首音/发送回执，不能仅用总时长推算解码速度。


## 历史听测验收（结构整改前，2026-11-26）

19个场景均已从干净集成提交完成真实gpt-realtime-1.5试验。首轮保留原样，
I04/R05/P03/M01针对性补测完成；最终选择19条均正常结束、目标无输出上限截断，
录音共52.01分钟。运行结束不代表语义能力全部通过。

[完整中文审阅、逐场景结论与逐轮延迟](local-user-integration-review.md)。
最终入口是 `data/simulations/local-user-integration/final-review/index.html`，
本地服务 `http://127.0.0.1:8796/`；原8795服务保留。最终目录复用结果链接，
离线拷贝须解引用（`cp -rL`）。真实Chromium已验证页面、媒体和file方式播放。

M01补测18分13秒、16阶段完整，三组回忆题实际听到的回答正确。本地B最大
判断输入只有1891 token，不能将它当成9B的超长上下文上限测试。
P03补测发现ASR“范前”使Python字段守卫误拒绝B已正确选中的饭钱字段，
因此多一次补问；这部分轮次不能归因目标模型。I03/I08等模拟用户措辞问题
也按原样标注，详见审阅报告。

最终页面支持可选 `review-notes.json`，每条日志审阅注释必须匹配对应
`summary.json` 的SHA256；不把旧结论套到新录音。首轮、补测和版本证据均可查看。
以上是当时的本地验收记录；后续规范整改、复验与草稿 PR 准备见下节。

## PR 前规范整改与复验（2026-11-26）

上述“历史听测验收”是结构整改前的历史录音，保留原文件和结论。
本节记录提交 PR 前的独立复验。实验已整体迁至
`src/experiments/tau_voice/local_user/`，保留本地剧本和执行器，
通过显式注册使用 τ 的 `FullDuplexUser`，核心不反向依赖实验代码。

最终运行代码提交为 `0ae4b9199e2f25a2961ec414d4260749733d4cf1`。
后续提交仅增加故障测试和审阅证据文档；不会把旧录音标成新提交运行。
[规范整改和测试审计](local-user-pr-audit.md)记录实现边界、生命周期修复、
Pydantic/类型/文档规范及核心和语音测试的基线失败比较。

新录音分为三组，均复用之前选定记录的场景与配置，原始失败不覆盖：

```bash
python -m experiments.tau_voice.duplex_bench local \
  --scenes I01,I02,I03,I05,I06,I07,I08,R01,R02,R03,R04,R06,P01,P02,P04 \
  --asr-model-path /path/to/x-asr-160ms \
  --output data/simulations/local-user-integration/pr-validation-20261126/standard

python -m experiments.tau_voice.duplex_bench local \
  --scenes I04,R05,P03 --max-seconds 300 --target-max-output-tokens 4096 \
  --asr-model-path /path/to/x-asr-160ms \
  --output data/simulations/local-user-integration/pr-validation-20261126/extended

python -m experiments.tau_voice.duplex_bench local \
  --scenes M01 --max-seconds 1800 --target-max-output-tokens 4096 \
  --asr-model-path /path/to/x-asr-160ms \
  --output data/simulations/local-user-integration/pr-validation-20261126/memory
```

本轮19条均正常结束，共47.89分钟；每条的场景和配置哈希与此前选定记录
一致，ASR观察已排空，目标无输出上限截断。正常结束不等于语义通过：
R01的附和触发重启、R03/R04回应旁人等问题按原样保留并标注。
P03在一次回答中发送两笔金额，净转账57/96元与独立账本一致。
M01为15分53秒、16阶段，三组回忆内容正确，但模拟用户中途重复了初始
预算，不能声称预算只暴露一次；本地B最大输入1688 token，也不代表9B
超长上下文能力已验证。详情见[机器可读证据](local-user-pr-validation.json)。

真实取消与断网测试2项通过；故障记录保留部分音频和摘要，并确认ASR、
后台任务、录音线程与连接释放。最终提交测试与已执行测试逐字节一致。

听测入口为
`data/simulations/local-user-integration/pr-validation-20261126/review/index.html`。
这是静态文件，不新开服务。其相邻场景目录是指向本次原始运行的相对链接，
离线复制整个听测目录时须解引用（`cp -rL`）。每条日志审阅注释绑定摘要哈希，
最终听感由用户审阅；不继承历史录音的语义判断。

取消和断网测试用真实 I01 播放音频后注入故障，检查部分录音、摘要、ASR
收尾和所有资源释放。它们不纳入正常场景的成功率。复现时提供新目录：

```bash
LOCAL_USER_LIVE_TEST_ENABLED=1 \
LOCAL_USER_ASR_MODEL_PATH=/path/to/x-asr-160ms \
LOCAL_USER_LIVE_OUTPUT=data/simulations/local-user-live-faults-new \
python -m pytest tests/test_streaming/test_local_user_live.py -q
```

PR 以 draft 交给用户最终审阅，不自动合并。完整核心/语音测试仍有在基线
复现的失败，未声称全套通过。所有实际结果和限制均随 PR 提交。
