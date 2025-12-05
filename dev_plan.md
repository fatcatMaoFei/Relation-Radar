# Relation Radar 开发计划（按小步 PR 拆分）

版本：v0.2（对齐当前仓库）  
目标：把 `prd.md` 里的整体愿景拆成一组“小而稳”的 PR，确保：
- 每个 PR 只做一件清晰的事，方便 review 和回滚；
- 每次改动都有最少一套「能跑通」的自检脚本；
- 计划始终和实际代码保持同步，方便多人协作。

约定：
- PR 编号形如：`PR-阶段-序号`，例如 `PR-0.2-07`；
- PR 描述要说明：对应条目、自检方式、行为变化；
- 本文件只记录已经规划/实施过的 PR，不当作日常任务看板。

---

## 阶段 v0.1：本地底座 & 最小可用

目标：先把「本地数据库 + 手动录入 + 基础 CRUD + RAG + CLI 问答」跑通。  
状态：本阶段所有 PR 均已完成。

### PR-0.1-01：项目初始化 & 依赖配置（已完成）
- 目标：搭好最小 Python 项目骨架和依赖列表。
- 范围：
  - `requirements.txt`：pydantic / langchain / chroma / sentence-transformers / fastapi 等；
  - `config/settings.toml`：数据库路径、向量模型、日志等级等；
  - 初始 README，后续开发说明已迁移到 `CONTRIBUTING.md`。
- 自检：
  - `pip install -r requirements.txt` 能成功；
  - 基础模块能被 Python 正常导入。

### PR-0.1-02：数据库连接与核心 Schema（Person / Event / Relationship）（已完成）
- 目标：在 SQLite 上实现核心表结构与初始化逻辑。
- 范围：
  - `backend/core/db.py`：获取连接、初始化建表；
  - `backend/core/models.py`：Pydantic 模型 `Person` / `Event` / `Relationship`；
  - `scripts/init_db.py`：一键初始化数据库脚本。
- 自检：
  - 运行 `python scripts/init_db.py` 后生成 SQLite 文件；
  - 简单查询能看到与 PRD 一致的字段。

### PR-0.1-03：Repository 层：Person / Event / Relationship CRUD（已完成）
- 目标：封装数据库访问，后续服务和 CLI 不直接写 SQL。
- 范围：
  - `backend/core/repositories.py`：
    - `PersonRepository`：create / get / list / update / delete；
    - `EventRepository`：create / `list_for_person`（支持时间范围 / limit）；
    - `RelationshipRepository`：create / get / list_for_person / update / delete；
  - `scripts/import_sample_data.py`：插入示例朋友和事件。
- 自检：
  - 跑示例脚本能插入若干 Person / Event / Relationship；
  - 手动查询 SQLite 数据与期望一致。

### PR-0.1-04：基础 CLI：管理 Person / Event（已完成）
- 目标：提供最小 CLI，能录入和查看数据。
- 范围：
  - `frontend/cli/main.py`：
    - `add-person` / `add-event`；
    - `list-persons` / `list-events`。
- 自检：
  - 本地运行 CLI 能完成上述操作且无未捕获异常；
  - 通过 SQLite 检查写入结果。

### PR-0.1-05：Embedding 管道与向量库封装（已完成）
- 目标：打通「文本 → 向量 → 存储」流程，为 RAG 做准备。
- 范围：
  - `backend/rag/embeddings.py`：embedding 客户端接口（真实模型 / mock 两种模式）；
  - `backend/rag/vector_store.py`：向量库（Chroma）封装，和 Event 建立关联。
- 自检：
  - demo 脚本可将若干文本向量化并写入向量库；
  - 能检索出语义相似文本。

### PR-0.1-06：最小 RAG 检索器 & QA Chain（已完成）
- 目标：实现基础 RAG 流：问题 → 检索 Event 片段 → prompt → LLM。
- 范围：
  - `backend/rag/retriever.py`：按 person / 时间 / 标签过滤并做向量检索；
  - `backend/rag/chains.py`：问答链，将检索结果拼接进 prompt；
  - `backend/llm/local_client.py`：本地 LLM 接口（支持 mock）。
- 自检：
  - demo 或 CLI 子命令能打印检索到的上下文和一个回答（即使是 mock）。

### PR-0.1-07：ingest_manual：手动文本录入 → 事件 + 向量（已完成）
- 目标：将“原始文本记录”转成结构化 Event，并落库 + 建索引。
- 范围：
  - `backend/core/ingest.py`：`ingest_manual` 管道（拆分 / 抽取 / 写入 / 索引）。
- 自检：
  - 用几条示例笔记跑 `ingest_manual`，能生成对应 Event，并在 RAG 中被检索到。

### PR-0.1-08：基础自检脚本与 CI 骨架（已完成）
- 目标：让每个 PR 都有最少一套“能跑通”的脚本，方便本地和 CI。
- 范围：
  - 若干 `scripts/test_*.py`；
  - GitHub Actions CI（compile / ruff 等基础检查）。
- 自检：
  - 本地运行脚本能通过；
  - 推到 GitHub 后 CI 绿灯。

---

## 阶段 v0.2：本地小模型 + 多模态录入 + Web 入口

目标：接入本地 Qwen 小模型，打通文本 / 截图 / 语音到 Event 的全链路，并提供 Web / 移动端可复用的 HTTP 后端。  
说明：本阶段逐步增强“第二大脑”能力，同时保持隐私与本地优先。

### PR-0.2-01：本地 LLM 接入（Qwen / Ollama）（已完成）
- 目标：用 Qwen2.5-3B（通过 Ollama 或本地推理服务）替换纯 mock LLM。
- 范围：
  - `backend/llm/local_client.py`：支持本地 Qwen、mock 两种模式；
  - `backend/llm/prompts.py`：抽取常用提示词模板。
- 自检：
  - `scripts/test_llm_local.py` 能在本地调用 Qwen 获得合理回答。

### PR-0.2-02：信息抽取链（Qwen 驱动）：文本 → 多事件 JSON（已完成）
- 目标：让系统能从一段聊天 / 笔记中抽取多条结构化 Event。
- 范围：
  - `backend/core/ingest.py`：`extract_events` 使用 Qwen 尝试生成 JSON；
  - 对 JSON 不规范的情况降级为简单规则抽取；
  - 提示词强调“只基于用户输入事实，不补充虚构内容”。
- 自检：
  - `scripts/test_extract_qwen.py` 覆盖多条示例输入，验证抽取结果合理；
  - 对坏 JSON 能自动 fallback，而不是直接异常。

### PR-0.2-03：ingest_ocr：聊天截图 → OCR → 文本 → 抽取链（已完成）
- 目标：让聊天截图可以像文本一样被纳入“第二大脑”。
- 范围：
  - `backend/core/ingest.py`：`ingest_ocr`，基于 Pillow + Tesseract；
  - `scripts/test_ingest_ocr.py`：单独验证 OCR 管道。
- 自检：
  - 对示例截图能成功识别出中文文本并写入事件；
  - 未安装 Tesseract 时给出明确错误提示，不影响其他功能。

### PR-0.2-04：ingest_audio：语音录音 → Whisper → 文本 → 抽取链（已完成）
- 目标：让语音记录同样能被结构化处理。
- 范围：
  - `backend/core/ingest.py`：`ingest_audio`，基于 openai-whisper + ffmpeg；
  - `scripts/test_ingest_audio.py`：验证语音到事件的链路。
- 自检：
  - 对示例音频文件能转写出文本并抽取成 Event；
  - Whisper / ffmpeg 缺失时给出清晰错误。

### PR-0.2-05：可调 RAG 检索范围（top_k）+ 产品 README（已完成）
- 目标：
  - 允许用户按需调整本次问答的检索条数（例如 5 / 10 / 30），满足“日常精简 + 偶尔大范围回顾”的需求；
  - README 变成更偏“产品介绍”和愿景说明的文档。
- 范围：
  - `backend/rag/retriever.py`：`top_k` 参数增加安全范围限制（如 [1, 50]）；
  - `backend/rag/chains.py`：问答链透传 `top_k`；
  - `frontend/cli/main.py`：`ask` 命令加入 `--top-k` 参数；
  - `README.md`：改为产品故事与使用场景介绍。
- 自检：
  - CLI 中用不同 `--top-k` 能明显看到上下文数量变化；
  - README 不再包含开发环境配置，开发说明移至 `CONTRIBUTING.md`。

### PR-0.2-06：“一人一本笔记本” Web API：按人 / 时间 / 标签查询（已完成）
- 目标：
  - 提供统一的 HTTP 后端，供 CLI / Web / 未来 iOS / Android 共用；
  - 对外暴露“一个人一本笔记本”的 JSON 视图（按人 / 时间 / 标签浏览）。
- 范围：
  - `backend/api/service.py`：
    - `GET /persons`：列出所有 Person，可按标签过滤；
    - `GET /persons/{id}`：获取单个 Person 详情；
    - `GET /persons/{id}/events`：按时间范围 / 标签 / limit 查询事件；
  - 保持接口与前端无框架耦合，作为统一后端契约；
  - `scripts/test_web_api.py`：用 FastAPI TestClient 做 200 / 404 冒烟测试。
- 自检：
  - 本地启动 API（或通过 `scripts/test_web_api.py` 间接启动）能正确返回 JSON；
  - 接口字段稳定，足以支撑 Web UI 和未来移动端直接调用。

### PR-0.2-07：“一人一本笔记本” Web UI（第一版）（已完成）
- 目标：
  - 实现最基础的 Web 界面：左侧朋友列表 + 中间单人时间线视图；
  - 从这一 PR 开始系统性地对 Web 交互做手动测试，为后续移动端沿用同一交互打样。
- 范围：
  - `frontend/web/app.py`（Streamlit）：
    - 调用 `GET /persons` 渲染朋友列表；
    - 调用 `GET /persons/{id}/events` 渲染时间线；
    - 支持按时间范围和标签过滤；
    - 提供 “Max events” 控件控制一次展示的事件条数；
    - 提供 `top_k` slider，占位同步 RAG 检索范围（值记录在 `st.session_state` 中，后续问答视图复用）。
  - 该 UI 只通过 Web API 获取数据，不直接访问数据库或向量库。
- 自检：
  - `python -m compileall backend frontend mcp_server scripts` 通过；
  - `ruff check backend frontend mcp_server scripts` 通过；
  - 启动 FastAPI（`python backend/api/service.py` 或 `uvicorn backend.api.service:app`）和
    `streamlit run frontend/web/app.py` 后，能在浏览器中选择朋友、按时间 / 标签过滤事件列表。

### PR-0.2-08：提醒机制（生日 / 长时间未联系 / 情绪波动）（已完成）
- 目标：实现基础提醒逻辑，先以 CLI / 日志方式输出，后续再接入 UI。
- 范围：
  - `backend/core/reminders.py`：
    - 扫描近期生日（默认 14 天内）；
    - 扫描长时间未记录互动的朋友（默认 90 天未有新事件）；
    - 扫描最近 30 天内连续出现负向情绪事件的朋友（基于若干关键词判断）。
  - `scripts/run_reminders.py`：运行扫描并在终端打印提醒建议。
- 自检：
  - `python -m compileall backend frontend mcp_server scripts` 通过；
  - `ruff check backend frontend mcp_server scripts` 通过；
  - 运行 `python scripts/run_reminders.py` 时能正常输出结果（当前示例数据可能没有提醒，至少不会报错）。

### PR-0.2-09：反馈机制：记录用户对回答的评价（待开发）
- 目标：为端到端训练闭环打基础，记录“这次回答好不好”的反馈。
- 范围：
  - 新增 Feedback 表：
    - question / answer / used_context_ids / rating（准确 / 不准 / 有风险） / created_at；
  - `backend/core/models.py` / `db.py` / `repositories.py`：对应模型与 CRUD；
  - CLI / Web UI：提供对上一条回答打分的入口。
- 自检：
  - 完成一次问答后可以进行评分，数据正确写入 Feedback 表；
  - （可选）`scripts/test_feedback_flow.py` 覆盖“问 → 答 → 评 → 查”的完整路径。

---

## 阶段 v0.3：MCP + 远端增强 + 本地微调（概略）

目标：通过 MCP 让远端大模型在“脱敏后的本地数据摘要”上做更强推理，并把远端大模型的回答作为“老师”，反哺本地小模型（LoRA / QLoRA 微调）。

> 具体 PR 编号会在进入 v0.3 前再细化，这里先锁定方向：

- MCP Server：
  - 封装本地数据库 / RAG / Web API，为远端大模型提供安全的“只读接口”；
  - 所有发送到远端的数据都做脱敏与裁剪，只保留必要事实。
- 端到端训练闭环：
  - 从问答 + Feedback + 使用的上下文，构造训练样本；
  - 远端大模型给出 “理想答案”，形成 `(question, facts, ideal_answer)` 三元组；
  - 定期用 LoRA / QLoRA 微调本地小模型，让其逐步接近用户自己的使用风格。
- 移动端（iOS / Android）探索：
  - 直接复用当前 Web API，先做简单原型（例如 Flutter / React Native）；
  - 保证所有能力都通过统一后端暴露，不需要为移动端额外复制业务逻辑。

---

## 测试节奏与统一后端策略（总结）

- 测试节奏：
  - CLI / 底层变更：至少跑一次相关 `scripts/test_*.py` 或核心命令；
  - LLM / 抽取 / RAG 变更：优先使用现有的 `scripts/test_ingest.py` / `scripts/test_rag.py`；
  - Web API：`scripts/test_web_api.py` 必须通过；
  - Web UI：按 `PR-0.2-07` 的 checklist 做手动冒烟，必要时补充简单脚本。
- 统一后端：
  - 所有前端（CLI / Web / iOS / Android / MCP 工具）都只调用统一的 Python 服务 / HTTP API；
  - 新能力优先在后端暴露，再由各前端接入，避免出现“只在某一端可用”的分裂实现。

