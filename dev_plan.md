# Relation Radar 开发计划（按小步 PR 拆分）

版本：v0.2（对齐当前代码库）  
目标：把 `prd.md` 里的愿景拆成一组“小而稳”的 PR，保持：
- 每个 PR 只做一件清晰的事，方便 review / 回滚；
- 每次改动都有最少一套可以本地跑通的自检脚本；
- 计划随代码演进持续更新。

约定：
- PR 编号形如：`PR-阶段-序号`，例如 `PR-0.2-07`；
- PR 描述需要说明：目标 / 范围 / 自检方式；
- 本文件是路线图快照，不当作日常任务看板。

---

## 阶段 v0.1：本地底座 & 最小可用

目标：先把「本地数据库 + 手动录入 + 基础 CRUD + RAG + CLI 问答」跑通。  
状态：本阶段所有 PR 均已完成。

### PR-0.1-01：项目初始化 & 依赖配置（已完成）
- 搭好基础目录结构；写入 `requirements.txt`、`config/settings.toml`；
- README / CONTRIBUTING 里给出最小开发说明。

### PR-0.1-02：SQLite Schema（Person / Event / Relationship）（已完成）
- 在 `backend/core/db.py` 中定义数据库路径和建表逻辑；
- 在 `backend/core/models.py` 中定义 Pydantic 模型；
- 用 `scripts/init_db.py` 初始化数据库。

### PR-0.1-03：Repository 层 CRUD（已完成）
- `backend/core/repositories.py` 中实现 Person / Event / Relationship 仓库；
- `scripts/import_sample_data.py` 插入示例朋友和事件。

### PR-0.1-04：基础 CLI：管理 Person / Event（已完成）
- `frontend/cli/main.py` 中实现 `add-person` / `add-event` / `list-persons` / `list-events`；
- CLI 完整走通“建人 → 记事件 → 查看事件”。

### PR-0.1-05：Embedding 管道与向量库封装（已完成）
- `backend/rag/embeddings.py`：embedding 客户端（真实模型 / mock）；
- `backend/rag/vector_store.py`：Chroma 向量库封装，与 Event 建立关联。

### PR-0.1-06：最小 RAG 检索器 & QA Chain（已完成）
- `backend/rag/retriever.py`：按 person / 时间 / 标签过滤并做向量检索；
- `backend/rag/chains.py`：问题 → 检索 → prompt → LLM 的 QAChain；
- `backend/llm/local_client.py`：本地 LLM 接口（含 mock）。

### PR-0.1-07：ingest_manual：手动文本 → 事件 + 向量（已完成）
- `backend/core/ingest.py`：`ingest_manual` 管道（抽取 → 入库 → 建索引）；
- `scripts/test_ingest.py`：验证手动文本到 RAG 的链路。

### PR-0.1-08：基础自检脚本 & CI 骨架（已完成）
- 几个 `scripts/test_*.py` 脚本作为 PR 自检入口；
- GitHub Actions 跑 `python -m compileall` 和 `ruff check`。

---

## 阶段 v0.2：本地小模型 + 多模态录入 + Web 入口

目标：接入本地 Qwen 小模型，打通文本 / 截图 / 语音到 Event 的全链路，并提供 Web / 移动端可复用的 HTTP 后端。  
状态：v0.2 计划中的 0.2‑01…0.2‑10 已完成。

### PR-0.2-01：本地 LLM 接入（Qwen / Ollama）（已完成）
- `backend/llm/local_client.py` 支持 `RELATION_RADAR_LLM_MODE=mock|qwen`；
- `scripts/test_llm_local.py` 用于本地测试 Qwen 是否连通。

### PR-0.2-02：信息抽取链：文本 → 多事件 JSON（已完成）
- `backend/core/ingest.py`：`extract_events` 用 Qwen 尝试产出 JSON，失败时回退到规则抽取；
- `backend/llm/prompts.py`：抽出抽取用 prompt 模板；
- `scripts/test_extract_qwen.py` 评估抽取效果。

### PR-0.2-03：ingest_ocr：聊天截图 → OCR → 文本 → 抽取链（已完成）
- `backend/core/ingest.py`：`ingest_ocr`（Pillow + Tesseract）；
- `scripts/test_ingest_ocr.py`：验证截图到事件链路。

### PR-0.2-04：ingest_audio：语音录音 → Whisper → 文本 → 抽取链（已完成）
- `backend/core/ingest.py`：`ingest_audio`（whisper + ffmpeg）；
- `scripts/test_ingest_audio.py`：验证音频到事件链路。

### PR-0.2-05：可调 RAG 检索范围（top_k）+ 产品 README（已完成）
- `backend/rag/retriever.py`：对 `top_k` 做 [1, 50] 的安全约束；
- `backend/rag/chains.py`、`frontend/cli/main.py`：`ask` 支持 `--top-k`；
- `README.md`：重写为产品介绍，而非开发文档。

### PR-0.2-06：“一人一本笔记本” Web API（已完成）
- `backend/api/service.py`：  
  - `GET /persons`、`GET /persons/{id}`、`GET /persons/{id}/events`；  
  - 响应结构适配未来 Web / iOS / Android。
- `scripts/test_web_api.py`：用 FastAPI TestClient 做 200 / 404 冒烟测试。

### PR-0.2-07：“一人一本笔记本” Web UI（已完成）
- `frontend/web/app.py`：Streamlit UI：左侧朋友列表 + 中间时间线视图；
- 支持按时间 / 标签过滤事件列表；
- 预留 `top_k` slider，用于后续问答检索范围选择。

### PR-0.2-08：提醒机制（生日 / 长时间未联系 / 情绪波动）（已完成）
- `backend/core/reminders.py`：扫描生日、久未互动、连续负面情绪事件；
- `scripts/run_reminders.py`：打印提醒建议，用于日常自查。

### PR-0.2-09：反馈机制：记录用户对回答的评价（已完成）
- 新增 Feedback 表：question / answer / used_context_ids / rating / created_at；
- `backend/core/models.py`、`db.py`、`repositories.py`：Feedback 模型 + 仓库；
- CLI `ask`：问答结束后可选择 1/2/3 评分并写入 Feedback；
- `scripts/test_feedback_flow.py`：验证写入 & 读取最近反馈。

### PR-0.2-10：Web 端问答 + 文本录入（最小版）（已完成）
- `backend/api/service.py`：  
  - `POST /persons/{id}/events`：纯文本 → `ingest_manual` → 事件 + 向量；  
  - `POST /persons/{id}/ask`：封装 RAG + LLM，返回答案和使用的 event_id 列表。  
- `frontend/web/app.py`：  
  - 时间线上方增加“Add a new note”折叠区，可以为当前朋友录入纯文本事件；  
  - 时间线下方增加 “Ask a question about this friend”，使用 `top_k` slider 控制检索范围；  
  - 在回答下方固定展示提醒文案：“Based on your recorded notes…”。  

---

## 阶段 v0.3：MCP + 远端增强 + 本地微调（规划中）

目标：通过 MCP 让远端大模型在“脱敏后的本地数据摘要”上做更强推理，同时把远端大模型的回答作为“老师”，反哺本地小模型（LoRA / QLoRA 微调），并系统化整理提示词体系。

以下 PR 编号暂定，进入 v0.3 前可以再微调。

### PR-0.3-01：MCP Server 基础实现（待规划）
- 目标：让远端大模型可以通过 MCP 调用 Relation Radar 的本地能力，而不是直接访问数据库。  
- 范围：
  - `mcp_server/server.py`：MCP server 启动入口，注册工具；
  - `mcp_server/tools/search_events.py`：封装按人 / 时间 / 标签 / 问句检索事件；
  - `mcp_server/tools/get_person_summary.py`：输出某个朋友的概要画像（基于现有 RAG + LLM）；
  - `mcp_server/tools/log_feedback.py`：将远端模型生成的回答和评估写入 Feedback 表。  
- 自检：  
  - 本地用简单 client 调用 MCP server，能完成“查事件 / 查概要 / 记反馈”三件事；
  - MCP 不返回原始敏感文本，只返回必要摘要。

### PR-0.3-02：Teacher 模式 & 训练数据流水线（待规划）
- 目标：让远端大模型在脱敏数据上生成“理想答案”，形成本地微调数据集。  
- 范围：
  - 设计一套专门给远端模型用的工具调用 prompt（强调：只能通过工具看数据、必须引用事实、要加风险提示）；
  - `scripts/build_teacher_dataset.py`：  
    - 输入：若干问题 + 本地事实（通过 MCP 工具查到的摘要）；  
    - 输出：`(question, facts, ideal_answer)` JSONL，用于训练；  
  - 与 Feedback 表打通：优先选用户标为“accurate”的样本进入数据集。  
- 自检：  
  - 抽样打印数据集中的若干条，确认回答风格贴近你的偏好、且有引用事实。

### PR-0.3-03：提示词体系整理 & A/B 测试（待规划）
- 目标：系统整理并验证提示词（prompt）设计，包括本地 Qwen 与远端 teacher 两侧。  
- 范围：
  - `backend/llm/prompts.py`：整理为清晰的多模板结构：  
    - 抽取类（extract_events）；  
    - 问答类（单人 / 多人 / 冲突规避 / 礼物推荐）；  
    - teacher 工具调用类（面向远端大模型）。  
  - `scripts/test_prompts_regression.py`（名称占位）：  
    - 固定一组典型用例（吃饭偏好、送礼、情绪关怀、多人场景）；  
    - 对比不同 prompt 版本的回答差异，并打印简要评分提示（人工主观评估）。  
- 微调方向（提示词层面）：  
  - 强调“只基于你记录的事实，不自行揣测”；  
  - 回答结构尽量短、先给结论再解释理由；  
  - 对可能冒犯 / 高风险建议（例如隐私、冲突升级）要主动给出提醒，而不是直接指导。

### PR-0.3-04：本地小模型微调（LoRA/QLoRA）与回归测试（待规划）
- 目标：在本地用 teacher 数据 + 用户反馈对小模型做一次完整的 LoRA/QLoRA 微调，并建立简单回归测试。  
- 范围：
  - `scripts/train_lora_qwen.py`：  
    - 使用 Hugging Face Transformers + peft，在你已有的 Qwen2.5-3B 基础上做参数高效微调；  
    - 支持从 JSONL 数据集中加载 `(question, facts, answer)`，按 batch 训练。  
  - `scripts/compare_llm_before_after.py`：  
    - 对同一组固定问题，分别调用“原始本地模型”和“微调后的模型”；  
    - 把两边回答并排打印，方便人工对比（准确度 / 风格 / 安全性）。  
- 微调方向（模型层面）：  
  - 更好地记住“一个人多条记录”里的偏好与忌讳，减少答非所问；  
  - 更稳定地复用时间、场景信息（例如“冬天”“生日”“上次吵架”）；  
  - 在用户标为“risky”的场景中学会收敛：多给提醒、少给激进建议。

---

## 测试节奏与统一后端策略（小结）

- 测试节奏：
  - CLI / 底层变更：至少跑一次相关 `scripts/test_*.py` 或核心命令；
  - LLM / 抽取 / RAG 变更：优先使用 `scripts/test_ingest.py`、`scripts/test_rag.py`、`scripts/test_llm_local.py`；
  - Web API：`scripts/test_web_api.py` 必须通过；
  - Web UI：按 `PR-0.2-07`、`PR-0.2-10` 的 checklist 做手动冒烟；
  - v0.3 阶段：再加 teacher 数据构建脚本和训练回归脚本作为自检。
- 统一后端原则：
  - 所有前端（CLI / Web / iOS / Android / MCP 工具）都只调用统一的 Python 服务 / HTTP API；
  - 新能力优先在后端暴露接口，再由各前端接入，避免“只在某个前端可用”的分裂实现。

