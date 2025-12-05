# Relation Radar 开发计划（按小步 PR 拆分）

版本：v0.2（对齐当前进度）  
目的：将 `prd.md` 中的整体规划拆解为可执行的小步 PR，方便多人协作与逐步集成。

约定：
- PR 号格式：`PR-阶段-序号`，例如 `PR-0.1-03`。  
- 每个 PR 尽量只做一件清晰的事，改动面可被一次 review 理解。  
- 每个 PR 条目包含：目标 / 范围 / 主要修改点 / 验收标准。

---

## 阶段 v0.1：本地底座 & 最小可用

目标：打好基础结构，先实现「本地数据库 + 手动录入 + 基础 CRUD + RAG + CLI 问答」。

### PR-0.1-01：项目初始化 & 依赖配置（已完成）

- 目标  
  - 初始化 Python 项目结构，配置基础依赖与运行方式。
- 范围  
  - 填写 `requirements.txt`（pydantic / langchain / chroma / sentence-transformers / fastapi 等）。  
  - 在 `README.md` 中加入最小开发说明（后续已迁移至 CONTRIBUTING）。  
  - 在 `config/settings.toml` 中定义基础配置（数据库路径、日志等级、向量模型等）。
- 主要修改点  
  - `requirements.txt`  
  - `README.md`  
  - `config/settings.toml`
- 验收标准  
  - 新环境下 `pip install -r requirements.txt` 能通过。  
  - README / CONTRIBUTING 中的本地安装步骤清晰可用。

---

### PR-0.1-02：数据库连接与基础 Schema（Person/Event/Relationship）（已完成）

- 目标  
  - 在 SQLite 上实现核心表结构与数据库初始化逻辑。
- 范围  
  - 在 `backend/core/db.py` 中实现：  
    - 获取数据库连接的方法。  
    - 初始化入口（建表 + 索引）。  
  - 在 `backend/core/models.py` 中定义 Pydantic 数据模型：Person、Event、Relationship。  
  - 在 `scripts/init_db.py` 中实现初始化脚本。
- 主要修改点  
  - `backend/core/db.py`  
  - `backend/core/models.py`  
  - `scripts/init_db.py`
- 验收标准  
  - 运行 `python scripts/init_db.py` 后，生成 SQLite 文件且包含三张主表。  
  - 简单查询能看到字段与 PRD 对齐。

---

### PR-0.1-03：Repository 层：Person / Event / Relationship CRUD（已完成）

- 目标  
  - 引入 Repository 层，封装基础 CRUD 操作，供后续服务 / CLI 使用。
- 范围  
  - 在 `backend/core/repositories.py` 中实现：  
    - `PersonRepository`：create / get / list / update / delete。  
    - `EventRepository`：create / list_for_person（支持时间范围 / limit）。  
    - `RelationshipRepository`：create / get / list_for_person / update / delete。  
  - 使用 JSON 序列化存储 tags / preferences / taboos。
- 主要修改点  
  - `backend/core/repositories.py`  
  - `scripts/import_sample_data.py`（示例数据）
- 验收标准  
  - 通过脚本插入/查询示例数据，验证 CRUD 正常工作。

---

### PR-0.1-04：基础 CLI：管理 Person / Event（已完成）

- 目标  
  - 提供最小 CLI，支持添加朋友、添加事件、查看某人的事件列表。
- 范围  
  - 在 `frontend/cli/main.py` 中实现命令：  
    - `add-person`  
    - `add-event`  
    - `list-events`  
    - `list-persons`
  - 使用 Repository 作为数据访问层。
- 主要修改点  
  - `frontend/cli/main.py`
- 验收标准  
  - 从命令行运行 CLI 能完成上述操作，无未捕获异常。  
  - 通过 SQLite 检查数据一致性。

---

### PR-0.1-05：Embedding 管道与向量库封装（已完成）

- 目标  
  - 打通「文本 → 向量 → 存储」的基础流程，为 RAG 做准备。
- 范围  
  - 在 `backend/rag/embeddings.py` 中封装 embedding 客户端接口（支持真实模型 / mock）。  
  - 在 `backend/rag/vector_store.py` 中封装向量库（Chroma）。  
  - 与 Event 关联：通过 `embedding_id` 与向量库文档 ID 关联。
- 主要修改点  
  - `backend/rag/embeddings.py`  
  - `backend/rag/vector_store.py`
- 验收标准  
  - demo 脚本可将若干文本向量化并写入向量库，随后能检索出相似文本。

---

### PR-0.1-06：最小 RAG 检索器 & QA Chain（已完成）

- 目标  
  - 实现基础 RAG 流：问题 → 检索 Event 片段 → 组装上下文 → 调用 LLM（mock/本地）。
- 范围  
  - 在 `backend/rag/retriever.py` 中实现：  
    - 按 person_id / 时间范围 / 标签过滤 + 向量检索。  
  - 在 `backend/rag/chains.py` 中实现 QA 链：  
    - `question -> retriever -> prompt -> llm`。
  - 在 `backend/llm/local_client.py` 中提供 mock LLM 接口。
- 主要修改点  
  - `backend/rag/retriever.py`  
  - `backend/rag/chains.py`  
  - `backend/llm/local_client.py`
- 验收标准  
  - demo 或 CLI 子命令输入问题 + person_id，能打印检索上下文和回答（即使是 mock）。

---

### PR-0.1-07：ingest_manual：手动文本录入 → 事件 + 向量（已完成）

- 目标  
  - 实现最小的信息抽取链，把手动文本转成 Event 并入库 + 向量化。
- 范围  
  - 在 `backend/core/ingest.py` 中实现：  
    - `TextExtractor`：规则版信息抽取器。  
    - `extract_events(text)`：抽取事件草稿（后续由 Qwen 增强）。  
    - `ingest_manual(person_ids, raw_text, auto_index)`：写 DB + 建向量索引。
  - 如果抽取失败则给出合理错误提示。
- 主要修改点  
  - `backend/core/ingest.py`  
  - `scripts/test_ingest.py`
- 验收标准  
  - 通过脚本：传入文本，对某个 person 调用 ingest_manual，DB 和向量库都有对应记录。

---

### PR-0.1-08：CLI 整合：基础问答（RAG + 本地 LLM）（已完成）

- 目标  
  - 在 CLI 中加入“问答”入口，打通最小端到端链路。
- 范围  
  - 在 `frontend/cli/main.py` 中新增命令：  
    - `ask "问题" --person-id 1 --top-k 5`。  
  - 内部调用 `backend/rag/chains.ask_question`。
- 主要修改点  
  - `frontend/cli/main.py`  
  - `scripts/test_rag.py`
- 验收标准  
  - 结合前面 PR，能完成“录入一条文本 → 通过 ask 问一个问题 → 得到基于记录的回答”的闭环。

---

## 阶段 v0.2：电脑端体验完善 & 多输入

目标：让电脑端实际可用，支持 Qwen、小模型抽取、OCR / 语音录入，“一人一本笔记本” Web 方向，以及提醒和反馈。

### PR-0.2-01：接入真实本地 LLM（Qwen 小模型）（已完成）

- 目标  
  - 用真实 Qwen 本地模型替换 local LLM 的占位实现，支持信息抽取和问答。
- 范围  
  - 在 `backend/llm/local_client.py` 中：  
    - 统一接口：`generate(prompt)` / `chat(messages)`。  
    - 支持两种模式：mock / qwen（通过 Ollama 调用 Qwen2.5:3B）。  
  - 在 `backend/llm/prompts.py` 和 `config/prompts/*.txt` 中填充抽取和 QA 的 prompt 模板。
- 主要修改点  
  - `backend/llm/local_client.py`  
  - `backend/llm/prompts.py`  
  - `config/prompts/*.txt`  
  - `scripts/test_llm_local.py`
- 验收标准  
  - 本地在 Qwen 模式下对简单问题和抽取任务返回合理输出。  
  - mock / qwen 两种模式都能跑通测试脚本。

---

### PR-0.2-02：信息抽取链（Qwen 驱动）：文本 → 多事件 JSON（已完成）

- 目标  
  - 将 `extract_events` 升级为 Qwen 驱动，生成结构化 JSON 事件列表。
- 范围  
  - 在 `backend/core/ingest.py` 中：  
    - 使用 Qwen + 抽取 prompt 返回 JSON 数组。  
    - 将 JSON 映射为 `EventDraft` / `Event`。  
    - 对 JSON 解析错误和异常情况做容错（回退规则抽取）。
- 主要修改点  
  - `backend/core/ingest.py`  
  - `backend/llm/prompts.py` / `config/prompts/extract_event.txt`  
  - `scripts/test_extract_qwen.py`
- 验收标准  
  - 输入包含多条信息的长文本，能生成合理的 1–N 个事件；  
  - 异常 JSON 不会导致崩溃，而是回退到规则抽取。

---

### PR-0.2-03：ingest_ocr：聊天截图 → OCR → 文本 → 抽取链（已完成）

- 目标  
  - 打通截图输入通道：从图片中获取文字并走现有抽取流水线。
- 范围  
  - 在 `backend/core/ingest.py` 中实现 `ingest_ocr(person_ids, image_path)`：  
    - 使用 Tesseract（pytesseract + Pillow）提取文本。  
    - 调用 `ingest_manual` 完成抽取 + 入库 + 建索引。  
  - 在 `requirements.txt` 中加入 OCR 依赖，并在 README 中提到安装要求。
- 主要修改点  
  - `backend/core/ingest.py`  
  - `requirements.txt`  
  - `scripts/test_ingest_ocr.py`
- 验收标准  
  - 使用一张包含聊天内容的截图，运行脚本可以生成对应事件记录；  
  - 失败时给出清晰的错误提示（未安装 Tesseract / 文本为空等）。

---

### PR-0.2-04：ingest_audio：语音录音 → Whisper → 文本 → 抽取链（已完成）

- 目标  
  - 打通语音输入通道，将录音转文字并进入抽取流水线。
- 范围  
  - 在 `backend/core/ingest.py` 中实现 `ingest_audio(person_ids, audio_path)`：  
    - 懒加载 Whisper（`openai-whisper`）和依赖。  
    - 对音频进行转写，文本为空时给出错误。  
    - 调用 `ingest_manual` 完成抽取 + 入库 + 建索引。  
  - 在 README 中说明 Whisper / ffmpeg 的安装方式。
- 主要修改点  
  - `backend/core/ingest.py`  
  - `scripts/test_ingest_audio.py`
- 验收标准  
  - 对示例音频跑通“语音 → 文本 → 抽取 → 入库”链路；  
  - 在未安装 Whisper / ffmpeg 时给出合理错误提示（不影响 CI）。

---

### PR-0.2-05：可调 RAG 检索范围 + 产品 README（已完成）

- 目标  
  - 允许用户在 CLI / 界面中调整本次问答使用的历史记录条数（top_k）；  
  - 将 README 调整为面向用户的产品介绍与愿景说明。
- 范围  
  - `backend/rag/retriever.py`：  
    - 对 `top_k` 做安全约束（例如将有效范围限制在 [1, 50]）。  
  - `backend/rag/chains.py`：  
    - 确保 QA 链 / chat 链接受 `top_k` 参数并向下传递。  
  - `frontend/cli/main.py`：  
    - `ask` 命令增加 `--top-k` 参数，默认 5，可调整。  
  - `README.md`：  
    - 改为产品和愿景向说明，不再包含虚拟环境/依赖安装等开发细节。
- 主要修改点  
  - `backend/rag/retriever.py`  
  - `backend/rag/chains.py`  
  - `frontend/cli/main.py`  
  - `README.md`
- 验收标准  
  - `ask` 命令中通过 `--top-k` 可明显感受到检索范围变化；  
  - README 更像产品介绍页，而不是开发文档。

---

### PR-0.2-06：“一人一本笔记本” Web API：按人 / 时间 / 标签查询（待开发）

- 目标  
  - 为 Web 界面提供查看数据的 HTTP API。
- 范围  
  - 在 `backend/api/service.py` 中实现：  
    - `GET /persons`：列出所有 Person（支持按标签过滤）；  
    - `GET /persons/{id}`：获取单个 Person 详情；  
    - `GET /persons/{id}/events`：支持分页、时间范围、标签过滤。  
  - 使用 FastAPI 或简单 Starlette 包装。  
  - 返回 JSON 结构友好，便于前端直接使用。
- 主要修改点  
  - `backend/api/service.py`  
  - 可能新增 `backend/api/__main__.py` 或启动脚本  
  - `scripts/test_web_api.py`（或轻量 pytest 用例）
- 验收标准  
  - 本地启动 API 服务后，通过 curl / Postman 能正确调用上述接口并返回 JSON；  
  - 脚本级测试覆盖主要 200 / 404 路径，并在 CI 中运行。

---

### PR-0.2-07：“一人一本笔记本” Web UI（第一版）（待规划）

- 目标  
  - 实现最基础的 Web 界面：朋友列表 + 单人时间线视图。
- 范围  
  - 在 `frontend/web/app.py` 中：  
    - 左侧好友列表（调用 `/persons`）。  
    - 中间区域展示当前好友的时间线（调用 `/persons/{id}/events`）。  
    - 支持按时间范围和标签过滤。  
  - 技术栈可用 Streamlit / Gradio 或轻量前端 + API。
- 主要修改点  
  - `frontend/web/app.py`
- 验收标准  
  - 启动 Web 应用后，可以选择某个朋友并浏览其事件列表，界面清晰可用；  
  - （可选）`scripts/smoke_web_timeline.py` 使用 `requests` 等做简单冒烟。

---

### PR-0.2-08：提醒机制（生日 / 长时间未联系 / 情绪波动）（待规划）

- 目标  
  - 实现基础提醒逻辑（可先输出到日志 / CLI）。
- 范围  
  - 在 `backend/core/reminders.py` 中：  
    - 实现扫描逻辑：  
      - 近期生日 / 纪念日；  
      - 长时间未联系的人；  
      - 根据 Event.emotion 判断近期连续负面情绪。  
  - 在 `scripts/run_reminders.py` 中调用扫描函数并打印结果。
- 主要修改点  
  - `backend/core/reminders.py`  
  - `scripts/run_reminders.py`
- 验收标准  
  - 插入一些示例数据，运行脚本能看到合理的提醒输出。

---

### PR-0.2-09：反馈机制：记录用户对回答的评价（待规划）

- 目标  
  - 为端到端训练闭环打基础，记录用户对回答的反馈。
- 范围  
  - 在 DB 中新增 Feedback 表：  
    - question / answer / used_context_ids / rating（准确/一般/不准/有风险） / created_at 等。  
  - 在 `backend/core/models.py`、`db.py`、`repositories.py` 中加入对应定义和 CRUD。  
  - 在 CLI / Web UI 中加入口标记上一条回答的质量。
- 主要修改点  
  - `backend/core/models.py`  
  - `backend/core/db.py`  
  - `backend/core/repositories.py`  
  - 前端/CLI 对应位置
- 验收标准  
  - 用户完成一次问答后，可以给出评分，数据被正确写入 Feedback 表；  
  - （可选）`scripts/test_feedback_flow.py` 覆盖“问 → 评 → 查”的完整路径。

---

## 阶段 v0.3：MCP + 远端增强 + 微调（概略）

目标：让远端大模型通过 MCP 使用本地数据，并开始训练本地小模型（LoRA/QLoRA）。

这里沿用之前的大致规划，具体内容可在正式进入 v0.3 前再细化。***
