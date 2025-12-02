# Relation Radar 开发计划（按小步 PR 拆分）

版本：v0.1  
目的：将 `prd.md` 中的整体规划拆解为可执行的小步 PR，方便多人协作与逐步集成。

> 约定：  
> - PR 号格式：`PR-阶段-序号`，例如 `PR-0.1-01`。  
> - 每个 PR 尽量聚焦一个清晰目标，改动面可被一次 review 理解。  
> - 每个 PR 条目包含：目标 / 范围 / 主要修改点 / 验收标准。

---

## 阶段 v0.1：本地底座 & 最小可用（1–2 周）

目标：打好基础结构，先实现「本地数据库 + 手动录入 + 基础 CRUD + 最小 RAG + CLI 问答」。

### PR-0.1-01：项目初始化 & 依赖配置

- **目标**  
  - 初始化 Python 项目结构，配置基础依赖与运行方式。
- **范围**  
  - 填写 `requirements.txt`（仅核心依赖：sqlite/ORM、pydantic、langchain、向量库客户端等）。  
  - 在 `README.md` 中加入本地开发快速启动说明（安装依赖、运行脚本）。  
  - 适当完善 `config/settings.toml` 的基本结构（数据库路径、日志等级等）。
- **主要修改点**  
  - `requirements.txt`  
  - `README.md`  
  - `config/settings.toml`
- **验收标准**  
  - 新环境下：`python -m venv .venv && pip install -r requirements.txt` 能通过。  
  - README 中的本地安装步骤清晰可用。

---

### PR-0.1-02：数据库连接与基础 Schema（Person/Event/Relationship）

- **目标**  
  - 在 SQLite 上实现核心表结构与数据库初始化逻辑。
- **范围**  
  - 在 `backend/core/db.py` 中实现：  
    - 获取数据库连接的方法。  
    - 初始化/迁移入口（可简单版本，如首次建表）。  
  - 在 `backend/core/models.py` 中定义 Pydantic 数据模型：Person、Event、Relationship。  
  - 在 `scripts/init_db.py` 中实现初始化脚本（创建表结构）。
- **主要修改点**  
  - `backend/core/db.py`  
  - `backend/core/models.py`  
  - `scripts/init_db.py`
- **验收标准**  
  - 运行 `python scripts/init_db.py` 后，生成 SQLite 文件且包含三张表。  
  - 能在 SQLite 浏览器或简单查询中看到表结构正确（字段名与 PRD 对齐）。

---

### PR-0.1-03：Repository 层：Person / Event / Relationship CRUD

- **目标**  
  - 引入 Repository 层，封装基础 CRUD 操作，供后续服务与 CLI 使用。
- **范围**  
  - 在 `backend/core/repositories.py` 中实现：  
    - `PersonRepository`：create / get / list / update / delete。  
    - `EventRepository`：create / query by person_id + time / tags。  
    - `RelationshipRepository`：create / get / list by person / update / delete。  
  - 约定：返回值使用 Pydantic 模型（models.py）。
- **主要修改点**  
  - `backend/core/repositories.py`
- **验收标准**  
  - 新增简单测试脚本或 demo（可在 `scripts/import_sample_data.py` 中演示插入和查询）。  
  - 手动运行脚本后，可以看到插入的数据并正确查出。

---

### PR-0.1-04：基础 CLI：管理 Person 与 Event

- **目标**  
  - 提供最小 CLI，支持添加朋友、添加事件、查看某人的事件列表。
- **范围**  
  - 在 `frontend/cli/main.py` 中实现简单命令行：  
    - `add-person`：创建 Person。  
    - `add-event`：为一个/多个 Person 添加 Event（先手动输入结构化字段即可）。  
    - `list-events`：按 person_id + 时间范围列出事件。  
  - 使用 `backend/core/repositories.py` 作为数据访问层。
- **主要修改点**  
  - `frontend/cli/main.py`
- **验收标准**  
  - 从命令行运行 CLI，能完成上述三类操作，无未捕获异常。  
  - 插入数据后，用 SQLite 浏览器或脚本确认数据一致。

---

### PR-0.1-05：Embedding 管道与向量库封装

- **目标**  
  - 打通「文本 → 向量 → 存储」的基础流程，为后续 RAG 做准备。
- **范围**  
  - 在 `backend/rag/embeddings.py` 中封装一个 embedding 客户端接口：  
    - 暂可使用开源 Sentence-Transformers 或 mock（先实现接口，后替换具体模型）。  
  - 在 `backend/rag/vector_store.py` 中封装向量库：  
    - 选型：Chroma / FAISS / SQLite+向量字段，任意其一。  
    - 实现：插入文档 + 按向量检索 top_k。  
  - 与 Event 关联：约定 `embedding_id` 与向量库 doc_id 的映射方式。
- **主要修改点**  
  - `backend/rag/embeddings.py`  
  - `backend/rag/vector_store.py`
- **验收标准**  
  - 写一个小 demo（可放 `scripts/import_sample_data.py` 或单独脚本）：  
    - 插入几条文本 → 向量化 → 写入向量库。  
    - 输入查询向量，能返回最相近的文本。

---

### PR-0.1-06：最小 RAG 检索器与 QA Chain（无真实 LLM 也可）

- **目标**  
  - 实现基础 RAG 流：问题 → 检索 Event 片段 → 组装上下文 → 调用 LLM（可先 mock）。
- **范围**  
  - 在 `backend/rag/retriever.py` 中实现：  
    - 按 person_id / 时间范围 / 标签 过滤 + 向量检索。  
    - 返回格式化好的片段列表（字符串数组或文档对象）。  
  - 在 `backend/rag/chains.py` 中实现：  
    - 简单 QA 链：`question -> retriever -> prompt -> llm`。  
    - 使用 `backend/llm/local_client.py` 的一个占位/假实现（先返回拼接文本），为之后换成 Qwen 留接口。
- **主要修改点**  
  - `backend/rag/retriever.py`  
  - `backend/rag/chains.py`  
  - `backend/llm/local_client.py`（添加基础接口）
- **验收标准**  
  - 写一个 demo 函数或 CLI 子命令：  
    - 输入问题 + 指定 person_id，能打印出检索到的上下文和 mock 回答。  
  - 检索逻辑正确：只在该人的事件中查找。

---

### PR-0.1-07：ingest_manual：手动文本录入 → 事件 + 向量化

- **目标**  
  - 实现最小的信息抽取链（可以先用简化规则/占位模型），把手动文本转成 Event 并入库 + 向量化。
- **范围**  
  - 在 `backend/core/ingest.py` 中实现：  
    - `ingest_manual(person_ids, raw_text)`：  
      - 目前可先简单处理：使用一个非常基础的规则或 mock 抽取（例如全部作为 summary，emotion 空）。  
      - 创建 Event 记录，写入 SQLite，并调用 embeddings + vector_store 建立向量。  
  - 为之后接入 Qwen 抽取链预留接口（例如，保留一个 `extract_events(text) -> List[EventDraft]` 的占位函数）。
- **主要修改点**  
  - `backend/core/ingest.py`  
  - 可能少量调整 `backend/core/models.py`（如 EventDraft 数据结构）。
- **验收标准**  
  - 通过 CLI：传入一段文本，对某个 person 调用 ingest_manual，能在 DB 里看到一条新的 Event，向量库也有对应记录。

---

### PR-0.1-08：CLI 整合：基础问答（RAG + 本地占位 LLM）

- **目标**  
  - 在 CLI 中加入“问答”入口，从而打通最小端到端链路。
- **范围**  
  - 在 `frontend/cli/main.py` 中新增命令：  
    - `ask`：参数包括 question、可选 person_id / 时间范围 / 标签过滤。  
  - 内部调用 `backend/rag/chains.py` 中的 QA 链。  
  - 结果输出到终端（包括：模型回答 + 可选显示用到的片段摘要）。
- **主要修改点**  
  - `frontend/cli/main.py`
- **验收标准**  
  - 基于少量样例数据，能完成“录入 → 问答”的最小闭环。  
  - 即便 LLM 只是占位实现，也能看出检索片段与拼接结果。

---

## 阶段 v0.2：电脑端体验完善 & 多输入（1–2 周）

目标：让电脑端实际可用，支持 OCR/语音录入，“一人一本笔记本” Web 界面，以及提醒和反馈。

### PR-0.2-01：接入真实本地 LLM（Qwen 小模型）基础接口

- **目标**  
  - 用真实 Qwen 本地模型替换 `backend/llm/local_client.py` 的占位实现，至少支持信息抽取和简单问答。
- **范围**  
  - 在 `backend/llm/local_client.py` 中：  
    - 定义统一接口（如 `chat(messages)` / `generate(prompt)`）。  
    - 实现调用 Qwen 本地运行的逻辑（可基于 Transformers、Ollama 或其它部署方式，具体可在 README 中说明）。  
  - 在 `backend/llm/prompts.py` / `config/prompts/*.txt` 中填充真实 prompt 文本（抽取、QA 等）。
- **主要修改点**  
  - `backend/llm/local_client.py`  
  - `backend/llm/prompts.py`  
  - `config/prompts/*.txt`
- **验收标准**  
  - 本地能成功调用 Qwen 小模型，对一个简单问题/抽取任务生成合理输出。  
  - 抽取链、RAG QA 链不再依赖 mock。

---

### PR-0.2-02：信息抽取链（Qwen 驱动）：文本 → 多事件 JSON

- **目标**  
  - 将 ingest_manual 中的抽取逻辑换成由 Qwen 驱动的 JSON 输出，生成结构合理的 Event 对象。
- **范围**  
  - 在 `backend/core/ingest.py` 中：  
    - 实现 `extract_events(text)`：调用 Qwen + 抽取 prompt，输出 JSON 数组。  
    - 将 JSON 映射为内部 EventDraft / Event 数据模型。  
  - 对错误 JSON/异常情况做容错（回退到简单摘要）。
- **主要修改点**  
  - `backend/core/ingest.py`  
  - `backend/llm/prompts.py` / `config/prompts/extract_event.txt`
- **验收标准**  
  - 输入一段包含多条信息的文本（例如一段聊天总结），能生成多条 Event 并入库。  
  - JSON 结构符合 PRD 约定，关键字段（人物、时间、情绪、偏好等）可被抽取。

---

### PR-0.2-03：ingest_ocr：聊天截图 → OCR → 文本 → 抽取链

- **目标**  
  - 打通截图输入通道：从图片中获取文字并走现有抽取流水线。
- **范围**  
  - 在 `backend/core/ingest.py` 中实现：  
    - `ingest_ocr(person_ids, image_path)`：  
      - 使用 Tesseract/PaddleOCR 提取文本。  
      - 调用 `extract_events(text)` → Event 入库。  
  - 在 `requirements.txt` 中加入 OCR 依赖，并在 README 说明本地安装要求。
- **主要修改点**  
  - `backend/core/ingest.py`  
  - `requirements.txt`  
  - `README.md`（安装 OCR 的说明）
- **验收标准**  
  - 使用一张包含聊天内容的截图，运行 ingest_ocr 能生成对应事件记录。

---

### PR-0.2-04：ingest_audio：语音录音 → Whisper → 文本 → 抽取链

- **目标**  
  - 打通语音输入通道，将录音转文字并进入抽取流水线。
- **范围**  
  - 在 `backend/core/ingest.py` 中实现：  
    - `ingest_audio(person_ids, audio_path)`：  
      - 使用 Whisper（本地或 API）转写文本。  
      - 调用 `extract_events(text)` → Event 入库。  
  - 更新依赖与 README：说明 Whisper 部署/安装方式。
- **主要修改点**  
  - `backend/core/ingest.py`  
  - `requirements.txt`  
  - `README.md`
- **验收标准**  
  - 提供一个测试音频文件，能被正确转写并生成事件记录。

---

### PR-0.2-05：“一人一本笔记本” Web API：按人/时间/标签查询

- **目标**  
  - 为 Web 界面提供查看数据的 HTTP API。
- **范围**  
  - 在 `backend/api/service.py` 中实现：  
    - `GET /persons`：列出所有 Person（支持按标签过滤）。  
    - `GET /persons/{id}`：获取单个 Person 详情。  
    - `GET /persons/{id}/events`：支持分页、时间范围、标签过滤。  
  - 可以使用 FastAPI 或简单 Flask/Starlette 包装。  
  - 考虑返回格式友好给前端直接使用。
- **主要修改点**  
  - `backend/api/service.py`  
  - （可能新增）`backend/api/__main__.py` 或启动脚本
- **验收标准**  
  - 本地启动 API 服务后，通过 curl/Postman 能调通上述接口，并返回 JSON。

---

### PR-0.2-06：“一人一本笔记本” Web UI（第一版）

- **目标**  
  - 实现最基础的 Web 界面：朋友列表 + 单人时间线视图。
- **范围**  
  - 在 `frontend/web/app.py` 中：  
    - 实现左侧好友列表（调用 `/persons`）。  
    - 中间区域展示当前好友的时间线（调用 `/persons/{id}/events`）。  
    - 支持按时间范围和标签过滤。  
  - 技术栈可用 Streamlit/Gradio 或轻量前端 + 后端 API。
- **主要修改点**  
  - `frontend/web/app.py`
- **验收标准**  
  - 启动 Web 应用后，可以选择某个朋友并浏览其事件列表，界面清晰可操作。

---

### PR-0.2-07：提醒机制（生日/长时间未联系/情绪波动）

- **目标**  
  - 实现基础提醒逻辑（暂可输出到日志或 CLI）。
- **范围**  
  - 在 `backend/core/reminders.py` 中：  
    - 实现扫描逻辑：  
      - 找出近期生日/纪念日。  
      - 找出长时间未联系的 Person。  
      - 根据 Event.emotion 判断近期连续负面情绪。  
  - 在 `scripts/run_reminders.py` 中调用扫描函数并打印/返回提醒结果。  
  - 未来可以挂到定时任务（cron）或前端展示。
- **主要修改点**  
  - `backend/core/reminders.py`  
  - `scripts/run_reminders.py`
- **验收标准**  
  - 插入一些示例数据，运行脚本能看到合理的提醒输出。

---

### PR-0.2-08：反馈机制：记录用户对回答的评价

- **目标**  
  - 为端到端训练闭环打基础，记录用户对回答的反馈。
- **范围**  
  - 在 DB 中新增 Feedback 表（可通过简单 schema 迁移）：  
    - question / answer / used_context_ids / rating（准确/一般/不准/有风险） / created_at 等。  
  - 在 `backend/core/models.py`、`db.py`、`repositories.py` 中加入对应定义和 CRUD。  
  - 在 CLI 或 Web UI 中加一个简单入口标记上一条回答的质量。
- **主要修改点**  
  - `backend/core/models.py`  
  - `backend/core/db.py`（迁移或重建）  
  - `backend/core/repositories.py`  
  - 前端/CLI 对应位置
- **验收标准**  
  - 用户完成一次问答后，可以给出评分，数据被正确写入 Feedback 表。

---

## 阶段 v0.3：MCP + 远端增强 + 微调

目标：让远端大模型通过 MCP 使用本地数据、并开始训练本地小模型。

### PR-0.3-01：MCP Server 骨架与基本工具实现

- **目标**  
  - 在本地实现 MCP server，暴露基础工具。
- **范围**  
  - 在 `mcp_server/server.py` 中实现一个最小可运行的 MCP server。  
  - 在 `mcp_server/tools/` 中实现：  
    - `search_events`：包装现有 RAG 检索接口，返回精简片段。  
    - `get_person_summary`：调用画像链，返回人物摘要。  
    - `log_feedback`：记录远端调用产生的反馈。  
  - 明确工具输入/输出 JSON schema。
- **主要修改点**  
  - `mcp_server/server.py`  
  - `mcp_server/tools/*.py`
- **验收标准**  
  - 使用 MCP 客户端（或简单测试脚本）调用工具，能正确返回数据。

---

### PR-0.3-02：远端大模型集成（作为“教师模型”/增强推理）

- **目标**  
  - 让远端大模型可以通过 MCP 工具拿本地摘要，并输出高质量回答。
- **范围**  
  - 在 `backend/llm/remote_client.py` 中封装远端模型调用（REST/API）。  
  - 在 `backend/rag/chains.py` 中增加一个“增强模式”链：  
    - 使用 MCP 工具获取上下文摘要 → 远端大模型生成回答。  
  - 在 config 中增加模式开关（本地/混合/云增强）。
- **主要修改点**  
  - `backend/llm/remote_client.py`  
  - `backend/rag/chains.py`  
  - `config/settings.toml`
- **验收标准**  
  - 在开启增强模式时，能看到远端大模型参与生成的回答（同时仍由 MCP 控制数据范围）。

---

### PR-0.3-03：训练数据构建与 LoRA/QLoRA 训练脚本

- **目标**  
  - 将 Feedback、问题、事实片段、远端“理想答案”组合为训练样本，并提供一键启动的微调脚本。
- **范围**  
  - 在 `scripts/` 下新增：  
    - `build_training_dataset.py`：从 DB 抽取 Feedback + 相关 Event + 远端回答，生成训练 JSONL。  
    - `train_lora.py`：使用 Transformers + PEFT 对 Qwen 小模型进行 LoRA/QLoRA 训练。  
  - 在 README 增加训练说明（硬件需求、参数配置）。
- **主要修改点**  
  - `scripts/build_training_dataset.py`  
  - `scripts/train_lora.py`  
  - `README.md`
- **验收标准**  
  - 能从真实/样例数据生成训练集文件。  
  - 在具备 GPU 的环境下跑通一次小规模训练，并产出 LoRA 权重。

---

### PR-0.3-04：本地小模型加载 LoRA 权重 & 模型版本管理

- **目标**  
  - 支持加载训练好的 LoRA 权重，并在运行时选择模型版本。
- **范围**  
  - 在 `backend/llm/local_client.py` 中增加加载 LoRA 权重能力。  
  - 在 `config/settings.toml` 中加入模型路径/版本配置项（base 模型 + LoRA 权重路径）。  
  - 在运行时提供简单日志，说明当前使用的模型版本。
- **主要修改点**  
  - `backend/llm/local_client.py`  
  - `config/settings.toml`
- **验收标准**  
  - 替换不同 LoRA 权重后，回答风格/能力有明显差异（可通过验证集检查）。

---

## 阶段 v0.4+：图谱可视化 & 移动端探索（概略）

这一阶段的 PR 可以更粗粒度规划，后续再细化：

- PR-0.4-01：关系图谱构建（NetworkX + 简单可视化 API）。  
- PR-0.4-02：Web 前端展示关系图谱视图（按 Person 展开周边关系）。  
- PR-0.4-03：手机 App 原型（例如使用 React Native/Flutter），仅做“遥控器”模式，调本地/家用服务器 API。  
- PR-0.4-04：移动端小模型探索（1B/3B Qwen 小模型的集成方案设计文档）。

---

> 提示：  
> - 如果团队人数不多，可以按阶段优先级顺序执行；多人时可以适度并行（例如一人做后端 Repo + RAG，一人做 Web UI）。  
> - 每个 PR 在正式实现前，建议在 issue 中附上对应 `dev_plan.md` 条目链接，保持需求与实现的一致性。

