# Relation Radar

Relation Radar 是一个本地优先的 AI 关系助手，把每个朋友当作可以持续更新的“活数据库”，帮助你记录事实、理解情绪、避免踩雷。

当前仓库为私有仓库，主要用于原型开发与架构探索。

## Project Docs

- 产品需求文档：`prd.md`  
- 开发计划（按小步 PR 拆分）：`dev_plan.md`  
- 协作与开发规范：`CONTRIBUTING.md`

## Development（简要）

> 详细流程请阅读 `CONTRIBUTING.md`。

### 1. 环境准备

- 推荐 Python 版本：3.10+
- 克隆仓库后，创建虚拟环境（示例）：
  - `python -m venv .venv`
  - `.\.venv\Scripts\activate`（Windows）或 `source .venv/bin/activate`（macOS/Linux）
- 安装依赖：
  - `pip install -r requirements.txt`

### 2. 本地运行（后续逐步完善）

- 初始化数据库（v0.1 完成后可用）：
  - `python scripts/init_db.py`
- 启动 CLI（完成基础命令后）：
  - `python -m frontend.cli.main --help`
- 启动 Web（实现后）：
  - `python -m frontend.web.app` 或按文档说明。

### 3. 开发流程建议

- 从 `main` 切出分支（参照 `dev_plan.md` 中的 PR 编号）：
  - 例如：`git checkout -b feature/pr-0.1-02-db-schema`
- 小步提交、小步 PR，合并前跑一遍基础检查（见 `CONTRIBUTING.md` 第 4 节）。
- 当出现重要行为变化时，记得同步更新：`prd.md` / `dev_plan.md` / `README.md`。

