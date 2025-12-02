# Relation Radar

Relation Radar 是一个本地优先的 AI 关系助手，把每个朋友当作可以持续更新的“活数据库”，帮助你记录事实、理解情绪、避免踩雷。

当前仓库为私有仓库，主要用于原型开发与架构探索。

## Project Docs

- 产品需求文档：`prd.md`  
- 开发计划（按小步 PR 拆分）：`dev_plan.md`  
- 协作与开发规范：`CONTRIBUTING.md`

## Development (简要)

> 详细请阅读 `CONTRIBUTING.md`。

- 推荐环境：Python 3.10+  
- 安装依赖：
  - `pip install -r requirements.txt`
- 初始化数据库（后续实现后可用）：
  - `python scripts/init_db.py`
- 开发流程建议：
  - 从 `main` 切出分支（参考 `dev_plan.md` 中的 PR 编号）。  
  - 小步提交、小步 PR，合并前跑一遍基础检查。  
  - 重要行为变化时同步更新 `prd.md` / `dev_plan.md`。

