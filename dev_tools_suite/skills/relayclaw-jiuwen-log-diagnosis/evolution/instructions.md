# Instructions

> Auto-generated from evolutions.json. Do not edit directly.

### [ev_6d61d387] ## 大体积API日志的grep搜索策略
- API日志常超过1MB，单独用宽泛关键词（如 `invocation`、`thread`）grep 会返回数十万字符的无效输出，应避免
- 采用"时间窗口+关键词"组合搜索：先用精确时间范围（如 `08:2[8-9]`）限定行数，再叠加 invocationId/session_id 等具体ID过滤
- 若单次 grep 输出超过 5000 字符，说明搜索条件过宽，应追加过滤条件缩小范围后再分析，而非直接阅读全量结果

*Source: conversation_review | 2026-05-12T09:09:21.013803+00:00*

---
