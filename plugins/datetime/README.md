# DateTime Plugin（openJiuwen）

获取当前时间，支持指定时区和多种输出格式。仅使用 Python 标准库，无需额外依赖与 API Key。

## 功能

| 工具 | 说明 |
|------|------|
| **get_current_time** | 获取当前时间，可指定时区与格式，返回格式化字符串及年月日等字段 |

## 安装

```bash
cd plugins/datetime
pip install -e .
```

## 参数说明

- **timezone**（可选）：IANA 时区名称，默认 `UTC`。
  常用：`Asia/Shanghai`、`America/New_York`、`Europe/London`、`Japan` 等。
- **format**（可选）：输出格式，默认 `datetime`。
  - 预设：`iso`、`datetime`（%Y-%m-%d %H:%M:%S）、`date`、`time`、`full`（中文完整）
  - 也可传入任意 strftime 格式，如 `%Y/%m/%d %H:%M`

## 返回示例

```json
{
  "current_time": "2025-03-14 18:30:00",
  "timezone": "Asia/Shanghai",
  "iso": "2025-03-14T18:30:00+08:00",
  "year": 2025,
  "month": 3,
  "day": 14,
  "hour": 18,
  "minute": 30,
  "second": 0,
  "weekday": "Friday"
}
```
