# Maths Plugin（OpenJiuwen）

使用 `NumExpr` 在本地计算数学表达式的插件（无需 API Key）。

## 工具

- `eval_expression`：计算数学表达式结果。

## 安装

```bash
cd plugins/maths
pip install -e .
```

## 参数

- `expression`（必填）：数学表达式，例如：
  - `1+(2+3)*4`
  - `cos(60)`
  - `sqrt(2)`

## 返回

返回 `report`（可读字符串）以及 `result`（计算结果字符串）。

