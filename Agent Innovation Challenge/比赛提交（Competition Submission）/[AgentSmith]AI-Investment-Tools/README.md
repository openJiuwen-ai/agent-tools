# AI 投资分析师

基于 FastAPI 的 AI 驱动投资分析服务，集成 18 位专业投资分析师的智慧，涵盖价值投资、成长投资、技术分析和风险管理等多个维度。

## 功能特性

### 投资风格分析师 (12 种)

| 分析师 | 投资风格 | 核心理念 |
|--------|----------|----------|
| 价值投资分析师 | 巴菲特风格 | 护城河、安全边际、长期持有 |
| 深度价值分析师 | 格雷厄姆风格 | 净净法、格雷厄姆数、深度价值挖掘 |
| 增长创新分析师 | 颠覆性技术 | 专注科技创新和商业模式变革 |
| 成长价值分析师 | Peter Lynch 风格 | PEG 比率、可理解性、十倍股寻找 |
| 质量成长分析师 | Phil Fisher 风格 | 管理层评估、创新能力、长期成长潜力 |
| 理性价值分析师 | Charlie Munger 风格 | 优质生意、合理价格、多学科思维 |
| 逆向投资分析师 | Michael Burry 风格 | 寻找被忽视的低估资产、市场失效 |
| 宏观趋势分析师 | Druckenmiller 风格 | 宏观趋势跟踪、大周期把握 |
| 估值专家分析师 | Damodaran 风格 | DCF 估值、内在价值计算 |
| 激进投资分析师 | Ackman 风格 | 价值释放、催化剂挖掘、积极股东 |
| 克隆投资分析师 | Pabrai 风格 | 价值克隆、低风险高不确定性 |
| 新兴市场增长分析师 | Jhunjhunwala 风格 | 消费升级、新兴市场机会 |

### 技术分析师 (6 种)

| 分析师 | 分析方向 |
|--------|----------|
| 技术分析师 | 图表形态、技术指标、价格趋势 |
| 基本面分析师 | 财务报表分析、盈利质量评估 |
| 增长分析师 | 收入和盈利增长趋势分析 |
| 估值分析师 | P/E、P/B、EV/EBITDA 多重估值 |
| 新闻情绪分析师 | 新闻文本情绪分析、舆论导向 |
| 市场情绪分析师 | 内部交易分析、价格动量分析 |

### 决策服务

| 服务 | 功能 | LLM 调用 |
|------|------|----------|
| 风险管理服务 | 基于波动率的持仓限制调整 | 否 |
| 投资组合管理 | 交易决策、仓位管理、风险控制 | 是 |

## 技术栈

- **Python**: >= 3.11
- **FastAPI**: 高性能异步 Web 框架
- **Pydantic**: 数据验证和序列化
- **Uvicorn**: ASGI 服务器

## 快速开始

### 环境准备

```bash
# 克隆项目
git clone https://github.com/your-username/ai-investment-analyst.git
cd ai-investment-analyst

# 安装 uv（推荐方式）
pip install uv

# 使用 uv 创建虚拟环境并安装依赖
uv sync
```

### 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入必要的 API Key
```

环境变量说明：

| 变量名 | 说明 | 是否必需 |
|--------|------|----------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | 是（部分功能） |
| `ANTHROPIC_API_KEY` | Anthropic API 密钥 | 是（部分功能） |
| `HOST` | 服务器地址 | 否，默认 0.0.0.0 |
| `PORT` | 服务器端口 | 否，默认 8000 |

### 启动服务

```bash
# 开发模式（自动重载）
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 生产模式
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 运行测试

```bash
uv run pytest
```

## API 使用

### 服务端点

服务启动后，可通过以下地址访问 API 文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 核心接口示例

#### 1. 获取分析师列表

```http
GET /analysts/list
```

返回所有可用的投资风格分析师和技术分析师列表。

#### 2. 价值投资分析

```http
GET /analysts/value-investor/prompt?ticker=AAPL&end_date=2024-12-31
```

获取巴菲特风格的价值投资分析提示词。

#### 3. 技术分析

```http
GET /analysts/technical/prompt?ticker=AAPL&start_date=2024-01-01&end_date=2024-12-31
```

获取技术分析提示词，包含图表形态和技术指标分析。

#### 4. 风险管理分析

```http
POST /decision/risk-management/analyze
Content-Type: application/json

{
  "ticker": "AAPL",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "cash": 100000
}
```

基于历史波动率计算持仓限制。

#### 5. 投资组合管理决策

```http
POST /decision/portfolio-management/prompt
Content-Type: application/json

{
  "ticker": "AAPL",
  "cash": 100000,
  "current_price": 180.5,
  "position_limit": 0.2,
  "signals": "{\"signal\":\"buy\",\"confidence\":0.85,\"reasoning\":\"...\",\"analyst_name\":\"价值投资分析师\"}|{\"signal\":\"hold\",\"confidence\":0.7,\"reasoning\":\"...\",\"analyst_name\":\"技术分析师\"}"
}
```

基于多个分析师信号生成交易决策。

### 数据获取接口

| 接口 | 功能 |
|------|------|
| `GET /data/prices` | 获取历史价格数据 |
| `GET /data/financial-metrics` | 获取财务指标 |
| `GET /data/insider-trades` | 获取内部交易数据 |
| `GET /data/news` | 获取公司新闻 |
| `POST /data/line-items` | 获取财务明细项目 |
| `GET /data/market-cap` | 获取市值数据 |

## 项目结构

```
ai-investment-analyst/
├── app/
│   ├── main.py                  # FastAPI 应用入口
│   ├── api/                     # API 路由
│   │   ├── analysts.py          # 投资分析师接口
│   │   ├── data.py              # 数据获取接口
│   │   └── decision.py          # 决策服务接口
│   ├── models/                  # Pydantic 数据模型
│   │   ├── analyst.py           # 分析师相关模型
│   │   └── data.py              # 数据相关模型
│   └── services/                # 业务逻辑服务
│       ├── data_fetching.py     # 数据获取
│       ├── line_items.py        # 财务明细处理
│       ├── risk_management.py   # 风险管理
│       ├── portfolio_management.py  # 投资组合管理
│       ├── value_investor_analysis.py     # 价值投资分析
│       ├── deep_value_analysis.py         # 深度价值分析
│       ├── growth_innovation_analysis.py   # 增长创新分析
│       ├── growth_value_analysis.py       # 成长价值分析
│       ├── quality_growth_analysis.py     # 质量成长分析
│       ├── rational_value_analysis.py     # 理性价值分析
│       ├── contrarian_analysis.py         # 逆向投资分析
│       ├── macro_trend_analysis.py        # 宏观趋势分析
│       ├── valuation_expert_analysis.py   # 估值专家分析
│       ├── activist_analysis.py           # 激进投资分析
│       ├── clone_investor_analysis.py     # 克隆投资分析
│       ├── emerging_growth_analysis.py    # 新兴市场增长分析
│       ├── technical_analysis.py          # 技术分析
│       ├── fundamentals_analysis.py       # 基本面分析
│       ├── growth_analysis.py             # 增长分析
│       ├── valuation_analysis.py          # 估值分析
│       └── sentiment_analysis.py          # 情绪分析
├── tests/                          # 测试文件
├── pyproject.toml                  # 项目依赖配置
├── .env.example                    # 环境变量模板
└── README.md                       # 项目说明文档
```

## 健康检查

```http
GET /
# 返回: {"message": "AI Investment Analyst API", "version": "0.1.0"}

GET /health
# 返回: {"status": "healthy"}
```

## 开发指南

### 添加新的分析师

1. 在 `app/services/` 下创建新的分析服务文件
2. 实现分析函数并返回结构化的分析结果
3. 在 `app/api/analysts.py` 中添加对应的 API 端点
4. 更新 `/analysts/list` 接口的返回数据

### 运行开发服务器

```bash
# 开发模式（代码变更自动重载）
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
